import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import RDLogger
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data

warnings.simplefilter('ignore', PDBConstructionWarning)
RDLogger.DisableLog('rdApp.*')

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def compute_laplacian_positional_encoding(adj_matrix, k=8):
    degree = adj_matrix.sum(dim=1)
    d_inv_sqrt = torch.pow(degree.float(), -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    eye = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
    laplacian = eye - torch.mm(torch.mm(d_mat_inv_sqrt, adj_matrix.float()), d_mat_inv_sqrt)
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    num_nodes = adj_matrix.size(0)
    lpe_dim = min(k, num_nodes - 1)
    lpe = eigenvectors[:, 1:lpe_dim + 1]
    if lpe_dim < k:
        padding = torch.zeros(num_nodes, k - lpe_dim, device=adj_matrix.device)
        lpe = torch.cat([lpe, padding], dim=1)
    return lpe


def extract_ligand_3d_features(mol, tokenizer, model, device):
    num_atoms = mol.GetNumAtoms()
    smiles = Chem.MolToSmiles(mol)

    inputs = tokenizer(smiles, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    full_embeddings = outputs.last_hidden_state[0]

    atom_features = []
    for idx, token in enumerate(tokens):
        if token.startswith('<') and token.endswith('>'):
            continue
        if re.search(r'[a-zA-Z]', token):
            atom_features.append(full_embeddings[idx].unsqueeze(0))
    atom_tensor = torch.cat(atom_features, dim=0)

    if atom_tensor.shape[0] != num_atoms:
        raise ValueError("分词对齐断裂")

    conf = mol.GetConformer()
    positions = conf.GetPositions()

    brics_fragments = list(BRICS.FindBRICSBonds(mol))
    motif_indices_list = []
    if not brics_fragments:
        motif_indices_list.append(list(range(num_atoms)))
    else:
        broken_mol = BRICS.BreakBRICSBonds(mol)
        fragments = Chem.GetMolFrags(broken_mol, asMols=False)
        for frag in fragments:
            valid_indices = [idx for idx in frag if idx < num_atoms]
            if valid_indices:
                motif_indices_list.append(valid_indices)

    motif_tensors = []
    motif_coords = []
    for indices in motif_indices_list:
        motif_nodes = atom_tensor[indices, :]
        motif_representation = torch.mean(motif_nodes, dim=0)
        motif_tensors.append(motif_representation)
        motif_coords.append(np.mean(positions[indices], axis=0))

    final_motif_tensor = torch.stack(motif_tensors)
    all_nodes_features = torch.cat([atom_tensor, final_motif_tensor], dim=0)

    all_coords = np.vstack([positions, motif_coords])
    coords_tensor = torch.tensor(all_coords, dtype=torch.float32)

    num_motifs = final_motif_tensor.shape[0]
    total_nodes = num_atoms + num_motifs
    adj_matrix = torch.zeros((total_nodes, total_nodes), device=device)
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adj_matrix[u, v] = 1.0
        adj_matrix[v, u] = 1.0
    for motif_idx, indices in enumerate(motif_indices_list):
        global_motif_idx = num_atoms + motif_idx
        for atom_idx in indices:
            adj_matrix[atom_idx, global_motif_idx] = 1.0
            adj_matrix[global_motif_idx, atom_idx] = 1.0

    lpe_tensor = compute_laplacian_positional_encoding(adj_matrix, k=8)

    return all_nodes_features.cpu(), lpe_tensor.cpu(), coords_tensor.cpu()


def extract_pocket_3d_features(protein_pdb, pocket_pdb, tokenizer, model, device):
    parser = PDBParser()
    full_structure = parser.get_structure("protein", protein_pdb)
    full_seq = ""
    res_id_to_idx = {}
    idx = 0

    for chain in full_structure.get_chains():
        chain_id = chain.get_id()
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            resname = residue.get_resname()
            if resname in AA_3TO1:
                full_seq += AA_3TO1[resname]
                res_key = (chain_id, residue.id)
                res_id_to_idx[res_key] = idx
                idx += 1

    if not full_seq:
        raise ValueError("未能提取全长序列")

    max_len = 1022
    if len(full_seq) > max_len:
        full_seq = full_seq[:max_len]

    inputs = tokenizer(full_seq, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    full_embeddings = outputs.last_hidden_state[0, 1:-1, :]

    pocket_structure = parser.get_structure("pocket", pocket_pdb)
    pocket_indices = []
    pocket_coords = []

    for chain in pocket_structure.get_chains():
        chain_id = chain.get_id()
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            res_key = (chain_id, residue.id)
            if res_key in res_id_to_idx and 'CA' in residue:
                seq_idx = res_id_to_idx[res_key]
                if seq_idx < max_len:
                    pocket_indices.append(seq_idx)
                    pocket_coords.append(residue['CA'].get_coord())

    if not pocket_indices:
        raise ValueError("口袋残基坐标提取失败")

    pocket_indices = torch.tensor(pocket_indices, dtype=torch.long, device=full_embeddings.device)
    pocket_embeddings = torch.index_select(full_embeddings, 0, pocket_indices)
    pocket_coords_tensor = torch.tensor(np.array(pocket_coords), dtype=torch.float32)

    return pocket_embeddings.cpu(), pocket_coords_tensor.cpu()


def build_3d_caches():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("装载 MoLFormer 专家...")
    mol_name = "ibm-research/MoLFormer-XL-both-10pct"
    mol_tokenizer = AutoTokenizer.from_pretrained(mol_name, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(mol_name, deterministic_eval=True, trust_remote_code=True).to(device)
    mol_model.eval()

    print("装载 ESM-2 专家...")
    esm_name = "facebook/esm2_t30_150M_UR50D"
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_name)
    esm_model = AutoModel.from_pretrained(esm_name).to(device)
    esm_model.eval()

    datasets = [
        ("train.csv", "train", "cache_3d_train"),
        ("valid.csv", "valid", "cache_3d_valid"),
        ("test.csv", "test", "cache_3d_test"),
        ("test_2013.csv", "test_2013", "cache_3d_test_2013"),
        ("test_2016.csv", "test_2016", "cache_3d_test_2016"),
        ("test_hiq.csv", "test_hiq", "cache_3d_test_hiq")
    ]

    for csv_file, split_dir, cache_dir in datasets:
        if not os.path.exists(csv_file):
            continue
        os.makedirs(cache_dir, exist_ok=True)
        df = pd.read_csv(csv_file)

        for idx in tqdm(range(len(df)), desc=f"构建 {cache_dir}"):
            row = df.iloc[idx]
            pdb_id = str(row['id'])
            affinity = torch.tensor([float(row['affinity'])], dtype=torch.float32)
            save_path = os.path.join(cache_dir, f"{pdb_id}.pt")

            if os.path.exists(save_path):
                continue

            try:
                sdf_path = os.path.join(split_dir, pdb_id, f"{pdb_id}_ligand.sdf")
                mol2_path = os.path.join(split_dir, pdb_id, f"{pdb_id}_ligand.mol2")
                protein_path = os.path.join(split_dir, pdb_id, f"{pdb_id}_protein.pdb")
                pocket_path = os.path.join(split_dir, pdb_id, f"{pdb_id}_pocket.pdb")

                if not os.path.exists(protein_path):
                    continue

                mol = None
                if os.path.exists(sdf_path):
                    supplier = Chem.SDMolSupplier(sdf_path)
                    try:
                        mol = next(supplier)
                    except Exception:
                        pass
                if mol is None and os.path.exists(mol2_path):
                    mol = Chem.MolFromMol2File(mol2_path)
                if mol is None:
                    continue

                ligand_x, ligand_lpe, ligand_pos = extract_ligand_3d_features(mol, mol_tokenizer, mol_model, device)
                ligand_data = Data(x=ligand_x, lpe=ligand_lpe, pos=ligand_pos)

                target_esm, target_pos = extract_pocket_3d_features(protein_path, pocket_path, esm_tokenizer, esm_model,
                                                                    device)

                torch.save({
                    'ligand': ligand_data,
                    'target_x': target_esm,
                    'target_pos': target_pos,
                    'affinity': affinity
                }, save_path)

            except Exception:
                pass


if __name__ == "__main__":
    build_3d_caches()