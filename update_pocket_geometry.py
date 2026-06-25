import os
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModel
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def extract_pocket_full_geometry(protein_pdb, pocket_pdb, tokenizer, model, device):
    parser = PDBParser(QUIET=True)
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
        raise ValueError("fail")

    max_len = 1022
    if len(full_seq) > max_len:
        full_seq = full_seq[:max_len]

    inputs = tokenizer(full_seq, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    full_embeddings = outputs.last_hidden_state[0, 1:-1, :]

    pocket_structure = parser.get_structure("pocket", pocket_pdb)
    pocket_indices = []
    ca_coords = []
    sc_coords = []
    sc_vectors = []

    backbone_atoms = {'N', 'CA', 'C', 'O'}

    for chain in pocket_structure.get_chains():
        chain_id = chain.get_id()
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            res_key = (chain_id, residue.id)
            if res_key in res_id_to_idx and 'CA' in residue:
                seq_idx = res_id_to_idx[res_key]
                if seq_idx < max_len:
                    ca_coord = residue['CA'].get_coord()

                    sc_atoms = [atom.get_coord() for atom in residue if
                                atom.get_name() not in backbone_atoms and atom.element != 'H']

                    if sc_atoms:
                        sc_centroid = np.mean(sc_atoms, axis=0)
                    else:
                        sc_centroid = ca_coord

                    sc_vec = sc_centroid - ca_coord

                    pocket_indices.append(seq_idx)
                    ca_coords.append(ca_coord)
                    sc_coords.append(sc_centroid)
                    sc_vectors.append(sc_vec)

    if not pocket_indices:
        raise ValueError("fail")

    pocket_indices = torch.tensor(pocket_indices, dtype=torch.long, device=full_embeddings.device)
    pocket_embeddings = torch.index_select(full_embeddings, 0, pocket_indices)

    ca_tensor = torch.tensor(np.array(ca_coords), dtype=torch.float32)
    sc_tensor = torch.tensor(np.array(sc_coords), dtype=torch.float32)
    vec_tensor = torch.tensor(np.array(sc_vectors), dtype=torch.float32)

    return pocket_embeddings.cpu(), ca_tensor.cpu(), sc_tensor.cpu(), vec_tensor.cpu()


def process_dataset(source_cache_dir, target_cache_dir, raw_root_dir, tokenizer, model, device):
    if not os.path.exists(source_cache_dir) or not os.path.exists(raw_root_dir):
        print(f"skip: {source_cache_dir}")
        return

    os.makedirs(target_cache_dir, exist_ok=True)
    pt_files = [f for f in os.listdir(source_cache_dir) if f.endswith('.pt')]

    print(f" {source_cache_dir} to {target_cache_dir} ...")
    success_count = 0

    for pt_file in tqdm(pt_files):
        pdb_id = pt_file.replace('.pt', '')
        source_pt_path = os.path.join(source_cache_dir, pt_file)
        target_pt_path = os.path.join(target_cache_dir, pt_file)

        protein_path = os.path.join(raw_root_dir, pdb_id, f"{pdb_id}_protein.pdb")
        pocket_path = os.path.join(raw_root_dir, pdb_id, f"{pdb_id}_pocket.pdb")

        if not os.path.exists(protein_path) or not os.path.exists(pocket_path):
            continue

        try:
            data = torch.load(source_pt_path, weights_only=False)

            target_x, target_pos, target_sc_pos, target_sc_vec = extract_pocket_full_geometry(
                protein_path, pocket_path, tokenizer, model, device
            )

            data['target_x'] = target_x
            data['target_pos'] = target_pos
            data['target_sc_pos'] = target_sc_pos
            data['target_sc_vec'] = target_sc_vec

            torch.save(data, target_pt_path)
            success_count += 1
        except Exception as e:
            pass

    print(f"successed : {success_count}/{len(pt_files)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_name = "facebook/esm2_t30_150M_UR50D"
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_name)
    esm_model = AutoModel.from_pretrained(esm_name).to(device)
    esm_model.eval()

    datasets = [
        ("cache_3d_train", "cache_3d_train_full", "train"),
        ("cache_3d_valid", "cache_3d_valid_full", "valid"),
        ("cache_3d_test", "cache_3d_test_full", "test"),
        ("cache_3d_test_2013", "cache_3d_test_2013_full", "test_2013"),
        ("cache_3d_test_2016", "cache_3d_test_2016_full", "test_2016"),
        ("cache_3d_test_hiq", "cache_3d_test_hiq_full", "test_hiq")
    ]

    for src_cache, tgt_cache, raw_dir in datasets:
        process_dataset(src_cache, tgt_cache, raw_dir, esm_tokenizer, esm_model, device)


if __name__ == "__main__":
    main()