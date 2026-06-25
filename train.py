import os
import random

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Cached3DDataset(Dataset):
    def __init__(self, csv_file, cache_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.cache_dir = cache_dir

        valid_ids = []
        for pid in self.data_frame['id']:
            if os.path.exists(os.path.join(self.cache_dir, f"{pid}.pt")):
                valid_ids.append(pid)
        self.data_frame = self.data_frame[self.data_frame['id'].isin(valid_ids)]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        pdb_id = str(self.data_frame.iloc[idx]['id'])
        pt_path = os.path.join(self.cache_dir, f"{pdb_id}.pt")
        data = torch.load(pt_path, weights_only=False)
        return data['ligand'], data['target_x'], data['target_pos'], data['target_sc_pos'], data['target_sc_vec'], data[
            'affinity']


def custom_collate(data_list):
    ligand_list = []
    target_list = []
    affinity_list = []

    for d in data_list:
        ligand_list.append(d[0])
        t_data = Data(x=d[1], pos=d[2], sc_pos=d[3], sc_vec=d[4])
        target_list.append(t_data)
        affinity_list.append(d[5])

    ligand_batch = Batch.from_data_list(ligand_list)
    target_batch = Batch.from_data_list(target_list)
    affinity_batch = torch.cat(affinity_list, dim=0)

    return ligand_batch, target_batch, affinity_batch


def compute_radius_graph(pos, r, batch):
    with torch.no_grad():
        pos_detach = pos.detach()
        dist = torch.cdist(pos_detach, pos_detach)
        mask = (dist < r) & (dist > 1e-5)
        batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        edge_index = (mask & batch_mask).nonzero(as_tuple=False).t().contiguous()
    return edge_index


class EGNNConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=False)
        )

    def forward(self, x, pos, edge_index):
        row, col = edge_index
        pos_f32 = pos.float()
        coord_diff = pos_f32[row] - pos_f32[col]
        sq_dist = torch.sum(coord_diff ** 2, dim=-1, keepdim=True).to(x.dtype)
        m_ij = self.edge_mlp(torch.cat([x[row], x[col], sq_dist], dim=-1))
        m_i = scatter(m_ij, row, dim=0, dim_size=x.size(0), reduce='mean')
        pos_update = coord_diff.to(x.dtype) * self.coord_mlp(m_ij)
        pos_update = torch.clamp(pos_update, min=-5.0, max=5.0)
        pos_i_update = scatter(pos_update, row, dim=0, dim_size=pos.size(0), reduce='mean')
        x_out = x + self.node_mlp(torch.cat([x, m_i], dim=-1))
        pos_out = pos + pos_i_update
        return x_out, pos_out


class EGNNExpert(nn.Module):
    def __init__(self, d_model=768, lpe_dim=8, num_layers=3, radius=4.0):
        super().__init__()
        self.radius = radius
        self.use_lpe = lpe_dim > 0
        if self.use_lpe:
            self.lpe_proj = nn.Linear(lpe_dim, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EGNNConv(d_model, d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, node_features, node_pos, batch_ptr, lpe_features=None, edge_index=None):
        h = node_features
        if self.use_lpe and lpe_features is not None:
            h = h + self.lpe_proj(lpe_features)
        pos = node_pos
        if edge_index is None:
            edge_index = compute_radius_graph(pos, r=self.radius, batch=batch_ptr)
        for conv in self.layers:
            h, pos = conv(x=h, pos=pos, edge_index=edge_index)
        h = self.norm(h)
        dense_x, mask = to_dense_batch(h, batch_ptr)
        dense_pos, _ = to_dense_batch(pos, batch_ptr)
        return dense_x, mask, dense_pos


class SpatialCrossAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=8, num_rbf=16, cutoff=8.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.cutoff = cutoff
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.centers = nn.Parameter(torch.linspace(0, 20, num_rbf))
        self.widths = nn.Parameter(torch.ones(num_rbf) * (20 / num_rbf))
        self.bias_proj = nn.Sequential(
            nn.Linear(num_rbf + 2, num_heads),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, pos_q, pos_k, key_padding_mask, query_padding_mask=None, vec_q=None,
                vec_k=None):
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape
        Q = self.q_proj(query).view(B, L_q, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, L_k, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, L_k, self.num_heads, self.d_head).transpose(1, 2)
        pos_q_f32 = pos_q.float()
        pos_k_f32 = pos_k.float()
        diff = pos_q_f32.unsqueeze(2) - pos_k_f32.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        dir_vec = diff / dist.unsqueeze(-1)

        if vec_q is not None:
            local_ref_q_norm = vec_q.float() / torch.sqrt(torch.sum(vec_q.float() ** 2, dim=-1, keepdim=True) + 1e-8)
        else:
            if query_padding_mask is not None:
                valid_mask_q = (~query_padding_mask).float().unsqueeze(-1)
                center_q = torch.sum(pos_q_f32 * valid_mask_q, dim=1, keepdim=True) / torch.clamp(
                    torch.sum(valid_mask_q, dim=1, keepdim=True), min=1e-9)
            else:
                center_q = pos_q_f32.mean(dim=1, keepdim=True)
            local_ref_q = pos_q_f32 - center_q
            local_ref_q_norm = local_ref_q / torch.sqrt(torch.sum(local_ref_q ** 2, dim=-1, keepdim=True) + 1e-8)

        if vec_k is not None:
            local_ref_k_norm = vec_k.float() / torch.sqrt(torch.sum(vec_k.float() ** 2, dim=-1, keepdim=True) + 1e-8)
        else:
            if key_padding_mask is not None:
                valid_mask_k = (~key_padding_mask).float().unsqueeze(-1)
                center_k = torch.sum(pos_k_f32 * valid_mask_k, dim=1, keepdim=True) / torch.clamp(
                    torch.sum(valid_mask_k, dim=1, keepdim=True), min=1e-9)
            else:
                center_k = pos_k_f32.mean(dim=1, keepdim=True)
            local_ref_k = pos_k_f32 - center_k
            local_ref_k_norm = local_ref_k / torch.sqrt(torch.sum(local_ref_k ** 2, dim=-1, keepdim=True) + 1e-8)

        cos_theta_q = torch.sum(dir_vec * local_ref_q_norm.unsqueeze(2), dim=-1)
        cos_theta_k = torch.sum((-dir_vec) * local_ref_k_norm.unsqueeze(1), dim=-1)
        dist = dist.to(query.dtype)
        cos_theta_q = cos_theta_q.to(query.dtype)
        cos_theta_k = cos_theta_k.to(query.dtype)
        rbf_feat = torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.widths ** 2))
        geom_feat = torch.cat([rbf_feat, cos_theta_q.unsqueeze(-1), cos_theta_k.unsqueeze(-1)], dim=-1)
        spatial_bias = self.bias_proj(geom_feat)
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = attn_weights + spatial_bias
        dist_mask = dist > self.cutoff
        dist_mask = dist_mask.unsqueeze(1).expand(B, self.num_heads, L_q, L_k)
        attn_weights = attn_weights.masked_fill(dist_mask, float('-inf'))
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.out_proj(out)


class DTIPredictor(nn.Module):
    def __init__(self, ligand_dim=768, target_dim=768, hidden_dims=[1024, 256], dropout_rate=0.4):
        super().__init__()
        input_dim = ligand_dim + target_dim
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ligand_features, target_features):
        fused_features = torch.cat([ligand_features, target_features], dim=1)
        return self.mlp(fused_features)


class DualTowerDTI(nn.Module):
    def __init__(self, hidden_dim=768, esm_dim=640, num_heads=8, dropout_rate=0.4):
        super().__init__()
        self.ligand_expert = EGNNExpert(d_model=hidden_dim, lpe_dim=8, radius=4.0)
        self.target_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.target_expert = EGNNExpert(d_model=hidden_dim, lpe_dim=0, radius=8.0)

        self.spatial_attn_ligand = SpatialCrossAttention(d_model=hidden_dim, num_heads=num_heads, cutoff=8.0)
        self.spatial_attn_target = SpatialCrossAttention(d_model=hidden_dim, num_heads=num_heads, cutoff=8.0)
        self.head = DTIPredictor(ligand_dim=hidden_dim, target_dim=hidden_dim, dropout_rate=dropout_rate)

    def forward(self, ligand_data, target_data):
        ligand_seq, ligand_mask, ligand_pos_dense = self.ligand_expert(
            node_features=ligand_data.x,
            node_pos=ligand_data.pos,
            batch_ptr=ligand_data.batch,
            lpe_features=ligand_data.lpe,
            edge_index=ligand_data.edge_index
        )
        t_x = self.target_proj(target_data.x)
        target_seq, target_mask, target_sc_pos_dense = self.target_expert(
            node_features=t_x,
            node_pos=target_data.sc_pos,
            batch_ptr=target_data.batch,
            lpe_features=None,
            edge_index=None
        )
        ligand_pad_mask = ~ligand_mask
        target_pad_mask = ~target_mask
        target_sc_vec_dense, _ = to_dense_batch(target_data.sc_vec, target_data.batch)

        ligand_cross = self.spatial_attn_ligand(
            query=ligand_seq, key=target_seq, value=target_seq,
            pos_q=ligand_pos_dense, pos_k=target_sc_pos_dense,
            key_padding_mask=target_pad_mask, query_padding_mask=ligand_pad_mask,
            vec_q=None, vec_k=target_sc_vec_dense
        )
        ligand_seq = ligand_seq + ligand_cross

        target_cross = self.spatial_attn_target(
            query=target_seq, key=ligand_seq, value=ligand_seq,
            pos_q=target_sc_pos_dense, pos_k=ligand_pos_dense,
            key_padding_mask=ligand_pad_mask, query_padding_mask=target_pad_mask,
            vec_q=target_sc_vec_dense, vec_k=None
        )
        target_seq = target_seq + target_cross

        ligand_mask_exp = ligand_mask.unsqueeze(-1).float()
        ligand_rep = torch.sum(ligand_seq * ligand_mask_exp, dim=1) / torch.clamp(torch.sum(ligand_mask_exp, dim=1),
                                                                                  min=1e-9)
        target_valid_mask = target_mask.unsqueeze(-1).float()
        target_rep = torch.sum(target_seq * target_valid_mask, dim=1) / torch.clamp(torch.sum(target_valid_mask, dim=1),
                                                                                    min=1e-9)

        pred = self.head(ligand_rep, target_rep)
        return pred, ligand_rep, target_rep


def tta_ensemble_evaluate(model_paths, device, num_tta=10, noise_scale=0.05):

    test_sets = [
        ["test_2016", "test_2016.csv", "cache_3d_test_2016_full"],
        ["test_hiq", "test_hiq.csv", "cache_3d_test_hiq_full"],
        ["test", "test.csv", "cache_3d_test_full"],
        ["test_2013", "test_2013.csv", "cache_3d_test_2013_full"]
    ]

    valid_paths = [p for p in model_paths if os.path.exists(p)]
    if not valid_paths:
        return

    for name, csv_file, cache_dir in test_sets:
        if not os.path.exists(csv_file) or not os.path.exists(cache_dir):
            continue

        dataset = Cached3DDataset(csv_file, cache_dir)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_collate, num_workers=16,
                                 pin_memory=True)
        all_models_preds, targets_list = [], []

        for path in valid_paths:
            model = DualTowerDTI().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            model_preds, targets_current = [], []

            with torch.no_grad():
                for ligand_b, target_b, affinity_b in tqdm(data_loader,
                                                           desc=f"{name} -> {path.split('_')[-1].split('.')[0]}",
                                                           leave=False):
                    ligand_b, target_b = ligand_b.to(device), target_b.to(device)
                    orig_lig_pos, orig_tgt_sc_pos = ligand_b.pos.clone(), target_b.sc_pos.clone()
                    batch_tta_preds = []

                    for _ in range(num_tta):
                        ligand_b.pos = orig_lig_pos + torch.randn_like(orig_lig_pos) * noise_scale
                        target_b.sc_pos = orig_tgt_sc_pos + torch.randn_like(orig_tgt_sc_pos) * noise_scale
                        with torch.amp.autocast('cuda'):
                            p, _, _ = model(ligand_b, target_b)
                            batch_tta_preds.append(p.view(-1).cpu().numpy())

                    ligand_b.pos, target_b.sc_pos = orig_lig_pos, orig_tgt_sc_pos
                    model_preds.extend(np.mean(batch_tta_preds, axis=0))
                    if len(all_models_preds) == 0:
                        targets_current.extend(affinity_b.view(-1).numpy())

            all_models_preds.append(model_preds)
            if len(all_models_preds) == 1:
                targets_list = targets_current

        ensemble_preds, targets_array = np.mean(all_models_preds, axis=0), np.array(targets_list)
        rmse = np.sqrt(np.mean((ensemble_preds - targets_array) ** 2))
        mae = np.mean(np.abs(ensemble_preds - targets_array))
        pearson_r, _ = pearsonr(ensemble_preds, targets_array)

        print(f"=== {name} Evaluate ===")
        print(f"RMSE:      {rmse:.4f}")
        print(f"MAE:       {mae:.4f}")
        print(f"Pearson R: {pearson_r:.4f}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Cached3DDataset("train.csv", "cache_3d_train_full")
    valid_dataset = Cached3DDataset("valid.csv", "cache_3d_valid_full")

    seeds = [52, 2407, 4888, 6234, 8026]
    num_epochs = 60
    noise_scale = 0.1
    cl_weight = 0.1
    temperature = 0.07
    saved_model_paths = []

    for seed in seeds:
        print(f"\n======================================")
        print(f"Training {seed} ...")
        print(f"======================================")
        set_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate, drop_last=True,
                                  num_workers=16, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate,
                                  drop_last=False, num_workers=16, pin_memory=True)

        model = DualTowerDTI().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda')

        best_valid_loss = float('inf')
        save_path = f"best_model_train23_seed_{seed}.pth"
        saved_model_paths.append(save_path)

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            pbar_train = tqdm(train_loader, desc=f"Seed {seed} | Epoch {epoch + 1}/{num_epochs} Train",
                              dynamic_ncols=True)

            for ligand_b, target_b, affinity_b in pbar_train:
                ligand_b, target_b, affinity_b = ligand_b.to(device), target_b.to(device), affinity_b.to(device)

                ligand_b.pos = ligand_b.pos + torch.randn_like(ligand_b.pos) * noise_scale
                target_b.sc_pos = target_b.sc_pos + torch.randn_like(target_b.sc_pos) * noise_scale

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    pred, lig_rep, tgt_rep = model(ligand_b, target_b)

                    loss_reg = criterion(pred.view(-1), affinity_b.view(-1))

                    lig_rep_norm = F.normalize(lig_rep, p=2, dim=-1)
                    tgt_rep_norm = F.normalize(tgt_rep, p=2, dim=-1)
                    logits = torch.matmul(lig_rep_norm, tgt_rep_norm.transpose(0, 1)) / temperature
                    labels = torch.arange(logits.size(0), device=device)
                    loss_cl = F.cross_entropy(logits, labels)

                    loss = loss_reg + cl_weight * loss_cl

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss_reg.item()
                pbar_train.set_postfix({'MSE': f"{loss_reg.item():.4f}", 'CL': f"{loss_cl.item():.4f}"})

            model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for ligand_b, target_b, affinity_b in valid_loader:
                    ligand_b, target_b, affinity_b = ligand_b.to(device), target_b.to(device), affinity_b.to(device)
                    with torch.amp.autocast('cuda'):
                        pred, _, _ = model(ligand_b, target_b)
                        loss = criterion(pred.view(-1), affinity_b.view(-1))
                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)
            scheduler.step(avg_valid_loss)

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), save_path)
                print(f" MSE : {best_valid_loss:.4f} saved")

    tta_ensemble_evaluate(saved_model_paths, device, num_tta=10, noise_scale=0.05)


if __name__ == "__main__":
    main()