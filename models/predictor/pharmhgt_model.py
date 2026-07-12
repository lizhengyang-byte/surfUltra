"""
pharmhgt_model.py — PharmHGT heterogeneous graph Transformer model for pCMC prediction.

Includes model definition (PharmHGTModel + sub-modules) and SMILES-to-PyG conversion.

Usage:
    from pharmhgt_model import load_pharmhgt_model, predict_pharmhgt
    model = load_pharmhgt_model("weights/pharmhgt_best_model.pth", device="cpu")
    pred = predict_pharmhgt(model, "CCO", device="cpu")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.data import HeteroData

from .featurizer import (
    get_atom_features, get_bond_features,
    get_pharmacophore_features, get_reaction_features,
    detect_surfactant, SURF_TYPE_TO_IDX,
)


# ===========================================================================
# Sub-modules
# ===========================================================================

class MultiViewCrossAttention(nn.Module):
    """Multi-View Cross Attention (Eq.1).

    Attention(Q, K, V) = Σ_p Ω_p · softmax(Q_p K_pᵀ / √d_k) · V_p
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # View weights Ω_p (α=atom, β=pharm, γ=react)
        self.view_weights = nn.Parameter(torch.ones(3) / 3.0)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_atom, x_pharm, x_react):
        """(N_atom, H) ← cross-attend to pharm(1,H) + react(1,H)."""
        N, H = x_atom.shape
        x_src = torch.cat([x_pharm, x_react], dim=0)  # (2, H)

        Q = self.q_proj(x_atom).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x_src).view(2, self.num_heads, self.head_dim)
        V = self.v_proj(x_src).view(2, self.num_heads, self.head_dim)

        # (h, N, 2) — no inplace ops
        attn = torch.einsum('nhd,shd->hns', Q, K) * self.scale

        view_w = F.softmax(self.view_weights, dim=0)
        attn_w = torch.empty_like(attn)
        attn_w[:, :, 0] = attn[:, :, 0] * view_w[1]  # β-pharm
        attn_w[:, :, 1] = attn[:, :, 1] * view_w[2]  # γ-reaction

        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.dropout(attn_w)

        out = torch.einsum('hns,shd->nhd', attn_w, V).reshape(N, H)
        out = self.out_proj(out)
        return out


class SimpleGNNLayer(nn.Module):
    """Simple GNN layer with explicit message passing — no inplace ops.

    Message:   MLP( cat(x_src, x_dst, e) )
    Aggregate: mean over incoming neighbors
    Update:    x + MLP(cat(x, agg))
    """

    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Message: MLP(x_src || x_dst || e)
        msg = self.msg_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        # Aggregate: mean over incoming (col) neighbors
        from torch_geometric.utils import scatter
        agg = scatter(msg, col, dim=0, dim_size=x.size(0), reduce='mean')
        # Update: residual + MLP
        upd = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return x + upd


# ===========================================================================
# PharmHGTModel
# ===========================================================================

class PharmHGTModel(nn.Module):
    """PharmHGT — Pharmacophoric-constrained Heterogeneous Graph Transformer.

    Architecture:
      1. Atom-level GNN (Gα)
      2. Pharmacophore encoding (Gβ)
      3. Surfactant attention (head/tail masks)
      4. MVMP — Multi-view message passing
      5. Hierarchical readout
      6. Output MLP → LogCMC
    """

    def __init__(self, atom_dim=55, bond_dim=14, pharm_dim=194, react_dim=34,
                 hidden_dim=256, num_layers=4, dropout=0.2, num_heads=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Feature projectors
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bond_embed = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.pharm_embed = nn.Sequential(
            nn.Linear(pharm_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.react_embed = nn.Sequential(
            nn.Linear(react_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # Surfactant attention
        self.head_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tail_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-view GNN layers
        self.atom_conv = nn.ModuleList()
        self.pharm_mlp = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in range(num_layers):
            self.atom_conv.append(
                SimpleGNNLayer(hidden_dim, hidden_dim, hidden_dim, dropout))
            self.pharm_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)))
            self.cross_attn.append(
                MultiViewCrossAttention(hidden_dim, num_heads, dropout))
            self.norm.append(nn.ModuleDict({
                'atom': nn.LayerNorm(hidden_dim),
                'pharm': nn.LayerNorm(hidden_dim),
                'react': nn.LayerNorm(hidden_dim),
            }))

        # Hierarchical readout
        self.fusion1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.readout_attn = nn.Linear(hidden_dim, 1)

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def _apply_surfactant_attention(self, h_atom, head_mask, tail_mask, batch):
        """Apply head/tail mask-guided attention."""
        n_atoms = h_atom.size(0)
        device = h_atom.device
        delta = torch.zeros_like(h_atom)

        for g in batch.unique():
            mask = (batch == g)
            hg = h_atom[mask]
            hm = head_mask[mask]
            tm = tail_mask[mask]

            head_proto = hg[hm].mean(dim=0, keepdim=True).tanh() if hm.any() else \
                torch.zeros(1, h_atom.size(1), device=device)
            tail_proto = hg[tm].mean(dim=0, keepdim=True).tanh() if tm.any() else \
                torch.zeros(1, h_atom.size(1), device=device)

            head_bias = self.head_proj(head_proto)
            tail_bias = self.tail_proj(tail_proto)

            hm_f = hm.float().unsqueeze(1)
            tm_f = tm.float().unsqueeze(1)
            delta_g = head_bias * hm_f + tail_bias * tm_f
            g_indices = mask.nonzero(as_tuple=True)[0]
            delta = delta.index_add(0, g_indices, delta_g)

        return h_atom + delta

    def forward(self, data):
        """Forward pass.

        Args:
            data: batched HeteroData
        Returns:
            logcmc_pred: (B,)
        """
        x_atom = data['atom'].x
        x_pharm = data['pharmacophore'].x
        x_react = data['reaction'].x

        edge_index = data['atom', 'bond', 'atom'].edge_index
        edge_attr = data['atom', 'bond', 'atom'].edge_attr

        head_mask = data['atom'].head_mask
        tail_mask = data['atom'].tail_mask
        batch = data['atom'].batch if hasattr(data['atom'], 'batch') else \
            torch.zeros(x_atom.size(0), dtype=torch.long, device=x_atom.device)

        n_graphs = x_pharm.size(0)

        # Embed
        h_atom = self.atom_embed(x_atom)
        h_pharm = self.pharm_embed(x_pharm)
        h_react = self.react_embed(x_react)
        h_bond = self.bond_embed(edge_attr)

        # Surfactant attention
        h_atom = self._apply_surfactant_attention(h_atom, head_mask, tail_mask, batch)

        # Per-graph atom counts
        atom_counts = []
        for g in range(n_graphs):
            atom_counts.append((batch == g).sum().item())

        # Multi-view message passing
        for i in range(self.num_layers):
            h_atom_in = h_atom
            h_atom = self.norm[i]['atom'](self.atom_conv[i](h_atom, edge_index, h_bond))

            h_pharm_in = h_pharm
            h_pharm = self.pharm_mlp[i](h_pharm)
            h_pharm = self.norm[i]['pharm'](h_pharm + h_pharm_in)

            h_atom_out_list = []
            start = 0
            for g in range(n_graphs):
                n_a = atom_counts[g]
                h_a = h_atom[start:start + n_a]
                h_p = h_pharm[g:g + 1]
                h_r = h_react[g:g + 1]
                h_a_out = self.cross_attn[i](h_a, h_p, h_r)
                h_atom_out_list.append(h_a_out)
                start += n_a
            h_atom = torch.cat(h_atom_out_list, dim=0)

        # Hierarchical readout
        preds = []
        start = 0
        for g in range(n_graphs):
            n_a = atom_counts[g]
            z_alpha = h_atom[start:start + n_a]
            z_beta = h_pharm[g:g + 1]
            z_gamma = h_react[g:g + 1]

            z_gamma_beta = self.fusion1(
                torch.cat([z_gamma, z_beta], dim=-1)).tanh()

            attn_scores = self.readout_attn(z_alpha)
            attn_w = F.softmax(attn_scores, dim=0)
            z_alpha_pooled = (z_alpha * attn_w).sum(dim=0, keepdim=True)

            z_fused = self.fusion2(
                torch.cat([z_gamma_beta, z_alpha_pooled], dim=-1)).relu()

            preds.append(self.output_mlp(z_fused).squeeze(-1))
            start += n_a

        return torch.cat(preds)


# ===========================================================================
# SMILES → PyG HeteroData
# ===========================================================================

def build_molecule_data(smiles: str):
    """Convert a SMILES string to a PyG HeteroData object for PharmHGT.

    Node types: 'atom' (55-dim), 'pharmacophore' (194-dim), 'reaction' (34-dim)
    Edge types: ('atom','bond','atom') with 14-dim features + cross-edges
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    n_atoms = mol.GetNumAtoms()

    # Atom features
    atom_feats = np.array([get_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Bond features (bidirectional)
    edge_index = []
    bond_feats = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_index.append([u, v])
        edge_index.append([v, u])
        bond_feats.append(bf)
        bond_feats.append(bf.copy())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    bond_feats = torch.tensor(np.array(bond_feats), dtype=torch.float32)

    # Pharmacophore & reaction features
    pharm_feats = torch.tensor(
        get_pharmacophore_features(mol).reshape(1, -1), dtype=torch.float32)
    react_feats = torch.tensor(
        get_reaction_features(mol).reshape(1, -1), dtype=torch.float32)

    # HeteroData
    data = HeteroData()
    data['atom'].x = torch.tensor(atom_feats, dtype=torch.float32)
    data['pharmacophore'].x = pharm_feats
    data['reaction'].x = react_feats

    data['atom', 'bond', 'atom'].edge_index = edge_index
    data['atom', 'bond', 'atom'].edge_attr = bond_feats

    # Cross edges
    data['pharmacophore', 'to_atom', 'atom'].edge_index = torch.tensor(
        [[0] * n_atoms, list(range(n_atoms))], dtype=torch.long)
    data['atom', 'to_pharmacophore', 'pharmacophore'].edge_index = torch.tensor(
        [list(range(n_atoms)), [0] * n_atoms], dtype=torch.long)
    data['reaction', 'to_atom', 'atom'].edge_index = torch.tensor(
        [[0] * n_atoms, list(range(n_atoms))], dtype=torch.long)
    data['atom', 'to_reaction', 'reaction'].edge_index = torch.tensor(
        [list(range(n_atoms)), [0] * n_atoms], dtype=torch.long)
    data['pharmacophore', 'to_reaction', 'reaction'].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long)
    data['reaction', 'to_pharmacophore', 'pharmacophore'].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long)

    # Surfactant info
    head_mask, tail_mask, surf_type = detect_surfactant(smiles)
    if len(head_mask) != n_atoms:
        head_mask = np.zeros(n_atoms, dtype=bool)
        tail_mask = np.zeros(n_atoms, dtype=bool)
    data['atom'].head_mask = torch.tensor(head_mask, dtype=torch.bool)
    data['atom'].tail_mask = torch.tensor(tail_mask, dtype=torch.bool)
    data.surf_type = surf_type
    data.surf_type_idx = SURF_TYPE_TO_IDX.get(surf_type, 2)

    return data


# ===========================================================================
# Load & Predict helpers
# ===========================================================================

def load_pharmhgt_model(ckpt_path: str, device: str = 'cpu'):
    """Load a trained PharmHGT model from a checkpoint file.

    The checkpoint must contain 'state_dict' and 'params' keys.
    'params' must include: hidden_dim, num_layers, dropout, num_heads.

    Returns:
        model (PharmHGTModel in eval mode)
        params (dict of hyperparameters)
        metrics (dict of test metrics, or None)
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw['state_dict']
    params = raw.get('params', {})
    metrics = raw.get('metrics', None)

    model = PharmHGTModel(
        hidden_dim=params.get('hidden_dim', 256),
        num_layers=params.get('num_layers', 4),
        dropout=params.get('dropout', 0.2),
        num_heads=params.get('num_heads', 8),
    ).to(device).float()

    # Filter state_dict to only matching keys
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state}
    if len(filtered) != len(model_state):
        missing = set(model_state.keys()) - set(filtered.keys())
        extra = set(state_dict.keys()) - set(model_state.keys())
        if missing:
            print(f"  [Warning] Missing keys in checkpoint: {missing}")
        if extra:
            print(f"  [Info] Skipped extra checkpoint keys: {extra}")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, params, metrics


@torch.no_grad()
def predict_pharmhgt(model, smiles: str, device: str = 'cpu'):
    """Predict pCMC for a single SMILES string using a trained PharmHGT model."""
    data = build_molecule_data(smiles)
    data = data.to(device)

    # Add batch dimension via PyG's Batch.from_data_list
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data])

    pred = model(batch)
    return pred.cpu().item()


@torch.no_grad()
def predict_pharmhgt_batch(model, smiles_list, device: str = 'cpu'):
    """Predict pCMC for a list of SMILES strings."""
    from torch_geometric.data import Batch
    data_list = [build_molecule_data(smi) for smi in smiles_list]
    batch = Batch.from_data_list(data_list).to(device)
    preds = model(batch)
    return preds.cpu().numpy()
