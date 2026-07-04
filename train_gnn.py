import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import rdchem
import optuna
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==================== SMILES → 分子图转换 ====================

ATOM_ELEMENTS = ["H", "C", "N", "O", "F", "Na", "S", "Cl", "Br", "P", "I", "other"]
ATOM_DEGREES = list(range(6))  # 0–5+
ATOM_CHARGES = [-2, -1, 0, 1, 2]
ATOM_HYBRID = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    None,  # other
]
ATOM_NUMH = list(range(5))  # 0–4+


def _one_hot(val, choices):
    v = [0] * len(choices)
    try:
        idx = choices.index(val)
        v[idx] = 1
    except (ValueError, IndexError):
        pass
    return v


def smiles_to_graph(smiles: str) -> Data | None:
    """SMILES → PyG Data 对象 (原子 39 维, 键 11 维).

    原子特征布局:
      元素(12) + 度数(6) + 形式电荷(5) + 杂化(4) + 芳香(1)
      + 总H(5) + 手性(3) + 在环(1) + 环尺寸标记(1) + 质量归一化(1)

    键特征布局:
      键类型(4) + 共轭(1) + 在环(1) + 立体化学(4) + 自环(1)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ---- 原子特征 ----
    atom_feats = []
    for atom in mol.GetAtoms():
        # 1) 元素 12-dim
        elem = atom.GetSymbol()
        f_elem = _one_hot(elem if elem in ATOM_ELEMENTS[:-1] else "other", ATOM_ELEMENTS)

        # 2) 度数 6-dim
        f_deg = _one_hot(min(atom.GetDegree(), 5), ATOM_DEGREES)

        # 3) 形式电荷 5-dim
        f_charge = _one_hot(atom.GetFormalCharge(), ATOM_CHARGES)

        # 4) 杂化 4-dim
        f_hyb = _one_hot(atom.GetHybridization(), ATOM_HYBRID)

        # 5) 芳香 1-dim
        f_arom = [1.0 if atom.GetIsAromatic() else 0.0]

        # 6) 总 H 数 5-dim
        f_h = _one_hot(min(atom.GetTotalNumHs(), 4), ATOM_NUMH)

        # 7) 手性 3-dim
        f_chiral = _one_hot(atom.GetChiralTag(), [
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_UNSPECIFIED,
        ])

        # 8) 在环 1-dim
        f_ring = [1.0 if atom.IsInRing() else 0.0]

        # 9) 在 3-6 元环标记 1-dim
        if atom.IsInRing():
            ring_sizes = mol.GetRingInfo().AtomRings()
            sizes = [len(r) for r in ring_sizes if atom.GetIdx() in r]
            f_ring_size = [1.0 if any(3 <= s <= 6 for s in sizes) else 0.0]
        else:
            f_ring_size = [0.0]

        # 10) 原子质量归一化 [0, 1]
        f_mass = [min(atom.GetMass() / 200.0, 1.0)]

        feat = f_elem + f_deg + f_charge + f_hyb + f_arom + f_h + f_chiral + f_ring + f_ring_size + f_mass
        assert len(feat) == 39, f"Atom feature has {len(feat)} dims, expected 39"
        atom_feats.append(feat)

    # ---- 键特征 ----
    row, col, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # 1) 键类型 4-dim
        f_type = _one_hot(bond.GetBondType(), [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
        ])

        # 2) 共轭 1-dim
        f_conj = [1.0 if bond.GetIsConjugated() else 0.0]

        # 3) 在环 1-dim
        f_ring_b = [1.0 if bond.IsInRing() else 0.0]

        # 4) 立体化学 4-dim
        f_stereo = _one_hot(bond.GetStereo(), [
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREONONE,
        ])

        # 5) 自环标记 1-dim（非自环时为 0）
        f_self = [0.0]

        e = f_type + f_conj + f_ring_b + f_stereo + f_self
        assert len(e) == 11, f"Edge feature has {len(e)} dims, expected 11"

        # 无向图 → i→j 和 j→i
        row += [i, j]
        col += [j, i]
        edge_feats += [e, e]

    # 自环 (self-loop)
    n = mol.GetNumAtoms()
    row += list(range(n))
    col += list(range(n))
    self_edge = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    edge_feats += [self_edge] * n

    x = torch.tensor(np.array(atom_feats, dtype=np.float32))
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feats, dtype=np.float32))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main():
    # ==================== 数据加载 ====================

    data_train_file_path = "data/surfpro_train.csv"
    data_test_file_path = "data/surfpro_test.csv"
    
    df_train = pd.read_csv(data_train_file_path)
    df_train = df_train.dropna(subset=["pCMC"])
    
    print("将 SMILES 转换为分子图 ...")
    graph_list, valid_idx = [], []
    for i, smi in enumerate(df_train["SMILES"]):
        g = smiles_to_graph(smi)
        if g is not None:
            g.y = torch.tensor([df_train["pCMC"].values[i]], dtype=torch.float)
            graph_list.append(g)
            valid_idx.append(i)
    
    y_train = df_train["pCMC"].values[valid_idx]
    print(f"有效样本: {len(graph_list)}/{len(df_train)}")
    print(f"平均原子数: {np.mean([g.num_nodes for g in graph_list]):.1f}")
    
    # ==================== 设备 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ==================== 分层划分 ====================
    indices = np.arange(len(graph_list))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    graphs_train = [graph_list[i] for i in train_idx]
    graphs_val   = [graph_list[i] for i in val_idx]
    graphs_test  = [graph_list[i] for i in test_idx]
    
    y_all = y_train.copy()  # 完整标签
    y_train_sub = y_all[train_idx]
    y_val_sub   = y_all[val_idx]
    y_test_sub  = y_all[test_idx]
    
    print(f"Train: {len(graphs_train)}, Val: {len(graphs_val)}, Test: {len(graphs_test)}")
    
    # ==================== 训练 / 评估工具函数 ====================
    
    
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)
    
    
    @torch.no_grad()
    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, y_true, y_pred = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            total_loss += criterion(pred, batch.y).item() * batch.num_graphs
            y_true.append(batch.y.cpu())
            y_pred.append(pred.cpu())
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        return (
            total_loss / len(loader.dataset),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred),
        )
    
    
    def build_model(hidden_dim, num_layers, num_timesteps, dropout):
        return AttentiveFP(
            in_channels=39,
            hidden_channels=hidden_dim,
            out_channels=1,
            edge_dim=11,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        ).to(device)
    
    
    # ==================== Optuna 超参数搜索 ====================
    
    print("\n" + "=" * 60)
    print("超参数搜索: Optuna (TPE) ...")
    
    N_TRIALS = 30
    
    
    def train_and_evaluate(params, max_epochs=500):
        loader = DataLoader(graphs_train, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(graphs_val, batch_size=params["batch_size"], shuffle=False)
    
        model = build_model(
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            num_timesteps=params["num_timesteps"],
            dropout=params["dropout"],
        )
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=1e-6)
        criterion = nn.MSELoss()
    
        best_val_loss = float("inf")
        trigger = 0
        for _ in range(max_epochs):
            train_epoch(model, loader, optimizer, criterion)
            val_loss, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger = 0
            else:
                trigger += 1
                if trigger >= 40:
                    break
        return best_val_loss
    
    
    def objective(trial):
        params = {
            "lr":            trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "dropout":       trial.suggest_float("dropout", 0.05, 0.4),
            "wd":            trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "hidden_dim":    trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers":    trial.suggest_int("num_layers", 2, 5),
            "num_timesteps": trial.suggest_int("num_timesteps", 2, 4),
            "batch_size":    trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        best_val_loss = train_and_evaluate(params)
        return -best_val_loss
    
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"\n最佳参数: {best_params}")
    print(f"最佳验证 MSE: {-study.best_value:.6f}")
    
    # ==================== 用最佳参数训练最终模型 ====================
    
    print("\n" + "=" * 60)
    print("用最佳参数训练最终模型 ...")
    
    final_loader = DataLoader(graphs_train, batch_size=best_params["batch_size"], shuffle=True)
    val_loader   = DataLoader(graphs_val,   batch_size=best_params["batch_size"], shuffle=False)
    test_loader  = DataLoader(graphs_test,  batch_size=best_params["batch_size"], shuffle=False)
    
    model = build_model(
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        num_timesteps=best_params["num_timesteps"],
        dropout=best_params["dropout"],
    )
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    trigger = 0
    for epoch in range(500):
        train_epoch(model, final_loader, optimizer, criterion)
        val_loss, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
        else:
            trigger += 1
            if trigger >= 50:
                print(f"Early stop @ epoch {epoch + 1}")
                break
    
    # ==================== 评估 ====================
    
    
    def print_metrics(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    
    with torch.no_grad():
        model.eval()
        splits = [
            ("Train", graphs_train, y_train_sub),
            ("Val", graphs_val, y_val_sub),
            ("Test", graphs_test, y_test_sub),
        ]
        all_preds = {}
        for name, gs, yt in splits:
            loader = DataLoader(gs, batch_size=best_params["batch_size"], shuffle=False)
            preds = []
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                preds.append(out.cpu())
            y_pred = torch.cat(preds).numpy()
            print_metrics(yt, y_pred, name)
            all_preds[name] = y_pred
    
    # ==================== 保存预测图 ====================
    
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_sub, all_preds["Test"], alpha=0.6)
    vmin = min(y_test_sub.min(), all_preds["Test"].min())
    vmax = max(y_test_sub.max(), all_preds["Test"].max())
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5)
    plt.xlabel("True pCMC")
    plt.ylabel("Predicted pCMC")
    plt.title(f"AttentiveFP — Test Set\nR² = {r2_score(y_test_sub, all_preds['Test']):.4f}")
    plt.tight_layout()
    plt.savefig("reports/gnn_pred_vs_true.png", dpi=150)
    plt.close()
    print("\n预测结果已保存至 reports/gnn_pred_vs_true.png")

if __name__ == "__main__":
    main()
