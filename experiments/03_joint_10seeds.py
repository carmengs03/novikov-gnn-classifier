"""Retrain the joint (dim2+dim3) model over N seeds, using the same
architecture, hyperparameters and pipeline as the joint model reported
in the paper. At the end of each seed the model is evaluated on
`dataset_noisy_topologico_dim3_enriquecido.pt`.

Output: `semillas_10_joint.json` with in-distribution test accuracy and
dim=3 noisy accuracy per seed, plus mean and standard deviation
across the N seeds.

Required files in the same directory:
  - dataset_combinado_dim2_dim3_dificiles.pt
  - dataset_noisy_topologico_dim3_enriquecido.pt

Usage:
    python -u 03_joint_10seeds.py --n_seeds 10 --n_epocas 100
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool


DATASET_TRAIN = "dataset_combinado_dim2_dim3_dificiles.pt"
DATASET_NOISY = "dataset_noisy_topologico_dim3_enriquecido.pt"
# Joint-model hyperparameters
LR = 0.0011940188529959736
DIM_OCULTA = 64
BATCH_SIZE = 64


def fijar_semilla(semilla):
    random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CapaAlgebraica(MessagePassing):
    def __init__(self, dim_in, dim_aristas, dim_oculta):
        super().__init__(aggr="max")
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + dim_aristas, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))


class RedNovikov(nn.Module):
    def __init__(self, dim_nodos=6, dim_aristas=3, dim_oculta=64, n_capas=3, residual=True):
        super().__init__()
        self.residual = residual
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta)
        self.capas_mp = nn.ModuleList([
            CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta)
            for _ in range(n_capas)
        ])
        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, datos):
        x = F.relu(self.proyeccion_entrada(datos.x))
        for capa in self.capas_mp:
            h = capa(x, datos.edge_index, datos.edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)
        return self.clasificador(global_mean_pool(x, datos.batch))


def entrenar_una_semilla(dataset, dispositivo, n_epocas, semilla,
                          batch_size, dim_oculta, lr):
    """Train one seed following the same methodology as the joint model
    reported in the paper:

      - 70/15/15 split per seed (`torch.manual_seed(42+semilla)`)
      - network init with `torch.manual_seed(100+semilla)`
      - keep the state dict of the epoch with the lowest val_loss
      - always evaluate test with that best checkpoint, not the last one
    """
    fijar_semilla(42 + semilla)
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    datos_train, datos_val, datos_test = random_split(
        dataset, [n_train, n_val, n_test])

    torch.manual_seed(100 + semilla)
    modelo = RedNovikov(dim_nodos=6, dim_aristas=3,
                        dim_oculta=dim_oculta).to(dispositivo)
    criterio = nn.BCELoss()
    optim = torch.optim.Adam(modelo.parameters(), lr=lr)

    train_loader = DataLoader(datos_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datos_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datos_test, batch_size=batch_size, shuffle=False)

    mejor_val_loss = float("inf")
    mejor_state_dict = None
    mejor_epoca = 0

    for epoca in range(n_epocas):
        t = time.time()
        modelo.train()
        loss_ac = 0.0
        for lote in train_loader:
            lote = lote.to(dispositivo)
            optim.zero_grad()
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            loss = criterio(pred, etiq)
            loss.backward()
            optim.step()
            loss_ac += loss.item() * lote.num_graphs
        train_loss = loss_ac / len(datos_train)

        modelo.eval()
        val_ac = 0.0
        with torch.no_grad():
            for lote in val_loader:
                lote = lote.to(dispositivo)
                pred = modelo(lote).view(-1)
                etiq = lote.y.float().view(-1)
                val_ac += criterio(pred, etiq).item() * lote.num_graphs
        val_loss = val_ac / len(datos_val)

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_epoca = epoca + 1
            mejor_state_dict = {k: v.detach().cpu().clone()
                                 for k, v in modelo.state_dict().items()}

        if epoca == 0 or (epoca + 1) % max(1, n_epocas // 5) == 0 or epoca == n_epocas - 1:
            print(f"    epoch {epoca+1:3d}/{n_epocas} | train={train_loss:.4f} | "
                  f"val={val_loss:.4f} | best_val={mejor_val_loss:.4f} "
                  f"@ep{mejor_epoca} | t={time.time()-t:.1f}s", flush=True)

    # Restore best checkpoint by val_loss and evaluate test with it
    modelo.load_state_dict(mejor_state_dict)
    modelo.eval()
    correctos = total = 0
    with torch.no_grad():
        for lote in test_loader:
            lote = lote.to(dispositivo)
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            correctos += ((pred >= 0.5).float() == etiq).sum().item()
            total += etiq.size(0)
    test_acc = correctos / total * 100

    return modelo, test_acc, mejor_val_loss, mejor_epoca


def evaluar_gnn(modelo, dataset, dispositivo, batch_size=128):
    modelo.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for lote in loader:
            lote = lote.to(dispositivo)
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            pb = (pred >= 0.5).float()
            tp += ((pb == 1) & (etiq == 1)).sum().item()
            fp += ((pb == 1) & (etiq == 0)).sum().item()
            tn += ((pb == 0) & (etiq == 0)).sum().item()
            fn += ((pb == 0) & (etiq == 1)).sum().item()
    total = tp + fp + tn + fn
    acc = (tp + tn) / total * 100 if total else 0.0
    return {"accuracy": acc, "n": total, "tp": int(tp),
            "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_epocas", type=int, default=150)
    parser.add_argument("--output", default="semillas_10_joint.json")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dispositivo}", flush=True)
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"[hp] lr={LR}  dim_oculta={DIM_OCULTA}  batch_size={BATCH_SIZE}",
          flush=True)

    t0 = time.time()
    print(f"\nLoading {DATASET_TRAIN}...", flush=True)
    dataset_train = torch.load(DATASET_TRAIN, weights_only=False)
    print(f"  {len(dataset_train):,} graphs ({time.time()-t0:.1f}s)",
          flush=True)

    t1 = time.time()
    print(f"Loading {DATASET_NOISY}...", flush=True)
    dataset_noisy = torch.load(DATASET_NOISY, weights_only=False)
    print(f"  {len(dataset_noisy):,} graphs ({time.time()-t1:.1f}s)",
          flush=True)

    resultados = []
    t_global = time.time()
    for s in range(args.n_seeds):
        print(f"\n=== Seed {s+1}/{args.n_seeds} ===", flush=True)
        t_s = time.time()
        modelo, test_acc, best_val, best_ep = entrenar_una_semilla(
            dataset_train, dispositivo, args.n_epocas, s,
            batch_size=BATCH_SIZE, dim_oculta=DIM_OCULTA, lr=LR)
        eval_noisy = evaluar_gnn(modelo, dataset_noisy, dispositivo)
        print(f"  best_val_loss: {best_val:.4f} @ epoch {best_ep}", flush=True)
        print(f"  in-dist test_acc: {test_acc:.2f}%", flush=True)
        print(f"  noisy dim3 acc:     {eval_noisy['accuracy']:.2f}% "
              f"(TP={eval_noisy['tp']} FP={eval_noisy['fp']} "
              f"TN={eval_noisy['tn']} FN={eval_noisy['fn']})", flush=True)
        print(f"  seed time: {(time.time()-t_s)/60:.1f} min", flush=True)
        resultados.append({
            "semilla": s,
            "in_dist_acc": float(test_acc),
            "best_val_loss": float(best_val),
            "mejor_epoca": int(best_ep),
            "noisy_dim3": eval_noisy,
        })
        # Incremental save after each seed
        with open(args.output, "w") as f:
            json.dump({"hp": {"lr": LR, "dim_oculta": DIM_OCULTA,
                              "batch_size": BATCH_SIZE},
                        "n_epocas": args.n_epocas,
                        "por_semilla": resultados,
                        "t_min": (time.time()-t_global)/60}, f, indent=2)

    if resultados:
        indist = np.array([r["in_dist_acc"] for r in resultados])
        noisy = np.array([r["noisy_dim3"]["accuracy"] for r in resultados])
        resumen = {
            "in_dist_mean": float(indist.mean()),
            "in_dist_std": float(indist.std(ddof=0)),
            "noisy_dim3_mean": float(noisy.mean()),
            "noisy_dim3_std": float(noisy.std(ddof=0)),
            "n_seeds": len(resultados),
        }
        print(f"\n=== SUMMARY over {len(resultados)} seeds ===", flush=True)
        print(f"  in-dist: {resumen['in_dist_mean']:.2f} +/- {resumen['in_dist_std']:.2f}",
              flush=True)
        print(f"  noisy d3:  {resumen['noisy_dim3_mean']:.2f} +/- {resumen['noisy_dim3_std']:.2f}",
              flush=True)
        with open(args.output, "w") as f:
            json.dump({"hp": {"lr": LR, "dim_oculta": DIM_OCULTA,
                              "batch_size": BATCH_SIZE},
                        "n_epocas": args.n_epocas,
                        "por_semilla": resultados,
                        "resumen": resumen,
                        "t_min": (time.time()-t_global)/60}, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
