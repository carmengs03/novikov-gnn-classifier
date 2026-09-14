"""LR + MLP baselines on the topology-preserving noisy datasets
(both dimensions)."""
import json
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


N_SEEDS = 3


def aplanar(g, dim):
    C = torch.zeros((dim, dim, dim), dtype=torch.float)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k]
    return C.flatten().numpy()


def evaluar(archivo, dim):
    ds = torch.load(archivo, weights_only=False)
    X = np.array([aplanar(g, dim) for g in ds])
    y = np.array([int(g.y.item()) for g in ds])
    print(f"  {archivo}: {len(ds):,} graphs "
          f"({int((y==1).sum()):,} pos / {int((y==0).sum()):,} noisy-neg)", flush=True)

    accs_lr, accs_mlp = [], []
    for semilla in range(N_SEEDS):
        t0 = time.time()
        torch.manual_seed(42 + semilla)
        perm = torch.randperm(len(ds)).numpy()
        n_tr = int(0.70 * len(ds))
        n_val = int(0.15 * len(ds))
        idx_tr = perm[:n_tr]
        idx_te = perm[n_tr + n_val:]

        sc = StandardScaler().fit(X[idx_tr])
        Xtr, Xte = sc.transform(X[idx_tr]), sc.transform(X[idx_te])

        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42 + semilla)
        lr.fit(Xtr, y[idx_tr])
        a_lr = lr.score(Xte, y[idx_te]) * 100
        accs_lr.append(a_lr)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                            random_state=42 + semilla)
        mlp.fit(Xtr, y[idx_tr])
        a_mlp = mlp.score(Xte, y[idx_te]) * 100
        accs_mlp.append(a_mlp)
        print(f"    Seed {42+semilla}: LR={a_lr:.2f}%  MLP={a_mlp:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"  Logistic regression: {np.mean(accs_lr):.2f}% +/- {np.std(accs_lr):.2f}%", flush=True)
    print(f"  Flat MLP:            {np.mean(accs_mlp):.2f}% +/- {np.std(accs_mlp):.2f}%", flush=True)
    return {
        "dataset": archivo,
        "lr_mean": float(np.mean(accs_lr)),
        "lr_std": float(np.std(accs_lr)),
        "mlp_mean": float(np.mean(accs_mlp)),
        "mlp_std": float(np.std(accs_mlp)),
        "lr_por_semilla": accs_lr,
        "mlp_por_semilla": accs_mlp,
    }


def main():
    resultados = {}
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("dim3", "both"):
        print("\n=== TOPOLOGICAL NOISY DIM 3 ===", flush=True)
        resultados["dim3"] = evaluar("dataset_noisy_topologico_dim3.pt", dim=3)

    if which in ("dim2", "both"):
        print("\n=== TOPOLOGICAL NOISY DIM 2 ===", flush=True)
        resultados["dim2"] = evaluar("dataset_noisy_topologico_dim2.pt", dim=2)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY -- Topology-preserving noisy", flush=True)
    print("=" * 60, flush=True)
    for k, v in resultados.items():
        print(f"  {k}: LR={v['lr_mean']:.2f}%  MLP={v['mlp_mean']:.2f}%", flush=True)

    with open("baseline_noisy_topologico.json", "w") as f:
        json.dump(resultados, f, indent=2)
    print("Saved: baseline_noisy_topologico.json", flush=True)


if __name__ == "__main__":
    main()
