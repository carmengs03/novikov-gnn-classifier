"""LR + MLP baseline on the noisy nearest-boundary dim 3 dataset."""
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DATASET = "dataset_noisy_dim3.pt"
N_SEEDS = 3


def aplanar(g, dim=3):
    C = torch.zeros((dim, dim, dim), dtype=torch.float)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k]
    return C.flatten().numpy()


def main():
    print(f"Loading {DATASET}...", flush=True)
    ds = torch.load(DATASET, weights_only=False)
    X = np.array([aplanar(g) for g in ds])
    y = np.array([int(g.y.item()) for g in ds])
    print(f"  {len(ds):,} graphs ({int((y==1).sum()):,} pos / {int((y==0).sum()):,} noisy-neg)", flush=True)

    accs_lr, accs_mlp = [], []
    for semilla in range(N_SEEDS):
        t0 = time.time()
        torch.manual_seed(42 + semilla)
        perm = torch.randperm(len(ds)).numpy()
        n_train = int(0.70 * len(ds))
        n_val = int(0.15 * len(ds))
        idx_tr = perm[:n_train]
        idx_te = perm[n_train + n_val:]

        sc = StandardScaler().fit(X[idx_tr])
        Xtr = sc.transform(X[idx_tr])
        Xte = sc.transform(X[idx_te])

        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42 + semilla)
        lr.fit(Xtr, y[idx_tr])
        acc_lr = lr.score(Xte, y[idx_te]) * 100
        accs_lr.append(acc_lr)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                            random_state=42 + semilla)
        mlp.fit(Xtr, y[idx_tr])
        acc_mlp = mlp.score(Xte, y[idx_te]) * 100
        accs_mlp.append(acc_mlp)
        print(f"  Seed {42+semilla}: LR={acc_lr:.2f}%  MLP={acc_mlp:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"NOISY BASELINE DIM 3 ({N_SEEDS} seeds)", flush=True)
    print("=" * 60, flush=True)
    print(f"Logistic regression: {np.mean(accs_lr):.2f}% +/- {np.std(accs_lr):.2f}%", flush=True)
    print(f"Flat MLP:            {np.mean(accs_mlp):.2f}% +/- {np.std(accs_mlp):.2f}%", flush=True)
    print(f"\nComparison with the other dim 3 regimes:", flush=True)
    print(f"  Easy (original):       LR=46.14%  MLP=96.00%", flush=True)
    print(f"  Boundary (structural): LR=46.35%  MLP=92.96%", flush=True)
    print(f"  Random (control):      LR=56.01%  MLP=100.00%", flush=True)
    print(f"  Noisy (new):     LR={np.mean(accs_lr):.2f}%  MLP={np.mean(accs_mlp):.2f}%", flush=True)

    with open("baseline_noisy_dim3.json", "w") as f:
        json.dump({
            "dataset": DATASET,
            "n_seeds": N_SEEDS,
            "lr_mean": float(np.mean(accs_lr)),
            "lr_std": float(np.std(accs_lr)),
            "mlp_mean": float(np.mean(accs_mlp)),
            "mlp_std": float(np.std(accs_mlp)),
            "lr_por_semilla": accs_lr,
            "mlp_por_semilla": accs_mlp,
        }, f, indent=2)
    print("Saved: baseline_noisy_dim3.json", flush=True)


if __name__ == "__main__":
    main()
