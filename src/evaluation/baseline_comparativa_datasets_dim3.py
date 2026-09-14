"""LR + MLP baselines on the three dim 3 difficulty regimes
(easy / near-boundary / random).

Analogous to baseline_comparativa_datasets.py but for dim 3: each algebra
is described by a flattened C tensor with 3^3 = 27 coefficients.
"""
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "1_facil":     "dataset_novikov_dim3_facil.pt",
    "2_frontera":  "dataset_novikov_dim3_dificiles.pt",
    "3_aleatorio": "dataset_aleatorio_dim3.pt",
}
N_SEEDS = 3
MLP_HIDDEN = (128, 64, 32)


def aplanar(g, dim=3):
    C = torch.zeros((dim, dim, dim), dtype=torch.float)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k]
    return C.flatten().numpy()


def evaluar(dataset):
    X = np.array([aplanar(g) for g in dataset])
    y = np.array([int(g.y.item()) for g in dataset])
    accs_lr, accs_mlp = [], []
    for semilla in range(N_SEEDS):
        torch.manual_seed(42 + semilla)
        perm = torch.randperm(len(dataset)).numpy()
        n_train = int(0.70 * len(dataset))
        n_val = int(0.15 * len(dataset))
        idx_tr = perm[:n_train]
        idx_te = perm[n_train + n_val:]

        sc = StandardScaler().fit(X[idx_tr])
        Xtr = sc.transform(X[idx_tr])
        Xte = sc.transform(X[idx_te])

        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42 + semilla)
        lr.fit(Xtr, y[idx_tr])
        accs_lr.append(lr.score(Xte, y[idx_te]) * 100)

        mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=300,
                            random_state=42 + semilla)
        mlp.fit(Xtr, y[idx_tr])
        accs_mlp.append(mlp.score(Xte, y[idx_te]) * 100)
    return {
        "lr_mean": float(np.mean(accs_lr)),
        "lr_std": float(np.std(accs_lr)),
        "mlp_mean": float(np.mean(accs_mlp)),
        "mlp_std": float(np.std(accs_mlp)),
        "lr_por_semilla": accs_lr,
        "mlp_por_semilla": accs_mlp,
    }


def main():
    resultados = {}
    for nombre, path in DATASETS.items():
        print(f"\n=== {nombre.upper()} ({path}) ===", flush=True)
        t0 = time.time()
        ds = torch.load(path, weights_only=False)
        y = np.array([int(g.y.item()) for g in ds])
        print(f"  {len(ds):,} graphs ({int((y==1).sum()):,} pos / {int((y==0).sum()):,} neg)",
              flush=True)

        r = evaluar(ds)
        resultados[nombre] = {"dataset": path, **r}
        print(f"  Logistic regression: {r['lr_mean']:.2f}% +/- {r['lr_std']:.2f}%", flush=True)
        print(f"  Flat MLP:            {r['mlp_mean']:.2f}% +/- {r['mlp_std']:.2f}%", flush=True)
        print(f"  (time: {time.time() - t0:.0f}s)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("COMPARATIVE TABLE -- dim 3", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Regime':<20} {'Logistic reg.':<25} {'Flat MLP':<25}", flush=True)
    print("-" * 70, flush=True)
    for nombre in DATASETS:
        r = resultados[nombre]
        print(f"{nombre:<20} "
              f"{r['lr_mean']:>6.2f}% +/- {r['lr_std']:.2f}%       "
              f"{r['mlp_mean']:>6.2f}% +/- {r['mlp_std']:.2f}%", flush=True)
    print("=" * 70, flush=True)

    with open("baseline_comparativa_datasets_dim3.json", "w") as f:
        json.dump(resultados, f, indent=2)
    print("Saved: baseline_comparativa_datasets_dim3.json", flush=True)


if __name__ == "__main__":
    main()
