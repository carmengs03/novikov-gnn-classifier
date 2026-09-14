"""Non-graph baseline: logistic regression on the flattened tensor of
structure constants.

Setup:
    - Dataset: dataset_novikov_dim2_50000_DATOS.pt (44,386 graphs)
    - Features: the 8 flattened c_{ijk} constants, no topological info.
    - Model: sklearn.LogisticRegression (L2, C=1.0)
    - Methodology: 10 seeds x random_split 70/15/15.

Output: baseline_logreg_metricas.json
"""
import json
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "dataset_novikov_dim2_50000_DATOS.pt"
N_SEEDS = 10


def aplanar_grafo(grafo):
    """Turn a graph into a vector of 8 features = structure constants.
    Rebuilds the tensor C[source, target, :] = edge_attr[k] and flattens
    it in row-major order. The 8 entries are: c111, c112, c121, c122,
    c211, c212, c221, c222.
    """
    C = torch.zeros((2, 2, 2), dtype=torch.float)
    for k in range(grafo.edge_index.shape[1]):
        i, j = grafo.edge_index[0, k].item(), grafo.edge_index[1, k].item()
        C[i, j, :] = grafo.edge_attr[k]
    return C.flatten().numpy()


def main():
    print(f"Loading {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"  {len(dataset):,} graphs")

    print("Flattening graphs into 8-feature vectors (row-major C tensor)...")
    X = np.array([aplanar_grafo(g) for g in dataset])
    y = np.array([int(g.y.item()) for g in dataset])
    print(f"  X: {X.shape}  |  y: {y.shape}  |  positives: {y.sum():,}")

    accuracies = []
    for semilla in range(N_SEEDS):
        # Same split as the GNNs (torch.manual_seed + torch.randperm) so
        # that the test set is identical.
        torch.manual_seed(42 + semilla)
        perm = torch.randperm(len(dataset)).numpy()
        n_train = int(0.70 * len(dataset))
        n_val = int(0.15 * len(dataset))
        idx_train = perm[:n_train]
        idx_test = perm[n_train + n_val:]  # test set identical to the GNN one

        # Without standardisation, logistic regression struggles with mixed magnitudes
        scaler = StandardScaler().fit(X[idx_train])
        X_train = scaler.transform(X[idx_train])
        X_test = scaler.transform(X[idx_test])

        modelo = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
            random_state=42 + semilla,
        )
        modelo.fit(X_train, y[idx_train])
        acc = modelo.score(X_test, y[idx_test]) * 100
        accuracies.append(acc)
        print(f"  Seed {42+semilla}: test_acc = {acc:.2f}%")

    accuracies = np.array(accuracies)
    print()
    print("=" * 60)
    print(f"LOGISTIC REGRESSION BASELINE ({N_SEEDS} seeds)")
    print("=" * 60)
    print(f"  Features: 8 (flattened C tensor, no graph structure)")
    print(f"  Model: LogisticRegression(L2, C=1.0)")
    print(f"  Test accuracy: {accuracies.mean():.2f}% +/- {accuracies.std():.2f}%")
    print()
    print("Comparison against GNN models (all on 44k graphs):")
    print(f"  - Logistic regression (non-graph):           {accuracies.mean():.2f}% +/- {accuracies.std():.2f}%")
    print(f"  - Baseline GNN (1 feature, unenriched):      91.94%")
    print(f"  - Enriched GNN (6 features):                 92.54%")
    print(f"  - Final GNN (Optuna-tuned):                  95.75% +/- 0.45%")
    print("=" * 60)

    resumen = {
        "n_features": 8,
        "modelo": "LogisticRegression(penalty='l2', C=1.0)",
        "n_seeds": N_SEEDS,
        "test_acc_mean": float(accuracies.mean()),
        "test_acc_std": float(accuracies.std()),
        "test_acc_por_semilla": accuracies.tolist(),
        "comparativa": {
            "logreg_no_grafo": float(accuracies.mean()),
            "gnn_baseline_1feat": 91.94,
            "gnn_enriquecido_6feat": 92.54,
            "gnn_expc_optuna": 95.75,
        },
    }
    with open("baseline_logreg_metricas.json", "w") as f:
        json.dump(resumen, f, indent=2)
    print("\nSaved: baseline_logreg_metricas.json")


if __name__ == "__main__":
    main()
