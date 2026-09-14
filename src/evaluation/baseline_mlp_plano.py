"""Second-level non-graph baseline: flat MLP over the flattened tensor of
structure constants. Complements the linear lower bound of
baseline_regresion_logistica.py with a non-linear one.

Setup:
    - Dataset: dataset_novikov_dim2_50000_DATOS.pt (44,386 graphs)
    - Features: the 8 flattened structure constants
    - Model: MLPClassifier with 3 hidden layers (128, 64, 32).
    - Methodology: 10 seeds x random_split 70/15/15.
"""
import json

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "dataset_novikov_dim2_50000_DATOS.pt"
N_SEEDS = 10
HIDDEN = (128, 64, 32)
MAX_ITER = 300


def aplanar_grafo(grafo):
    """Same procedure as the logistic regression baseline:
    rebuild C in R^{2x2x2} from edge_attr and flatten it."""
    C = torch.zeros((2, 2, 2), dtype=torch.float)
    for k in range(grafo.edge_index.shape[1]):
        i, j = grafo.edge_index[0, k].item(), grafo.edge_index[1, k].item()
        C[i, j, :] = grafo.edge_attr[k]
    return C.flatten().numpy()


def main():
    print(f"Loading {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"  {len(dataset):,} graphs")

    print("Flattening graphs to 8-feature vectors...")
    X = np.array([aplanar_grafo(g) for g in dataset])
    y = np.array([int(g.y.item()) for g in dataset])
    print(f"  X: {X.shape}  |  y: {y.shape}")

    print(f"\nModel: MLPClassifier(hidden={HIDDEN}, max_iter={MAX_ITER}, relu, adam)")
    print(f"Approx. parameters: 8*128 + 128*64 + 64*32 + 32*1 ~ {8*128 + 128*64 + 64*32 + 32:,}")

    accuracies = []
    for semilla in range(N_SEEDS):
        # Same split as the GNNs so that the test set is identical.
        torch.manual_seed(42 + semilla)
        perm = torch.randperm(len(dataset)).numpy()
        n_train = int(0.70 * len(dataset))
        n_val = int(0.15 * len(dataset))
        idx_train = perm[:n_train]
        idx_test = perm[n_train + n_val:]

        scaler = StandardScaler().fit(X[idx_train])
        X_train = scaler.transform(X[idx_train])
        X_test = scaler.transform(X[idx_test])

        modelo = MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=MAX_ITER,
            random_state=42 + semilla,
            verbose=False,
        )
        modelo.fit(X_train, y[idx_train])
        acc = modelo.score(X_test, y[idx_test]) * 100
        accuracies.append(acc)
        print(f"  Seed {42+semilla}: test_acc = {acc:.2f}%  "
              f"(iterations used={modelo.n_iter_})")

    accuracies = np.array(accuracies)
    print()
    print("=" * 70)
    print(f"FLAT MLP BASELINE ({N_SEEDS} seeds)")
    print("=" * 70)
    print(f"  Features: 8 (flattened C tensor, no graph structure)")
    print(f"  Model: MLP {HIDDEN}, ReLU + Adam, lr=1e-3")
    print(f"  Test accuracy: {accuracies.mean():.2f}% +/- {accuracies.std():.2f}%")
    print()
    print("Comparison against the reference models:")
    print(f"  Level 1 -- Logistic regression (linear, non-graph):    49.39% +/- 4.00%")
    print(f"  Level 2 -- Flat MLP (non-linear, non-graph):          {accuracies.mean():.2f}% +/- {accuracies.std():.2f}%")
    print(f"  Level 3 -- Baseline GNN (1 node feature):              91.94%")
    print(f"  Level 3 -- Enriched GNN (6 node features):             92.54%")
    print(f"  Level 3 -- Final GNN (Optuna-tuned):                  95.75% +/- 0.45%")
    print("=" * 70)

    resumen = {
        "n_features": 8,
        "modelo": f"MLPClassifier(hidden={HIDDEN}, relu, adam, lr=1e-3)",
        "n_seeds": N_SEEDS,
        "test_acc_mean": float(accuracies.mean()),
        "test_acc_std": float(accuracies.std()),
        "test_acc_por_semilla": accuracies.tolist(),
    }
    with open("baseline_mlp_metricas.json", "w") as f:
        json.dump(resumen, f, indent=2)
    print("\nSaved: baseline_mlp_metricas.json")


if __name__ == "__main__":
    main()
