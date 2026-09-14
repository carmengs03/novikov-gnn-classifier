"""OOD baseline: LR + MLP trained on every configuration EXCEPT d and xviii,
evaluated zero-shot on graphs drawn exclusively from d + xviii.

Purpose: empirically check that the flat MLP MEMORISES local patterns
without generalising to unseen topologies, while the GNN (evaluated on
GPU in another script) should generalise better thanks to its structural
bias.

Reports:
    - In-distribution accuracy on the training test split
    - Zero-shot OOD accuracy on d + xviii
    - Exact binomial test on the OOD accuracy for significance
"""
import json
import time

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ARCHIVO_TRAIN = "dataset_train_ood.pt"
ARCHIVO_TEST_OOD = "dataset_test_ood.pt"
N_SEEDS = 3
MLP_HIDDEN = (128, 64, 32)
DIM = 3   # the dataset ships with edge_attr padded to dim 3, x has 3 rows


def aplanar(g, dim=DIM):
    """Rebuild C in R^{d x d x d} from edge_attr (padded to dim 3)."""
    C = torch.zeros((dim, dim, dim), dtype=torch.float)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k]
    return C.flatten().numpy()


def main():
    print(f"Loading {ARCHIVO_TRAIN}...", flush=True)
    ds_train = torch.load(ARCHIVO_TRAIN, weights_only=False)
    print(f"  {len(ds_train):,} training graphs", flush=True)

    print(f"Loading {ARCHIVO_TEST_OOD}...", flush=True)
    ds_test = torch.load(ARCHIVO_TEST_OOD, weights_only=False)
    print(f"  {len(ds_test):,} OOD test graphs", flush=True)

    # Verify the composition of the OOD test set
    configs_test = set(g.config for g in ds_test)
    print(f"  Configs in OOD test: {configs_test}", flush=True)

    X_tr = np.array([aplanar(g) for g in ds_train])
    y_tr = np.array([int(g.y.item()) for g in ds_train])
    X_ood = np.array([aplanar(g) for g in ds_test])
    y_ood = np.array([int(g.y.item()) for g in ds_test])

    results = {"train_size": len(ds_train), "ood_size": len(ds_test), "por_semilla": []}

    accs_id_lr, accs_id_mlp = [], []
    accs_ood_lr, accs_ood_mlp = [], []

    for semilla in range(N_SEEDS):
        t0 = time.time()
        torch.manual_seed(42 + semilla)
        # 85/15 split of the training set to obtain an in-distribution test
        perm = torch.randperm(len(ds_train)).numpy()
        n_train = int(0.85 * len(ds_train))
        idx_tr = perm[:n_train]
        idx_id = perm[n_train:]

        sc = StandardScaler().fit(X_tr[idx_tr])
        Xtr = sc.transform(X_tr[idx_tr])
        Xid = sc.transform(X_tr[idx_id])
        Xood = sc.transform(X_ood)

        # --- LR ---
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42 + semilla)
        lr.fit(Xtr, y_tr[idx_tr])
        a_id_lr = lr.score(Xid, y_tr[idx_id]) * 100
        a_ood_lr = lr.score(Xood, y_ood) * 100
        aciertos_ood_lr = int((lr.predict(Xood) == y_ood).sum())

        # --- MLP ---
        mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=300,
                            random_state=42 + semilla)
        mlp.fit(Xtr, y_tr[idx_tr])
        a_id_mlp = mlp.score(Xid, y_tr[idx_id]) * 100
        a_ood_mlp = mlp.score(Xood, y_ood) * 100
        aciertos_ood_mlp = int((mlp.predict(Xood) == y_ood).sum())

        accs_id_lr.append(a_id_lr); accs_ood_lr.append(a_ood_lr)
        accs_id_mlp.append(a_id_mlp); accs_ood_mlp.append(a_ood_mlp)

        # Per-OOD-config breakdown
        preds_mlp = mlp.predict(Xood)
        preds_lr = lr.predict(Xood)
        por_config = {}
        configs_arr = np.array([g.config for g in ds_test])
        for cfg in sorted(set(configs_arr.tolist())):
            m = configs_arr == cfg
            por_config[cfg] = {
                "n": int(m.sum()),
                "lr_acc": float((preds_lr[m] == y_ood[m]).mean() * 100),
                "mlp_acc": float((preds_mlp[m] == y_ood[m]).mean() * 100),
            }

        print(f"  Seed {42+semilla}:  "
              f"LR   ID={a_id_lr:.2f}% OOD={a_ood_lr:.2f}% "
              f"| MLP  ID={a_id_mlp:.2f}% OOD={a_ood_mlp:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        for cfg, d in por_config.items():
            print(f"    {cfg:>10s} (n={d['n']:>5d}): LR={d['lr_acc']:6.2f}%  MLP={d['mlp_acc']:6.2f}%",
                  flush=True)

        results["por_semilla"].append({
            "semilla": 42 + semilla,
            "lr_id": a_id_lr, "lr_ood": a_ood_lr, "lr_ood_aciertos": aciertos_ood_lr,
            "mlp_id": a_id_mlp, "mlp_ood": a_ood_mlp, "mlp_ood_aciertos": aciertos_ood_mlp,
            "por_config": por_config,
        })

    print("\n" + "=" * 70, flush=True)
    print(f"OOD SUMMARY ({N_SEEDS} seeds)", flush=True)
    print("=" * 70, flush=True)
    print(f"Logistic regression:", flush=True)
    print(f"  ID   : {np.mean(accs_id_lr):.2f}% +/- {np.std(accs_id_lr):.2f}%", flush=True)
    print(f"  OOD  : {np.mean(accs_ood_lr):.2f}% +/- {np.std(accs_ood_lr):.2f}%", flush=True)
    print(f"  delta: {np.mean(accs_id_lr)-np.mean(accs_ood_lr):+.2f} pts (drop)", flush=True)
    print(f"Flat MLP:", flush=True)
    print(f"  ID   : {np.mean(accs_id_mlp):.2f}% +/- {np.std(accs_id_mlp):.2f}%", flush=True)
    print(f"  OOD  : {np.mean(accs_ood_mlp):.2f}% +/- {np.std(accs_ood_mlp):.2f}%", flush=True)
    print(f"  delta: {np.mean(accs_id_mlp)-np.mean(accs_ood_mlp):+.2f} pts (drop)", flush=True)

    # Binomial test on the mean MLP OOD accuracy
    aciertos_med = int(round(np.mean(accs_ood_mlp) / 100 * len(ds_test)))
    test = binomtest(k=aciertos_med, n=len(ds_test), p=0.5, alternative="greater")
    print(f"\nBinomial test (mean MLP OOD, one-sided p>0.5):", flush=True)
    print(f"  k={aciertos_med}, n={len(ds_test)}, p-value={test.pvalue:.4e}", flush=True)

    results["lr_id_mean"] = float(np.mean(accs_id_lr))
    results["lr_ood_mean"] = float(np.mean(accs_ood_lr))
    results["mlp_id_mean"] = float(np.mean(accs_id_mlp))
    results["mlp_ood_mean"] = float(np.mean(accs_ood_mlp))
    results["p_valor_mlp_ood"] = float(test.pvalue)

    with open("baseline_ood_mlp.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: baseline_ood_mlp.json", flush=True)


if __name__ == "__main__":
    main()
