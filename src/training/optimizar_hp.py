"""
Hyperparameter search with Optuna for RedNovikov on dim2.

Search space:
    - lr           : log-uniform in [1e-5, 1e-2]
    - dim_oculta   : {16, 32, 64, 128}
    - batch_size   : {16, 32, 64, 128}
    - n_epocas     : controlled by early stopping (max_epochs + patience)

Metric: BCE val_loss (to minimise).
Pruner: MedianPruner cuts trials worse than the median after a warmup.
Persistence: study kept in SQLite + best_params in JSON so runs can resume.

Typical usage:
    # local, quick smoke test
    python optimizar_hp.py --n-trials 3 --max-epochs 20

    # runpod, full search
    python optimizar_hp.py --n-trials 30 --max-epochs 100
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from modelo_mpnn import RedNovikov


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def split_indices(n: int, semilla: int, fracciones=(0.70, 0.15, 0.15)):
    """Same train/val/test split for every trial so hyperparameters are compared fairly."""
    g = torch.Generator().manual_seed(semilla)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(fracciones[0] * n)
    n_val = int(fracciones[1] * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def make_objective(dataset, train_idx, val_idx, dispositivo, max_epochs, patience, seed,
                   dim_aristas=2):
    criterio = nn.BCELoss()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dim_oculta = trial.suggest_categorical("dim_oculta", [16, 32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        fijar_semilla(seed)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        modelo = RedNovikov(dim_nodos=6, dim_aristas=dim_aristas, dim_oculta=dim_oculta).to(dispositivo)
        optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

        mejor_val_loss = float("inf")
        epocas_sin_mejora = 0

        for epoca in range(max_epochs):
            modelo.train()
            for lote in train_loader:
                lote = lote.to(dispositivo)
                optimizador.zero_grad()
                pred = modelo(lote).view(-1)
                etiq = lote.y.float().view(-1)
                loss = criterio(pred, etiq)
                loss.backward()
                optimizador.step()

            modelo.eval()
            loss_val_acum = 0.0
            with torch.no_grad():
                for lote in val_loader:
                    lote = lote.to(dispositivo)
                    pred = modelo(lote).view(-1)
                    etiq = lote.y.float().view(-1)
                    loss_val_acum += criterio(pred, etiq).item() * lote.num_graphs
            val_loss = loss_val_acum / len(val_idx)

            trial.report(val_loss, epoca)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < mejor_val_loss - 1e-4:
                mejor_val_loss = val_loss
                epocas_sin_mejora = 0
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= patience:
                    break

        return mejor_val_loss

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_enriquecido_dim2_50000_DATOS.pt")
    parser.add_argument("--study-name", default="novikov_dim2_hp")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim-aristas", type=int, default=2,
                        help="Edge-attribute dimension (2 for dim2, 3 for the joint dataset)")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Optimising on: {dispositivo}")
    if dispositivo.type == "cpu":
        print("[WARNING] Running on CPU. With 50k samples x 30 trials this will be slow.")
        print("          Consider reducing --n-trials, using a smaller dataset, or a GPU.")

    print(f"Loading {args.dataset}...")
    dataset = torch.load(args.dataset, weights_only=False)
    print(f"Dataset loaded: {len(dataset)} graphs")

    train_idx, val_idx, test_idx = split_indices(len(dataset), args.seed)
    print(f"Fixed split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test (reserved)")

    storage = f"sqlite:///{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    objective = make_objective(
        dataset, train_idx, val_idx, dispositivo,
        args.max_epochs, args.patience, args.seed,
        dim_aristas=args.dim_aristas,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print("\n=== BEST HYPERPARAMETERS ===")
    print(f"val_loss: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    n_completos = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_podados = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"\nTrials: {len(study.trials)} total | {n_completos} complete | {n_podados} pruned")

    salida = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completos": n_completos,
        "n_podados": n_podados,
        "dataset": args.dataset,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    out_path = Path(f"best_params_{args.study_name}.json")
    out_path.write_text(json.dumps(salida, indent=2))
    print(f"Best hyperparameters saved to {out_path}")
    print(f"Study persisted at {storage} (resume with --study-name {args.study_name})")


if __name__ == "__main__":
    main()
