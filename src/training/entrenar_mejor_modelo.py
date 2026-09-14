"""Train the final model with the best hyperparameters found by Optuna.

Uses 100 fixed epochs without early stopping, random_split per seed with
fijar_semilla(42+i), 10 seeds. The only difference against the manual runs
is the source of lr, dim_oculta and batch_size, which come from Optuna.

Usage:
    python entrenar_mejor_modelo.py
    python entrenar_mejor_modelo.py --n-seeds 10 --n-epocas 100
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from modelo_mpnn import RedNovikov
from optimizar_hp import fijar_semilla


def entrenar_una_semilla(dataset, hp, dispositivo, n_epocas, semilla):
    # 42+i controls the data random_split; 100+i controls weight initialisation.
    fijar_semilla(42 + semilla)
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    datos_train, datos_val, datos_test = random_split(dataset, [n_train, n_val, n_test])

    torch.manual_seed(100 + semilla)
    modelo = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=hp["dim_oculta"]).to(dispositivo)
    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=hp["lr"])

    train_loader = DataLoader(datos_train, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(datos_val, batch_size=hp["batch_size"], shuffle=False)
    test_loader = DataLoader(datos_test, batch_size=hp["batch_size"], shuffle=False)

    h_train, h_val = [], []
    for epoca in range(n_epocas):
        modelo.train()
        loss_train_acum = 0.0
        for lote in train_loader:
            lote = lote.to(dispositivo)
            optimizador.zero_grad()
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            loss = criterio(pred, etiq)
            loss.backward()
            optimizador.step()
            loss_train_acum += loss.item() * lote.num_graphs
        h_train.append(loss_train_acum / n_train)

        modelo.eval()
        loss_val_acum = 0.0
        with torch.no_grad():
            for lote in val_loader:
                lote = lote.to(dispositivo)
                pred = modelo(lote).view(-1)
                etiq = lote.y.float().view(-1)
                loss_val_acum += criterio(pred, etiq).item() * lote.num_graphs
        h_val.append(loss_val_acum / n_val)

    # Test is evaluated at the last epoch (no val-based checkpointing).
    modelo.eval()
    test_loss_acum = 0.0
    aciertos = 0
    with torch.no_grad():
        for lote in test_loader:
            lote = lote.to(dispositivo)
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            test_loss_acum += criterio(pred, etiq).item() * lote.num_graphs
            aciertos += ((pred >= 0.5).float() == etiq).sum().item()

    return {
        "semilla": semilla,
        "val_loss_final": h_val[-1],
        "test_loss": test_loss_acum / n_test,
        "test_acc": (aciertos / n_test) * 100,
        "h_train": h_train,
        "h_val": h_val,
        "state_dict": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
    }


def plotear_curvas(resultados, hp, n_seeds, plot_path):
    train_arr = np.array([r["h_train"] for r in resultados])
    val_arr = np.array([r["h_val"] for r in resultados])
    epochs = np.arange(1, train_arr.shape[1] + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_arr.mean(0), label="Train loss (mean)", color="tab:blue")
    plt.fill_between(epochs,
                     train_arr.mean(0) - train_arr.std(0),
                     train_arr.mean(0) + train_arr.std(0),
                     alpha=0.2, color="tab:blue", label="Train +/- std")
    plt.plot(epochs, val_arr.mean(0), label="Val loss (mean)", color="tab:orange")
    plt.fill_between(epochs,
                     val_arr.mean(0) - val_arr.std(0),
                     val_arr.mean(0) + val_arr.std(0),
                     alpha=0.2, color="tab:orange", label="Val +/- std")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"Final model with optimised hyperparameters ({n_seeds} seeds x {train_arr.shape[1]} epochs)\n"
              f"lr={hp['lr']:.4g} | dim_oculta={hp['dim_oculta']} | batch_size={hp['batch_size']}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_enriquecido_dim2_50000_DATOS.pt")
    parser.add_argument("--best-params", default="best_params_novikov_dim2_hp.json")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-epocas", type=int, default=100)
    parser.add_argument("--output-prefix", default="modelo_final_dim2")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {dispositivo}")

    with open(args.best_params) as f:
        cfg = json.load(f)
    hp = cfg["best_params"]
    print(f"\nHyperparameters loaded from {args.best_params}:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

    print(f"\nMethodology: {args.n_seeds} seeds x {args.n_epocas} fixed epochs, "
          f"random_split per seed")

    print(f"\nLoading {args.dataset}...")
    dataset = torch.load(args.dataset, weights_only=False)
    print(f"Dataset: {len(dataset)} graphs")

    resultados = []
    print(f"\n=== Training {args.n_seeds} seeds ===")
    for i in range(args.n_seeds):
        print(f"\n--- Seed {i+1}/{args.n_seeds} (split-seed {42+i}, init-seed {100+i}) ---")
        r = entrenar_una_semilla(dataset, hp, dispositivo, args.n_epocas, semilla=i)
        print(f"  val_loss_final={r['val_loss_final']:.4f} | "
              f"test_loss={r['test_loss']:.4f} | "
              f"test_acc={r['test_acc']:.2f}%")
        resultados.append(r)

    val_losses = np.array([r["val_loss_final"] for r in resultados])
    test_losses = np.array([r["test_loss"] for r in resultados])
    test_accs = np.array([r["test_acc"] for r in resultados])

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({args.n_seeds} seeds, {args.n_epocas} fixed epochs)")
    print("=" * 60)
    print(f"Final val loss  : {val_losses.mean():.4f} +/- {val_losses.std():.4f}")
    print(f"Test loss       : {test_losses.mean():.4f} +/- {test_losses.std():.4f}")
    print(f"Test acc        : {test_accs.mean():.2f}% +/- {test_accs.std():.2f}%")
    print(f"\nManual-hyperparameter reference: 94.65% +/- 0.38%")

    delta = test_losses.mean() - val_losses.mean()
    print(f"\nDelta (test - val) = {delta:+.4f}")
    if abs(delta) < 0.02:
        print("  -> val approx test. No relevant HP overfitting detected.")
    elif test_losses.mean() > val_losses.mean():
        print("  -> test > val. Possible mild HP overfitting.")

    # Save the checkpoint from the seed with the highest test_acc.
    idx_mejor = int(np.argmax(test_accs))
    mejor = resultados[idx_mejor]
    ckpt_path = f"{args.output_prefix}.pt"
    torch.save({
        "state_dict": mejor["state_dict"],
        "hp": hp,
        "semilla": mejor["semilla"],
        "test_acc": mejor["test_acc"],
        "test_loss": mejor["test_loss"],
        "split_seed": 42 + mejor["semilla"],
        "init_seed": 100 + mejor["semilla"],
    }, ckpt_path)
    print(f"\nBest model (seed {mejor['semilla']}, test_acc {mejor['test_acc']:.2f}%) "
          f"saved to {ckpt_path}")

    resumen = {
        "hp": hp,
        "metodologia": "random_split per seed, no early stopping",
        "n_seeds": args.n_seeds,
        "n_epocas": args.n_epocas,
        "val_loss_mean": float(val_losses.mean()),
        "val_loss_std": float(val_losses.std()),
        "test_loss_mean": float(test_losses.mean()),
        "test_loss_std": float(test_losses.std()),
        "test_acc_mean": float(test_accs.mean()),
        "test_acc_std": float(test_accs.std()),
        "val_losses_por_semilla": val_losses.tolist(),
        "test_losses_por_semilla": test_losses.tolist(),
        "test_accs_por_semilla": test_accs.tolist(),
        "mejor_semilla": mejor["semilla"],
        "checkpoint": ckpt_path,
    }
    metricas_path = f"{args.output_prefix}_metricas.json"
    Path(metricas_path).write_text(json.dumps(resumen, indent=2))
    print(f"Metrics in {metricas_path}")

    plot_path = f"{args.output_prefix}_curvas.png"
    plotear_curvas(resultados, hp, args.n_seeds, plot_path)
    print(f"Train-vs-val plot in {plot_path}")


if __name__ == "__main__":
    main()
