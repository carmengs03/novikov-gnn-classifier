"""
Self-contained RunPod script for the final training run with optimised
hyperparameters.

Uses 100 fixed epochs without early stopping, random_split per seed, 10 seeds
and reports test_acc as mean +/- std. Hyperparameters come from Optuna
instead of the manual settings.

Required files (same directory as this script):
    - dataset_enriquecido_dim2_50000_DATOS.pt
    - best_params_novikov_dim2_hp.json

Run:
    python entrenar_mejor_modelo_runpod.py
"""
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

N_SEEDS = 10
N_EPOCAS = 100
DATASET_PATH = "dataset_enriquecido_dim2_50000_DATOS_dificiles.pt"
BEST_PARAMS_PATH = "best_params_novikov_dim2_hp.json"
OUTPUT_PREFIX = "modelo_final_dim2_dificil"


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
    def __init__(self, dim_nodos=6, dim_aristas=2, dim_oculta=32, n_capas=3, residual=True):
        super().__init__()
        self.residual = residual
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta)
        self.capas_mp = nn.ModuleList(
            [CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta) for _ in range(n_capas)]
        )
        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, datos):
        x, edge_index, edge_attr, batch = datos.x, datos.edge_index, datos.edge_attr, datos.batch
        x = F.relu(self.proyeccion_entrada(x))
        for capa in self.capas_mp:
            h = capa(x, edge_index, edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)
        return self.clasificador(global_mean_pool(x, batch))


def entrenar_una_semilla(dataset, hp, dispositivo, n_epocas, semilla):
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

    # With small batch size and dataset in RAM, num_workers=0 is faster.
    train_loader = DataLoader(datos_train, batch_size=hp["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(datos_val, batch_size=hp["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(datos_test, batch_size=hp["batch_size"], shuffle=False,
                             num_workers=0, pin_memory=False)

    h_train, h_val = [], []
    mejor_val_loss = float("inf")
    mejor_state_dict = None
    mejor_epoca = 0

    for epoca in range(n_epocas):
        t_epoca = time.time()
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
        val_loss = loss_val_acum / n_val
        h_val.append(val_loss)

        # Keep best-by-val-loss checkpoint within the seed.
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_epoca = epoca + 1
            mejor_state_dict = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

        if epoca == 0 or (epoca + 1) % 5 == 0 or epoca == n_epocas - 1:
            print(f"    epoch {epoca+1:3d}/{n_epocas} | train={h_train[-1]:.4f} | "
                  f"val={val_loss:.4f} | time={time.time() - t_epoca:.1f}s", flush=True)

    # Test is always evaluated on the best-val-loss checkpoint.
    modelo.load_state_dict(mejor_state_dict)
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
        "val_loss_final": mejor_val_loss,   # val_loss of the chosen checkpoint
        "mejor_epoca": mejor_epoca,
        "test_loss": test_loss_acum / n_test,
        "test_acc": (aciertos / n_test) * 100,
        "h_train": h_train,
        "h_val": h_val,
        "state_dict": mejor_state_dict,
    }


def plotear_curvas(resultados, hp, n_seeds, plot_path):
    train_arr = np.array([r["h_train"] for r in resultados])
    val_arr = np.array([r["h_val"] for r in resultados])
    epochs = np.arange(1, train_arr.shape[1] + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_arr.mean(0), label="Train loss (mean)", color="tab:blue")
    plt.fill_between(epochs, train_arr.mean(0) - train_arr.std(0),
                     train_arr.mean(0) + train_arr.std(0),
                     alpha=0.2, color="tab:blue", label="Train +/- std")
    plt.plot(epochs, val_arr.mean(0), label="Val loss (mean)", color="tab:orange")
    plt.fill_between(epochs, val_arr.mean(0) - val_arr.std(0),
                     val_arr.mean(0) + val_arr.std(0),
                     alpha=0.2, color="tab:orange", label="Val +/- std")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"Final model with optimised hyperparameters ({n_seeds} seeds x {train_arr.shape[1]} epochs)\n"
              f"lr={hp['lr']:.4g} | dim_oculta={hp['dim_oculta']} | batch_size={hp['batch_size']}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dispositivo}")
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(BEST_PARAMS_PATH) as f:
        cfg = json.load(f)
    hp = cfg["best_params"]
    print(f"\nHyperparameters loaded from {BEST_PARAMS_PATH}:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

    print(f"\nMethodology: {N_SEEDS} seeds x {N_EPOCAS} fixed epochs, random_split per seed")

    print(f"\nLoading {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Dataset: {len(dataset)} graphs")

    t_global = time.time()
    resultados = []
    print(f"\n=== Training {N_SEEDS} seeds ===")
    for i in range(N_SEEDS):
        t_semilla = time.time()
        print(f"\n--- Seed {i+1}/{N_SEEDS} (split-seed {42+i}, init-seed {100+i}) ---")
        r = entrenar_una_semilla(dataset, hp, dispositivo, N_EPOCAS, semilla=i)
        elapsed = time.time() - t_semilla
        print(f"  val_loss_final={r['val_loss_final']:.4f} | "
              f"test_loss={r['test_loss']:.4f} | "
              f"test_acc={r['test_acc']:.2f}% | "
              f"time={elapsed/60:.1f} min")
        resultados.append(r)

    val_losses = np.array([r["val_loss_final"] for r in resultados])
    test_losses = np.array([r["test_loss"] for r in resultados])
    test_accs = np.array([r["test_acc"] for r in resultados])

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({N_SEEDS} seeds, {N_EPOCAS} fixed epochs)")
    print("=" * 60)
    print(f"Final val loss  : {val_losses.mean():.4f} +/- {val_losses.std():.4f}")
    print(f"Test loss       : {test_losses.mean():.4f} +/- {test_losses.std():.4f}")
    print(f"Test acc        : {test_accs.mean():.2f}% +/- {test_accs.std():.2f}%")
    print(f"\nManual-hyperparameter reference: 94.65% +/- 0.38%")
    print(f"Total time: {(time.time() - t_global)/60:.1f} min")

    delta = test_losses.mean() - val_losses.mean()
    print(f"\nDelta (test - val) = {delta:+.4f}")
    if abs(delta) < 0.02:
        print("  -> val approx test. No relevant HP overfitting detected.")
    elif test_losses.mean() > val_losses.mean():
        print("  -> test > val. Possible mild HP overfitting.")

    # Honest selection: pick by VAL LOSS (smaller is better), not by test.
    # We report the test_acc of that seed once, without peeking at test.
    idx_mejor = int(np.argmin(val_losses))
    mejor = resultados[idx_mejor]
    ckpt_path = f"{OUTPUT_PREFIX}.pt"
    torch.save({
        "state_dict": mejor["state_dict"],
        "hp": hp,
        "semilla": mejor["semilla"],
        "val_loss": mejor["val_loss_final"],
        "test_acc": mejor["test_acc"],
        "test_loss": mejor["test_loss"],
        "split_seed": 42 + mejor["semilla"],
        "init_seed": 100 + mejor["semilla"],
        "seleccion": "by val_loss (avoids peeking at test)",
    }, ckpt_path)
    print(f"\nBest model by val_loss (seed {mejor['semilla']}, "
          f"val_loss={mejor['val_loss_final']:.4f}, test_acc={mejor['test_acc']:.2f}%) "
          f"saved to {ckpt_path}")

    resumen = {
        "hp": hp,
        "metodologia": "random_split per seed, no early stopping",
        "n_seeds": N_SEEDS,
        "n_epocas": N_EPOCAS,
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
    metricas_path = f"{OUTPUT_PREFIX}_metricas.json"
    with open(metricas_path, "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"Metrics in {metricas_path}")

    plot_path = f"{OUTPUT_PREFIX}_curvas.png"
    plotear_curvas(resultados, hp, N_SEEDS, plot_path)
    print(f"Train-vs-val plot in {plot_path}")
