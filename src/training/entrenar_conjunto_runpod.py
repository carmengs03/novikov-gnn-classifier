"""
Joint training on dim2 + dim3 via zero-padding.

Self-contained script: includes the architecture, training loop, checkpoint
saving and plots. Only the combined dataset file is required.

Configuration:
    - aggr=max, pool=mean, 3 layers, residual connections
    - lr=0.00122, dim_oculta=64, batch_size=16
    - dim_aristas=3 (all graphs carry an edge_attr of dim 3: dim-2 graphs
      were padded with an extra zero channel)
    - 10 seeds x 150 fixed epochs, random_split per seed

Post-hoc "elbow" analysis:
    Saves a model checkpoint every CHECKPOINT_INTERVAL epochs and, when each
    seed finishes, evaluates test on every checkpoint. The result is a full
    per-epoch test_acc curve that lets us locate the optimal epoch by looking
    at the data itself, with no early stopping and no implicit best-checkpoint
    selection.

Required files:
    - dataset_combinado_dim2_dim3.pt

Run:
    python entrenar_conjunto_runpod.py
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
N_EPOCAS = 150
CHECKPOINT_INTERVAL = 10  # save a checkpoint every 10 epochs, plus epoch 1 and the last one
DATASET_PATH = "dataset_combinado_dim2_dim3_dificiles.pt"
# Preference order for hyperparameters: joint-model Optuna JSON if it exists,
# then the dim2 Optuna JSON, then the hardcoded fallback values below.
BEST_PARAMS_PATH = "best_params_novikov_conjunto_hp.json"
FALLBACK_PARAMS_PATH = "best_params_novikov_dim2_hp.json"
OUTPUT_PREFIX = "modelo_conjunto_dim23_dificil"

# Fallback hyperparameters if no Optuna JSON is available.
LR = 0.0012159853408094212
DIM_OCULTA = 64
BATCH_SIZE = 16


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


def evaluar_test(modelo, test_loader, criterio, dispositivo, n_test):
    modelo.eval()
    aciertos_total = total_total = 0
    aciertos_2d = total_2d = 0
    aciertos_3d = total_3d = 0
    test_loss_acum = 0.0
    with torch.no_grad():
        for lote in test_loader:
            lote = lote.to(dispositivo)
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            test_loss_acum += criterio(pred, etiq).item() * lote.num_graphs
            preds_binarias = (pred >= 0.5).float()
            correctos = (preds_binarias == etiq)

            dims = torch.tensor(
                [g.dim_algebra for g in lote.to_data_list()],
                device=dispositivo,
            )
            mask_2d = dims == 2
            mask_3d = dims == 3

            aciertos_total += correctos.sum().item()
            total_total += etiq.size(0)
            aciertos_2d += correctos[mask_2d].sum().item()
            total_2d += mask_2d.sum().item()
            aciertos_3d += correctos[mask_3d].sum().item()
            total_3d += mask_3d.sum().item()

    return {
        "test_loss": test_loss_acum / n_test,
        "test_acc_global": (aciertos_total / total_total) * 100 if total_total else 0.0,
        "test_acc_2d": (aciertos_2d / total_2d) * 100 if total_2d else 0.0,
        "test_acc_3d": (aciertos_3d / total_3d) * 100 if total_3d else 0.0,
        "n_test_2d": total_2d,
        "n_test_3d": total_3d,
    }


def entrenar_una_semilla(dataset, dispositivo, n_epocas, semilla):
    fijar_semilla(42 + semilla)
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    datos_train, datos_val, datos_test = random_split(dataset, [n_train, n_val, n_test])

    torch.manual_seed(100 + semilla)
    modelo = RedNovikov(dim_nodos=6, dim_aristas=3, dim_oculta=DIM_OCULTA).to(dispositivo)
    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    train_loader = DataLoader(datos_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(datos_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(datos_test, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    checkpoints = {}
    h_train, h_val = [], []

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
        h_val.append(loss_val_acum / n_val)

        # Save a checkpoint every CHECKPOINT_INTERVAL epochs, plus epoch 1 and the last one.
        epoca_1indexed = epoca + 1
        if (epoca_1indexed == 1
                or epoca_1indexed % CHECKPOINT_INTERVAL == 0
                or epoca_1indexed == n_epocas):
            checkpoints[epoca_1indexed] = {
                k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()
            }
            print(f"    epoch {epoca_1indexed:3d}/{n_epocas} | train={h_train[-1]:.4f} | "
                  f"val={h_val[-1]:.4f} | time={time.time() - t_epoca:.1f}s | "
                  f"[checkpoint]",
                  flush=True)

    # Test evaluation over every saved checkpoint.
    print(f"  evaluating test on {len(checkpoints)} checkpoints...", flush=True)
    eval_por_epoca = {}
    for epoca, sd in sorted(checkpoints.items()):
        modelo.load_state_dict(sd)
        eval_por_epoca[epoca] = evaluar_test(modelo, test_loader, criterio, dispositivo, n_test)

    print("    epoch | val_loss | test_loss | test_acc_global | dim2 | dim3", flush=True)
    for epoca in sorted(checkpoints.keys()):
        e = eval_por_epoca[epoca]
        vl = h_val[epoca - 1]
        print(f"    {epoca:5d} | {vl:8.4f} | {e['test_loss']:9.4f} | "
              f"{e['test_acc_global']:14.2f}% | {e['test_acc_2d']:5.2f}% | "
              f"{e['test_acc_3d']:5.2f}%", flush=True)

    return {
        "semilla": semilla,
        "h_train": h_train,
        "h_val": h_val,
        "eval_por_epoca": eval_por_epoca,
        "checkpoints": checkpoints,
    }


def plotear_curvas_loss(resultados, n_seeds, plot_path):
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
    plt.title(f"Joint model dim2 + dim3 ({n_seeds} seeds x {train_arr.shape[1]} epochs)\n"
              f"lr={LR:.4g} | dim_oculta={DIM_OCULTA} | batch_size={BATCH_SIZE}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plotear_acc_por_epoca(resultados, n_seeds, plot_path):
    """Mean per-epoch test_acc curve across checkpoints. Used to locate the elbow
    and pick the optimal epoch."""
    epocas = sorted(resultados[0]["eval_por_epoca"].keys())
    acc_global = np.array([[r["eval_por_epoca"][e]["test_acc_global"] for e in epocas] for r in resultados])
    acc_2d = np.array([[r["eval_por_epoca"][e]["test_acc_2d"] for e in epocas] for r in resultados])
    acc_3d = np.array([[r["eval_por_epoca"][e]["test_acc_3d"] for e in epocas] for r in resultados])

    plt.figure(figsize=(10, 6))
    for arr, label, color in [(acc_global, "Global", "tab:purple"),
                                (acc_2d, "dim 2", "tab:blue"),
                                (acc_3d, "dim 3", "tab:green")]:
        m = arr.mean(0)
        s = arr.std(0)
        plt.plot(epocas, m, label=f"{label} (mean)", color=color, marker="o")
        plt.fill_between(epocas, m - s, m + s, alpha=0.2, color=color)
    plt.xlabel("Checkpoint epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"Test accuracy per checkpoint epoch ({n_seeds} seeds)\n"
              f"Used to locate the optimal 'elbow'")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dispositivo}")
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    hp_source = None
    for candidato in [BEST_PARAMS_PATH, FALLBACK_PARAMS_PATH]:
        if os.path.exists(candidato):
            with open(candidato) as f:
                best = json.load(f)["best_params"]
            LR = best["lr"]
            DIM_OCULTA = best["dim_oculta"]
            BATCH_SIZE = best["batch_size"]
            hp_source = candidato
            break
    print(f"Hyperparameters loaded from: {hp_source or 'hardcoded defaults'}")

    print(f"\nOptuna hyperparameters:")
    print(f"  lr={LR}  dim_oculta={DIM_OCULTA}  batch_size={BATCH_SIZE}")
    print(f"  architecture: MAX + MEAN + 3 layers + residual, dim_aristas=3")
    print(f"\nMethodology: {N_SEEDS} seeds x {N_EPOCAS} fixed epochs, random_split per seed")
    print(f"Checkpoints saved every {CHECKPOINT_INTERVAL} epochs for post-hoc elbow analysis")

    print(f"\nLoading {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Dataset: {len(dataset):,} graphs")
    n_2d = sum(1 for g in dataset if g.dim_algebra == 2)
    n_3d = sum(1 for g in dataset if g.dim_algebra == 3)
    print(f"  - dim 2: {n_2d:,} ({n_2d/len(dataset)*100:.1f}%)")
    print(f"  - dim 3: {n_3d:,} ({n_3d/len(dataset)*100:.1f}%)")

    t_global = time.time()
    resultados = []
    print(f"\n=== Training {N_SEEDS} seeds ===")
    for i in range(N_SEEDS):
        t_semilla = time.time()
        print(f"\n--- Seed {i+1}/{N_SEEDS} (split-seed {42+i}, init-seed {100+i}) ---")
        r = entrenar_una_semilla(dataset, dispositivo, N_EPOCAS, semilla=i)
        elapsed = time.time() - t_semilla
        print(f"  seed time: {elapsed/60:.1f} min")
        resultados.append(r)

    print(f"\nTotal training time: {(time.time() - t_global)/60:.1f} min")

    # Post-hoc analysis: pick the epoch by mean VAL LOSS (avoids peeking at
    # test). Then report test_acc at that epoch once.
    epocas = sorted(resultados[0]["eval_por_epoca"].keys())
    val_loss_por_epoca = {
        e: np.mean([r["h_val"][e - 1] for r in resultados]) for e in epocas
    }
    acc_global_por_epoca = {
        e: np.mean([r["eval_por_epoca"][e]["test_acc_global"] for r in resultados])
        for e in epocas
    }
    epoca_optima = min(val_loss_por_epoca, key=val_loss_por_epoca.get)

    print("\n" + "=" * 70)
    print(f"PER-EPOCH RESULTS (mean over {N_SEEDS} seeds)")
    print("=" * 70)
    print(f"{'epoch':>6} | {'val_loss':>8} | {'test_acc_glob':>13} | {'dim2':>6} | {'dim3':>6}")
    for e in epocas:
        val_mean = np.mean([r["h_val"][e - 1] for r in resultados])
        acc_g = np.mean([r["eval_por_epoca"][e]["test_acc_global"] for r in resultados])
        acc_g_std = np.std([r["eval_por_epoca"][e]["test_acc_global"] for r in resultados])
        acc_2 = np.mean([r["eval_por_epoca"][e]["test_acc_2d"] for r in resultados])
        acc_3 = np.mean([r["eval_por_epoca"][e]["test_acc_3d"] for r in resultados])
        marca = "  <--" if e == epoca_optima else ""
        print(f"{e:>6} | {val_mean:>8.4f} | {acc_g:>10.2f}+/-{acc_g_std:.2f}% | "
              f"{acc_2:>5.2f}% | {acc_3:>5.2f}%{marca}")
    print("=" * 70)
    print(f"Best epoch (by mean VAL LOSS, avoids peeking at test): {epoca_optima}")
    print(f"  val_loss:        {val_loss_por_epoca[epoca_optima]:.4f}")
    print(f"  test_acc_global: {acc_global_por_epoca[epoca_optima]:.2f}%  (reported, not used for selection)")
    print(f"\nReference dim2-only model: 95.75% +/- 0.45%")

    # Store the best-by-val-loss checkpoint at that epoch (not by test_acc).
    val_losses_en_optima = [r["h_val"][epoca_optima - 1] for r in resultados]
    accs_optimas = [r["eval_por_epoca"][epoca_optima]["test_acc_global"] for r in resultados]
    idx_mejor = int(np.argmin(val_losses_en_optima))
    mejor = resultados[idx_mejor]
    ckpt_path = f"{OUTPUT_PREFIX}.pt"
    torch.save({
        "state_dict": mejor["checkpoints"][epoca_optima],
        "hp": {"lr": LR, "dim_oculta": DIM_OCULTA, "batch_size": BATCH_SIZE},
        "epoca_optima": epoca_optima,
        "semilla": mejor["semilla"],
        "eval": mejor["eval_por_epoca"][epoca_optima],
        "split_seed": 42 + mejor["semilla"],
        "init_seed": 100 + mejor["semilla"],
    }, ckpt_path)
    print(f"\nBest model (seed {mejor['semilla']}, epoch {epoca_optima}) "
          f"saved to {ckpt_path}")

    resumen = {
        "hp": {"lr": LR, "dim_oculta": DIM_OCULTA, "batch_size": BATCH_SIZE},
        "metodologia": (
            f"{N_SEEDS} seeds x {N_EPOCAS} fixed epochs, random_split per seed. "
            f"Checkpoints every {CHECKPOINT_INTERVAL} epochs, per-epoch test evaluation."
        ),
        "n_seeds": N_SEEDS,
        "n_epocas": N_EPOCAS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "epoca_optima": epoca_optima,
        "test_acc_global_en_epoca_optima": {
            "mean": float(acc_global_por_epoca[epoca_optima]),
            "std": float(np.std(accs_optimas)),
            "per_seed": [float(a) for a in accs_optimas],
        },
        "tabla_por_epoca": {
            int(e): {
                "val_loss_mean": float(np.mean([r["h_val"][e - 1] for r in resultados])),
                "val_loss_std": float(np.std([r["h_val"][e - 1] for r in resultados])),
                "test_acc_global_mean": float(np.mean([r["eval_por_epoca"][e]["test_acc_global"] for r in resultados])),
                "test_acc_global_std": float(np.std([r["eval_por_epoca"][e]["test_acc_global"] for r in resultados])),
                "test_acc_2d_mean": float(np.mean([r["eval_por_epoca"][e]["test_acc_2d"] for r in resultados])),
                "test_acc_2d_std": float(np.std([r["eval_por_epoca"][e]["test_acc_2d"] for r in resultados])),
                "test_acc_3d_mean": float(np.mean([r["eval_por_epoca"][e]["test_acc_3d"] for r in resultados])),
                "test_acc_3d_std": float(np.std([r["eval_por_epoca"][e]["test_acc_3d"] for r in resultados])),
                "test_loss_mean": float(np.mean([r["eval_por_epoca"][e]["test_loss"] for r in resultados])),
            }
            for e in epocas
        },
        "por_semilla": [
            {
                "semilla": r["semilla"],
                "h_train": r["h_train"],
                "h_val": r["h_val"],
                "eval_por_epoca": {
                    int(e): {k: float(v) if not isinstance(v, int) else int(v)
                             for k, v in r["eval_por_epoca"][e].items()}
                    for e in epocas
                },
            }
            for r in resultados
        ],
    }
    metricas_path = f"{OUTPUT_PREFIX}_metricas.json"
    with open(metricas_path, "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"Metrics in {metricas_path}")

    plotear_curvas_loss(resultados, N_SEEDS, f"{OUTPUT_PREFIX}_curvas_loss.png")
    print(f"Loss curves plot: {OUTPUT_PREFIX}_curvas_loss.png")
    plotear_acc_por_epoca(resultados, N_SEEDS, f"{OUTPUT_PREFIX}_acc_por_epoca.png")
    print(f"Test-accuracy-per-epoch plot: {OUTPUT_PREFIX}_acc_por_epoca.png")
