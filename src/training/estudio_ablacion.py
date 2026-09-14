import json

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
import matplotlib.pyplot as plt
import numpy as np
import random
import os

DATASET_PATH = "dataset_enriquecido_dim2_50000_DATOS_dificiles.pt"
SEMILLAS = list(range(42, 52))
EPOCAS = 100

# Output directory: if Google Drive is mounted we persist results there
# (survives session resets); otherwise fall back to the current directory.
DRIVE_DIR = "/content/drive/MyDrive/tfm_ablacion"
OUT_DIR = DRIVE_DIR if os.path.isdir("/content/drive/MyDrive") else "."


def fijar_semillas_absolutas(semilla=42):
    """Set every RNG so the experiment is fully deterministic and reproducible."""
    random.seed(semilla)
    os.environ['PYTHONHASHSEED'] = str(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

fijar_semillas_absolutas(42)


class CapaAlgebraica(MessagePassing):
    def __init__(self, dim_nodos_entrada, dim_aristas, dim_oculta, tipo_agregacion):
        super(CapaAlgebraica, self).__init__(aggr=tipo_agregacion)
        self.transformador_mensajes = nn.Sequential(
            nn.Linear(dim_nodos_entrada + dim_aristas, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        info_combinada = torch.cat([x_j, edge_attr], dim=1)
        return self.transformador_mensajes(info_combinada)

class RedNovikov(nn.Module):
    def __init__(self, dim_nodos=6, dim_aristas=2, dim_oculta=32,
                 tipo_agregacion='add', tipo_pooling='mean',
                 n_capas=2, residual=False):
        super(RedNovikov, self).__init__()
        self.tipo_pooling = tipo_pooling
        self.n_capas = n_capas
        self.residual = residual

        # Initial projection is only needed when residual connections are on
        # so that every layer operates in dim_oculta -> dim_oculta.
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta) if residual else None
        dim_entrada_cap1 = dim_oculta if residual else dim_nodos

        self.capas = nn.ModuleList()
        for i in range(n_capas):
            dim_in = dim_entrada_cap1 if i == 0 else dim_oculta
            self.capas.append(CapaAlgebraica(dim_in, dim_aristas, dim_oculta, tipo_agregacion))

        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, datos):
        x, edge_index, edge_attr, batch = datos.x, datos.edge_index, datos.edge_attr, datos.batch

        if self.residual:
            x = torch.relu(self.proyeccion_entrada(x))
            for capa in self.capas:
                h = capa(x, edge_index, edge_attr)
                x = torch.relu(x + h)
        else:
            for capa in self.capas:
                x = torch.relu(capa(x, edge_index, edge_attr))

        if self.tipo_pooling == 'mean':
            x_grafo = global_mean_pool(x, batch)
        elif self.tipo_pooling == 'add':
            x_grafo = global_add_pool(x, batch)
        elif self.tipo_pooling == 'max':
            x_grafo = global_max_pool(x, batch)

        return self.clasificador(x_grafo)


def entrenar_modelo(modelo, train_loader, val_loader, test_loader, device, epocas=EPOCAS):
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio = nn.BCELoss()

    historial_train_loss = []
    historial_val_loss = []

    for epoca in range(epocas):
        modelo.train()
        loss_train_acumulado = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizador.zero_grad()
            predicciones = modelo(batch).view(-1)
            etiquetas = batch.y.float().view(-1)
            loss = criterio(predicciones, etiquetas)
            loss.backward()
            optimizador.step()
            loss_train_acumulado += loss.item() * batch.num_graphs

        loss_train_medio = loss_train_acumulado / len(train_loader.dataset)
        historial_train_loss.append(loss_train_medio)

        modelo.eval()
        loss_val_acumulado = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                predicciones = modelo(batch).view(-1)
                etiquetas = batch.y.float().view(-1)
                loss = criterio(predicciones, etiquetas)
                loss_val_acumulado += loss.item() * batch.num_graphs

        loss_val_medio = loss_val_acumulado / len(val_loader.dataset)
        historial_val_loss.append(loss_val_medio)

    modelo.eval()
    correctos = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predicciones = modelo(batch).view(-1)
            etiquetas = batch.y.float().view(-1)
            predicciones_binarias = (predicciones >= 0.5).float()
            correctos += (predicciones_binarias == etiquetas).sum().item()
            total += etiquetas.size(0)

    precision_test = (correctos / total) * 100
    return precision_test, historial_train_loss, historial_val_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting full ablation study on: {device}")

    print(f"Loading {DATASET_PATH}...")
    dataset_original = torch.load(DATASET_PATH, weights_only=False)

    agregaciones = ['add', 'mean', 'max']
    poolings = ['add', 'mean', 'max']

    # precisiones[aggr][pool] = list of accuracies (one per seed)
    precisiones = {aggr: {pool: [] for pool in poolings} for aggr in agregaciones}
    historiales_loss = {aggr: {} for aggr in agregaciones}

    print(f"\n--- TRAINING {len(agregaciones)*len(poolings)} MODELS x {len(SEMILLAS)} SEEDS ---")

    for semilla in SEMILLAS:
        print(f"\n=== SEED {semilla} ===")
        fijar_semillas_absolutas(semilla)

        # Shuffle depends on the seed, so each seed sees a different split;
        # this is what gives us +/- std across the partition.
        dataset = list(dataset_original)
        random.seed(semilla)
        random.shuffle(dataset)

        n_train = int(len(dataset) * 0.7)
        n_val = int(len(dataset) * 0.15)
        train_dataset = dataset[:n_train]
        val_dataset = dataset[n_train:n_train + n_val]
        test_dataset = dataset[n_train + n_val:]

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        for aggr in agregaciones:
            for pool in poolings:
                torch.manual_seed(semilla)

                print(f"[seed {semilla}][{aggr.upper()} + {pool.upper()}] Training...")
                modelo = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=32,
                                    tipo_agregacion=aggr, tipo_pooling=pool).to(device)

                precision, h_train, h_val = entrenar_modelo(modelo, train_loader, val_loader, test_loader, device, epocas=EPOCAS)

                precisiones[aggr][pool].append(precision)
                if semilla == SEMILLAS[0]:
                    historiales_loss[aggr][pool] = (h_train, h_val)
                print(f"   -> Test accuracy: {precision:.2f}%")

        # Incremental save: if the session is interrupted mid-run we still
        # keep the seeds completed so far.
        os.makedirs(OUT_DIR, exist_ok=True)
        parcial_path = os.path.join(OUT_DIR, "ablacion_dificil_parcial.json")
        with open(parcial_path, "w") as f:
            json.dump({
                "dataset": DATASET_PATH,
                "epocas": EPOCAS,
                "semillas_completadas": SEMILLAS[:SEMILLAS.index(semilla) + 1],
                "por_semilla": precisiones,
            }, f, indent=2)
        print(f"[checkpoint] Seed {semilla} saved to {parcial_path}")

    # Means and standard deviations across seeds.
    resultados_precision = {aggr: [float(np.mean(precisiones[aggr][pool])) for pool in poolings]
                            for aggr in agregaciones}
    resultados_std = {aggr: [float(np.std(precisiones[aggr][pool])) for pool in poolings]
                      for aggr in agregaciones}

    print("\n=== SUMMARY (mean +/- std across seeds) ===")
    for aggr in agregaciones:
        for j, pool in enumerate(poolings):
            m, s = resultados_precision[aggr][j], resultados_std[aggr][j]
            print(f"  {aggr.upper():>4s} + {pool.upper():>4s}: {m:.2f}% +/- {s:.2f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    resultados_path = os.path.join(OUT_DIR, "ablacion_dificil_resultados.json")
    with open(resultados_path, "w") as f:
        json.dump({
            "dataset": DATASET_PATH,
            "epocas": EPOCAS,
            "semillas": SEMILLAS,
            "por_semilla": precisiones,
            "media": resultados_precision,
            "std": resultados_std,
        }, f, indent=2)
    print(f"\nSaved: {resultados_path}")

    print("\nGenerating plots...")

    x = np.arange(len(poolings))
    width = 0.25
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - width, resultados_precision['add'], width, yerr=resultados_std['add'],
                     label="Aggregation: ADD", color='#4C72B0', capsize=4)
    rects2 = ax1.bar(x, resultados_precision['mean'], width, yerr=resultados_std['mean'],
                     label="Aggregation: MEAN", color='#55A868', capsize=4)
    rects3 = ax1.bar(x + width, resultados_precision['max'], width, yerr=resultados_std['max'],
                     label="Aggregation: MAX", color='#C44E52', capsize=4)

    ax1.set_ylabel('Test accuracy (%)', fontsize=12)
    ax1.set_title(f'Final accuracy per architecture (Hard dataset, {len(SEMILLAS)} seeds)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['POOL: ADD', 'POOL: MEAN', 'POOL: MAX'], fontsize=11)
    ax1.legend(loc='lower right')
    ax1.set_ylim([40, 100])
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    fig1.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablacion_dificil_precision.png"), dpi=300)

    fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig2.suptitle('Overfitting analysis: Train vs Val loss', fontsize=16, fontweight='bold')

    for i, aggr in enumerate(agregaciones):
        for j, pool in enumerate(poolings):
            ax = axes[i, j]
            h_train, h_val = historiales_loss[aggr][pool]
            ax.plot(h_train, label='Train Loss', color='blue', linewidth=2)
            ax.plot(h_val, label='Val Loss', color='orange', linewidth=2)
            ax.set_title(f'Aggr: {aggr.upper()} | Pool: {pool.upper()}', fontsize=12)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)

    fig2.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(OUT_DIR, "ablacion_dificil_loss.png"), dpi=300)

    # Mini-ablation on the best combination (MAX + MEAN): depth and residual
    # connections. Configurations: {2, 3, 4 layers} x {without, with residual}.
    print("\n" + "=" * 70)
    print("MINI-ABLATION: depth + residual (MAX + MEAN)")
    print("=" * 70)

    configuraciones_mini = [
        ("n=2, no residual", 2, False),
        ("n=3, no residual", 3, False),
        ("n=4, no residual", 4, False),
        ("n=2, residual", 2, True),
        ("n=3, residual", 3, True),
        ("n=4, residual", 4, True),
    ]

    precisiones_mini = {nombre: [] for nombre, _, _ in configuraciones_mini}
    historiales_mini = {}

    for semilla in SEMILLAS:
        print(f"\n=== SEED {semilla} ===")
        fijar_semillas_absolutas(semilla)

        dataset = list(dataset_original)
        random.seed(semilla)
        random.shuffle(dataset)

        n_train = int(len(dataset) * 0.7)
        n_val = int(len(dataset) * 0.15)
        train_dataset = dataset[:n_train]
        val_dataset = dataset[n_train:n_train + n_val]
        test_dataset = dataset[n_train + n_val:]

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        for nombre, n_capas, residual in configuraciones_mini:
            torch.manual_seed(semilla)
            print(f"[seed {semilla}][{nombre}] Training...")
            modelo = RedNovikov(
                dim_nodos=6, dim_aristas=2, dim_oculta=32,
                tipo_agregacion='max', tipo_pooling='mean',
                n_capas=n_capas, residual=residual,
            ).to(device)
            precision, h_train, h_val = entrenar_modelo(
                modelo, train_loader, val_loader, test_loader, device, epocas=EPOCAS,
            )
            precisiones_mini[nombre].append(precision)
            if semilla == SEMILLAS[0]:
                historiales_mini[nombre] = (h_train, h_val)
            print(f"   -> Test accuracy: {precision:.2f}%")

    print("\n=== MINI-ABLATION SUMMARY ===")
    resumen_mini = {}
    for nombre, _, _ in configuraciones_mini:
        m = float(np.mean(precisiones_mini[nombre]))
        s = float(np.std(precisiones_mini[nombre]))
        resumen_mini[nombre] = {"media": m, "std": s,
                                 "por_semilla": precisiones_mini[nombre]}
        print(f"  {nombre:>20s}: {m:.2f}% +/- {s:.2f}%")

    mini_path = os.path.join(OUT_DIR, "ablacion_dificil_mini_resultados.json")
    with open(mini_path, "w") as f:
        json.dump({
            "dataset": DATASET_PATH,
            "epocas": EPOCAS,
            "semillas": SEMILLAS,
            "arquitectura_fija": "aggr=max, pool=mean, dim_oculta=32, lr=1e-3, batch=64",
            "resultados": resumen_mini,
        }, f, indent=2)
    print(f"\nSaved: {mini_path}")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    nombres_mini = [n for n, _, _ in configuraciones_mini]
    medias = [resumen_mini[n]["media"] for n in nombres_mini]
    stds = [resumen_mini[n]["std"] for n in nombres_mini]
    colores = ['#4C72B0', '#4C72B0', '#4C72B0', '#C44E52', '#C44E52', '#C44E52']
    xs = np.arange(len(nombres_mini))
    ax3.bar(xs, medias, yerr=stds, capsize=5, color=colores)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(nombres_mini, rotation=20, ha='right')
    ax3.set_ylabel('Test accuracy (%)', fontsize=12)
    ax3.set_title(f'Mini-ablation: depth x residual (MAX+MEAN, {len(SEMILLAS)} seeds)',
                  fontsize=13, fontweight='bold')
    ax3.grid(axis='y', linestyle=':', alpha=0.7)
    for xi, m, s in zip(xs, medias, stds):
        ax3.text(xi, m + s + 0.4, f'{m:.1f}', ha='center', fontsize=9)
    fig3.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablacion_dificil_mini_precision.png"), dpi=300)
    print("Saved: ablacion_dificil_mini_precision.png")
