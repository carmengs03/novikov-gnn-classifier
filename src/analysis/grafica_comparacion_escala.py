"""
Controlled large-scale comparison: baseline (1 feature) vs enriched (6 features).
Same dataset (44,386 graphs), same seed (42), same lr (0.001), same epochs (50).
The only differing factor is the node-feature dimensionality.
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from modelo_mpnn_baseline import RedNovikov


def entrenar(dataset, dim_nodos, nombre):
    torch.manual_seed(42)
    dispositivo = torch.device('cpu')

    total       = len(dataset)
    tam_train   = int(0.70 * total)
    tam_val     = int(0.15 * total)
    tam_test    = total - tam_train - tam_val

    datos_train, datos_val, datos_test = random_split(
        dataset, [tam_train, tam_val, tam_test]
    )
    cargador_train = DataLoader(datos_train, batch_size=32, shuffle=True)
    cargador_val   = DataLoader(datos_val,   batch_size=32, shuffle=False)
    cargador_test  = DataLoader(datos_test,  batch_size=32, shuffle=False)

    modelo      = RedNovikov(dim_nodos=dim_nodos, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio    = nn.BCELoss()

    epocas = 50
    hist_loss_train, hist_loss_val, hist_prec_val = [], [], []

    print(f"\n{'='*60}")
    print(f"  {nombre}  |  dim_nodos={dim_nodos}  |  train={tam_train}  |  test={tam_test}")
    print(f"{'='*60}")

    for epoca in range(epocas):
        modelo.train()
        perdida_train_total = 0
        for lote in cargador_train:
            lote = lote.to(dispositivo)
            optimizador.zero_grad()
            pred = modelo(lote)
            y    = lote.y.view(-1, 1)
            err  = criterio(pred, y)
            err.backward()
            optimizador.step()
            perdida_train_total += err.item()
        hist_loss_train.append(perdida_train_total / len(cargador_train))

        modelo.eval()
        perdida_val_total, aciertos_val = 0, 0
        with torch.no_grad():
            for lote in cargador_val:
                lote = lote.to(dispositivo)
                pred = modelo(lote)
                y    = lote.y.view(-1, 1)
                perdida_val_total += criterio(pred, y).item()
                aciertos_val += ((pred > 0.5).float() == y).sum().item()
        hist_loss_val.append(perdida_val_total / len(cargador_val))
        hist_prec_val.append(aciertos_val / tam_val * 100)

        print(f"Epoch {epoca+1:02d}/{epocas} | "
              f"Train Loss: {hist_loss_train[-1]:.4f} | "
              f"Val Loss: {hist_loss_val[-1]:.4f} | "
              f"Val Acc: {hist_prec_val[-1]:.2f}%")

    modelo.eval()
    aciertos_test = 0
    with torch.no_grad():
        for lote in cargador_test:
            lote = lote.to(dispositivo)
            pred = modelo(lote)
            y    = lote.y.view(-1, 1)
            aciertos_test += ((pred > 0.5).float() == y).sum().item()
    precision_test = aciertos_test / tam_test * 100
    print(f"\n>> Test accuracy ({nombre}): {precision_test:.2f}% <<\n")

    return hist_loss_train, hist_loss_val, hist_prec_val, precision_test


if __name__ == "__main__":
    dataset_base = torch.load("dataset_novikov_dim2_50000_DATOS.pt",    weights_only=False)
    dataset_enri = torch.load("dataset_enriquecido_dim2_50000_DATOS.pt", weights_only=False)

    lt_b, lv_b, pv_b, pt_b = entrenar(dataset_base, dim_nodos=1, nombre="BASELINE (1 feature)")
    lt_e, lv_e, pv_e, pt_e = entrenar(dataset_enri, dim_nodos=6, nombre="ENRICHED (6 features)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = np.arange(1, 51)

    # Left panel: Val Loss
    ax1.plot(epochs_range, lv_b, 's-', linewidth=2, markersize=4,
             label='Baseline val loss', color='#1f77b4')
    ax1.plot(epochs_range, lv_e, 'o-', linewidth=2, markersize=4,
             label='Enriched val loss', color='#d62728')
    ax1.axhline(y=0.693, color='green', linestyle='--', linewidth=1.5,
                label='Random Baseline (log 0.5)', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Loss: Baseline vs Enriched', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.5, 50.5)

    # Right panel: Val Accuracy
    ax2.plot(epochs_range, pv_b, 's-', linewidth=2, markersize=4,
             label='Baseline val accuracy', color='#1f77b4')
    ax2.plot(epochs_range, pv_e, 'o-', linewidth=2, markersize=4,
             label='Enriched val accuracy', color='#d62728')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.2,
                label='Random Baseline (50%)', alpha=0.6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy: Baseline vs Enriched', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(0.5, 50.5)
    ax2.set_ylim(40, 100)

    plt.suptitle(f'Controlled comparison - 44,386 graphs | lr=0.001 | 50 epochs | seed=42',
                 fontsize=11, style='italic', y=1.01)
    plt.tight_layout()
    plt.savefig('grafica_comparacion_escala.png', dpi=300, bbox_inches='tight')
    print("Saved: grafica_comparacion_escala.png")
