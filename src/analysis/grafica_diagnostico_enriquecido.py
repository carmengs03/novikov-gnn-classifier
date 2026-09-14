"""
Learning curve of the enriched model (6 features) on the small dataset with
initial hyperparameters: lr=0.01, 20 epochs.
Shows the oscillatory val-loss behaviour that motivates the diagnosis.
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

def entrenar_y_graficar():
    torch.manual_seed(42)
    dispositivo = torch.device('cpu')

    dataset = torch.load("dataset_enriquecido_dim2_2000_DATOS.pt", weights_only=False)
    total = len(dataset)

    tamano_train = int(0.70 * total)
    tamano_val   = int(0.15 * total)
    tamano_test  = total - tamano_train - tamano_val

    datos_train, datos_val, datos_test = random_split(
        dataset, [tamano_train, tamano_val, tamano_test]
    )

    cargador_train = DataLoader(datos_train, batch_size=32, shuffle=True)
    cargador_val   = DataLoader(datos_val,   batch_size=32, shuffle=False)
    cargador_test  = DataLoader(datos_test,  batch_size=32, shuffle=False)

    modelo      = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio    = nn.BCELoss()

    epocas = 50
    hist_loss_train, hist_loss_val, hist_prec_val = [], [], []

    print(f"Train: {tamano_train} | Val: {tamano_val} | Test: {tamano_test}\n")

    for epoca in range(epocas):
        modelo.train()
        perdida_train_total = 0
        for lote in cargador_train:
            lote = lote.to(dispositivo)
            optimizador.zero_grad()
            pred = modelo(lote)
            y = lote.y.view(-1, 1)
            err = criterio(pred, y)
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
                y = lote.y.view(-1, 1)
                perdida_val_total += criterio(pred, y).item()
                aciertos_val += ((pred > 0.5).float() == y).sum().item()
        hist_loss_val.append(perdida_val_total / len(cargador_val))
        hist_prec_val.append(aciertos_val / tamano_val * 100)

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
            y = lote.y.view(-1, 1)
            aciertos_test += ((pred > 0.5).float() == y).sum().item()
    precision_test = aciertos_test / tamano_test * 100
    print(f"\n>> Test accuracy: {precision_test:.2f}% <<\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = np.arange(1, epocas + 1)

    # Subplot 1: Loss — shows the val-loss oscillation
    ax1.plot(epochs_range, hist_loss_train, 'o-', linewidth=2, markersize=5,
             label='Training Loss', color='#1f77b4')
    ax1.plot(epochs_range, hist_loss_val, 's-', linewidth=2, markersize=5,
             label='Validation Loss', color='#ff7f0e')
    ax1.axhline(y=0.693, color='green', linestyle='--', linewidth=1.5,
                label='Random Baseline (log 0.5)', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss: Enriched Model (lr=0.01, 1766 graphs)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.5, epocas + 0.5)

    # Subplot 2: Accuracy
    ax2.plot(epochs_range, hist_prec_val, 'D-', linewidth=2, markersize=5,
             label='Validation Accuracy', color='#2ca02c')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                label='Random Baseline (50%)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy: Enriched Model (lr=0.01, 1766 graphs)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(0.5, epocas + 0.5)
    ax2.set_ylim(40, 100)

    plt.tight_layout()
    plt.savefig('grafica_diagnostico_enriquecido.png', dpi=300, bbox_inches='tight')
    print("Saved: grafica_diagnostico_enriquecido.png\n")

if __name__ == "__main__":
    entrenar_y_graficar()
