"""
Script that generates the proof-of-concept training figure.
Trains the model for 20 epochs on the 2000-graph Novikov dataset and plots
the evolution of loss and accuracy.
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
    print(f"Training on: {dispositivo}")

    print("Loading the 2000-graph Novikov dataset...")
    dataset = torch.load("dataset_novikov_dim2_2000_DATOS.pt", weights_only=False)
    total = len(dataset)
    print(f"Total graphs: {total}")

    # 70% Train, 15% Val, 15% Test (only train is used below)
    tamano_train = int(0.70 * total)
    tamano_val = int(0.15 * total)
    tamano_test = total - tamano_train - tamano_val

    datos_train, datos_val, datos_test = random_split(
        dataset, [tamano_train, tamano_val, tamano_test]
    )

    cargador_train = DataLoader(datos_train, batch_size=32, shuffle=True)
    cargador_val = DataLoader(datos_val, batch_size=32, shuffle=False)
    cargador_test = DataLoader(datos_test, batch_size=32, shuffle=False)

    modelo = RedNovikov(dim_nodos=1, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio = nn.BCELoss()

    epocas = 20
    historico_perdida_train = []
    historico_perdida_val = []
    historico_precision_val = []

    print(f"\n--- STARTING TRAINING (20 EPOCHS) ---\n")

    for epoca in range(epocas):
        modelo.train()
        perdida_train_total = 0

        for lote in cargador_train:
            lote = lote.to(dispositivo)
            optimizador.zero_grad()
            prediccion = modelo(lote)
            etiquetas_reales = lote.y.view(-1, 1)
            error = criterio(prediccion, etiquetas_reales)
            error.backward()
            optimizador.step()
            perdida_train_total += error.item()

        perdida_train_media = perdida_train_total / len(cargador_train)
        historico_perdida_train.append(perdida_train_media)

        modelo.eval()
        perdida_val_total = 0
        aciertos_val = 0

        with torch.no_grad():
            for lote in cargador_val:
                lote = lote.to(dispositivo)
                prediccion = modelo(lote)
                etiquetas_reales = lote.y.view(-1, 1)

                error_val = criterio(prediccion, etiquetas_reales)
                perdida_val_total += error_val.item()

                prediccion_binaria = (prediccion > 0.5).float()
                aciertos_val += (prediccion_binaria == etiquetas_reales).sum().item()

        perdida_val_media = perdida_val_total / len(cargador_val)
        precision_val = (aciertos_val / tamano_val) * 100

        historico_perdida_val.append(perdida_val_media)
        historico_precision_val.append(precision_val)

        print(f"Epoch {epoca+1:02d}/20 | Train Loss: {perdida_train_media:.4f} | Val Loss: {perdida_val_media:.4f} | Val Acc: {precision_val:.2f}%")

    print("\nTraining finished.\n")

    modelo.eval()
    aciertos_test = 0
    with torch.no_grad():
        for lote in cargador_test:
            lote = lote.to(dispositivo)
            prediccion = modelo(lote)
            etiquetas_reales = lote.y.view(-1, 1)
            aciertos_test += ((prediccion > 0.5).float() == etiquetas_reales).sum().item()
    precision_test = aciertos_test / tamano_test * 100
    print(f">> Test accuracy: {precision_test:.2f}% <<\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = np.arange(1, epocas + 1)

    # Subplot 1: Loss
    ax1.plot(epochs_range, historico_perdida_train, 'o-', linewidth=2.5,
             markersize=6, label='Training Loss', color='#1f77b4')
    ax1.plot(epochs_range, historico_perdida_val, 's-', linewidth=2.5,
             markersize=6, label='Validation Loss', color='#ff7f0e')
    ax1.axhline(y=0.693, color='green', linestyle='--', linewidth=1.5,
                label='Random Baseline (log(0.5))', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Evolution of Loss During Training', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim(0.5, epocas + 0.5)

    # Subplot 2: Accuracy
    ax2.plot(epochs_range, historico_precision_val, 'D-', linewidth=2.5,
             markersize=7, label='Validation Accuracy', color='#2ca02c')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                label='Random Baseline (50%)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolution of Accuracy During Training', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(0.5, epocas + 0.5)
    ax2.set_ylim(40, 100)

    plt.tight_layout()
    plt.savefig('grafica_prueba_concepto_entrenamiento.png', dpi=300, bbox_inches='tight')
    print("Figure saved: grafica_prueba_concepto_entrenamiento.png\n")
    plt.show()

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Initial loss (Epoch 1):     {historico_perdida_val[0]:.4f}")
    print(f"Final loss (Epoch 20):      {historico_perdida_val[-1]:.4f}")
    print(f"Loss reduction:             {historico_perdida_val[0] - historico_perdida_val[-1]:.4f}")
    print(f"\nInitial accuracy (Epoch 1): {historico_precision_val[0]:.2f}%")
    print(f"Final accuracy (Epoch 20):  {historico_precision_val[-1]:.2f}%")
    print(f"Accuracy improvement:       {historico_precision_val[-1] - historico_precision_val[0]:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    entrenar_y_graficar()
