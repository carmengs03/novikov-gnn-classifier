"""
Script that produces the figures for the "Evaluation and generalization" section.
Generates two figures:
  1. grafica_evaluacion_formal.png  — 70/15/15 training with test evaluation
  2. grafica_ood_zeroshot.png       — OOD experiment without configuration d
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


# FIGURE 1: formal training with 70 / 15 / 15 split
def figura_evaluacion_formal():
    torch.manual_seed(42)
    dispositivo = torch.device('cpu')

    print("=" * 60)
    print("FIGURE 1: Formal evaluation (70/15/15)")
    print("=" * 60)

    dataset = torch.load("dataset_novikov_dim2_2000_DATOS.pt", weights_only=False)
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

    modelo     = RedNovikov(dim_nodos=1, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio   = nn.BCELoss()

    epocas = 50
    hist_loss_train, hist_loss_val, hist_prec_val = [], [], []

    print(f"Graphs - Train: {tamano_train} | Val: {tamano_val} | Test: {tamano_test}\n")

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

    ax1.plot(epochs_range, hist_loss_train, 'o-', linewidth=2, markersize=4,
             label='Training Loss', color='#1f77b4')
    ax1.plot(epochs_range, hist_loss_val, 's-', linewidth=2, markersize=4,
             label='Validation Loss', color='#ff7f0e')
    ax1.axhline(y=0.693, color='green', linestyle='--', linewidth=1.5,
                label='Random Baseline (log 0.5)', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss During Formal Training', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.5, epocas + 0.5)

    ax2.plot(epochs_range, hist_prec_val, 'D-', linewidth=2, markersize=5,
             label='Validation Accuracy', color='#2ca02c')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                label='Random Baseline (50%)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy During Formal Training', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(0.5, epocas + 0.5)
    ax2.set_ylim(40, 100)

    plt.tight_layout()
    plt.savefig('grafica_evaluacion_formal.png', dpi=300, bbox_inches='tight')
    print("Saved: grafica_evaluacion_formal.png\n")


# FIGURE 2: OOD zero-shot experiment (without configuration d)
def figura_ood_zeroshot():
    torch.manual_seed(42)
    dispositivo = torch.device('cpu')

    print("=" * 60)
    print("FIGURE 2: OOD zero-shot experiment")
    print("=" * 60)

    dataset_estudio = torch.load("dataset_train_sin_d.pt", weights_only=False)
    dataset_examen  = torch.load("dataset_test_solo_d.pt",  weights_only=False)

    total_estudio = len(dataset_estudio)
    tamano_train  = int(0.80 * total_estudio)
    tamano_val    = total_estudio - tamano_train
    total_examen  = len(dataset_examen)

    datos_train, datos_val = random_split(dataset_estudio, [tamano_train, tamano_val])

    cargador_train = DataLoader(datos_train,    batch_size=32, shuffle=True)
    cargador_val   = DataLoader(datos_val,      batch_size=32, shuffle=False)
    cargador_test  = DataLoader(dataset_examen, batch_size=32, shuffle=False)

    modelo      = RedNovikov(dim_nodos=1, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio    = nn.BCELoss()

    epocas = 20
    hist_loss_train, hist_loss_val, hist_prec_val = [], [], []

    print(f"Graphs - Train: {tamano_train} | Val: {tamano_val} | OOD Test: {total_examen}\n")

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

    # Zero-shot evaluation on the unseen configuration d
    modelo.eval()
    aciertos_ood = 0
    with torch.no_grad():
        for lote in cargador_test:
            lote = lote.to(dispositivo)
            pred = modelo(lote)
            y = lote.y.view(-1, 1)
            aciertos_ood += ((pred > 0.5).float() == y).sum().item()
    precision_ood = aciertos_ood / total_examen * 100

    print(f"\n>> Zero-shot OOD accuracy (config d): {precision_ood:.2f}% <<\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = np.arange(1, epocas + 1)

    ax1.plot(epochs_range, hist_loss_train, 'o-', linewidth=2, markersize=5,
             label='Training Loss\n(without config. d)', color='#1f77b4')
    ax1.plot(epochs_range, hist_loss_val, 's-', linewidth=2, markersize=5,
             label='Validation Loss\n(without config. d)', color='#ff7f0e')
    ax1.axhline(y=0.693, color='green', linestyle='--', linewidth=1.5,
                label='Random Baseline (log 0.5)', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('OOD Training: Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.5, epocas + 0.5)

    ax2.plot(epochs_range, hist_prec_val, 'D-', linewidth=2, markersize=5,
             label='Validation Accuracy\n(without config. d)', color='#2ca02c')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                label='Random Baseline (50%)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('OOD Training: Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(0.5, epocas + 0.5)
    ax2.set_ylim(40, 100)

    plt.tight_layout()
    plt.savefig('grafica_ood_zeroshot.png', dpi=300, bbox_inches='tight')
    print("Saved: grafica_ood_zeroshot.png\n")


def figura_ood_zeroshot_50ep():
    torch.manual_seed(42)
    dispositivo = torch.device('cpu')

    print("=" * 60)
    print("OOD FIGURE 50 EPOCHS: OOD zero-shot experiment (lr=0.001)")
    print("=" * 60)

    dataset_estudio = torch.load("dataset_train_sin_d.pt", weights_only=False)
    dataset_examen  = torch.load("dataset_test_solo_d.pt",  weights_only=False)

    total_estudio = len(dataset_estudio)
    tamano_train  = int(0.80 * total_estudio)
    tamano_val    = total_estudio - tamano_train
    total_examen  = len(dataset_examen)

    datos_train, datos_val = random_split(dataset_estudio, [tamano_train, tamano_val])

    cargador_train = DataLoader(datos_train,    batch_size=32, shuffle=True)
    cargador_val   = DataLoader(datos_val,      batch_size=32, shuffle=False)
    cargador_test  = DataLoader(dataset_examen, batch_size=32, shuffle=False)

    modelo      = RedNovikov(dim_nodos=1, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio    = nn.BCELoss()

    epocas = 50
    hist_loss_train, hist_loss_val, hist_prec_val = [], [], []

    print(f"Graphs - Train: {tamano_train} | Val: {tamano_val} | OOD Test: {total_examen}\n")

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
    aciertos_ood = 0
    with torch.no_grad():
        for lote in cargador_test:
            lote = lote.to(dispositivo)
            pred = modelo(lote)
            y = lote.y.view(-1, 1)
            aciertos_ood += ((pred > 0.5).float() == y).sum().item()
    precision_ood = aciertos_ood / total_examen * 100

    print(f"\n>> Zero-shot OOD accuracy (config d, 50 ep): {precision_ood:.2f}% <<\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = np.arange(1, epocas + 1)

    ax1.plot(epochs_range, hist_loss_train, 'o-', linewidth=2, markersize=4,
             label='Training Loss\n(without config. d)', color='#1f77b4')
    ax1.plot(epochs_range, hist_loss_val, 's-', linewidth=2, markersize=4,
             label='Validation Loss\n(without config. d)', color='#ff7f0e')
    ax1.axhline(y=0.693, color='green', linestyle='--', linewidth=1.5,
                label='Random Baseline (log 0.5)', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('OOD Training: Loss (50 epochs, lr=0.001)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.5, epocas + 0.5)

    ax2.plot(epochs_range, hist_prec_val, 'D-', linewidth=2, markersize=5,
             label='Validation Accuracy\n(without config. d)', color='#2ca02c')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                label='Random Baseline (50%)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('OOD Training: Validation Accuracy (50 ep, lr=0.001)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(0.5, epocas + 0.5)
    ax2.set_ylim(40, 100)

    plt.tight_layout()
    plt.savefig('grafica_ood_zeroshot_50ep.png', dpi=300, bbox_inches='tight')
    print("Saved: grafica_ood_zeroshot_50ep.png\n")


if __name__ == "__main__":
    figura_evaluacion_formal()
    figura_ood_zeroshot()
    figura_ood_zeroshot_50ep()
