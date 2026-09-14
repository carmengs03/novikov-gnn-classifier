import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from modelo_mpnn_baseline import RedNovikov

def entrenar_con_validacion():
    dispositivo = torch.device('cpu')
    print(f"Training on: {dispositivo}")

    print("Loading full dataset...")
    dataset_completo = torch.load("dataset_enriquecido_dim2.pt", weights_only=False)

    dataset_reducido = dataset_completo
    total = len(dataset_reducido)
    print(f"Total graphs in experiment: {total}")

    # 70% Train, 15% Val, 15% Test
    tamano_train = int(0.70 * total)
    tamano_val = int(0.15 * total)
    tamano_test = total - tamano_train - tamano_val

    datos_train, datos_val, datos_test = random_split(
        dataset_reducido, [tamano_train, tamano_val, tamano_test]
    )

    cargador_train = DataLoader(datos_train, batch_size=32, shuffle=True)
    cargador_val = DataLoader(datos_val, batch_size=32, shuffle=False)
    cargador_test = DataLoader(datos_test, batch_size=32, shuffle=False)

    modelo = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterio = nn.BCELoss()

    epocas = 50
    print(f"\n--- STARTING TRAINING ({tamano_train} Train | {tamano_val} Val) ---")

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

        print(f"Epoch {epoca+1:02d}/{epocas} | Train Loss: {perdida_train_media:.4f} | Val Loss: {perdida_val_media:.4f} | Val Accuracy: {precision_val:.2f}%")

    print("\n--- FINAL EVALUATION (HELD-OUT TEST SET) ---")
    modelo.eval()
    aciertos_test = 0
    with torch.no_grad():
        for lote in cargador_test:
            lote = lote.to(dispositivo)
            prediccion = modelo(lote)
            etiquetas_reales = lote.y.view(-1, 1)
            prediccion_binaria = (prediccion > 0.5).float()
            aciertos_test += (prediccion_binaria == etiquetas_reales).sum().item()

    precision_test = (aciertos_test / tamano_test) * 100
    print(f"Test accuracy ({tamano_test} unseen graphs): {precision_test:.2f}%")

if __name__ == "__main__":
    entrenar_con_validacion()
