import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from modelo_mpnn_baseline import RedNovikov

def entrenar_experimento_ood():
    dispositivo = torch.device('cpu')
    print(f"Training on: {dispositivo}")

    print("Loading experiment datasets...")

    # Training dataset: everything except configuration 'd'.
    dataset_estudio = torch.load("dataset_train_sin_d.pt", weights_only=False)
    total_estudio = len(dataset_estudio)

    # 80% Train, 20% Val
    tamano_train = int(0.80 * total_estudio)
    tamano_val = total_estudio - tamano_train
    datos_train, datos_val = random_split(dataset_estudio, [tamano_train, tamano_val])

    # Held-out test set: only configuration 'd'.
    dataset_examen = torch.load("dataset_test_solo_d.pt", weights_only=False)
    total_examen = len(dataset_examen)

    print(f"-> Train graphs: {tamano_train}")
    print(f"-> Val graphs: {tamano_val}")
    print(f"-> OOD test graphs: {total_examen}")

    cargador_train = DataLoader(datos_train, batch_size=32, shuffle=True)
    cargador_val = DataLoader(datos_val, batch_size=32, shuffle=False)
    cargador_test = DataLoader(dataset_examen, batch_size=32, shuffle=False)

    modelo = RedNovikov(dim_nodos=1, dim_aristas=2, dim_oculta=32).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)
    criterio = nn.BCELoss()

    epocas = 20
    print("\n--- STARTING TRAINING ---")

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

    print("\n" + "="*60)
    print("--- ZERO-SHOT GENERALISATION ---")
    print("Evaluating on configuration 'd'...")
    print("="*60)

    modelo.eval()
    aciertos_test = 0
    with torch.no_grad():
        for lote in cargador_test:
            lote = lote.to(dispositivo)
            prediccion = modelo(lote)
            etiquetas_reales = lote.y.view(-1, 1)
            prediccion_binaria = (prediccion > 0.5).float()
            aciertos_test += (prediccion_binaria == etiquetas_reales).sum().item()

    precision_test = (aciertos_test / total_examen) * 100
    print(f"\n>> ACCURACY ON CONFIGURATION 'D': {precision_test:.2f}% <<\n")

if __name__ == "__main__":
    entrenar_experimento_ood()
