import torch
from torch_geometric.data import Data
import random

from verificador import es_novikov

import generadores_dim2.config_b as config_b
import generadores_dim2.config_c as config_c
import generadores_dim2.config_d as config_d
import generadores_dim2.config_e as config_e
import generadores_dim2.config_f as config_f
import generadores_dim2.config_g as config_g
import generadores_dim2.config_h as config_h
import generadores_dim2.config_i as config_i
import generadores_dim2.config_j as config_j

def fijar_semilla(semilla=42):
    """Fix the random seed for exact dataset reproducibility across
    machines."""
    random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    print(f"Random seed fixed at: {semilla}")

def generar_dataset_masivo(num_muestras_deseadas=1000000, dim_algebra=2):
    print(f"Starting the generation of {num_muestras_deseadas} graphs...")

    lista_configuraciones = [
        config_b.generar_grafo_b,
        config_c.generar_grafo_c,
        config_d.generar_grafo_d,
        config_e.generar_grafo_e,
        config_f.generar_grafo_f,
        config_g.generar_grafo_g,
        config_h.generar_grafo_h,
        config_i.generar_grafo_i,
        config_j.generar_grafo_j,
    ]

    grafos_clase_1 = []
    grafos_clase_0 = []

    # Initial nodes (vector of 1s for each basis element)
    x = torch.ones((dim_algebra, 1), dtype=torch.float)

    for i in range(num_muestras_deseadas):
        generador_elegido = random.choice(lista_configuraciones)
        edge_index, edge_attr = generador_elegido()

        # Exact verifier: returns 1.0 (Novikov) or 0.0 (near-miss)
        etiqueta_real = es_novikov(edge_index, edge_attr, dim_algebra)
        y = torch.tensor([etiqueta_real], dtype=torch.float)

        grafo = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if etiqueta_real == 1.0:
            grafos_clase_1.append(grafo)
        else:
            grafos_clase_0.append(grafo)

        if (i + 1) % 20000 == 0:
            print(f"Generated {i + 1} graphs...")

    # --- 50/50 balancing ---
    print("\n--- Balancing phase ---")
    print(f"Total class 1 (Novikov): {len(grafos_clase_1)}")
    print(f"Total class 0 (near-misses): {len(grafos_clase_0)}")

    min_cantidad = min(len(grafos_clase_1), len(grafos_clase_0))
    grafos_clase_1 = random.sample(grafos_clase_1, min_cantidad)
    grafos_clase_0 = random.sample(grafos_clase_0, min_cantidad)

    dataset_final = grafos_clase_1 + grafos_clase_0
    random.shuffle(dataset_final)

    print(f"Balanced dataset: {len(dataset_final)} graphs ready.")
    return dataset_final

if __name__ == "__main__":
    fijar_semilla(42)
    dataset = generar_dataset_masivo(num_muestras_deseadas=50000, dim_algebra=2)
    torch.save(dataset, "dataset_novikov_dim2.pt")
    print("\nDataset saved as 'dataset_novikov_dim2.pt'")
