"""Generates the OOD split for dim 2: train on every configuration except
'd', evaluate zero-shot on graphs drawn exclusively from configuration 'd'.
"""
import torch
import random
from torch_geometric.data import Data
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

def generar_subconjunto(lista_fábricas, num_muestras):
    grafos_1, grafos_0 = [], []
    x = torch.ones((2, 1), dtype=torch.float)

    for _ in range(num_muestras):
        gen = random.choice(lista_fábricas)
        edge_index, edge_attr = gen()
        etiqueta = es_novikov(edge_index, edge_attr, 2)
        y = torch.tensor([etiqueta], dtype=torch.float)

        grafo = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if etiqueta == 1.0: grafos_1.append(grafo)
        else: grafos_0.append(grafo)

    # 50/50 balancing
    min_cant = min(len(grafos_1), len(grafos_0))
    final = random.sample(grafos_1, min_cant) + random.sample(grafos_0, min_cant)
    random.shuffle(final)
    return final

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    # Training material: every configuration except 'd'
    print("Generating TRAINING dataset (everything except configuration d)...")
    fabricas_train = [
        config_b.generar_grafo_b, config_c.generar_grafo_c,
        config_e.generar_grafo_e, config_f.generar_grafo_f,
        config_g.generar_grafo_g, config_h.generar_grafo_h,
        config_i.generar_grafo_i, config_j.generar_grafo_j
    ]
    dataset_train = generar_subconjunto(fabricas_train, 10000)
    torch.save(dataset_train, "dataset_train_sin_d.pt")
    print(f"-> Saved: {len(dataset_train)} balanced graphs.\n")

    # Zero-shot test: only configuration 'd'
    print("Generating dataset with only configuration d...")
    fabricas_test = [config_d.generar_grafo_d]
    dataset_test = generar_subconjunto(fabricas_test, 2000)
    torch.save(dataset_test, "dataset_test_solo_d.pt")
    print(f"-> Saved: {len(dataset_test)} balanced graphs.")
