"""Generate a balanced sample of dim2 graphs with explicit `.config`
metadata for the t-SNE / PCA analysis of the latent space.

Output: dataset_etiquetado_dim2.pt -- 9 configs x N_POR_CONFIG graphs
each, with .config (str: 'b', 'c', ..., 'j'), .y (0/1) and x enriched to
6 features.

Seed 12345 (different from the 42 used at training time) guarantees that
these graphs are new and have not been seen while fitting the model.
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov
from enriquecer_datos import inyectar_caracteristicas

import generadores_dim2.config_b as config_b
import generadores_dim2.config_c as config_c
import generadores_dim2.config_d as config_d
import generadores_dim2.config_e as config_e
import generadores_dim2.config_f as config_f
import generadores_dim2.config_g as config_g
import generadores_dim2.config_h as config_h
import generadores_dim2.config_i as config_i
import generadores_dim2.config_j as config_j


CONFIGS = {
    "b": config_b.generar_grafo_b,
    "c": config_c.generar_grafo_c,
    "d": config_d.generar_grafo_d,
    "e": config_e.generar_grafo_e,
    "f": config_f.generar_grafo_f,
    "g": config_g.generar_grafo_g,
    "h": config_h.generar_grafo_h,
    "i": config_i.generar_grafo_i,
    "j": config_j.generar_grafo_j,
}

N_POR_CONFIG = 400
SEMILLA = 12345
ARCHIVO_SALIDA = "dataset_etiquetado_dim2.pt"


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    print(f"Generating {N_POR_CONFIG} graphs per configuration (seed {SEMILLA})\n")

    x_base = torch.ones((2, 1), dtype=torch.float)
    dataset = []
    resumen = {}

    for nombre, generador in CONFIGS.items():
        ys_config = []
        for _ in range(N_POR_CONFIG):
            edge_index, edge_attr = generador()
            y_val = es_novikov(edge_index, edge_attr, dim_algebra=2)
            grafo = Data(
                x=x_base,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([y_val], dtype=torch.float),
            )
            grafo_enriquecido = inyectar_caracteristicas(grafo, dim_algebra=2)
            grafo_enriquecido.config = nombre
            dataset.append(grafo_enriquecido)
            ys_config.append(y_val)

        n_pos = sum(int(y) for y in ys_config)
        n_neg = len(ys_config) - n_pos
        resumen[nombre] = (n_pos, n_neg)
        print(f"  config {nombre}: {n_pos:4d} Novikov / {n_neg:4d} near-miss")

    random.shuffle(dataset)
    torch.save(dataset, ARCHIVO_SALIDA)
    print(f"\nSaved: {ARCHIVO_SALIDA} ({len(dataset)} graphs)")
    total_pos = sum(p for p, _ in resumen.values())
    total_neg = sum(n for _, n in resumen.values())
    print(f"Total: {total_pos} Novikov / {total_neg} near-miss")


if __name__ == "__main__":
    main()
