"""Generate the control dataset with PURELY RANDOM near-misses for dim 3.

Output: dataset_aleatorio_dim3.pt
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov


SEMILLA = 42
DIM = 3
N_MUESTRAS = 300000
ARCHIVO_SALIDA = "dataset_aleatorio_dim3.pt"


def generar_grafo_aleatorio(dim=DIM):
    """Sample dim*dim weight vectors iid Uniform(-5, 5)."""
    pesos = torch.empty((dim * dim, dim), dtype=torch.float).uniform_(-5.0, 5.0)
    origenes = [i for i in range(dim) for _ in range(dim)]
    destinos = [j for _ in range(dim) for j in range(dim)]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)
    return edge_index, pesos


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    print(f"Sampling {N_MUESTRAS:,} random dim {DIM} tensors...")
    grafos_novikov_azar = []
    grafos_near_miss_azar = []
    x_base = torch.ones((DIM, 1), dtype=torch.float)

    for i in range(N_MUESTRAS):
        edge_index, edge_attr = generar_grafo_aleatorio(DIM)
        etiqueta = es_novikov(edge_index, edge_attr, DIM)
        y = torch.tensor([etiqueta], dtype=torch.float)
        grafo = Data(x=x_base, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if etiqueta == 1.0:
            grafos_novikov_azar.append(grafo)
        else:
            grafos_near_miss_azar.append(grafo)

        if (i + 1) % 25000 == 0:
            print(f"  processed {i+1:,}  "
                  f"(Nov={len(grafos_novikov_azar):,}, nm={len(grafos_near_miss_azar):,})")

    print(f"\nEmpirical ratio of random Novikov in dim {DIM}: "
          f"{len(grafos_novikov_azar)/N_MUESTRAS*100:.6f}%")

    print("\nCombining with structured Novikov from the hard dim3 dataset...")
    ds_estruct = torch.load("dataset_novikov_dim3_dificiles.pt", weights_only=False)
    positivos_estruct = [g for g in ds_estruct if int(g.y.item()) == 1]
    print(f"  {len(positivos_estruct):,} structured positives available")

    n_balance = min(len(positivos_estruct), len(grafos_near_miss_azar))
    positivos = random.sample(positivos_estruct, n_balance)
    negativos_azar = random.sample(grafos_near_miss_azar, n_balance)

    dataset = positivos + negativos_azar
    random.shuffle(dataset)

    torch.save(dataset, ARCHIVO_SALIDA)
    print(f"\nSaved: {ARCHIVO_SALIDA}")
    print(f"  Total: {len(dataset):,} graphs ({n_balance:,} pos + {n_balance:,} neg)")


if __name__ == "__main__":
    main()
