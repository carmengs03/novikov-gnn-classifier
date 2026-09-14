"""Generate a control dataset with PURELY RANDOM near-misses for dim 2.

Motivation:
    The near-misses in the main dataset are built by controlled violation
    of a specific condition ('noisy samples near the boundary'). This
    dataset adds a third category: random C tensors without topological
    constraint, which serves as an upper bound on easiness (we expect both
    the MLP and the GNN to saturate accuracy on this control).

Design:
    - Each near-miss: 8 coefficients c_ijh sampled iid Uniform(-5, 5),
      not respecting any topology. The verifier labels each one.
    - Positives come from the structured generators of the hard dataset,
      to obtain a balanced 50/50 dataset.

Output: dataset_aleatorio_dim2.pt
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov


SEMILLA = 42
DIM = 2
N_MUESTRAS = 50000     # initial generation (balanced afterwards)
ARCHIVO_SALIDA = "dataset_aleatorio_dim2.pt"


def generar_grafo_aleatorio(dim=DIM):
    """Sample 8 coefficients iid Uniform(-5, 5) and build the graph with
    the standard edge_index (all 4 edges of dim 2)."""
    pesos = torch.empty((dim * dim, dim), dtype=torch.float).uniform_(-5.0, 5.0)
    origenes = [i for i in range(dim) for _ in range(dim)]
    destinos = [j for _ in range(dim) for j in range(dim)]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)
    return edge_index, pesos


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    print(f"Sampling {N_MUESTRAS:,} random tensors and labeling with the verifier...")
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

        if (i + 1) % 10000 == 0:
            print(f"  processed {i+1:,}  "
                  f"(Nov={len(grafos_novikov_azar):,}, nm={len(grafos_near_miss_azar):,})")

    print(f"\nEmpirical ratio of Novikov in random tensors: "
          f"{len(grafos_novikov_azar)/N_MUESTRAS*100:.4f}%")

    # The random Novikov ratio is approx 0, so to balance we take positives
    # from the structured generator (hard dataset) and mix them with random
    # near-misses sampled here.
    print("\nCombining with structured Novikov from the hard dataset...")
    ds_estruct = torch.load("dataset_novikov_dim2_50000_DATOS_dificiles.pt",
                            weights_only=False)
    positivos_estruct = [g for g in ds_estruct if int(g.y.item()) == 1]
    print(f"  {len(positivos_estruct):,} structured positives available")

    n_balance = min(len(positivos_estruct), len(grafos_near_miss_azar))
    positivos = random.sample(positivos_estruct, n_balance)
    negativos_azar = random.sample(grafos_near_miss_azar, n_balance)

    dataset = positivos + negativos_azar
    random.shuffle(dataset)

    torch.save(dataset, ARCHIVO_SALIDA)
    print(f"\nRandom control dataset saved: {ARCHIVO_SALIDA}")
    print(f"  Total: {len(dataset):,} graphs ({n_balance:,} positives + {n_balance:,} random near-miss)")


if __name__ == "__main__":
    main()
