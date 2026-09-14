"""Merge the dim2 (44k existing) and dim3 (freshly generated) datasets into
one file ready for joint training.

Steps:
    1. Load the existing enriched dim2 (edge_attr dim 2, x dim (2, 6)).
    2. Load the freshly generated dim3 (edge_attr dim 3, x raw dim (3, 1)).
    3. Enrich dim3 with `inyectar_caracteristicas(grafo, dim_algebra=3)`
       -> x dim (3, 6).
    4. Pad dim2 edge_attr to dim 3 by appending a zero column (trivial ideal).
    5. Tag each graph with `.dim_algebra` (2 or 3) for later analysis.
    6. Concatenate, shuffle with fixed seed and save.

Output: dataset_combinado_dim2_dim3.pt
"""
import random
import torch
from torch_geometric.data import Data

from enriquecer_datos import inyectar_caracteristicas


SEMILLA = 42
ARCHIVO_2D = "dataset_enriquecido_dim2_50000_DATOS.pt"
ARCHIVO_3D = "dataset_novikov_dim3.pt"
ARCHIVO_SALIDA = "dataset_combinado_dim2_dim3.pt"


def pad_a_dim3(grafo_2d):
    """Pad edge_attr from dim 2 to dim 3 with a zero column and mark the
    graph with `.dim_algebra = 2` for traceability."""
    pad = torch.zeros((grafo_2d.edge_attr.shape[0], 1), dtype=grafo_2d.edge_attr.dtype)
    nuevo_attr = torch.cat([grafo_2d.edge_attr, pad], dim=1)
    nuevo = Data(
        x=grafo_2d.x,
        edge_index=grafo_2d.edge_index,
        edge_attr=nuevo_attr,
        y=grafo_2d.y,
    )
    nuevo.dim_algebra = 2
    return nuevo


def enriquecer_3d(grafo_3d):
    enriquecido = inyectar_caracteristicas(grafo_3d, dim_algebra=3)
    enriquecido.dim_algebra = 3
    return enriquecido


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    print(f"Loading enriched dim2: {ARCHIVO_2D}")
    ds_2d = torch.load(ARCHIVO_2D, weights_only=False)
    print(f"  {len(ds_2d):,} dim 2 graphs (edge_attr dim 2)")

    print(f"\nLoading generated dim3: {ARCHIVO_3D}")
    ds_3d_crudo = torch.load(ARCHIVO_3D, weights_only=False)
    print(f"  {len(ds_3d_crudo):,} dim 3 graphs (edge_attr dim 3)")

    print("\nProcessing dim 2 graphs (padding edge_attr 2 -> 3)...")
    ds_2d_padded = [pad_a_dim3(g) for g in ds_2d]
    print(f"  done ({len(ds_2d_padded):,} graphs)")

    print("\nProcessing dim 3 graphs (enriching node features)...")
    ds_3d_enriquecido = [enriquecer_3d(g) for g in ds_3d_crudo]
    print(f"  done ({len(ds_3d_enriquecido):,} graphs)")

    print("\nConcatenating and shuffling...")
    combinado = ds_2d_padded + ds_3d_enriquecido
    random.shuffle(combinado)
    print(f"  total combined: {len(combinado):,} graphs")

    n_2d = sum(1 for g in combinado if g.dim_algebra == 2)
    n_3d = sum(1 for g in combinado if g.dim_algebra == 3)
    n_pos = sum(1 for g in combinado if int(g.y.item()) == 1)
    n_neg = sum(1 for g in combinado if int(g.y.item()) == 0)
    print(f"  - dim 2: {n_2d:,} ({n_2d/len(combinado)*100:.1f}%)")
    print(f"  - dim 3: {n_3d:,} ({n_3d/len(combinado)*100:.1f}%)")
    print(f"  - Novikov:   {n_pos:,} ({n_pos/len(combinado)*100:.1f}%)")
    print(f"  - near-miss: {n_neg:,} ({n_neg/len(combinado)*100:.1f}%)")

    print(f"\nSaving to {ARCHIVO_SALIDA}...")
    torch.save(combinado, ARCHIVO_SALIDA)

    g0 = combinado[0]
    print(f"\nSanity check (first graph): "
          f"x={tuple(g0.x.shape)} edge_attr={tuple(g0.edge_attr.shape)} "
          f"dim_algebra={g0.dim_algebra} y={int(g0.y.item())}")


if __name__ == "__main__":
    main()
