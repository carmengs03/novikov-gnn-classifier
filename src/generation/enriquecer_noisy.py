"""Enrich the noisy datasets with the 6 node features.

The noisy datasets were generated with `x = torch.ones((dim, 1))`
(trivial feature). The trained GNN expects `x` with 6 enriched features
(in_degree, out_degree, in_mass, out_mass, self_loop_mass, centralidad).

Usage:
    python enriquecer_noisy.py
    -> dataset_noisy_topologico_dim2_enriquecido.pt
    -> dataset_noisy_topologico_dim3_enriquecido.pt
"""
import torch
from enriquecer_datos import inyectar_caracteristicas


def enriquecer(path_in, path_out, dim_algebra):
    print(f"Loading {path_in}...")
    ds = torch.load(path_in, weights_only=False)
    print(f"  {len(ds):,} graphs, original x.shape: {ds[0].x.shape}")
    ds_enr = [inyectar_caracteristicas(g, dim_algebra=dim_algebra) for g in ds]
    print(f"  enriched x.shape: {ds_enr[0].x.shape}")
    torch.save(ds_enr, path_out)
    print(f"  Saved: {path_out}")


if __name__ == "__main__":
    enriquecer(
        "dataset_noisy_topologico_dim2.pt",
        "dataset_noisy_topologico_dim2_enriquecido.pt",
        dim_algebra=2,
    )
    enriquecer(
        "dataset_noisy_topologico_dim3.pt",
        "dataset_noisy_topologico_dim3_enriquecido.pt",
        dim_algebra=3,
    )
