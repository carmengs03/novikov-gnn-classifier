"""Topology-preserving Noisy Nearest-Boundary Sampling.

Correction over the previous method: we perturb only the NON-ZERO
coefficients of the positive. Structural zeros (imposed by the graph
topology) are preserved as exact zeros.

Rationale: in dim 3 many c_ijk are structurally zero. Applying Gaussian
noise to those zeros produces values approx 0.001 that an MLP easily
picks up as an noisy signature. This method removes that signature
while keeping the same topology as the positive.

Outputs:
    - dataset_noisy_topologico_dim2.pt
    - dataset_noisy_topologico_dim3.pt
"""
import random
import sys
import torch
from torch_geometric.data import Data

from verificador import es_novikov


SEMILLA = 42
SIGMA_INICIAL = 0.001
SIGMA_FACTOR = 1.5
SIGMA_MAX = 5.0
MAX_INTENTOS = 30
UMBRAL_NO_NULO = 1e-6


def perturbar_hasta_frontera_topologico(edge_index, edge_attr_pos, dim,
                                        sigma0=SIGMA_INICIAL, factor=SIGMA_FACTOR,
                                        sigma_max=SIGMA_MAX, max_intentos=MAX_INTENTOS):
    """Perturb only the NON-ZERO coefficients of the positive. Structural
    zeros are preserved.
    """
    mascara = (edge_attr_pos.abs() > UMBRAL_NO_NULO).float()  # 1 where perturbation is applied
    if mascara.sum() == 0:
        return None, sigma0   # trivial positive (all zeros), nothing to perturb

    sigma = sigma0
    for _ in range(max_intentos):
        ruido = torch.randn_like(edge_attr_pos) * sigma * mascara
        edge_attr_noisy = edge_attr_pos + ruido
        if es_novikov(edge_index, edge_attr_noisy, dim) == 0.0:
            return edge_attr_noisy, sigma
        sigma *= factor
        if sigma > sigma_max:
            break
    return None, sigma


def procesar_dataset(archivo_entrada, archivo_salida, dim):
    print(f"\n=== DIM {dim}: {archivo_entrada} ===", flush=True)
    ds = torch.load(archivo_entrada, weights_only=False)
    positivos = [g for g in ds if int(g.y.item()) == 1]
    print(f"  {len(positivos):,} positives available", flush=True)

    x_base = torch.ones((dim, 1), dtype=torch.float)
    noisy, sigmas = [], []
    fallos, trivial = 0, 0

    for i, g_pos in enumerate(positivos):
        edge_attr_noisy, sigma = perturbar_hasta_frontera_topologico(
            g_pos.edge_index, g_pos.edge_attr, dim=dim
        )
        if edge_attr_noisy is None:
            if (g_pos.edge_attr.abs() > UMBRAL_NO_NULO).sum() == 0:
                trivial += 1
            else:
                fallos += 1
            continue
        g_noisy = Data(
            x=x_base,
            edge_index=g_pos.edge_index.clone(),
            edge_attr=edge_attr_noisy,
            y=torch.tensor([0.0], dtype=torch.float),
        )
        noisy.append(g_noisy)
        sigmas.append(sigma)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} processed (noisy OK: {len(noisy):,}, "
                  f"trivial: {trivial}, failures: {fallos})", flush=True)

    if sigmas:
        s = torch.tensor(sigmas)
        print(f"  Total noisy samples: {len(noisy):,} "
              f"(trivial: {trivial}, failures: {fallos})", flush=True)
        print(f"  Effective sigma: min={s.min():.4f}, median={s.median():.4f}, "
              f"max={s.max():.4f}, mean={s.mean():.4f}", flush=True)

    random.seed(SEMILLA)
    n = min(len(positivos), len(noisy))
    pos_bal = random.sample(positivos, n)
    dataset = pos_bal + noisy[:n]
    random.shuffle(dataset)

    torch.save(dataset, archivo_salida)
    print(f"  Saved: {archivo_salida} ({len(dataset):,} graphs, "
          f"{n:,}+{n:,})", flush=True)


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    if len(sys.argv) > 1 and sys.argv[1] == "dim3":
        procesar_dataset(
            "dataset_novikov_dim3_facil.pt",
            "dataset_noisy_topologico_dim3.pt",
            dim=3,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "dim2":
        procesar_dataset(
            "dataset_novikov_dim2_50000_DATOS.pt",
            "dataset_noisy_topologico_dim2.pt",
            dim=2,
        )
    else:
        procesar_dataset(
            "dataset_novikov_dim2_50000_DATOS.pt",
            "dataset_noisy_topologico_dim2.pt",
            dim=2,
        )
        procesar_dataset(
            "dataset_novikov_dim3_facil.pt",
            "dataset_noisy_topologico_dim3.pt",
            dim=3,
        )


if __name__ == "__main__":
    main()
