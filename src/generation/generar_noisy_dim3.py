"""Noisy Nearest-Boundary Sampling for dim 3.

Identical to generar_noisy_dim2.py except for the dimension and the
input/output paths.
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov


SEMILLA = 42
DIM = 3
SIGMA_INICIAL = 0.001
SIGMA_FACTOR = 1.5
SIGMA_MAX = 5.0
MAX_INTENTOS = 30

ARCHIVO_ENTRADA = "dataset_novikov_dim3_facil.pt"
ARCHIVO_SALIDA = "dataset_noisy_dim3.pt"


def perturbar_hasta_frontera(edge_index, edge_attr_pos, dim=DIM,
                             sigma0=SIGMA_INICIAL, factor=SIGMA_FACTOR,
                             sigma_max=SIGMA_MAX, max_intentos=MAX_INTENTOS):
    sigma = sigma0
    for _ in range(max_intentos):
        ruido = torch.randn_like(edge_attr_pos) * sigma
        edge_attr_noisy = edge_attr_pos + ruido
        if es_novikov(edge_index, edge_attr_noisy, dim) == 0.0:
            return edge_attr_noisy, sigma
        sigma *= factor
        if sigma > sigma_max:
            break
    return None, sigma


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    print(f"Loading positives from {ARCHIVO_ENTRADA}...", flush=True)
    ds = torch.load(ARCHIVO_ENTRADA, weights_only=False)
    positivos = [g for g in ds if int(g.y.item()) == 1]
    print(f"  {len(positivos):,} positives available", flush=True)

    print("\nGenerating nearest-boundary noisy near-misses...", flush=True)
    x_base = torch.ones((DIM, 1), dtype=torch.float)
    noisy = []
    sigmas = []
    fallos = 0

    for i, g_pos in enumerate(positivos):
        edge_attr_noisy, sigma = perturbar_hasta_frontera(
            g_pos.edge_index, g_pos.edge_attr, dim=DIM
        )
        if edge_attr_noisy is None:
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
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1:,}  (noisy OK: {len(noisy):,}, "
                  f"failures: {fallos}, sigma_med={torch.tensor(sigmas).median():.4f})",
                  flush=True)

    print(f"\nGenerated {len(noisy):,} noisy samples (failures={fallos})", flush=True)
    if sigmas:
        s = torch.tensor(sigmas)
        print(f"  Effective sigma: min={s.min():.4f}, median={s.median():.4f}, "
              f"max={s.max():.4f}, mean={s.mean():.4f}", flush=True)

    print("\nBalancing 50/50...", flush=True)
    n = min(len(positivos), len(noisy))
    positivos_bal = random.sample(positivos, n)
    dataset = positivos_bal + noisy[:n]
    random.shuffle(dataset)

    torch.save(dataset, ARCHIVO_SALIDA)
    print(f"\nSaved: {ARCHIVO_SALIDA}", flush=True)
    print(f"  Total: {len(dataset):,} graphs ({n:,} pos + {n:,} noisy near-miss)", flush=True)


if __name__ == "__main__":
    main()
