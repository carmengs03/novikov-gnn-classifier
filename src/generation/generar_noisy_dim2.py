"""Generate noisy near-misses by sampling on the Novikov boundary.

Method (Noisy Nearest-Boundary Sampling):
    For each positive algebra X in the original Novikov dataset:
        1. Fix the topology (edge_index) of the positive.
        2. Perturb edge_attr with Gaussian noise N(0, sigma) over ALL
           coefficients simultaneously.
        3. If the perturbed algebra is still Novikov (according to the
           exact verifier), increase sigma and try again.
        4. As soon as it stops being Novikov, keep the perturbation as
           an noisy near-miss.

This produces near-misses on the geometric boundary of the Novikov variety
without depending on the internal logic of any specific config, and
without modifying the generators.

Output: dataset_noisy_dim2.pt (positives + noisy near-misses
        balanced 50/50).
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov


SEMILLA = 42
DIM = 2
SIGMA_INICIAL = 0.001
SIGMA_FACTOR = 1.5
SIGMA_MAX = 5.0
MAX_INTENTOS = 30

ARCHIVO_ENTRADA = "dataset_novikov_dim2_50000_DATOS.pt"
ARCHIVO_SALIDA = "dataset_noisy_dim2.pt"


def perturbar_hasta_frontera(edge_index, edge_attr_pos, dim=DIM,
                             sigma0=SIGMA_INICIAL, factor=SIGMA_FACTOR,
                             sigma_max=SIGMA_MAX, max_intentos=MAX_INTENTOS):
    """Iteratively perturb edge_attr with increasing Gaussian noise until
    the Novikov identities no longer hold. Return the perturbed edge_attr
    or None if Novikov was not broken with sigma <= sigma_max.
    """
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

    print(f"Loading original positives from {ARCHIVO_ENTRADA}...")
    ds = torch.load(ARCHIVO_ENTRADA, weights_only=False)
    positivos = [g for g in ds if int(g.y.item()) == 1]
    print(f"  {len(positivos):,} positives available")

    print(f"\nGenerating nearest-boundary noisy near-misses...")
    x_base = torch.ones((DIM, 1), dtype=torch.float)
    noisy = []
    sigmas_efectivas = []
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
        sigmas_efectivas.append(sigma)
        if (i + 1) % 2500 == 0:
            print(f"  processed {i+1:,}  (noisy OK: {len(noisy):,}, "
                  f"failures: {fallos}, sigma_med={torch.tensor(sigmas_efectivas).median():.4f})")

    print(f"\nGenerated {len(noisy):,} noisy near-misses.")
    print(f"  Failures: {fallos} (positives that did not break Novikov with sigma <= {SIGMA_MAX})")
    if sigmas_efectivas:
        s = torch.tensor(sigmas_efectivas)
        print(f"  Effective sigma: min={s.min():.4f}, median={s.median():.4f}, "
              f"max={s.max():.4f}, mean={s.mean():.4f}")

    print("\nBalancing 50/50 positives + noisy samples...")
    n = min(len(positivos), len(noisy))
    positivos_bal = random.sample(positivos, n)
    dataset = positivos_bal + noisy[:n]
    random.shuffle(dataset)

    torch.save(dataset, ARCHIVO_SALIDA)
    print(f"\nSaved: {ARCHIVO_SALIDA}")
    print(f"  Total: {len(dataset):,} graphs ({n:,} pos + {n:,} noisy near-miss)")


if __name__ == "__main__":
    main()
