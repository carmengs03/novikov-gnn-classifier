"""Histogram of ||R||_inf on noisy negatives vs positives.

Characterises how far the residual distribution of the noisy
negatives sits from the verifier tolerance (epsilon = 1e-4).

Required files:
  - dataset_noisy_topologico_dim2_enriquecido.pt
  - dataset_noisy_topologico_dim3_enriquecido.pt
  - verificador.py (next to this script)
"""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verificador import residuos


def tensor_C_de_grafo(g, dim):
    C = torch.zeros((dim, dim, dim))
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k, :dim]
    return C


def norma_res(C):
    R1, R2 = residuos(C)
    return max(R1.abs().max().item(), R2.abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="continuidad_4a.json")
    args = parser.parse_args()

    salida = {}
    for dim, path in [
        (2, "dataset_noisy_topologico_dim2_enriquecido.pt"),
        (3, "dataset_noisy_topologico_dim3_enriquecido.pt"),
    ]:
        print(f"\n=== dim {dim}: {path} ===", flush=True)
        t = time.time()
        ds = torch.load(path, weights_only=False)
        print(f"  {len(ds):,} graphs ({time.time()-t:.1f}s)", flush=True)

        residuos_pos = []
        residuos_neg = []
        t = time.time()
        for i, g in enumerate(ds):
            C = tensor_C_de_grafo(g, dim)
            r = norma_res(C)
            if float(g.y) == 1.0:
                residuos_pos.append(r)
            else:
                residuos_neg.append(r)
            if (i + 1) % 20000 == 0:
                print(f"    {i+1:,}/{len(ds):,} ({time.time()-t:.0f}s)",
                      flush=True)

        def stats(a):
            if not a:
                return None
            arr = np.array(a)
            return {
                "n": len(arr),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "median": float(np.median(arr)),
                "q05": float(np.quantile(arr, 0.05)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "q95": float(np.quantile(arr, 0.95)),
                "mean": float(arr.mean()),
                # Distribution samples used for the histogram
                "muestras": arr[:5000].tolist(),
            }

        salida[f"dim{dim}"] = {
            "positivos": stats(residuos_pos),
            "noisy_negativos": stats(residuos_neg),
            "tolerancia_verificador": 1e-4,
        }
        pos_stats = salida[f"dim{dim}"]["positivos"]
        neg_stats = salida[f"dim{dim}"]["noisy_negativos"]
        print(f"  positives:      median={pos_stats['median']:.6f}, "
              f"max={pos_stats['max']:.6f}", flush=True)
        print(f"  noisy negatives:  median={neg_stats['median']:.6f}, "
              f"max={neg_stats['max']:.6f}", flush=True)

    with open(args.output, "w") as f:
        json.dump(salida, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
