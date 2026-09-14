"""Audit of the near-misses in the hard dim2 dataset.

Goal: verify that near-misses close to the boundary are correctly
detected by the verifier (tolerance 1e-4) and that there is no
contamination of the Novikov class by perturbations that are too small.

Reported metrics:
    - Distribution of perturbation magnitudes per configuration
    - Novikov / true near-miss ratio vs. the label assigned by the verifier
    - Boundary cases (magnitude < 10 * tolerance)
"""
import numpy as np
import torch

from verificador import es_novikov, tensor_a_matriz_C


DATASET = "dataset_novikov_dim2_50000_DATOS_dificiles.pt"
TOLERANCIA = 1e-4


def magnitud_violacion(edge_index, edge_attr, dim=2):
    """Return how far the algebra is from satisfying the Novikov identities.
    The output is the maximum residual of the two identities across all indices."""
    C = tensor_a_matriz_C(edge_index, edge_attr, dim)
    # Identity 2: (x*y)*z = (x*z)*y
    izq2 = torch.einsum('ijm,mkh->ijkh', C, C)
    der2 = torch.einsum('ikm,mjh->ijkh', C, C)
    res_id2 = (izq2 - der2).abs().max().item()

    # Identity 1
    term1_izq = torch.einsum('jkm,imh->ijkh', C, C)
    term1_der = torch.einsum('ikm,jmh->ijkh', C, C)
    term2_izq = torch.einsum('ijm,mkh->ijkh', C, C)
    term2_der = torch.einsum('jim,mkh->ijkh', C, C)
    res_id1 = ((term1_izq - term2_izq) - (term1_der - term2_der)).abs().max().item()

    return max(res_id1, res_id2)


def main():
    print(f"Loading {DATASET}...")
    dataset = torch.load(DATASET, weights_only=False)
    print(f"  {len(dataset):,} graphs")

    ys = np.array([int(g.y.item()) for g in dataset])
    n_nov = int((ys == 1).sum())
    n_nm = int((ys == 0).sum())
    print(f"  Novikov: {n_nov:,} ({n_nov/len(ys)*100:.1f}%)")
    print(f"  Near-miss: {n_nm:,} ({n_nm/len(ys)*100:.1f}%)")

    print("\nComputing violation magnitude per graph...")
    magnitudes = np.array([magnitud_violacion(g.edge_index, g.edge_attr)
                           for g in dataset])

    print("\n=== NEAR-MISSES (y=0): violation distribution ===")
    m_nm = magnitudes[ys == 0]
    print(f"  min        : {m_nm.min():.6f}")
    print(f"  percentile 1 : {np.percentile(m_nm, 1):.6f}")
    print(f"  percentile 5 : {np.percentile(m_nm, 5):.6f}")
    print(f"  median     : {np.median(m_nm):.6f}")
    print(f"  mean       : {m_nm.mean():.6f}")
    print(f"  percentile 95: {np.percentile(m_nm, 95):.6f}")
    print(f"  max        : {m_nm.max():.6f}")

    # Dangerous cases: close to the verifier tolerance
    n_borde = int((m_nm < 10 * TOLERANCIA).sum())
    n_muy_borde = int((m_nm < 2 * TOLERANCIA).sum())
    print(f"\n  Near-misses with violation < 10*tol (1e-3): {n_borde} ({n_borde/len(m_nm)*100:.2f}%)")
    print(f"  Near-misses with violation < 2*tol  (2e-4):  {n_muy_borde} ({n_muy_borde/len(m_nm)*100:.2f}%)")

    print("\n=== NOVIKOV (y=1): violation (should be 0) ===")
    m_nov = magnitudes[ys == 1]
    print(f"  min: {m_nov.min():.2e}  max: {m_nov.max():.2e}  mean: {m_nov.mean():.2e}")
    n_novikov_sospechoso = int((m_nov > TOLERANCIA).sum())
    print(f"  Novikov with violation > tol: {n_novikov_sospechoso}")

    print("\n=== Verdict ===")
    if n_muy_borde == 0 and n_novikov_sospechoso == 0:
        print("  OK: near-misses have a wide margin above the verifier tolerance.")
        print("      No risk of cross-class contamination.")
    elif n_muy_borde < 50:
        print(f"  Acceptable: {n_muy_borde} near-misses very close to the threshold, marginal.")
    else:
        print(f"  WARNING: {n_muy_borde} near-misses too close to the threshold.")
        print("             Consider increasing the minimum perturbation magnitude.")


if __name__ == "__main__":
    main()
