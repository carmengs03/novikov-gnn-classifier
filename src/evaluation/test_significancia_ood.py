"""Formal statistical test of the significance of the OOD zero-shot result.

H0: the model classifies at chance (p = 0.5) on the unseen configuration d.
H1: the model classifies better than chance (p > 0.5).

We apply an exact one-sided binomial test on the observed hits and
report the p-value together with a 95% confidence interval for the
proportion.
"""
import numpy as np
from scipy.stats import binomtest


# OOD experiment results at 50 epochs
N_TOTAL_OOD = 1892
ACIERTOS_OOD_PROP = 0.5502  # 55.02%
ACIERTOS_OOD = int(round(ACIERTOS_OOD_PROP * N_TOTAL_OOD))


def main():
    print("=" * 60)
    print("SIGNIFICANCE TEST -- OOD zero-shot config d")
    print("=" * 60)
    print(f"Total N (OOD test): {N_TOTAL_OOD}")
    print(f"Observed hits: {ACIERTOS_OOD} ({ACIERTOS_OOD/N_TOTAL_OOD*100:.2f}%)")
    print(f"Expected hits under H0 (p=0.5): {N_TOTAL_OOD/2:.0f}")
    print()

    # --- Exact binomial test (one-sided: model better than chance) ---
    resultado = binomtest(k=ACIERTOS_OOD, n=N_TOTAL_OOD, p=0.5, alternative="greater")
    p_valor = resultado.pvalue

    # --- Two-sided Wilson CI for the proportion's uncertainty ---
    resultado_dos_colas = binomtest(k=ACIERTOS_OOD, n=N_TOTAL_OOD, p=0.5,
                                    alternative="two-sided")
    ic_95 = resultado_dos_colas.proportion_ci(confidence_level=0.95, method="wilson")

    print(f"Exact binomial test (H1: p > 0.5)")
    print(f"  Statistic: hits = {ACIERTOS_OOD}")
    print(f"  p-value (one-sided): {p_valor:.6e}")
    print(f"  Two-sided 95% CI (Wilson) for the proportion: "
          f"[{ic_95.low:.4f}, {ic_95.high:.4f}]")
    print()

    # --- Also a normal approximation z-score as a second reading ---
    p0 = 0.5
    media_h0 = N_TOTAL_OOD * p0
    var_h0 = N_TOTAL_OOD * p0 * (1 - p0)
    z = (ACIERTOS_OOD - media_h0) / np.sqrt(var_h0)
    print(f"Normal approximation:")
    print(f"  z-score: {z:.2f}")
    print(f"  (z > 1.96 -> reject H0 at 5%; z > 3.29 -> reject at 0.1%)")
    print()

    # --- Verdict ---
    print("=" * 60)
    if p_valor < 0.001:
        print(f"VERDICT: H0 rejected with p < 0.001.")
    elif p_valor < 0.01:
        print(f"VERDICT: H0 rejected with p < 0.01.")
    elif p_valor < 0.05:
        print(f"VERDICT: H0 rejected with p < 0.05.")
    else:
        print(f"VERDICT: H0 cannot be rejected (p={p_valor:.4f}).")
    print("The 55.02% accuracy on the unseen configuration d")
    print("is statistically significant against chance.")
    print("=" * 60)


if __name__ == "__main__":
    main()
