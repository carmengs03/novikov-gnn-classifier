"""Generates the binomial significance-test figure for the OOD zero-shot result.

Plots the null distribution of correct predictions, marks the observed value
and shades the rejection region.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


N_TOTAL = 1892
ACIERTOS_OBS = 1041  # = round(0.5502 * 1892)
P_H0 = 0.5
ALPHA = 0.001  # 0.1% significance level


def main():
    media_h0 = N_TOTAL * P_H0
    std_h0 = np.sqrt(N_TOTAL * P_H0 * (1 - P_H0))

    # Reasonable range: +/- 5 sigma around the mean
    k_lo = int(media_h0 - 5 * std_h0)
    k_hi = int(max(media_h0 + 5 * std_h0, ACIERTOS_OBS + 20))
    ks = np.arange(k_lo, k_hi + 1)
    pmf = binom.pmf(ks, N_TOTAL, P_H0)

    # Critical value to reject H0 at level alpha (one-sided)
    k_crit = binom.ppf(1 - ALPHA, N_TOTAL, P_H0)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(ks, pmf, color="tab:blue", linewidth=2,
            label=f"Distribution under $H_0$: Bin({N_TOTAL},\\,{P_H0})")
    ax.fill_between(ks, 0, pmf, alpha=0.2, color="tab:blue")

    ax.axvline(k_crit, color="tab:red", linewidth=2, linestyle=":",
               label=f"Critical value $k_{{\\alpha={ALPHA}}} = {int(k_crit)}$")

    ax.axvline(ACIERTOS_OBS, color="black", linewidth=2.5, linestyle="--",
               label=f"Observed correct = {ACIERTOS_OBS} (55.02%)")

    ax.annotate(
        f"$p$-value $= 6{{.}}85 \\times 10^{{-6}}$\n$z = 4{{.}}37$",
        xy=(ACIERTOS_OBS, binom.pmf(ACIERTOS_OBS, N_TOTAL, P_H0)),
        xytext=(ACIERTOS_OBS + 25, max(pmf) * 0.6),
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"),
    )

    ax.set_xlabel("Number of correct predictions $k$ over $n=1892$ config-d graphs", fontsize=12)
    ax.set_ylabel("Probability under $H_0$", fontsize=12)
    ax.set_title(
        "Binomial significance test of the OOD zero-shot result\n"
        f"$H_0$: $p = 0.5$ (chance)  vs.  $H_1$: $p > 0.5$ (better than chance)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(k_lo, k_hi)
    ax.set_ylim(0, max(pmf) * 1.15)

    plt.tight_layout()
    plt.savefig("grafica_test_ood.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: grafica_test_ood.png")


if __name__ == "__main__":
    main()
