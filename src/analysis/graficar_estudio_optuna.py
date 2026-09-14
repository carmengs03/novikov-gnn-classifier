"""
Generates two analytical figures for the Optuna study from the SQLite .db:
    - optuna_convergencia.png : val_loss per trial + best-so-far curve
    - optuna_importancia.png  : relative importance of each hyperparameter
"""
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.importance import get_param_importances

STUDY_NAME = "novikov_dim2_hp"
STORAGE = f"sqlite:///{STUDY_NAME}.db"


def main():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

    trials_completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_podados = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"Total trials: {len(study.trials)}")
    print(f"  Completed: {len(trials_completos)}")
    print(f"  Pruned:    {len(trials_podados)}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Convergence
    xs_completos = [t.number for t in trials_completos]
    ys_completos = [t.value for t in trials_completos]

    # For pruned trials we use their last intermediate_value (val_loss at the
    # epoch where they were pruned) — useful to see where they were when killed
    xs_podados, ys_podados = [], []
    for t in trials_podados:
        if t.intermediate_values:
            xs_podados.append(t.number)
            ys_podados.append(list(t.intermediate_values.values())[-1])

    # Best-so-far curve
    trials_ordenados = sorted(trials_completos, key=lambda t: t.number)
    best_so_far, mejor_actual = [], float("inf")
    for t in trials_ordenados:
        mejor_actual = min(mejor_actual, t.value)
        best_so_far.append(mejor_actual)
    xs_best = [t.number for t in trials_ordenados]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(xs_completos, ys_completos,
               c="tab:blue", s=70, alpha=0.7, edgecolor="black", linewidth=0.6,
               label=f"Completed trial (n={len(trials_completos)})", zorder=3)
    if xs_podados:
        ax.scatter(xs_podados, ys_podados,
                   c="tab:gray", s=70, marker="x", alpha=0.6, linewidth=1.5,
                   label=f"Pruned trial (n={len(trials_podados)})", zorder=2)
    ax.plot(xs_best, best_so_far,
            c="tab:red", linewidth=2.2, label="Best val_loss so far", zorder=4)

    idx_mejor = int(np.argmin(ys_completos))
    ax.scatter(xs_completos[idx_mejor], ys_completos[idx_mejor],
               c="tab:red", s=280, marker="*", edgecolor="black", linewidth=1.5,
               label=f"Optimum (trial #{xs_completos[idx_mejor]}, "
                     f"val_loss={ys_completos[idx_mejor]:.4f})",
               zorder=5)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("val_loss (BCE)")
    ax.set_title(f"Optuna study convergence ({len(study.trials)} trials, "
                 f"TPE + MedianPruner)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig("optuna_convergencia.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved: optuna_convergencia.png")

    # Hyperparameter importance (fANOVA, requires scikit-learn)
    importancias = get_param_importances(study)
    print("\nRelative importance per hyperparameter (fANOVA):")
    for k, v in sorted(importancias.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} = {v:.4f}  ({v*100:.1f}%)")

    params = list(importancias.keys())
    valores = list(importancias.values())
    orden = np.argsort(valores)[::-1]
    params = [params[i] for i in orden]
    valores = [valores[i] for i in orden]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colores = ["tab:blue", "tab:orange", "tab:green"][: len(params)]
    ax.barh(params, valores, color=colores, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Relative importance (fANOVA, sums to 1)")
    ax.set_title("Hyperparameter importance on val_loss")
    for i, v in enumerate(valores):
        ax.text(v + max(valores) * 0.02, i, f"{v:.3f}", va="center", fontsize=11)
    ax.set_xlim(0, max(valores) * 1.20)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("optuna_importancia.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: optuna_importancia.png")


if __name__ == "__main__":
    main()
