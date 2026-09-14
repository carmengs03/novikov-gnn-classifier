"""
Latent-space analysis of the final model (Exp. C + Optuna HPs) on dim2
using 2D t-SNE and 3D PCA.

Outputs:
    tsne_por_clase.png      — 2D projection coloured by Novikov / near-miss
    tsne_por_config.png     — 2D projection coloured by topological configuration
    pca3d_por_clase.png     — 3D projection coloured by class
    pca3d_por_config.png    — 3D projection coloured by configuration
    espacio_latente_resumen.json — PCA explained variance, centroid distances
                                   and separability metrics

Reads:
    modelo_final_dim2.pt          (checkpoint)
    dataset_etiquetado_dim2.pt    (3600 graphs with .config and .y)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from modelo_mpnn import RedNovikov

PERPLEXITY_TSNE = 30
SEED = 42


def extraer_embeddings(modelo, dataset, batch_size=64):
    """Replicates the model forward pass but returns the POST-pooling vector
    (input to the classifier) rather than the final prediction."""
    modelo.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for lote in loader:
            x = F.relu(modelo.proyeccion_entrada(lote.x))
            for capa in modelo.capas_mp:
                h = capa(x, lote.edge_index, lote.edge_attr)
                x = F.relu(x + h) if modelo.residual else F.relu(h)
            x_grafo = global_mean_pool(x, lote.batch)
            embeddings.append(x_grafo.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def plot_2d_por_clase(emb_2d, ys, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color, name in [
        (0, "tab:red", "Non-Novikov (y=0)"),
        (1, "tab:blue", "Novikov (y=1)"),
    ]:
        mask = ys == label
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=color, label=f"{name} (n={mask.sum()})",
                   s=18, alpha=0.55, edgecolor="none")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_2d_por_config(emb_2d, configs, path):
    config_names = sorted(set(configs))
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, name in enumerate(config_names):
        mask = np.array(configs) == name
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(i)], label=f"config {name}",
                   s=18, alpha=0.7, edgecolor="none")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", ncol=2, fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_por_clase(emb_3d, ys, varianza, path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for label, color, name in [
        (0, "tab:red", "Non-Novikov (y=0)"),
        (1, "tab:blue", "Novikov (y=1)"),
    ]:
        mask = ys == label
        ax.scatter(emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                   c=color, label=f"{name} (n={mask.sum()})",
                   s=15, alpha=0.55, depthshade=True)
    ax.set_xlabel(f"PC1 ({varianza[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({varianza[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({varianza[2]*100:.1f}%)")
    ax.legend(loc="upper right")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_por_config(emb_3d, configs, varianza, path):
    config_names = sorted(set(configs))
    cmap = plt.cm.tab10
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, name in enumerate(config_names):
        mask = np.array(configs) == name
        ax.scatter(emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                   c=[cmap(i)], label=f"config {name}",
                   s=15, alpha=0.7, depthshade=True)
    ax.set_xlabel(f"PC1 ({varianza[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({varianza[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({varianza[2]*100:.1f}%)")
    ax.legend(loc="best", ncol=2, fontsize=9)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap_centroides(metricas, path):
    """Symmetric heatmap of L2 distances between configuration centroids
    in the original (non-projected) latent space."""
    nombres = metricas["config_names"]
    M = np.array(metricas["matriz_distancias_centroides"])

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(nombres)))
    ax.set_yticks(range(len(nombres)))
    ax.set_xticklabels([f"config {n}" for n in nombres], rotation=45, ha="right")
    ax.set_yticklabels([f"config {n}" for n in nombres])

    for i in range(len(nombres)):
        for j in range(len(nombres)):
            txt = f"{M[i, j]:.0f}"
            color = "white" if M[i, j] < M.max() * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="L2 distance")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def calcular_metricas(embeddings, configs, ys):
    """Per-config centroids plus centroid-to-centroid distance matrix
    (computed on the original 64-D space, not on the projections)."""
    config_names = sorted(set(configs))
    configs_arr = np.array(configs)

    centroides = {}
    for name in config_names:
        centroides[name] = embeddings[configs_arr == name].mean(axis=0)

    n = len(config_names)
    distancias = np.zeros((n, n))
    for i, a in enumerate(config_names):
        for j, b in enumerate(config_names):
            distancias[i, j] = np.linalg.norm(centroides[a] - centroides[b])

    # Mean intra-config distance (compactness)
    distancias_intra = {}
    for name in config_names:
        pts = embeddings[configs_arr == name]
        c = centroides[name]
        distancias_intra[name] = float(np.mean(np.linalg.norm(pts - c, axis=1)))

    # Mean distance between the Novikov and near-miss centroids
    centroide_pos = embeddings[ys == 1].mean(axis=0)
    centroide_neg = embeddings[ys == 0].mean(axis=0)
    dist_pos_neg = float(np.linalg.norm(centroide_pos - centroide_neg))
    radio_pos = float(np.mean(np.linalg.norm(
        embeddings[ys == 1] - centroide_pos, axis=1)))
    radio_neg = float(np.mean(np.linalg.norm(
        embeddings[ys == 0] - centroide_neg, axis=1)))

    return {
        "config_names": config_names,
        "matriz_distancias_centroides": distancias.tolist(),
        "distancias_intra_config": distancias_intra,
        "novikov_vs_near_miss": {
            "dist_centroides": dist_pos_neg,
            "radio_medio_novikov": radio_pos,
            "radio_medio_near_miss": radio_neg,
            "ratio_separabilidad": dist_pos_neg / max(radio_pos, radio_neg),
        },
    }


def main():
    print("Loading checkpoint...")
    ckpt = torch.load("modelo_final_dim2.pt", weights_only=False)
    hp = ckpt["hp"]
    print(f"  HP: {hp}")
    print(f"  Checkpoint test_acc: {ckpt['test_acc']:.2f}%")

    modelo = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=hp["dim_oculta"])
    modelo.load_state_dict(ckpt["state_dict"])

    print("\nLoading labelled dataset...")
    dataset = torch.load("dataset_etiquetado_dim2.pt", weights_only=False)
    configs = [g.config for g in dataset]
    ys = np.array([int(g.y.item()) for g in dataset])
    print(f"  {len(dataset)} graphs")

    print("\nExtracting post-pooling embeddings (dim 64)...")
    embeddings = extraer_embeddings(modelo, dataset)
    print(f"  Embeddings: {embeddings.shape}")

    print("\nComputing 2D t-SNE...")
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY_TSNE,
                random_state=SEED, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)

    print("Computing 3D PCA (on standardised embeddings)...")
    # StandardScaler before PCA prevents features with a large absolute
    # magnitude from dominating the components — required because the model
    # does not normalise the post-pooling outputs and some embeddings have a
    # very large raw magnitude.
    embeddings_std = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=3, random_state=SEED)
    emb_3d = pca.fit_transform(embeddings_std)
    varianza = pca.explained_variance_ratio_
    print(f"  Explained variance PC1/PC2/PC3: "
          f"{varianza[0]*100:.1f}% / {varianza[1]*100:.1f}% / {varianza[2]*100:.1f}%")
    print(f"  Cumulative: {sum(varianza)*100:.1f}%")

    print("\nGenerating figures...")
    plot_2d_por_clase(emb_2d, ys, "tsne_por_clase.png")
    print("  tsne_por_clase.png")
    plot_2d_por_config(emb_2d, configs, "tsne_por_config.png")
    print("  tsne_por_config.png")
    plot_3d_por_clase(emb_3d, ys, varianza, "pca3d_por_clase.png")
    print("  pca3d_por_clase.png")
    plot_3d_por_config(emb_3d, configs, varianza, "pca3d_por_config.png")
    print("  pca3d_por_config.png")

    print("\nComputing analytical metrics...")
    metricas = calcular_metricas(embeddings, configs, ys)
    plot_heatmap_centroides(metricas, "heatmap_distancias_configs.png")
    print("  heatmap_distancias_configs.png")
    metricas["pca_varianza"] = {
        "pc1": float(varianza[0]),
        "pc2": float(varianza[1]),
        "pc3": float(varianza[2]),
        "acumulada_3D": float(sum(varianza)),
    }
    metricas["perplexity_tsne"] = PERPLEXITY_TSNE
    metricas["random_state"] = SEED

    with open("espacio_latente_resumen.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print("  espacio_latente_resumen.json")

    print("\n=== Novikov vs near-miss separability ===")
    sep = metricas["novikov_vs_near_miss"]
    print(f"  Centroid distance:           {sep['dist_centroides']:.3f}")
    print(f"  Mean Novikov radius:         {sep['radio_medio_novikov']:.3f}")
    print(f"  Mean near-miss radius:       {sep['radio_medio_near_miss']:.3f}")
    print(f"  Separability ratio:          {sep['ratio_separabilidad']:.3f}")
    print("  (Ratio > 1 means the clusters are farther apart than their own radii)")

    print("\n=== Per-configuration compactness ===")
    for name, d in metricas["distancias_intra_config"].items():
        print(f"  config {name}: mean radius = {d:.3f}")


if __name__ == "__main__":
    main()
