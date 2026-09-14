"""dim2 -> dim3 transition experiment via in-memory model expansion.

Nothing is retrained: modelo_final_dim2.pt is loaded as-is and the weights of
the first Linear layer of each CapaAlgebraica MLP are "expanded" by adding a
zero column for the third edge_attr channel. On 2D graphs with that channel
padded to zero, the behaviour matches the original model exactly. On genuine
3D graphs, the e3 channel is ignored due to the null column.

Outputs:
    transicion3d_tsne.png          — joint t-SNE 2D (background) + 3D (points)
    transicion3d_pca.png           — joint 3D PCA
    transicion3d_predicciones.csv  — predictions on each 3D config
    transicion3d_resumen.json      — per-config metrics
"""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from enriquecer_datos import inyectar_caracteristicas
from modelo_mpnn import CapaAlgebraica, RedNovikov
from verificador import es_novikov

import generadores_dim3.generadores_digrafos_no_conexos.config_i as cfg_i
import generadores_dim3.generadores_digrafos_no_conexos.config_iv as cfg_iv
import generadores_dim3.generadores_digrafos_no_conexos.config_viii as cfg_viii
import generadores_dim3.generadores_digrafos_no_conexos.config_xviii as cfg_xviii
import generadores_dim3.generadores_digrafos_conexos.config_1 as cfg_1
import generadores_dim3.generadores_digrafos_conexos.config_2 as cfg_2


CONFIGS_3D = {
    "3D-i":    (cfg_i.generar_grafo_i,    "Trivial (no edges)"),
    "3D-iv":   (cfg_iv.generar_grafo_iv,  "3 independent loops"),
    "3D-viii": (cfg_viii.generar_grafo_viii, "Forbidden (loop e3 + e2->e3)"),
    "3D-xviii": (cfg_xviii.generar_grafo_xviii, "Dense (loops + e2<->e3)"),
    "3D-1":    (cfg_1.generar_grafo_1,    "Connected e1->e2, e3->e2"),
    "3D-2":    (cfg_2.generar_grafo_2,    "Forbidden connected e1->e2->e3"),
}
N_POR_CONFIG_3D = 100
# Distinct from 12345 (2D labelling) and 42 (training) to avoid seed reuse
SEMILLA = 54321


def expandir_modelo(modelo_original):
    """Builds a model with dim_aristas=3 whose first Linear of each MLP has
    the original weights plus a null column for the third edge_attr channel.
    On 2D graphs padded to 3 channels the output matches the original model;
    on genuine 3D graphs the third channel contributes zero.
    """
    dim_oculta = modelo_original.proyeccion_entrada.out_features
    modelo_3d = RedNovikov(dim_nodos=6, dim_aristas=3, dim_oculta=dim_oculta,
                           n_capas=len(modelo_original.capas_mp), residual=True)
    # Projection and classifier do not depend on edge_attr: copied verbatim.
    modelo_3d.proyeccion_entrada.load_state_dict(
        modelo_original.proyeccion_entrada.state_dict()
    )
    modelo_3d.clasificador.load_state_dict(
        modelo_original.clasificador.state_dict()
    )
    # Expand the first Linear of each MLP: (dim_oculta + 2) -> (dim_oculta + 3)
    for capa_orig, capa_new in zip(modelo_original.capas_mp, modelo_3d.capas_mp):
        lin0_orig = capa_orig.mlp[0]
        lin0_new = capa_new.mlp[0]
        with torch.no_grad():
            lin0_new.weight[:, :dim_oculta + 2].copy_(lin0_orig.weight)
            lin0_new.weight[:, dim_oculta + 2].zero_()   # e3 channel = 0
            lin0_new.bias.copy_(lin0_orig.bias)
        capa_new.mlp[2].load_state_dict(capa_orig.mlp[2].state_dict())
    return modelo_3d


def pad_edge_attr_a_3d(grafo):
    """Converts a graph with 2-dim edge_attr into one with 3-dim edge_attr
    by appending a zero column. Nothing else is modified."""
    if grafo.edge_attr.shape[1] == 3:
        return grafo
    pad = torch.zeros((grafo.edge_attr.shape[0], 1), dtype=grafo.edge_attr.dtype)
    nuevo_attr = torch.cat([grafo.edge_attr, pad], dim=1)
    return Data(x=grafo.x, edge_index=grafo.edge_index,
                edge_attr=nuevo_attr, y=grafo.y,
                **{k: getattr(grafo, k) for k in ["config"] if hasattr(grafo, k)})


def generar_dataset_3d():
    """Generates N_POR_CONFIG_3D graphs per chosen 3D configuration, enriches
    them with dim_algebra=3 features and tags each one with .config."""
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    dataset = []
    resumen_clases = {}
    x_base = torch.ones((3, 1), dtype=torch.float)
    for nombre, (generador, _) in CONFIGS_3D.items():
        n_pos = n_neg = 0
        for _ in range(N_POR_CONFIG_3D):
            edge_index, edge_attr = generador(dim_algebra=3)
            y_val = es_novikov(edge_index, edge_attr, dim_algebra=3)
            grafo = Data(x=x_base, edge_index=edge_index, edge_attr=edge_attr,
                         y=torch.tensor([y_val], dtype=torch.float))
            grafo = inyectar_caracteristicas(grafo, dim_algebra=3)
            grafo.config = nombre
            dataset.append(grafo)
            n_pos += int(y_val == 1.0)
            n_neg += int(y_val == 0.0)
        resumen_clases[nombre] = (n_pos, n_neg)
        print(f"  {nombre}: {n_pos:3d} Novikov / {n_neg:3d} near-miss")
    return dataset, resumen_clases


def extraer_embeddings_y_pred(modelo, dataset, batch_size=64):
    """Runs graphs through the model, returning (a) the post-pooling embedding
    and (b) the Novikov probability."""
    modelo.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, preds = [], []
    with torch.no_grad():
        for lote in loader:
            x = F.relu(modelo.proyeccion_entrada(lote.x))
            for capa in modelo.capas_mp:
                h = capa(x, lote.edge_index, lote.edge_attr)
                x = F.relu(x + h) if modelo.residual else F.relu(h)
            x_grafo = global_mean_pool(x, lote.batch)
            prob = modelo.clasificador(x_grafo).cpu().numpy().reshape(-1)
            embeddings.append(x_grafo.cpu().numpy())
            preds.append(prob)
    return np.concatenate(embeddings, axis=0), np.concatenate(preds, axis=0)


def plot_tsne_transicion(emb_2d_total, n_grafos_2d, configs_3d_arr, path):
    fig, ax = plt.subplots(figsize=(12, 9))
    # Background: original 2D graphs in light grey
    ax.scatter(emb_2d_total[:n_grafos_2d, 0], emb_2d_total[:n_grafos_2d, 1],
               c="lightgray", s=12, alpha=0.4, edgecolor="none",
               label="Dimension-2 graphs (context)")


    # Foreground: 3D graphs coloured by configuration
    config_names_3d = list(CONFIGS_3D.keys())
    cmap = plt.cm.tab10
    for i, name in enumerate(config_names_3d):
        mask = configs_3d_arr == name
        # Offset because emb_2d_total concatenates 2D + 3D
        idx = np.where(mask)[0] + n_grafos_2d
        ax.scatter(emb_2d_total[idx, 0], emb_2d_total[idx, 1],
                   c=[cmap(i)], s=42, alpha=0.85,
                   edgecolor="black", linewidth=0.5,
                   label=name.removeprefix("3D-"))
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_transicion(emb_3d_total, n_grafos_2d, configs_3d_arr, varianza, path):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(emb_3d_total[:n_grafos_2d, 0], emb_3d_total[:n_grafos_2d, 1],
               emb_3d_total[:n_grafos_2d, 2],
               c="lightgray", s=10, alpha=0.25, depthshade=True,
               label="Dimension-2 graphs (context)")

    config_names_3d = list(CONFIGS_3D.keys())
    cmap = plt.cm.tab10
    for i, name in enumerate(config_names_3d):
        mask = configs_3d_arr == name
        idx = np.where(mask)[0] + n_grafos_2d
        ax.scatter(emb_3d_total[idx, 0], emb_3d_total[idx, 1], emb_3d_total[idx, 2],
                   c=[cmap(i)], s=40, alpha=0.9, depthshade=True,
                   label=name.removeprefix("3D-"))
    ax.set_xlabel(f"PC1 ({varianza[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({varianza[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({varianza[2]*100:.1f}%)")
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Loading 2D model checkpoint...")
    ckpt = torch.load("modelo_final_dim2.pt", weights_only=False)
    hp = ckpt["hp"]
    modelo_2d = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=hp["dim_oculta"])
    modelo_2d.load_state_dict(ckpt["state_dict"])
    print(f"  2D checkpoint test_acc: {ckpt['test_acc']:.2f}%")

    print("\nExpanding model to dim_aristas=3 (null column on e3 channel)...")
    modelo_3d_expandido = expandir_modelo(modelo_2d)

    print("\nVerifying mathematical equivalence of the expanded model...")
    dataset_2d = torch.load("dataset_etiquetado_dim2.pt", weights_only=False)
    grafo_test = dataset_2d[0]
    grafo_test_padded = pad_edge_attr_a_3d(grafo_test)
    with torch.no_grad():
        from torch_geometric.data import Batch
        b_orig = Batch.from_data_list([grafo_test])
        b_padded = Batch.from_data_list([grafo_test_padded])
        pred_orig = modelo_2d(b_orig).item()
        pred_padded = modelo_3d_expandido(b_padded).item()
    diff = abs(pred_orig - pred_padded)
    print(f"  Original 2D model pred    : {pred_orig:.8f}")
    print(f"  Expanded model pred (pad) : {pred_padded:.8f}")
    print(f"  |difference|              : {diff:.2e}  "
          f"{'(OK)' if diff < 1e-5 else '(WARNING)'}")
    assert diff < 1e-5, "Expanded model is NOT mathematically equivalent"

    print("\nGenerating labelled 3D dataset...")
    dataset_3d, resumen_clases = generar_dataset_3d()
    print(f"  Total: {len(dataset_3d)} 3D graphs")

    print("\nPadding 2D dataset edge_attr to 3 channels...")
    dataset_2d_padded = [pad_edge_attr_a_3d(g) for g in dataset_2d]
    print(f"  Done: {len(dataset_2d_padded)} padded 2D graphs")

    print("\nExtracting embeddings and predictions on the padded 2D dataset...")
    emb_2d, pred_2d = extraer_embeddings_y_pred(modelo_3d_expandido, dataset_2d_padded)
    print(f"  2D embeddings: {emb_2d.shape}")

    print("Extracting embeddings and predictions on the 3D dataset...")
    emb_3d, pred_3d = extraer_embeddings_y_pred(modelo_3d_expandido, dataset_3d)
    print(f"  3D embeddings: {emb_3d.shape}")

    # Combined for joint t-SNE / PCA
    embeddings_total = np.concatenate([emb_2d, emb_3d], axis=0)
    configs_3d_arr = np.array([g.config for g in dataset_3d])
    ys_3d = np.array([int(g.y.item()) for g in dataset_3d])

    print("\nComputing joint t-SNE (2D + 3D)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                init="pca", learning_rate="auto")
    emb_2d_proj = tsne.fit_transform(embeddings_total)

    print("Computing joint 3D PCA (with StandardScaler)...")
    embeddings_std = StandardScaler().fit_transform(embeddings_total)
    pca = PCA(n_components=3, random_state=42)
    emb_3d_proj = pca.fit_transform(embeddings_std)
    varianza = pca.explained_variance_ratio_
    print(f"  Explained variance: "
          f"PC1={varianza[0]*100:.1f}% PC2={varianza[1]*100:.1f}% PC3={varianza[2]*100:.1f}%")

    print("\nGenerating figures...")
    plot_tsne_transicion(emb_2d_proj, len(emb_2d), configs_3d_arr,
                          "transicion3d_tsne.png")
    print("  transicion3d_tsne.png")
    plot_pca_transicion(emb_3d_proj, len(emb_2d), configs_3d_arr, varianza,
                         "transicion3d_pca.png")
    print("  transicion3d_pca.png")

    print("\nExpanded model predictions per 3D config:")
    filas = []
    for name in CONFIGS_3D.keys():
        mask = configs_3d_arr == name
        probs = pred_3d[mask]
        clases = ys_3d[mask]
        pct_pred_novikov = float(np.mean(probs >= 0.5) * 100)
        pct_real_novikov = float(np.mean(clases == 1) * 100)
        acc_modelo = float(np.mean((probs >= 0.5).astype(int) == clases) * 100)
        filas.append({
            "config": name,
            "descripcion": CONFIGS_3D[name][1],
            "n": int(mask.sum()),
            "pct_real_novikov": pct_real_novikov,
            "pct_pred_novikov": pct_pred_novikov,
            "accuracy_modelo": acc_modelo,
            "prob_media": float(np.mean(probs)),
        })
        print(f"  {name:9s}: real={pct_real_novikov:5.1f}%N | "
              f"pred={pct_pred_novikov:5.1f}%N | "
              f"acc={acc_modelo:5.1f}% | "
              f"<p>={np.mean(probs):.3f}")
    pd.DataFrame(filas).to_csv("transicion3d_predicciones.csv", index=False)
    print("  transicion3d_predicciones.csv")

    resumen = {
        "n_grafos_2d": len(dataset_2d_padded),
        "n_grafos_3d": len(dataset_3d),
        "diferencia_modelo_expandido": diff,
        "varianza_pca": [float(v) for v in varianza],
        "predicciones_por_config": filas,
    }
    Path("transicion3d_resumen.json").write_text(json.dumps(resumen, indent=2))
    print("  transicion3d_resumen.json")


if __name__ == "__main__":
    main()
