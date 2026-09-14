"""Generates latent-space plots for the hard models:

Dim2 model:
    - latente_dim2_tsne_por_clase.png
    - latente_dim2_tsne_por_config.png
    - latente_dim2_pca3d_por_config.png

Joint model:
    - latente_conjunto_tsne.png
    - latente_conjunto_pca3d.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

from enriquecer_datos import inyectar_caracteristicas
from verificador import es_novikov


# Architecture (mirrors evaluar_gnn.py / entrenar_conjunto_runpod)
class CapaAlgebraica(MessagePassing):
    def __init__(self, dim_in, dim_aristas, dim_oculta):
        super().__init__(aggr="max")
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + dim_aristas, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))


class RedNovikov(nn.Module):
    def __init__(self, dim_nodos=6, dim_aristas=2, dim_oculta=64, n_capas=3, residual=True):
        super().__init__()
        self.residual = residual
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta)
        self.capas_mp = nn.ModuleList(
            [CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta) for _ in range(n_capas)]
        )
        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def embedding(self, datos):
        x = F.relu(self.proyeccion_entrada(datos.x))
        for capa in self.capas_mp:
            h = capa(x, datos.edge_index, datos.edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)
        return global_mean_pool(x, datos.batch)

    def forward(self, datos):
        return self.clasificador(self.embedding(datos))


# Generate a per-config balanced dataset
def generar_dim2_muestra(n_por_config=200):
    """Returns a list of (graph, config) covering the 9 dim2 configs."""
    from generadores_dim2 import (
        config_b, config_c, config_d, config_e, config_f,
        config_g, config_h, config_i, config_j,
    )
    generadores = {
        "b": config_b.generar_grafo_b, "c": config_c.generar_grafo_c,
        "d": config_d.generar_grafo_d, "e": config_e.generar_grafo_e,
        "f": config_f.generar_grafo_f, "g": config_g.generar_grafo_g,
        "h": config_h.generar_grafo_h, "i": config_i.generar_grafo_i,
        "j": config_j.generar_grafo_j,
    }
    datos = []
    for nombre, gen in generadores.items():
        for _ in range(n_por_config):
            ei, ea = gen()
            y = es_novikov(ei, ea, 2)
            g_crudo = Data(x=torch.ones((2, 1)), edge_index=ei, edge_attr=ea,
                           y=torch.tensor([y]))
            g_enr = inyectar_caracteristicas(g_crudo, dim_algebra=2)
            g_enr.config = nombre
            g_enr.dim_algebra = 2
            datos.append(g_enr)
    return datos


def generar_dim3_muestra(n_por_config=100):
    """Returns a sample of dim3 graphs spanning the 31 configurations."""
    from generadores_dim3.generadores_digrafos_no_conexos import (
        config_i     as cfg3_i,     config_ii    as cfg3_ii,
        config_iii   as cfg3_iii,   config_iv    as cfg3_iv,
        config_v     as cfg3_v,     config_vi    as cfg3_vi,
        config_vii   as cfg3_vii,   config_viii  as cfg3_viii,
        config_ix    as cfg3_ix,    config_x     as cfg3_x,
        config_xi    as cfg3_xi,    config_xii   as cfg3_xii,
        config_xiii  as cfg3_xiii,  config_xiv   as cfg3_xiv,
        config_xv    as cfg3_xv,    config_xvi   as cfg3_xvi,
        config_xvii  as cfg3_xvii,  config_xviii as cfg3_xviii,
    )
    from generadores_dim3.generadores_digrafos_conexos import (
        config_1 as cfg3_1, config_2 as cfg3_2, config_3 as cfg3_3,
        config_4 as cfg3_4, config_5 as cfg3_5, config_6 as cfg3_6,
        config_7 as cfg3_7, config_8 as cfg3_8, config_9 as cfg3_9,
        config_10 as cfg3_10, config_11 as cfg3_11, config_12 as cfg3_12,
        config_13 as cfg3_13,
    )
    generadores = {
        "3D-i": cfg3_i.generar_grafo_i,     "3D-ii": cfg3_ii.generar_grafo_ii,
        "3D-iii": cfg3_iii.generar_grafo_iii, "3D-iv": cfg3_iv.generar_grafo_iv,
        "3D-v": cfg3_v.generar_grafo_v,     "3D-vi": cfg3_vi.generar_grafo_vi,
        "3D-vii": cfg3_vii.generar_grafo_vii, "3D-viii": cfg3_viii.generar_grafo_viii,
        "3D-ix": cfg3_ix.generar_grafo_ix,  "3D-x": cfg3_x.generar_grafo_x,
        "3D-xi": cfg3_xi.generar_grafo_xi,  "3D-xii": cfg3_xii.generar_grafo_xii,
        "3D-xiii": cfg3_xiii.generar_grafo_xiii, "3D-xiv": cfg3_xiv.generar_grafo_xiv,
        "3D-xv": cfg3_xv.generar_grafo_xv,  "3D-xvi": cfg3_xvi.generar_grafo_xvi,
        "3D-xvii": cfg3_xvii.generar_grafo_xvii, "3D-xviii": cfg3_xviii.generar_grafo_xviii,
        "3D-1": cfg3_1.generar_grafo_1, "3D-2": cfg3_2.generar_grafo_2,
        "3D-3": cfg3_3.generar_grafo_3, "3D-4": cfg3_4.generar_grafo_4,
        "3D-5": cfg3_5.generar_grafo_5, "3D-6": cfg3_6.generar_grafo_6,
        "3D-7": cfg3_7.generar_grafo_7, "3D-8": cfg3_8.generar_grafo_8,
        "3D-9": cfg3_9.generar_grafo_9, "3D-10": cfg3_10.generar_grafo_10,
        "3D-11": cfg3_11.generar_grafo_11, "3D-12": cfg3_12.generar_grafo_12,
        "3D-13": cfg3_13.generar_grafo_13,
    }
    datos = []
    for nombre, gen in generadores.items():
        for _ in range(n_por_config):
            ei, ea = gen(dim_algebra=3)
            y = es_novikov(ei, ea, 3)
            g_crudo = Data(x=torch.ones((3, 1)), edge_index=ei, edge_attr=ea,
                           y=torch.tensor([y]))
            g_enr = inyectar_caracteristicas(g_crudo, dim_algebra=3)
            g_enr.config = nombre
            g_enr.dim_algebra = 3
            datos.append(g_enr)
    return datos


def pad_a_dim3(dataset):
    """Pads dim2 graphs to dim_aristas=3 and to 3 node-feature rows."""
    out = []
    for g in dataset:
        if g.dim_algebra == 2:
            pad_ea = torch.zeros((g.edge_attr.shape[0], 1), dtype=g.edge_attr.dtype)
            ea = torch.cat([g.edge_attr, pad_ea], dim=1)
            pad_x = torch.zeros((1, g.x.shape[1]), dtype=g.x.dtype)
            xnew = torch.cat([g.x, pad_x], dim=0)
            gnew = Data(x=xnew, edge_index=g.edge_index, edge_attr=ea, y=g.y)
        else:
            gnew = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y)
        gnew.config = g.config
        gnew.dim_algebra = g.dim_algebra
        out.append(gnew)
    return out


# Embedding extraction
def extraer_embeddings(modelo, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeds, ys, configs, dims = [], [], [], []
    modelo.eval()
    with torch.no_grad():
        idx = 0
        for lote in loader:
            lote = lote.to(device)
            z = modelo.embedding(lote).cpu().numpy()
            embeds.append(z)
            n = z.shape[0]
            for k in range(n):
                ys.append(int(dataset[idx + k].y.item()))
                configs.append(dataset[idx + k].config)
                dims.append(dataset[idx + k].dim_algebra)
            idx += n
    return np.vstack(embeds), np.array(ys), np.array(configs), np.array(dims)


# Plots
def plot_tsne_por_clase(embeds, ys, out_path, title):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    Z = tsne.fit_transform(embeds)
    plt.figure(figsize=(8, 6))
    for cls, color, name in [(1, "#2E86C1", "Novikov"), (0, "#C0392B", "Non-Novikov")]:
        mask = ys == cls
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6, c=color, label=name)
    plt.legend(loc="best", frameon=True)
    plt.xlabel("t-SNE dimension 1"); plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()
    print(f"  -> {out_path}")


def plot_tsne_por_config(embeds, configs, out_path, title):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    Z = tsne.fit_transform(embeds)
    uniques = sorted(set(configs))
    cmap = plt.cm.get_cmap("tab20" if len(uniques) <= 20 else "gist_ncar", len(uniques))
    plt.figure(figsize=(9, 6.5))
    for i, cfg in enumerate(uniques):
        mask = configs == cfg
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7, c=[cmap(i)], label=cfg)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
               ncol=1, frameon=False)
    plt.xlabel("t-SNE dimension 1"); plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  -> {out_path}")


def plot_pca3d_por_config(embeds, configs, out_path, title):
    pca = PCA(n_components=3, random_state=42)
    Z = pca.fit_transform(embeds)
    uniques = sorted(set(configs))
    cmap = plt.cm.get_cmap("tab20" if len(uniques) <= 20 else "gist_ncar", len(uniques))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, cfg in enumerate(uniques):
        mask = configs == cfg
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=8, alpha=0.6,
                   c=[cmap(i)], label=cfg)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({100*pca.explained_variance_ratio_[2]:.1f}%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=6, ncol=1,
              frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  -> {out_path}")


def plot_transicion_tsne(embeds, dims, configs, out_path, title):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    Z = tsne.fit_transform(embeds)
    plt.figure(figsize=(9, 6.5))
    m2 = dims == 2; m3 = dims == 3
    plt.scatter(Z[m2, 0], Z[m2, 1], s=8, alpha=0.35, c="#95A5A6", label="Dimension-2 graphs (context)")
    # dim 3 coloured by config
    uniques3 = sorted(set(configs[m3]))
    cmap = plt.cm.get_cmap("tab20" if len(uniques3) <= 20 else "gist_ncar", len(uniques3))
    for i, cfg in enumerate(uniques3):
        mask = (configs == cfg) & m3
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7, c=[cmap(i)], label=cfg)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, ncol=1,
               frameon=False)
    plt.xlabel("t-SNE dimension 1"); plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  -> {out_path}")


def plot_transicion_pca(embeds, dims, configs, out_path, title):
    pca = PCA(n_components=3, random_state=42)
    Z = pca.fit_transform(embeds)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    m2 = dims == 2; m3 = dims == 3
    ax.scatter(Z[m2, 0], Z[m2, 1], Z[m2, 2], s=8, alpha=0.35, c="#95A5A6",
               label="Dimension-2 graphs (context)")
    uniques3 = sorted(set(configs[m3]))
    cmap = plt.cm.get_cmap("tab20" if len(uniques3) <= 20 else "gist_ncar", len(uniques3))
    for i, cfg in enumerate(uniques3):
        mask = (configs == cfg) & m3
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=8, alpha=0.7,
                   c=[cmap(i)], label=cfg)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({100*pca.explained_variance_ratio_[2]:.1f}%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=6, ncol=1,
              frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  -> {out_path}")


def main():
    device = torch.device("cpu")

    print("== Hard dim2 model ==")
    ck = torch.load("modelo_final_dim2_dificil.pt", map_location=device, weights_only=False)
    dim_oculta = ck["hp"]["dim_oculta"]
    modelo_dim2 = RedNovikov(dim_nodos=6, dim_aristas=2, dim_oculta=dim_oculta).to(device)
    modelo_dim2.load_state_dict(ck["state_dict"])
    print(f"  dim_oculta={dim_oculta}")

    print("Generating dim2 sample (9 configs x 200)...")
    ds2 = generar_dim2_muestra(n_por_config=200)
    print(f"  {len(ds2)} graphs")

    print("Extracting dim2 embeddings...")
    E2, y2, c2, d2 = extraer_embeddings(modelo_dim2, ds2, device)
    print(f"  embeddings shape: {E2.shape}")

    OUT = "LaTeX/Paper_JCAM_39paginas/figures"
    plot_tsne_por_clase(E2, y2, f"{OUT}/latente_dim2_tsne_por_clase.png",
                        "Dimension-2 latent space, coloured by class")
    plot_tsne_por_config(E2, c2, f"{OUT}/latente_dim2_tsne_por_config.png",
                         "Dimension-2 latent space, coloured by topological configuration")
    plot_pca3d_por_config(E2, c2, f"{OUT}/latente_dim2_pca3d_por_config.png",
                          "Dimension-2 latent space, PCA-3D projection by configuration")

    print("\n== Joint dim2 + dim3 model ==")
    ck = torch.load("modelo_conjunto_dim23_dificil.pt", map_location=device, weights_only=False)
    dim_oculta = ck["hp"]["dim_oculta"]
    modelo_conj = RedNovikov(dim_nodos=6, dim_aristas=3, dim_oculta=dim_oculta).to(device)
    modelo_conj.load_state_dict(ck["state_dict"])
    print(f"  dim_oculta={dim_oculta}")

    print("Generating joint sample (9 dim2 configs + 31 dim3 configs, 60 each)...")
    ds2_pad = pad_a_dim3(generar_dim2_muestra(n_por_config=60))
    ds3 = pad_a_dim3(generar_dim3_muestra(n_por_config=60))
    ds_conj = ds2_pad + ds3
    print(f"  {len(ds_conj)} graphs (dim2: {len(ds2_pad)}, dim3: {len(ds3)})")

    print("Extracting joint embeddings...")
    E, y, c, d = extraer_embeddings(modelo_conj, ds_conj, device)
    print(f"  embeddings shape: {E.shape}")

    plot_transicion_tsne(E, d, c, f"{OUT}/latente_conjunto_tsne.png",
                         "Joint dim-2 and dim-3 latent space, t-SNE projection")
    plot_transicion_pca(E, d, c, f"{OUT}/latente_conjunto_pca3d.png",
                        "Joint dim-2 and dim-3 latent space, PCA-3D projection")

    print("\nOK.")


if __name__ == "__main__":
    main()
