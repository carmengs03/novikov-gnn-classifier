"""Generate the datasets for the zero-shot OOD experiment of the joint model.

Configuration split:
    - Excluded from training: config d (dim 2) and config xviii (dim 3).
    - Train: all other configs of dim 2 and dim 3.
    - Test OOD: only graphs from d + xviii, never seen during training.

Outputs:
    - dataset_train_ood.pt (train without d, without xviii)
    - dataset_test_ood.pt (only d + xviii)
    Both with edge_attr padded to dim 3 and node features enriched to
    dim 3, ready to consume by the joint model.
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov
from enriquecer_datos import inyectar_caracteristicas

# --- Generators dim 2 (9 configs) ---
from generadores_dim2 import (
    config_b, config_c, config_d, config_e, config_f,
    config_g, config_h, config_i, config_j,
)
# --- Generators dim 3 disconnected (18 configs) ---
from generadores_dim3.generadores_digrafos_no_conexos import (
    config_i     as cfg3_i,
    config_ii    as cfg3_ii,
    config_iii   as cfg3_iii,
    config_iv    as cfg3_iv,
    config_v     as cfg3_v,
    config_vi    as cfg3_vi,
    config_vii   as cfg3_vii,
    config_viii  as cfg3_viii,
    config_ix    as cfg3_ix,
    config_x     as cfg3_x,
    config_xi    as cfg3_xi,
    config_xii   as cfg3_xii,
    config_xiii  as cfg3_xiii,
    config_xiv   as cfg3_xiv,
    config_xv    as cfg3_xv,
    config_xvi   as cfg3_xvi,
    config_xvii  as cfg3_xvii,
    config_xviii as cfg3_xviii,
)
# --- Generators dim 3 connected (13 configs) ---
from generadores_dim3.generadores_digrafos_conexos import (
    config_1 as cfg3_1, config_2 as cfg3_2, config_3 as cfg3_3,
    config_4 as cfg3_4, config_5 as cfg3_5, config_6 as cfg3_6,
    config_7 as cfg3_7, config_8 as cfg3_8, config_9 as cfg3_9,
    config_10 as cfg3_10, config_11 as cfg3_11, config_12 as cfg3_12,
    config_13 as cfg3_13,
)


SEMILLA = 42

# --- Configs INCLUDED in train ---
GENERADORES_2D_TRAIN = {
    "b": config_b.generar_grafo_b,
    "c": config_c.generar_grafo_c,
    "e": config_e.generar_grafo_e,
    "f": config_f.generar_grafo_f,
    "g": config_g.generar_grafo_g,
    "h": config_h.generar_grafo_h,
    "i": config_i.generar_grafo_i,
    "j": config_j.generar_grafo_j,
}   # config d excluded
GENERADORES_3D_TRAIN = {
    "3D-i":    cfg3_i.generar_grafo_i,
    "3D-ii":   cfg3_ii.generar_grafo_ii,
    "3D-iii":  cfg3_iii.generar_grafo_iii,
    "3D-iv":   cfg3_iv.generar_grafo_iv,
    "3D-v":    cfg3_v.generar_grafo_v,
    "3D-vi":   cfg3_vi.generar_grafo_vi,
    "3D-vii":  cfg3_vii.generar_grafo_vii,
    "3D-viii": cfg3_viii.generar_grafo_viii,
    "3D-ix":   cfg3_ix.generar_grafo_ix,
    "3D-x":    cfg3_x.generar_grafo_x,
    "3D-xi":   cfg3_xi.generar_grafo_xi,
    "3D-xii":  cfg3_xii.generar_grafo_xii,
    "3D-xiii": cfg3_xiii.generar_grafo_xiii,
    "3D-xiv":  cfg3_xiv.generar_grafo_xiv,
    "3D-xv":   cfg3_xv.generar_grafo_xv,
    "3D-xvi":  cfg3_xvi.generar_grafo_xvi,
    "3D-xvii": cfg3_xvii.generar_grafo_xvii,
    "3D-1": cfg3_1.generar_grafo_1, "3D-2": cfg3_2.generar_grafo_2,
    "3D-3": cfg3_3.generar_grafo_3, "3D-4": cfg3_4.generar_grafo_4,
    "3D-5": cfg3_5.generar_grafo_5, "3D-6": cfg3_6.generar_grafo_6,
    "3D-7": cfg3_7.generar_grafo_7, "3D-8": cfg3_8.generar_grafo_8,
    "3D-9": cfg3_9.generar_grafo_9, "3D-10": cfg3_10.generar_grafo_10,
    "3D-11": cfg3_11.generar_grafo_11, "3D-12": cfg3_12.generar_grafo_12,
    "3D-13": cfg3_13.generar_grafo_13,
}   # config xviii excluded

# --- Configs EXCLUDED (test OOD) ---
GENERADORES_OOD = {
    "d":       (config_d.generar_grafo_d,       2),
    "3D-xviii":(cfg3_xviii.generar_grafo_xviii, 3),
}

N_TRAIN_POR_CONFIG_2D = 3000
N_TRAIN_POR_CONFIG_3D = 3000
N_TEST_POR_CONFIG_OOD = 2000


def pad_a_dim3(edge_index, edge_attr):
    """Pad edge_attr from dim 2 to dim 3 with a zero column."""
    pad = torch.zeros((edge_attr.shape[0], 1), dtype=edge_attr.dtype)
    return torch.cat([edge_attr, pad], dim=1)


def construir_grafo(edge_index, edge_attr, dim_algebra, config_name):
    y_val = es_novikov(edge_index, edge_attr, dim_algebra)
    if dim_algebra == 2:
        edge_attr_out = pad_a_dim3(edge_index, edge_attr)
    else:
        edge_attr_out = edge_attr
    grafo_crudo = Data(
        x=torch.ones((dim_algebra, 1), dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr_out,
        y=torch.tensor([y_val], dtype=torch.float),
    )
    # Enrich with the REAL dim_algebra (2 or 3): topological features are
    # computed over the edges BEFORE padding.
    g_enr = inyectar_caracteristicas(grafo_crudo, dim_algebra=dim_algebra)
    # However, for the dim 3 model x must have 3 rows. Re-pad.
    if dim_algebra == 2:
        pad_x = torch.zeros((1, g_enr.x.shape[1]), dtype=g_enr.x.dtype)
        g_enr.x = torch.cat([g_enr.x, pad_x], dim=0)
    g_enr.dim_algebra = dim_algebra
    g_enr.config = config_name
    return g_enr


def generar_bloque(nombre, generador, dim, n_muestras):
    dataset = []
    for _ in range(n_muestras):
        if dim == 3:
            ei, ea = generador(dim_algebra=3)
        else:
            ei, ea = generador()
        dataset.append(construir_grafo(ei, ea, dim, nombre))
    return dataset


def main():
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    print(f"Generating TRAIN (without d, without xviii)...")
    ds_train = []
    for nombre, gen in GENERADORES_2D_TRAIN.items():
        bloque = generar_bloque(nombre, gen, 2, N_TRAIN_POR_CONFIG_2D)
        ds_train += bloque
    print(f"  dim 2 (8 configs x {N_TRAIN_POR_CONFIG_2D}): {sum(1 for g in ds_train if g.dim_algebra==2):,} graphs")
    for nombre, gen in GENERADORES_3D_TRAIN.items():
        bloque = generar_bloque(nombre, gen, 3, N_TRAIN_POR_CONFIG_3D)
        ds_train += bloque
    n_2d = sum(1 for g in ds_train if g.dim_algebra == 2)
    n_3d = sum(1 for g in ds_train if g.dim_algebra == 3)
    print(f"  dim 3 (30 configs x {N_TRAIN_POR_CONFIG_3D}): {n_3d:,} graphs")

    # Balance 50/50 in train
    pos = [g for g in ds_train if int(g.y.item()) == 1]
    neg = [g for g in ds_train if int(g.y.item()) == 0]
    n_balance = min(len(pos), len(neg))
    print(f"\nTrain balance: pos={len(pos):,}, neg={len(neg):,} -> {n_balance:,} per class")
    ds_train_bal = random.sample(pos, n_balance) + random.sample(neg, n_balance)
    random.shuffle(ds_train_bal)

    torch.save(ds_train_bal, "dataset_train_ood.pt")
    print(f"Saved: dataset_train_ood.pt ({len(ds_train_bal):,} graphs)")

    print(f"\nGenerating TEST OOD (only d + xviii)...")
    ds_test = []
    for nombre, (gen, dim) in GENERADORES_OOD.items():
        bloque = generar_bloque(nombre, gen, dim, N_TEST_POR_CONFIG_OOD)
        ds_test += bloque
    pos_t = sum(1 for g in ds_test if int(g.y.item()) == 1)
    neg_t = sum(1 for g in ds_test if int(g.y.item()) == 0)
    print(f"  {len(ds_test):,} graphs ({pos_t:,} Novikov / {neg_t:,} near-miss)")
    random.shuffle(ds_test)

    torch.save(ds_test, "dataset_test_ood.pt")
    print(f"Saved: dataset_test_ood.pt ({len(ds_test):,} graphs)")


if __name__ == "__main__":
    main()
