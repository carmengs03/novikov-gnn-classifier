"""Control for the change-of-basis experiment: also evaluates negatives.

Motivation
----------
The main experiment (experimento_cambio_base.py) showed that the GNN
keeps a "positive" rate of 99% on positives transformed by O(n), while
the MLP collapses to 20%. However, that high rate alone does not prove
invariance under isomorphism: it could just mean that the GNN is biased
towards always predicting "positive".

This control applies the same change of basis to NEGATIVES and checks:
  - Applying O(n) to a non-Novikov tensor produces another non-Novikov
    tensor (by invariance of the Novikov property, and its negation,
    under isomorphism). Sanity: `es_novikov(C_tilde) == 0`.
  - If the GNN keeps predicting "negative" for the transformed
    negatives -> it learns a real discrimination (passes the control).
  - If it starts predicting "positive" for the transformed negatives
    -> it was biased, it does not learn the variety.

Output: results/from_experiments/experimento_cambio_base_control.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "core"))
from verificador import es_novikov
from enriquecer_datos import inyectar_caracteristicas


# =====================================================================
# GNN identical to the paper (same class as in experimento_cambio_base.py)
# =====================================================================
class CapaAlgebraica(MessagePassing):
    def __init__(self, dim_in, dim_aristas, dim_oculta):
        super().__init__(aggr="max")
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + dim_aristas, dim_oculta), nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
        )
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))


class RedNovikov(nn.Module):
    def __init__(self, dim_nodos=6, dim_aristas=3, dim_oculta=64,
                  n_capas=3, residual=True):
        super().__init__()
        self.residual = residual
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta)
        self.capas_mp = nn.ModuleList([
            CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta)
            for _ in range(n_capas)])
        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, datos):
        x = F.relu(self.proyeccion_entrada(datos.x))
        for capa in self.capas_mp:
            h = capa(x, datos.edge_index, datos.edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)
        return self.clasificador(global_mean_pool(x, datos.batch))


# =====================================================================
def haar_orthogonal(n, generator):
    G = torch.randn(n, n, generator=generator)
    Q, R = torch.linalg.qr(G)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    return Q * d.unsqueeze(0)


def cambio_base(C, P):
    Pinv = torch.linalg.inv(P)
    return torch.einsum('ai,bj,hc,abc->ijh', P, P, Pinv, C)


def C_desde_grafo(g, n):
    C = torch.zeros(n, n, n)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k, :n]
    return C


def es_novikov_C(C, n, tol=1e-4):
    edges = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                          dtype=torch.long).T
    ea = torch.stack([C[a, b, :] for a in range(n) for b in range(n)])
    return es_novikov(edges, ea, n, tolerancia=tol) == 1.0


def data_desde_C(C, n, y, tol_arista=1e-6):
    edges_list, ea_list = [], []
    for a in range(n):
        for b in range(n):
            if C[a, b, :].abs().max().item() > tol_arista:
                edges_list.append([a, b])
                ea_list.append(C[a, b, :].clone())
    if edges_list:
        ei = torch.tensor(edges_list, dtype=torch.long).T
        ea = torch.stack(ea_list)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, n))
    g_raw = Data(x=torch.ones((n, 1)), edge_index=ei, edge_attr=ea,
                  y=torch.tensor([y]))
    return inyectar_caracteristicas(g_raw, dim_algebra=n)


def edge_attr_padded_a_3(g, dim_algebra):
    if g.edge_attr.shape[1] >= 3:
        return g
    pad = torch.zeros(g.edge_attr.shape[0], 3 - g.edge_attr.shape[1],
                       dtype=g.edge_attr.dtype)
    return Data(x=g.x.clone(),
                 edge_index=g.edge_index.clone(),
                 edge_attr=torch.cat([g.edge_attr, pad], dim=1),
                 y=g.y.clone())


def predecir_gnn(modelo, grafos, dispositivo, dim_aristas_target,
                  batch_size=64):
    if dim_aristas_target == 3:
        grafos = [edge_attr_padded_a_3(g, 2) if g.edge_attr.shape[1] < 3
                   else g for g in grafos]
    loader = DataLoader(grafos, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for lote in loader:
            lote = lote.to(dispositivo)
            p = modelo(lote).view(-1).cpu().numpy()
            preds.extend((p >= 0.5).astype(int).tolist())
    return np.array(preds)


def cargar_gnn(dim, dispositivo):
    if dim == 2:
        path = "models/modelo_final_dim2_dificil.pt"
        dim_aristas = 2
    else:
        path = "models/modelo_conjunto_dim23_dificil.pt"
        dim_aristas = 3
    ckpt = torch.load(path, map_location=dispositivo, weights_only=False)
    hp = ckpt["hp"]
    modelo = RedNovikov(dim_nodos=6, dim_aristas=dim_aristas,
                        dim_oculta=hp["dim_oculta"]).to(dispositivo)
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, dim_aristas


# =====================================================================
# MLP baseline (same scheme as baseline_noisy_dim2.py)
# =====================================================================
def aplanar(C):
    return C.flatten().numpy()


def entrenar_mlp_baseline(dataset, dim, semilla):
    X = np.array([aplanar(C_desde_grafo(g, dim)) for g in dataset])
    y = np.array([int(g.y.item()) for g in dataset])
    torch.manual_seed(42 + semilla)
    perm = torch.randperm(len(dataset)).numpy()
    n_tr = int(0.70 * len(dataset))
    n_vl = int(0.15 * len(dataset))
    idx_tr = perm[:n_tr]
    idx_te = perm[n_tr + n_vl:]
    sc = StandardScaler().fit(X[idx_tr])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                         random_state=42 + semilla)
    mlp.fit(sc.transform(X[idx_tr]), y[idx_tr])
    return {"mlp": mlp, "scaler": sc, "idx_test": idx_te,
            "acc_test_mlp": mlp.score(sc.transform(X[idx_te]), y[idx_te]) * 100}


# =====================================================================
# Experiment on BOTH classes (positives + negatives)
# =====================================================================
def experimento(dataset_path, dim, k_variantes, n_semillas,
                 semilla_haar, salida_por_dim, dispositivo):
    print(f"\n=== dim = {dim}: {dataset_path} ===", flush=True)
    ds = torch.load(dataset_path, weights_only=False)
    print(f"  {len(ds):,} graphs", flush=True)

    gnn, dim_aristas_target = cargar_gnn(dim, dispositivo)
    print(f"  GNN loaded (dim_aristas={dim_aristas_target})", flush=True)

    gen = torch.Generator().manual_seed(semilla_haar)
    matrices_P = [haar_orthogonal(dim, gen) for _ in range(k_variantes * 200)]

    resultados_semilla = []
    for s in range(n_semillas):
        print(f"\n  [seed {s+1}/{n_semillas}]", flush=True)
        t = time.time()
        m = entrenar_mlp_baseline(ds, dim, s)
        print(f"    MLP in-dist: {m['acc_test_mlp']:.2f}% ({time.time()-t:.0f}s)",
              flush=True)

        # Positives and negatives on the test split
        pos_idx = [i for i in m["idx_test"] if float(ds[i].y) == 1.0]
        neg_idx = [i for i in m["idx_test"] if float(ds[i].y) == 0.0]
        print(f"    positives in test: {len(pos_idx)}, negatives: {len(neg_idx)}",
              flush=True)

        # --- Process each class ---
        rs = {}
        for etiqueta, idx_list, y_real in [("positivos", pos_idx, 1),
                                             ("negativos", neg_idx, 0)]:
            C_orig = [C_desde_grafo(ds[i], dim) for i in idx_list]

            # Sanity: the originals have the expected class
            #   (positives: es_novikov=1; negatives: es_novikov=0)
            #   Only verified on a sample to avoid a slowdown.
            sanity_originales = sum(
                1 for C in C_orig[:200]
                if es_novikov_C(C, dim) == (y_real == 1))
            n_muestra = min(200, len(C_orig))
            if sanity_originales < n_muestra * 0.98:
                print(f"    [sanity {etiqueta} orig] "
                      f"{sanity_originales}/{n_muestra} consistent with y={y_real}",
                      flush=True)

            # Original predictions
            X_o = np.array([aplanar(C) for C in C_orig])
            p_mlp_o = m["mlp"].predict(m["scaler"].transform(X_o))
            g_o = [data_desde_C(C, dim, float(y_real)) for C in C_orig]
            p_gnn_o = predecir_gnn(gnn, g_o, dispositivo, dim_aristas_target)

            # Orthogonal variants with invariance sanity
            C_var, orig_i, sanity_ok, sanity_fail = [], [], 0, 0
            for i, C in enumerate(C_orig):
                for k in range(k_variantes):
                    P = matrices_P[(i * k_variantes + k) % len(matrices_P)]
                    Cn = cambio_base(C, P)
                    # sanity: the class must be preserved (positive->Novikov, negative->non-Novikov)
                    if es_novikov_C(Cn, dim) == (y_real == 1):
                        sanity_ok += 1
                    else:
                        sanity_fail += 1
                    C_var.append(Cn)
                    orig_i.append(i)
            n_eval = len(C_var)

            X_v = np.array([aplanar(C) for C in C_var])
            p_mlp_v = m["mlp"].predict(m["scaler"].transform(X_v))
            g_v = [data_desde_C(C, dim, float(y_real)) for C in C_var]
            p_gnn_v = predecir_gnn(gnn, g_v, dispositivo, dim_aristas_target)

            orig_i_arr = np.array(orig_i)
            acc_mlp = int((p_mlp_v == y_real).sum()) / n_eval * 100  # predicts the correct class?
            acc_gnn = int((p_gnn_v == y_real).sum()) / n_eval * 100
            acu_mlp = int((p_mlp_v == p_mlp_o[orig_i_arr]).sum()) / n_eval * 100  # agreement with original
            acu_gnn = int((p_gnn_v == p_gnn_o[orig_i_arr]).sum()) / n_eval * 100

            print(f"    [{etiqueta:9s}] variants={n_eval}  "
                  f"sanity(inv)={sanity_ok}/{n_eval} ({sanity_fail} fail)",
                  flush=True)
            print(f"      MLP  accuracy(C_tilde==y)={acc_mlp:.2f}%   "
                  f"agreement(C,C_tilde)={acu_mlp:.2f}%", flush=True)
            print(f"      GNN  accuracy(C_tilde==y)={acc_gnn:.2f}%   "
                  f"agreement(C,C_tilde)={acu_gnn:.2f}%", flush=True)

            rs[etiqueta] = {
                "n_originales": len(idx_list),
                "n_variantes": n_eval,
                "sanity_ok": sanity_ok,
                "sanity_fail": sanity_fail,
                "mlp_precision_variantes_pct": float(acc_mlp),
                "mlp_acuerdo_pct": float(acu_mlp),
                "gnn_precision_variantes_pct": float(acc_gnn),
                "gnn_acuerdo_pct": float(acu_gnn),
            }

        resultados_semilla.append({
            "semilla": s,
            "mlp_test_acc_in_dist": float(m["acc_test_mlp"]),
            "positivos": rs["positivos"],
            "negativos": rs["negativos"],
        })

    # Aggregate summary
    def agg(key1, key2):
        vals = np.array([r[key1][key2] for r in resultados_semilla])
        return float(vals.mean()), float(vals.std(ddof=0))

    resumen = {}
    for et in ("positivos", "negativos"):
        for m_ in ("mlp_precision_variantes_pct", "mlp_acuerdo_pct",
                   "gnn_precision_variantes_pct", "gnn_acuerdo_pct"):
            mean, std = agg(et, m_)
            resumen[f"{et}_{m_}"] = {"mean": mean, "std": std}

    salida_por_dim[f"dim{dim}"] = {
        "n_semillas": n_semillas,
        "k_variantes_por_grafo": k_variantes,
        "resumen": resumen,
        "por_semilla": resultados_semilla,
    }

    print(f"\n  SUMMARY dim {dim} ({n_semillas} seeds):", flush=True)
    for et in ("positivos", "negativos"):
        for etiq_mod, key in [
            ("MLP prec ", f"{et}_mlp_precision_variantes_pct"),
            ("GNN prec ", f"{et}_gnn_precision_variantes_pct"),
        ]:
            r = resumen[key]
            print(f"    [{et:9s}] {etiq_mod}: {r['mean']:.2f} +/- {r['std']:.2f}%",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_variantes", type=int, default=5)
    parser.add_argument("--n_semillas", type=int, default=10)
    parser.add_argument("--semilla_haar", type=int, default=12345)
    parser.add_argument("--solo_dim", type=int, default=None)
    parser.add_argument("--output",
                        default="results/from_experiments/experimento_cambio_base_control.json")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resultados = {}

    if args.solo_dim in (None, 2):
        experimento("data/dataset_enriquecido_dim2_50000_DATOS_dificiles.pt",
                     dim=2, k_variantes=args.k_variantes,
                     n_semillas=args.n_semillas,
                     semilla_haar=args.semilla_haar,
                     salida_por_dim=resultados,
                     dispositivo=dispositivo)
    if args.solo_dim in (None, 3):
        print(f"\n=== dim = 3: filtering from the combined dataset ===",
              flush=True)
        ds_c = torch.load("data/dataset_combinado_dim2_dim3_dificiles.pt",
                            weights_only=False)
        ds3 = [g for g in ds_c if int(g.dim_algebra) == 3]
        tmp = "/tmp/tmp_dim3_control.pt"
        torch.save(ds3, tmp)
        experimento(tmp, dim=3, k_variantes=args.k_variantes,
                     n_semillas=args.n_semillas,
                     semilla_haar=args.semilla_haar + 1,
                     salida_por_dim=resultados,
                     dispositivo=dispositivo)
        os.remove(tmp)

    salida = {
        "hp": {
            "k_variantes_por_grafo": args.k_variantes,
            "n_semillas": args.n_semillas,
            "semilla_haar": args.semilla_haar,
        },
        "descripcion": (
            "Control for the change-of-basis experiment: evaluates positives "
            "AND negatives after applying an orthogonal O(n) transformation. "
            "Sanity: preservation of the Novikov / non-Novikov class."
        ),
        "referencias": [
            "Bai C. & Meng D. (2001) J. Phys. A 34:1581",
            "Gutierrez Silva C. (2025) BSc Thesis, Univ. Loyola Andalucia, p. 24",
            "Mezzadri F. (2007) Notices AMS 54(5):592",
        ],
        "resultados": resultados,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(salida, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
