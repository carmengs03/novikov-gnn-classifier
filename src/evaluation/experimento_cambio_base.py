"""Experiment: robustness of the MLP and the GNN under orthogonal
change of basis.

Methodology
-----------
Uniform change of basis over the orthogonal group O(n), sampled with
the Haar measure (Mezzadri 2007). A Novikov algebra is invariant under
any isomorphism of vector spaces (Bai & Meng 2001), and in particular
under O(n) subset GL(n). Gutierrez Silva (2025, p. 24) shows explicitly
that two bases of the same algebra give different coefficient tensors
and possibly non-isomorphic graphs.

Transformation law of the structure tensor under $f_i = P_{ai} e_a$:

    \\tilde{C}^h_{ij} = \\sum_{a,b,c} P_{ai}\\, P_{bj}\\, (P^{-1})_{hc}\\, C^c_{ab}

In einsum: torch.einsum('ai,bj,hc,abc->ijh', P, P, Pinv, C).

Design
------
For each dimension (2 and 3) and each of N_SEMILLAS seeds:
1. Retrain the MLP baseline (same configuration as
   src/evaluation/baseline_noisy_dim2.py).
2. Load the published GNN (models/modelo_final_dim2_dificil.pt for
   dim=2; models/modelo_conjunto_dim23_dificil.pt for dim=3).
3. On the test split, for every positive:
    - Sample K orthogonal variants.
    - Verify with the exact verifier (sanity).
    - Evaluate MLP -> measure agreement with the original prediction.
    - Rebuild edge_index and node features for the transformed graph
      and evaluate the GNN -> measure agreement with the original
      prediction.
4. Report, per model:
    - Agreement between pred(C) and pred(C_tilde).
    - Rate of C_tilde classified as positive (expected ~100% if the
      model learns the variety; ~50% if it captures only a canonical
      signature).

Interpretation
--------------
- MLP collapses (agreement ~50%) + GNN holds (~100%) -> the GNN learns
  something invariant under isomorphism; the MLP only a canonical
  signature. Reinforces the paper's thesis.
- Both collapse -> neither learns the variety; both detect specific
  parameterisations. Reword the conclusions.
- Only the GNN collapses -> unexpected hypothesis; investigate.
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
from sklearn.linear_model import LogisticRegression
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
# GNN architecture (identical to the paper)
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
# Mathematical utilities
# =====================================================================
def haar_orthogonal(n: int, generator: torch.Generator) -> torch.Tensor:
    """Sample Q in O(n) under the Haar measure (Mezzadri 2007)."""
    G = torch.randn(n, n, generator=generator)
    Q, R = torch.linalg.qr(G)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    return Q * d.unsqueeze(0)


def cambio_base(C: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """C_tilde^h_{ij} = P_{ai} P_{bj} (P^-1)_{hc} C^c_{ab}"""
    Pinv = torch.linalg.inv(P)
    return torch.einsum('ai,bj,hc,abc->ijh', P, P, Pinv, C)


def C_desde_grafo(g, n: int) -> torch.Tensor:
    C = torch.zeros(n, n, n)
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k, :n]
    return C


def aplanar(C: torch.Tensor) -> np.ndarray:
    return C.flatten().numpy()


def verificar_novikov(C: torch.Tensor, n: int,
                       tolerancia: float = 1e-4) -> bool:
    edges = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                          dtype=torch.long).T
    ea = torch.stack([C[a, b, :] for a in range(n) for b in range(n)])
    return es_novikov(edges, ea, n, tolerancia=tolerancia) == 1.0


def data_desde_C(C: torch.Tensor, n: int, y: float,
                  tol_arista: float = 1e-6) -> Data:
    """Rebuild a PyG `Data` from the tensor C:
      - edge_index: only edges with at least one coefficient |C^h_{ij}| > tol.
      - edge_attr: coefficients per edge (shape (E, n)).
      - x: 6 node features produced by `inyectar_caracteristicas`.
    """
    edges_list = []
    ea_list = []
    for a in range(n):
        for b in range(n):
            if C[a, b, :].abs().max().item() > tol_arista:
                edges_list.append([a, b])
                ea_list.append(C[a, b, :].clone())
    if edges_list:
        edge_index = torch.tensor(edges_list, dtype=torch.long).T
        edge_attr = torch.stack(ea_list)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, n))
    x0 = torch.ones((n, 1))  # placeholder; will be replaced by 6 features
    g_raw = Data(x=x0, edge_index=edge_index, edge_attr=edge_attr,
                  y=torch.tensor([y]))
    g_enr = inyectar_caracteristicas(g_raw, dim_algebra=n)
    return g_enr


def edge_attr_padded_a_3(g: Data, dim_algebra: int) -> Data:
    """Return a copy of g with edge_attr padded to 3 columns when
    dim_algebra < 3 (needed by the joint model)."""
    if g.edge_attr.shape[1] >= 3:
        return g
    pad = torch.zeros(g.edge_attr.shape[0], 3 - g.edge_attr.shape[1],
                       dtype=g.edge_attr.dtype)
    return Data(x=g.x.clone(),
                 edge_index=g.edge_index.clone(),
                 edge_attr=torch.cat([g.edge_attr, pad], dim=1),
                 y=g.y.clone())


# =====================================================================
# Retraining of the MLP baseline
# =====================================================================
def entrenar_mlp_baseline(dataset, dim: int, semilla: int):
    """LR + MLP with the SAME configuration as
    src/evaluation/baseline_noisy_dim2.py."""
    X = np.array([aplanar(C_desde_grafo(g, dim)) for g in dataset])
    y = np.array([int(g.y.item()) for g in dataset])

    torch.manual_seed(42 + semilla)
    perm = torch.randperm(len(dataset)).numpy()
    n_train = int(0.70 * len(dataset))
    n_val = int(0.15 * len(dataset))
    idx_tr = perm[:n_train]
    idx_te = perm[n_train + n_val:]

    sc = StandardScaler().fit(X[idx_tr])
    Xtr = sc.transform(X[idx_tr])
    Xte = sc.transform(X[idx_te])

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42 + semilla)
    lr.fit(Xtr, y[idx_tr])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                         random_state=42 + semilla)
    mlp.fit(Xtr, y[idx_tr])

    return {
        "lr": lr, "mlp": mlp, "scaler": sc,
        "idx_test": idx_te,
        "acc_test_lr": lr.score(Xte, y[idx_te]) * 100,
        "acc_test_mlp": mlp.score(Xte, y[idx_te]) * 100,
    }


# =====================================================================
# Load the published GNN
# =====================================================================
def cargar_gnn(dim: int, dispositivo):
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
    return modelo, dim_aristas, hp


def predecir_gnn(modelo: RedNovikov, grafos, dispositivo,
                  dim_aristas_target: int, batch_size: int = 64):
    """Return a 0/1 array with one prediction per graph."""
    # pad if necessary
    if dim_aristas_target == 3:
        grafos = [edge_attr_padded_a_3(g, 2) if g.edge_attr.shape[1] < 3 else g
                   for g in grafos]
    loader = DataLoader(grafos, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for lote in loader:
            lote = lote.to(dispositivo)
            p = modelo(lote).view(-1).cpu().numpy()
            preds.extend((p >= 0.5).astype(int).tolist())
    return np.array(preds)


# =====================================================================
# Experiment
# =====================================================================
def experimento(dataset_path: str, dim: int, k_variantes: int,
                 n_semillas: int, semilla_haar: int,
                 salida_por_dim: dict, dispositivo):
    print(f"\n=== dim = {dim}: {dataset_path} ===", flush=True)
    ds = torch.load(dataset_path, weights_only=False)
    n_pos_ds = sum(1 for g in ds if float(g.y) == 1.0)
    n_neg_ds = sum(1 for g in ds if float(g.y) == 0.0)
    print(f"  {len(ds):,} graphs ({n_pos_ds:,} pos / {n_neg_ds:,} neg)",
          flush=True)

    # Load the published GNN (only once)
    gnn, dim_aristas_target, hp_gnn = cargar_gnn(dim, dispositivo)
    print(f"  GNN loaded: dim_aristas={dim_aristas_target}, hp={hp_gnn}",
          flush=True)

    # Reproducible pool of Haar matrices
    gen = torch.Generator().manual_seed(semilla_haar)
    n_matrices = k_variantes * 100
    matrices_P = [haar_orthogonal(dim, gen) for _ in range(n_matrices)]
    print(f"  precomputed {n_matrices} Haar matrices in O({dim})",
          flush=True)

    resultados_semilla = []
    for s in range(n_semillas):
        print(f"\n  [seed {s+1}/{n_semillas}]", flush=True)
        t = time.time()
        modelo_mlp = entrenar_mlp_baseline(ds, dim, semilla=s)
        print(f"    MLP: {modelo_mlp['acc_test_mlp']:.2f}%   "
              f"LR: {modelo_mlp['acc_test_lr']:.2f}%   "
              f"({time.time()-t:.0f}s)", flush=True)

        # Test-set positives
        pos_idx = [i for i in modelo_mlp["idx_test"] if float(ds[i].y) == 1.0]
        print(f"    positives in test split: {len(pos_idx)}", flush=True)
        C_orig = [C_desde_grafo(ds[i], dim) for i in pos_idx]

        # ------- ORIGINAL predictions -------
        # MLP
        X_o = np.array([aplanar(C) for C in C_orig])
        X_o_std = modelo_mlp["scaler"].transform(X_o)
        p_mlp_orig = modelo_mlp["mlp"].predict(X_o_std)
        # GNN
        grafos_orig = [data_desde_C(C, dim, 1.0) for C in C_orig]
        p_gnn_orig = predecir_gnn(gnn, grafos_orig, dispositivo,
                                    dim_aristas_target)

        # ------- Generate orthogonal variants -------
        fallos_sanity = 0
        C_var, pos_i_var, P_used = [], [], []
        for pos_i, C in enumerate(C_orig):
            for k in range(k_variantes):
                P = matrices_P[(pos_i * k_variantes + k) % n_matrices]
                Cn = cambio_base(C, P)
                if not verificar_novikov(Cn, dim):
                    fallos_sanity += 1
                    continue
                C_var.append(Cn)
                pos_i_var.append(pos_i)
                P_used.append(P)
        n_eval = len(C_var)
        if n_eval == 0:
            print("    (no valid variants)", flush=True)
            continue
        print(f"    valid variants: {n_eval}  "
              f"(sanity failures: {fallos_sanity})", flush=True)

        # Evaluate MLP on the variants
        X_var = np.array([aplanar(C) for C in C_var])
        X_var_std = modelo_mlp["scaler"].transform(X_var)
        p_mlp_var = modelo_mlp["mlp"].predict(X_var_std)

        # Evaluate GNN on the variants
        grafos_var = [data_desde_C(C, dim, 1.0) for C in C_var]
        p_gnn_var = predecir_gnn(gnn, grafos_var, dispositivo,
                                   dim_aristas_target)

        # ------- Metrics -------
        pos_i_arr = np.array(pos_i_var)
        acc_mlp = int((p_mlp_var == p_mlp_orig[pos_i_arr]).sum()) / n_eval * 100
        acc_gnn = int((p_gnn_var == p_gnn_orig[pos_i_arr]).sum()) / n_eval * 100
        tas_mlp = int((p_mlp_var == 1).sum()) / n_eval * 100
        tas_gnn = int((p_gnn_var == 1).sum()) / n_eval * 100

        print(f"    MLP  agreement(C,C_tilde)={acc_mlp:.2f}%   "
              f"pos_rate(C_tilde)={tas_mlp:.2f}%", flush=True)
        print(f"    GNN  agreement(C,C_tilde)={acc_gnn:.2f}%   "
              f"pos_rate(C_tilde)={tas_gnn:.2f}%", flush=True)

        resultados_semilla.append({
            "semilla": s,
            "n_positivos": len(pos_idx),
            "n_evaluaciones": n_eval,
            "fallos_sanity": fallos_sanity,
            "mlp_test_acc_in_dist": float(modelo_mlp["acc_test_mlp"]),
            "lr_test_acc_in_dist": float(modelo_mlp["acc_test_lr"]),
            "mlp_acuerdo_pct": float(acc_mlp),
            "mlp_tasa_positivo_pct": float(tas_mlp),
            "gnn_acuerdo_pct": float(acc_gnn),
            "gnn_tasa_positivo_pct": float(tas_gnn),
        })

    # Aggregate per dim
    if resultados_semilla:
        mlp_a = np.array([r["mlp_acuerdo_pct"] for r in resultados_semilla])
        gnn_a = np.array([r["gnn_acuerdo_pct"] for r in resultados_semilla])
        mlp_t = np.array([r["mlp_tasa_positivo_pct"] for r in resultados_semilla])
        gnn_t = np.array([r["gnn_tasa_positivo_pct"] for r in resultados_semilla])
        salida_por_dim[f"dim{dim}"] = {
            "n_semillas": n_semillas,
            "k_variantes_por_positivo": k_variantes,
            "mlp_acuerdo_mean": float(mlp_a.mean()),
            "mlp_acuerdo_std": float(mlp_a.std(ddof=0)),
            "mlp_tasa_pos_mean": float(mlp_t.mean()),
            "mlp_tasa_pos_std": float(mlp_t.std(ddof=0)),
            "gnn_acuerdo_mean": float(gnn_a.mean()),
            "gnn_acuerdo_std": float(gnn_a.std(ddof=0)),
            "gnn_tasa_pos_mean": float(gnn_t.mean()),
            "gnn_tasa_pos_std": float(gnn_t.std(ddof=0)),
            "por_semilla": resultados_semilla,
        }
        print(f"\n  SUMMARY dim {dim} ({n_semillas} seeds):", flush=True)
        print(f"    MLP: agreement = {mlp_a.mean():.2f} +/- {mlp_a.std(ddof=0):.2f}%   "
              f"pos_rate = {mlp_t.mean():.2f} +/- {mlp_t.std(ddof=0):.2f}%",
              flush=True)
        print(f"    GNN: agreement = {gnn_a.mean():.2f} +/- {gnn_a.std(ddof=0):.2f}%   "
              f"pos_rate = {gnn_t.mean():.2f} +/- {gnn_t.std(ddof=0):.2f}%",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_variantes", type=int, default=5)
    parser.add_argument("--n_semillas", type=int, default=10)
    parser.add_argument("--semilla_haar", type=int, default=12345)
    parser.add_argument("--solo_dim", type=int, default=None,
                        help="run only this dimension (2 or 3)")
    parser.add_argument("--output",
                        default="results/from_experiments/experimento_cambio_base.json")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial sanity check
    print("=== Sanity of the mathematical implementation ===", flush=True)
    torch.manual_seed(0)
    for n_test, path in [
        (2, "data/dataset_enriquecido_dim2_50000_DATOS_dificiles.pt"),
        (3, "data/dataset_combinado_dim2_dim3_dificiles.pt"),
    ]:
        ds = torch.load(path, weights_only=False)
        pos = next(g for g in ds
                    if int(getattr(g, "dim_algebra", n_test)) == n_test
                    and float(g.y) == 1.0)
        C = C_desde_grafo(pos, n_test)
        C_id = cambio_base(C, torch.eye(n_test))
        assert (C - C_id).abs().max() < 1e-6, f"P=I does not reconstruct (dim={n_test})"
        gen = torch.Generator().manual_seed(0)
        P = haar_orthogonal(n_test, gen)
        Cn = cambio_base(C, P)
        assert abs(torch.norm(C).item() - torch.norm(Cn).item()) < 1e-4, \
            f"Orthogonal did not preserve the norm (dim={n_test})"
        assert verificar_novikov(Cn, n_test), \
            f"Change of basis pushed C off the variety (dim={n_test})"
    print("  sanity OK", flush=True)

    resultados = {}
    if args.solo_dim in (None, 2):
        experimento("data/dataset_enriquecido_dim2_50000_DATOS_dificiles.pt",
                     dim=2, k_variantes=args.k_variantes,
                     n_semillas=args.n_semillas,
                     semilla_haar=args.semilla_haar,
                     salida_por_dim=resultados,
                     dispositivo=dispositivo)
    if args.solo_dim in (None, 3):
        # dim=3 extracted from the combined dataset by filtering
        print(f"\n=== dim = 3: filtering from the combined dataset ===",
              flush=True)
        ds_combinado = torch.load(
            "data/dataset_combinado_dim2_dim3_dificiles.pt",
            weights_only=False)
        ds_dim3 = [g for g in ds_combinado if int(g.dim_algebra) == 3]
        print(f"  {len(ds_dim3):,} dim=3 graphs", flush=True)
        tmp_path = "/tmp/tmp_dim3_only.pt"
        torch.save(ds_dim3, tmp_path)
        experimento(tmp_path, dim=3, k_variantes=args.k_variantes,
                     n_semillas=args.n_semillas,
                     semilla_haar=args.semilla_haar + 1,
                     salida_por_dim=resultados,
                     dispositivo=dispositivo)
        os.remove(tmp_path)

    salida = {
        "hp": {
            "k_variantes_por_positivo": args.k_variantes,
            "n_semillas": args.n_semillas,
            "semilla_haar": args.semilla_haar,
        },
        "arquitectura_mlp": (
            "MLPClassifier(hidden_layer_sizes=(128, 64, 32), "
            "max_iter=300)"
        ),
        "arquitectura_gnn": (
            "RedNovikov(dim_nodos=6, dim_oculta=64, n_capas=3, "
            "aggr=max, pool=mean, residual=True)"
        ),
        "referencia_metodologica": (
            "MLP identical to src/evaluation/baseline_noisy_dim2.py "
            "(split 70/15/15, StandardScaler fitted on train). "
            "GNN: published models 'models/modelo_final_dim2_dificil.pt' "
            "(dim=2) and 'models/modelo_conjunto_dim23_dificil.pt' (dim=3)."
        ),
        "referencia_matematica": [
            "Bai C. & Meng D. (2001) 'The classification of Novikov algebras "
            "in low dimensions'. J. Phys. A 34(8):1581. Novikov invariance "
            "under vector space isomorphism.",
            "Gutierrez Silva C. (2025) 'Novikov algebras and combinatorial "
            "structures' (BSc Thesis, Universidad Loyola Andalucia), p. 24. "
            "Two bases of the same algebra give possibly different tensors "
            "and graphs.",
            "Mezzadri F. (2007) 'How to generate random matrices from the "
            "classical compact groups'. Notices AMS 54(5):592-604. Haar "
            "sampling via corrected QR.",
            "Villar S., Hogg D.W., Storey-Fisher K., Yao W., Blum-Smith B. "
            "(2021) 'Scalars are universal: Equivariant machine learning, "
            "structured like classical physics'. NeurIPS 34:28848. "
            "Equivariance testing.",
        ],
        "resultados": resultados,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(salida, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
