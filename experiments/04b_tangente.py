"""Control with positives re-perturbed in a direction tangent to the manifold.

Ruled-out check for a possible noisy-sampler artefact. A
perturbation is applied to positives in a direction tangent to the
manifold (approximation by projection onto the orthogonal complement of
the gradient of the squared residual). If the perturbed positive is
still verified as Novikov, it is kept. The GNN is then evaluated on
{positives, tangentially re-perturbed positives, sampler noisy
negatives}. If the model still separates positives from negatives when
tangential re-perturbations are present, the paper's claim holds; if it
degrades, there is an artefact.

Required files:
  - dataset_combinado_dim2_dim3_dificiles.pt
  - dataset_noisy_topologico_dim3_enriquecido.pt (noisy negatives)
  - modelo_conjunto_dim23_dificil.pt
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verificador import residuos, es_novikov_tensor


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
    def __init__(self, dim_nodos=6, dim_aristas=3, dim_oculta=64, n_capas=3, residual=True):
        super().__init__()
        self.residual = residual
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta)
        self.capas_mp = nn.ModuleList([
            CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta) for _ in range(n_capas)])
        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, datos):
        x = F.relu(self.proyeccion_entrada(datos.x))
        for capa in self.capas_mp:
            h = capa(x, datos.edge_index, datos.edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)
        return self.clasificador(global_mean_pool(x, datos.batch))


def tensor_C_de_grafo(g, dim_algebra):
    C = torch.zeros((dim_algebra, dim_algebra, dim_algebra))
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        C[i, j, :] = g.edge_attr[k, :dim_algebra]
    return C


def data_desde_tensor(C, referencia_g, dim_algebra, y_val, dim_aristas_salida=3):
    ea = []
    for k in range(referencia_g.edge_index.shape[1]):
        i, j = referencia_g.edge_index[0, k].item(), referencia_g.edge_index[1, k].item()
        ea.append(C[i, j, :].clone())
    ea = torch.stack(ea) if ea else torch.zeros((0, dim_algebra))
    if ea.shape[1] < dim_aristas_salida:
        pad = torch.zeros(ea.shape[0], dim_aristas_salida - ea.shape[1])
        ea = torch.cat([ea, pad], dim=1)
    return Data(
        x=referencia_g.x.clone(),
        edge_index=referencia_g.edge_index.clone(),
        edge_attr=ea,
        y=torch.tensor([float(y_val)]),
        dim_algebra=dim_algebra,
    )


def perturbacion_tangente(C_pos, sigma, mask, max_intentos=10):
    """Apply an approximate tangent perturbation to C_pos.

    Low-cost approximation: project Gaussian noise onto the orthogonal
    complement of the gradient of the total squared residual. Verify
    that the result is still Novikov; if not, retry. Returns None if it
    fails after max_intentos attempts.
    """
    for _ in range(max_intentos):
        C = C_pos.clone().detach().requires_grad_(True)
        R1, R2 = residuos(C)
        f = (R1 ** 2).sum() + (R2 ** 2).sum()
        g = torch.autograd.grad(f, C, create_graph=False)[0].detach()

        eta = torch.randn_like(C_pos) * mask
        norma_g_sq = (g ** 2).sum().clamp_min(1e-30)
        eta_tan = eta - (eta * g).sum() / norma_g_sq * g

        norm_eta_tan = eta_tan.norm().clamp_min(1e-30)
        eta_tan = eta_tan / norm_eta_tan * sigma * mask.numel() ** 0.5 * mask.mean().clamp_min(1e-8)

        C_pert = (C_pos + eta_tan).detach()
        if es_novikov_tensor(C_pert, tolerancia=1e-4):
            return C_pert
    return None


def normalizar(g, y_val):
    """Return a Data with only (x, edge_index, edge_attr, y). Removes
    attribute inconsistencies between positives from the combined set,
    re-perturbed positives, and noisy negatives (some lack
    dim_algebra)."""
    return Data(
        x=g.x.clone(),
        edge_index=g.edge_index.clone(),
        edge_attr=g.edge_attr.clone(),
        y=torch.tensor([float(y_val)]),
    )


def evaluar_gnn(modelo, dataset, dispositivo, batch_size=64):
    modelo.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for lote in loader:
            lote = lote.to(dispositivo)
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            pb = (pred >= 0.5).float()
            tp += ((pb == 1) & (etiq == 1)).sum().item()
            fp += ((pb == 1) & (etiq == 0)).sum().item()
            tn += ((pb == 0) & (etiq == 0)).sum().item()
            fn += ((pb == 0) & (etiq == 1)).sum().item()
    total = tp + fp + tn + fn
    return {"accuracy": (tp + tn) / total * 100 if total else 0.0,
            "n": total, "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_positivos", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--output", default="continuidad_4b.json")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dispositivo}", flush=True)

    # Load joint model
    ckpt = torch.load("modelo_conjunto_dim23_dificil.pt",
                       map_location=dispositivo, weights_only=False)
    hp = ckpt["hp"]
    modelo = RedNovikov(dim_nodos=6, dim_aristas=3,
                        dim_oculta=hp["dim_oculta"]).to(dispositivo)
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    print(f"[ckpt] hp={hp}", flush=True)

    # Load combined dataset (source of positives)
    print("\nLoading combined dataset...", flush=True)
    t = time.time()
    dataset = torch.load("dataset_combinado_dim2_dim3_dificiles.pt",
                          weights_only=False)
    print(f"  {len(dataset):,} graphs ({time.time()-t:.1f}s)", flush=True)

    # Load enriched dim=3 noisy negatives
    print("Loading enriched dim=3 noisy samples...", flush=True)
    t = time.time()
    noisy = torch.load("dataset_noisy_topologico_dim3_enriquecido.pt",
                      weights_only=False)
    print(f"  {len(noisy):,} graphs ({time.time()-t:.1f}s)", flush=True)

    # Sample dim=3 positives from the combined set (consistent edge_attr)
    random.seed(0); torch.manual_seed(0)
    positivos = [g for g in dataset if int(g.dim_algebra) == 3
                                        and float(g.y) == 1.0]
    random.shuffle(positivos)
    positivos = positivos[:args.n_positivos]
    print(f"Selected dim=3 positives: {len(positivos)}", flush=True)

    # Generate tangentially re-perturbed positives
    print(f"\nGenerating tangential positives (sigma={args.sigma})...", flush=True)
    t = time.time()
    positivos_tang = []
    for g in positivos:
        C = tensor_C_de_grafo(g, 3)
        mask = (C.abs() > 1e-6).float()
        C_tang = perturbacion_tangente(C, args.sigma, mask)
        if C_tang is not None:
            positivos_tang.append(data_desde_tensor(C_tang, g, 3, y_val=1.0))
    print(f"  accepted tangential positives: {len(positivos_tang)}/"
          f"{len(positivos)} ({time.time()-t:.0f}s)", flush=True)

    # Noisy dim=3 negatives, normalised so batching is consistent
    negativos_noisy_raw = [g for g in noisy if float(g.y) == 0.0][:len(positivos_tang)]
    negativos_noisy = [normalizar(g, 0.0) for g in negativos_noisy_raw]
    print(f"Noisy negatives: {len(negativos_noisy)}", flush=True)

    # Normalise positives (original and tangential) to the same schema
    pos_originales = [normalizar(g, 1.0)
                       for g in positivos[:len(positivos_tang)]]
    positivos_tang = [normalizar(g, 1.0) for g in positivos_tang]

    print("\nEvaluating GNN on each subset...", flush=True)
    eval_pos = evaluar_gnn(modelo, pos_originales, dispositivo)
    eval_pos_tang = evaluar_gnn(modelo, positivos_tang, dispositivo)
    eval_neg = evaluar_gnn(modelo, negativos_noisy, dispositivo)
    eval_mezcla = evaluar_gnn(modelo,
                                pos_originales + positivos_tang + negativos_noisy,
                                dispositivo)

    print(f"  original positives      : {eval_pos['accuracy']:.2f}% "
          f"(TP={eval_pos['tp']} FN={eval_pos['fn']})", flush=True)
    print(f"  re-perturbed positives  : {eval_pos_tang['accuracy']:.2f}% "
          f"(TP={eval_pos_tang['tp']} FN={eval_pos_tang['fn']})", flush=True)
    print(f"  noisy negatives   : {eval_neg['accuracy']:.2f}% "
          f"(TN={eval_neg['tn']} FP={eval_neg['fp']})", flush=True)
    print(f"  MIXED (three sources)   : {eval_mezcla['accuracy']:.2f}%", flush=True)

    salida = {
        "sigma": args.sigma,
        "n_positivos_seleccionados": len(positivos),
        "n_positivos_tang_aceptados": len(positivos_tang),
        "n_negativos_noisy": len(negativos_noisy),
        "eval": {
            "positivos_originales": eval_pos,
            "positivos_tangenciales": eval_pos_tang,
            "negativos_noisy": eval_neg,
            "mezcla_las_tres_fuentes": eval_mezcla,
        },
        "interpretacion": (
            "If the GNN keeps a high TP/(TP+FN) on tangential positives "
            "and a high TN/(TN+FP) on noisy negatives, the paper's "
            "claim holds: the model learns something structural that "
            "separates true positives from near-misses. If TP collapses "
            "on tangential positives, there is an artefact: the model "
            "was memorising the canonical signature that the tangent "
            "perturbation destroys."
        ),
    }
    with open(args.output, "w") as f:
        json.dump(salida, f, indent=2)
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
