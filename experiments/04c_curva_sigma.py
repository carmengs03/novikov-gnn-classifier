"""Graph-classifier accuracy as a function of the perturbation magnitude sigma.

Generates negatives by perturbing positives from the test split with a
log-spaced range of sigma values. For each sigma:
  - Sample N positives from the test split.
  - For each, generate one negative by adding Gaussian noise of
    magnitude sigma to the non-zero coefficients.
  - Check that the perturbed sample leaves the manifold (using the
    verifier).
  - Evaluate the GNN on {positives + their sigma-negatives}.
  - Record accuracy, TP/FP/TN/FN.

Reuses the joint-model architecture (dim=3) because positives are drawn
from the combined dataset. Dim=2 positives are padded to 3, edge_attr
shape (n, 3).

Required files:
  - dataset_combinado_dim2_dim3_dificiles.pt
  - modelo_conjunto_dim23_dificil.pt
  - verificador.py (next to this script)
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
from torch.utils.data import random_split
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


def data_desde_tensor(C, referencia_g, dim_algebra, dim_aristas_salida=3):
    """Build a Data padded to `dim_aristas_salida` columns."""
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
        y=torch.tensor([0.0]),
        dim_algebra=dim_algebra,
    )


def evaluar_gnn(modelo, dataset, dispositivo, batch_size=128):
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
    parser.add_argument("--sigmas", default="1e-5,1e-4,1e-3,1e-2,1e-1,1",
                        help="comma-separated list of sigma values")
    parser.add_argument("--n_positivos", type=int, default=800,
                        help="number of positives to sample from the test split")
    parser.add_argument("--max_iter_neg", type=int, default=30,
                        help="how many attempts before giving up on finding a negative")
    parser.add_argument("--output", default="continuidad_4c.json")
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
    print(f"[ckpt] hp={hp}, seed={ckpt.get('semilla', '?')}", flush=True)

    # Reproduce the checkpoint's split
    print("\nLoading combined dataset...", flush=True)
    t = time.time()
    dataset = torch.load("dataset_combinado_dim2_dim3_dificiles.pt",
                          weights_only=False)
    print(f"  {len(dataset):,} graphs ({time.time()-t:.1f}s)", flush=True)

    semilla = ckpt.get("semilla", 2)
    random.seed(42 + semilla); np.random.seed(42 + semilla)
    torch.manual_seed(42 + semilla)
    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    _, _, datos_test = random_split(dataset, [n_train, n_val, n_test])
    print(f"Test split: {len(datos_test):,} graphs", flush=True)

    # Sample positives from the test split
    positivos = [g for g in datos_test if float(g.y) == 1.0]
    random.seed(0)
    random.shuffle(positivos)
    positivos = positivos[:args.n_positivos]
    print(f"  selected positives: {len(positivos)}", flush=True)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    print(f"  sigmas: {sigmas}", flush=True)

    salida = {"hp": hp, "n_positivos_deseados": args.n_positivos, "por_sigma": {}}

    for sigma in sigmas:
        t_s = time.time()
        print(f"\n=== sigma = {sigma} ===", flush=True)
        # Generate one negative per positive
        negativos = []
        rejects = 0
        for g in positivos:
            dim = int(g.dim_algebra)
            C = tensor_C_de_grafo(g, dim)
            mask = (C.abs() > 1e-6).float()
            neg_found = None
            for _ in range(args.max_iter_neg):
                eta = torch.randn_like(C) * sigma * mask
                C_neg = C + eta
                if not es_novikov_tensor(C_neg, tolerancia=1e-4):
                    neg_found = C_neg
                    break
            if neg_found is not None:
                negativos.append(data_desde_tensor(neg_found, g, dim,
                                                    dim_aristas_salida=3))
            else:
                rejects += 1

        n_pos_efec = len(positivos) - rejects
        # Test-split positives are already padded to 3 columns, use as is.
        pos_muestra = positivos[:n_pos_efec]
        conjunto = list(pos_muestra) + negativos
        eval_gnn = evaluar_gnn(modelo, conjunto, dispositivo)
        print(f"  positives: {len(pos_muestra)}, negatives: {len(negativos)}, "
              f"rejected: {rejects}", flush=True)
        print(f"  accuracy: {eval_gnn['accuracy']:.2f}%  "
              f"(TP={eval_gnn['tp']} FP={eval_gnn['fp']} "
              f"TN={eval_gnn['tn']} FN={eval_gnn['fn']}) "
              f"({time.time()-t_s:.0f}s)", flush=True)
        salida["por_sigma"][str(sigma)] = {
            "n_positivos": len(pos_muestra),
            "n_negativos": len(negativos),
            "rechazados": rejects,
            **eval_gnn,
        }
        with open(args.output, "w") as f:
            json.dump(salida, f, indent=2)

    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
