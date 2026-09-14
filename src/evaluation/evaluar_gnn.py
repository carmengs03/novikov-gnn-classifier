"""Evaluate a trained checkpoint on an arbitrary dataset.

Usage:
    python evaluar_gnn.py \\
        --checkpoint modelo_conjunto_dim23_dificil_best.pt \\
        --dataset dataset_test_ood.pt \\
        --dim_aristas 3 \\
        --nombre "GNN OOD zero-shot"

Rebuilds the final architecture (aggr=max, pool=mean, 3 layers, residual)
and stores a JSON with the global accuracy.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool


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
    def __init__(self, dim_nodos=6, dim_aristas=3, dim_oculta=64, n_capas=3, residual=True):
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

    def forward(self, datos):
        x = F.relu(self.proyeccion_entrada(datos.x))
        for capa in self.capas_mp:
            h = capa(x, datos.edge_index, datos.edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)
        return self.clasificador(global_mean_pool(x, datos.batch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dim_nodos", type=int, default=6)
    p.add_argument("--dim_aristas", type=int, default=3)
    p.add_argument("--dim_oculta", type=int, default=None,
                   help="If None, read from the checkpoint (hp.dim_oculta) or inferred from the state_dict.")
    p.add_argument("--nombre", default="GNN eval")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", default=None,
                   help="Output JSON file. Default: eval_<nombre>.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Determine dim_oculta: (1) explicit argument, (2) hp from the checkpoint,
    # (3) infer from the state_dict (last dim of clasificador.0.weight).
    dim_oculta = args.dim_oculta
    if dim_oculta is None and isinstance(ckpt, dict) and "hp" in ckpt:
        dim_oculta = ckpt["hp"].get("dim_oculta")
    if dim_oculta is None:
        # clasificador.0.weight has shape (16, dim_oculta)
        dim_oculta = state_dict["clasificador.0.weight"].shape[1]
    print(f"  inferred dim_oculta: {dim_oculta}")

    modelo = RedNovikov(
        dim_nodos=args.dim_nodos, dim_aristas=args.dim_aristas, dim_oculta=dim_oculta,
    ).to(device)
    modelo.load_state_dict(state_dict)
    modelo.eval()

    print(f"Loading dataset: {args.dataset}")
    dataset = torch.load(args.dataset, weights_only=False)
    print(f"  {len(dataset):,} graphs")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Global confusion matrix: TP, FP, TN, FN
    # TP: pred=1 and label=1 (says Novikov and it is)
    # FP: pred=1 and label=0 (says Novikov, it is not -- "leaks through")
    # TN: pred=0 and label=0 (says non-Novikov and it is not)
    # FN: pred=0 and label=1 (says non-Novikov, but it was -- "missed")
    tp = fp = tn = fn = 0
    por_config = {}    # cfg -> {tp, fp, tn, fn, n}
    with torch.no_grad():
        for lote in loader:
            lote = lote.to(device)
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            pred_bin = (pred >= 0.5).float()

            tp += ((pred_bin == 1) & (etiq == 1)).sum().item()
            fp += ((pred_bin == 1) & (etiq == 0)).sum().item()
            tn += ((pred_bin == 0) & (etiq == 0)).sum().item()
            fn += ((pred_bin == 0) & (etiq == 1)).sum().item()

            if hasattr(lote, "config") and isinstance(lote.config, list):
                pb = pred_bin.tolist()
                et = etiq.tolist()
                for cfg, p, e in zip(lote.config, pb, et):
                    d = por_config.setdefault(cfg, {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0})
                    d["n"] += 1
                    if p == 1 and e == 1: d["tp"] += 1
                    elif p == 1 and e == 0: d["fp"] += 1
                    elif p == 0 and e == 0: d["tn"] += 1
                    else: d["fn"] += 1

    total = tp + fp + tn + fn
    aciertos = tp + tn
    acc = aciertos / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n[{args.nombre}]")
    print(f"  Accuracy      : {acc:.2f}%   ({aciertos:,}/{total:,})")
    print(f"  Precision     : {precision:.2f}%   (TP / (TP+FP))")
    print(f"  Recall        : {recall:.2f}%   (TP / (TP+FN))")
    print(f"  F1            : {f1:.2f}")
    print(f"  Confusion matrix (global):")
    print(f"    TP={tp:6d}    FP={fp:6d}")
    print(f"    FN={fn:6d}    TN={tn:6d}")

    resultado = {
        "nombre": args.nombre,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n": total,
        "aciertos": aciertos,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    if por_config:
        print("\nPer-config breakdown:")
        print(f"  {'config':>12s}  {'n':>6s}  {'acc':>7s}  {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}")
        resultado["por_config"] = {}
        for cfg, d in sorted(por_config.items()):
            a = (d["tp"] + d["tn"]) / d["n"] * 100
            print(f"  {cfg:>12s}  {d['n']:>6d}  {a:6.2f}%  "
                  f"{d['tp']:>5d} {d['fp']:>5d} {d['tn']:>5d} {d['fn']:>5d}")
            resultado["por_config"][cfg] = {
                **d, "accuracy": a,
            }

    out = args.out or f"eval_{args.nombre.lower().replace(' ', '_')}.json"
    with open(out, "w") as f:
        json.dump(resultado, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
