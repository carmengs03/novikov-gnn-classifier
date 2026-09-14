"""Leave-one-out zero-shot evaluation over the 9+31 catalogue topologies.

For each catalogue configuration c:
  1. Identify graphs in the combined dataset belonging to c using
     `identificar_topologia.CatalogoTopologias`.
  2. Train the joint model excluding those graphs entirely.
  3. Evaluate on the held-out graphs (zero-shot).

Reuses the exact joint-model architecture and hyperparameters (same code
as in `03_joint_10seeds.py`).

Required files in the same directory:
  - dataset_combinado_dim2_dim3_dificiles.pt
  - identificar_topologia.py + verificador.py + generadores_dim{2,3}/

Usage:
    python -u 01_ood_leaveoneout.py --n_seeds 1 --n_epocas 80
    python -u 01_ood_leaveoneout.py --solo d,xviii --n_seeds 3
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from identificar_topologia import CatalogoTopologias


DATASET_TRAIN = "dataset_combinado_dim2_dim3_dificiles.pt"
# Joint-model hyperparameters
LR = 0.0011940188529959736
DIM_OCULTA = 64
BATCH_SIZE = 64

TOPS_DIM2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
TOPS_DIM3 = (["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
              "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii"]
             + [str(k) for k in range(1, 14)])


def fijar_semilla(s):
    random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def entrenar(datos_train_full, dispositivo, n_epocas, semilla):
    """Train the joint model excluding the held-out configuration.

    Uses an 85/15 split inside train (no test split here: the "test" is
    the full held-out configuration). Keeps the best state_dict by
    val_loss.
    """
    fijar_semilla(42 + semilla)
    n = len(datos_train_full)
    n_tr = int(0.85 * n)
    n_vl = n - n_tr
    datos_tr, datos_vl = random_split(datos_train_full, [n_tr, n_vl])

    torch.manual_seed(100 + semilla)
    modelo = RedNovikov(dim_nodos=6, dim_aristas=3,
                        dim_oculta=DIM_OCULTA).to(dispositivo)
    criterio = nn.BCELoss()
    optim = torch.optim.Adam(modelo.parameters(), lr=LR)

    train_loader = DataLoader(datos_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datos_vl, batch_size=BATCH_SIZE, shuffle=False)

    mejor_val_loss = float("inf")
    mejor_state_dict = None
    mejor_epoca = 0

    for epoca in range(n_epocas):
        modelo.train()
        for lote in train_loader:
            lote = lote.to(dispositivo)
            optim.zero_grad()
            pred = modelo(lote).view(-1)
            etiq = lote.y.float().view(-1)
            criterio(pred, etiq).backward()
            optim.step()

        modelo.eval()
        val_ac = 0.0
        with torch.no_grad():
            for lote in val_loader:
                lote = lote.to(dispositivo)
                val_ac += criterio(modelo(lote).view(-1),
                                    lote.y.float().view(-1)).item() * lote.num_graphs
        val_loss = val_ac / n_vl
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_epoca = epoca + 1
            mejor_state_dict = {k: v.detach().cpu().clone()
                                 for k, v in modelo.state_dict().items()}

        if epoca == 0 or (epoca + 1) % max(1, n_epocas // 4) == 0 or epoca == n_epocas - 1:
            print(f"    epoch {epoca+1}/{n_epocas} val={val_loss:.4f} "
                  f"best={mejor_val_loss:.4f} @ep{mejor_epoca}", flush=True)

    # Restore best checkpoint for the zero-shot evaluation
    modelo.load_state_dict(mejor_state_dict)
    return modelo


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


def indexar_por_config(dataset, cat2, cat3):
    """Return dict {key -> list of indices} in a single pass."""
    print(f"Indexing {len(dataset):,} graphs by configuration...", flush=True)
    t0 = time.time()
    indices = {}
    for i, g in enumerate(dataset):
        d = int(g.dim_algebra)
        cat = cat2 if d == 2 else cat3
        ea = g.edge_attr[:, :d] if g.edge_attr.shape[1] > d else g.edge_attr
        r = cat.identificar(ea, g.edge_index)
        cfg = r.config if r.config else "unk"
        indices.setdefault(f"dim{d}_{cfg}", []).append(i)
        if (i + 1) % 40000 == 0:
            print(f"  {i+1:,}/{len(dataset):,} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"Indexed in {time.time()-t0:.1f}s. Configs: {len(indices)}",
          flush=True)
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_epocas", type=int, default=80)
    parser.add_argument("--solo", default=None,
                        help="Comma-separated list: 'd,xviii' or 'dim2_d,dim3_xviii'")
    parser.add_argument("--output", default="ood_leaveoneout.json")
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dispositivo}", flush=True)
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"[hp] lr={LR} dim_oculta={DIM_OCULTA} batch={BATCH_SIZE}", flush=True)

    t0 = time.time()
    print(f"\nLoading {DATASET_TRAIN}...", flush=True)
    dataset = torch.load(DATASET_TRAIN, weights_only=False)
    print(f"  {len(dataset):,} graphs ({time.time()-t0:.1f}s)", flush=True)

    print("Building topology catalogues...", flush=True)
    cat2 = CatalogoTopologias(2)
    cat3 = CatalogoTopologias(3)
    indices = indexar_por_config(dataset, cat2, cat3)

    if args.solo:
        objetivos = []
        for t in args.solo.split(","):
            t = t.strip()
            if t.startswith("dim"):
                objetivos.append(t)
            elif t in TOPS_DIM2:
                objetivos.append(f"dim2_{t}")
            elif t in TOPS_DIM3:
                objetivos.append(f"dim3_{t}")
    else:
        objetivos = ([f"dim2_{c}" for c in TOPS_DIM2]
                     + [f"dim3_{c}" for c in TOPS_DIM3])

    objetivos = [o for o in objetivos if o in indices]
    print(f"\nSweep over {len(objetivos)} configs, seeds={args.n_seeds}, "
          f"epochs={args.n_epocas}", flush=True)

    resultados = {}
    t_barrido = time.time()
    for k, clave in enumerate(objetivos):
        idx_test = indices[clave]
        idx_train = [i for c, ii in indices.items() if c != clave for i in ii]
        print(f"\n[{k+1}/{len(objetivos)}] held-out={clave} "
              f"n_train={len(idx_train):,} n_test={len(idx_test):,}",
              flush=True)

        datos_train = Subset(dataset, idx_train)
        datos_test = Subset(dataset, idx_test)

        res_semilla = []
        for s in range(args.n_seeds):
            t_s = time.time()
            modelo = entrenar(datos_train, dispositivo, args.n_epocas, s)
            e = evaluar_gnn(modelo, datos_test, dispositivo)
            print(f"  seed {s} zero-shot={e['accuracy']:.2f}% "
                  f"(TP={e['tp']} FP={e['fp']} TN={e['tn']} FN={e['fn']}) "
                  f"({(time.time()-t_s)/60:.1f} min)", flush=True)
            res_semilla.append({"semilla": s, **e})

        accs = np.array([r["accuracy"] for r in res_semilla])
        resultados[clave] = {
            "n_train": len(idx_train),
            "n_test": len(idx_test),
            "n_seeds": args.n_seeds,
            "n_epocas": args.n_epocas,
            "zero_shot_mean": float(accs.mean()),
            "zero_shot_std": float(accs.std(ddof=0)),
            "por_semilla": res_semilla,
        }
        with open(args.output, "w") as f:
            json.dump({"hp": {"lr": LR, "dim_oculta": DIM_OCULTA,
                              "batch_size": BATCH_SIZE},
                        "n_seeds": args.n_seeds, "n_epocas": args.n_epocas,
                        "t_min": (time.time()-t_barrido)/60,
                        "resultados": resultados}, f, indent=2)
        print(f"  [saved {len(resultados)}/{len(objetivos)}]", flush=True)


if __name__ == "__main__":
    main()
