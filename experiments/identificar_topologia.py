"""Configuration (topology) identifier for a graph of the dataset.

Signature
---------
Each of the 9 dim=2 and 31 dim=3 generators fixes:
  (a) an `edge_index` (ordered set of arcs of the pseudo-digraph);
  (b) a fixed pattern of `edge_attr` positions that may be non-zero,
      both for positives and for negatives.

Denote by `envelope_pos(c)` the union of `|edge_attr| > tol` masks over
all reachable positives of generator `c`, and by `envelope_neg(c)` the
union of masks over all negatives. A dataset graph is identified as
configuration `c` when:
  - `edge_set(g) == edge_set(c)` (same edge_index as a set), and
  - either `mask(g) subseteq envelope_pos(c)` (positive),
    or `envelope_pos(c) subset mask(g) subseteq envelope_neg(c)`
    (negative with perturbed positions).

If two configs share signature, the graph is marked ambiguous. The
identifier reports separate counts for positives, negatives, ambiguous
and unknown so they can be audited.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TOL_ESTRUCTURAL = 1e-6
K_MUESTRAS = 200
SEMILLA = 12345


@dataclass
class ResultadoIdentificacion:
    config: Optional[str]
    tipo: str  # 'positivo' | 'negativo' | 'ambiguo' | 'desconocido'
    envelope_pos_size: int
    envelope_neg_size: int
    mask_size: int
    nucleo_pos_size: int = -1
    nucleo_neg_size: int = -1


def _mascara(edge_attr: torch.Tensor, tol: float = TOL_ESTRUCTURAL) -> torch.Tensor:
    return edge_attr.abs() > tol


def _edges(edge_index: torch.Tensor) -> FrozenSet[Tuple[int, int]]:
    if edge_index.numel() == 0:
        return frozenset()
    return frozenset(zip(edge_index[0].tolist(), edge_index[1].tolist()))


class CatalogoTopologias:
    """Catalogue with signature (edges, envelope_pos, envelope_neg) per config."""

    def __init__(self, dim: int, k: int = K_MUESTRAS, semilla: int = SEMILLA):
        assert dim in (2, 3)
        self.dim = dim
        # entry: edges, envelope_pos, envelope_neg, nucleo_pos, nucleo_neg
        self._firmas: Dict[str, Tuple[FrozenSet[Tuple[int, int]],
                                       torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]] = {}
        self._construir(k, semilla)

    # ------------------------------------------------------------------
    def _generadores_dim2(self):
        from generadores_dim2 import (config_b, config_c, config_d, config_e,
                                       config_f, config_g, config_h, config_i,
                                       config_j)
        return {
            "b": config_b.generar_grafo_b, "c": config_c.generar_grafo_c,
            "d": config_d.generar_grafo_d, "e": config_e.generar_grafo_e,
            "f": config_f.generar_grafo_f, "g": config_g.generar_grafo_g,
            "h": config_h.generar_grafo_h, "i": config_i.generar_grafo_i,
            "j": config_j.generar_grafo_j,
        }

    def _generadores_dim3(self):
        from generadores_dim3.generadores_digrafos_no_conexos import (
            config_i as ci, config_ii as cii, config_iii as ciii,
            config_iv as civ, config_v as cv, config_vi as cvi,
            config_vii as cvii, config_viii as cviii, config_ix as cix,
            config_x as cx, config_xi as cxi, config_xii as cxii,
            config_xiii as cxiii, config_xiv as cxiv, config_xv as cxv,
            config_xvi as cxvi, config_xvii as cxvii, config_xviii as cxviii,
        )
        from generadores_dim3.generadores_digrafos_conexos import (
            config_1 as c1, config_2 as c2, config_3 as c3, config_4 as c4,
            config_5 as c5, config_6 as c6, config_7 as c7, config_8 as c8,
            config_9 as c9, config_10 as c10, config_11 as c11,
            config_12 as c12, config_13 as c13,
        )
        return {
            "i": ci.generar_grafo_i, "ii": cii.generar_grafo_ii,
            "iii": ciii.generar_grafo_iii, "iv": civ.generar_grafo_iv,
            "v": cv.generar_grafo_v, "vi": cvi.generar_grafo_vi,
            "vii": cvii.generar_grafo_vii, "viii": cviii.generar_grafo_viii,
            "ix": cix.generar_grafo_ix, "x": cx.generar_grafo_x,
            "xi": cxi.generar_grafo_xi, "xii": cxii.generar_grafo_xii,
            "xiii": cxiii.generar_grafo_xiii, "xiv": cxiv.generar_grafo_xiv,
            "xv": cxv.generar_grafo_xv, "xvi": cxvi.generar_grafo_xvi,
            "xvii": cxvii.generar_grafo_xvii, "xviii": cxviii.generar_grafo_xviii,
            "1": c1.generar_grafo_1, "2": c2.generar_grafo_2,
            "3": c3.generar_grafo_3, "4": c4.generar_grafo_4,
            "5": c5.generar_grafo_5, "6": c6.generar_grafo_6,
            "7": c7.generar_grafo_7, "8": c8.generar_grafo_8,
            "9": c9.generar_grafo_9, "10": c10.generar_grafo_10,
            "11": c11.generar_grafo_11, "12": c12.generar_grafo_12,
            "13": c13.generar_grafo_13,
        }

    def _construir(self, k: int, semilla: int) -> None:
        random.seed(semilla)
        torch.manual_seed(semilla)
        from verificador import es_novikov
        gens = self._generadores_dim2() if self.dim == 2 else self._generadores_dim3()
        for nombre, fn in gens.items():
            edges_ref = None
            env_pos = None
            env_neg = None
            nuc_pos = None
            nuc_neg = None
            intentos = 0
            n_pos = n_neg = 0
            while (n_pos < k or n_neg < k) and intentos < 60 * k:
                intentos += 1
                ei, ea = fn()
                if edges_ref is None:
                    edges_ref = _edges(ei)
                y = es_novikov(ei, ea, self.dim)
                m = _mascara(ea)
                if y == 1.0 and n_pos < k:
                    env_pos = m if env_pos is None else (env_pos | m)
                    nuc_pos = m if nuc_pos is None else (nuc_pos & m)
                    n_pos += 1
                elif y == 0.0 and n_neg < k:
                    env_neg = m if env_neg is None else (env_neg | m)
                    nuc_neg = m if nuc_neg is None else (nuc_neg & m)
                    n_neg += 1
            # Fill empty envelopes with zeros for configs with no
            # reachable positives or no reachable negatives.
            if env_pos is None and env_neg is not None:
                env_pos = torch.zeros_like(env_neg)
                nuc_pos = torch.zeros_like(env_neg)
            if env_neg is None and env_pos is not None:
                env_neg = env_pos.clone()
                nuc_neg = nuc_pos.clone() if nuc_pos is not None else env_pos.clone()
            if env_pos is None and env_neg is None:
                ei, ea = fn()
                env_pos = torch.zeros_like(_mascara(ea))
                env_neg = env_pos.clone()
                nuc_pos = env_pos.clone()
                nuc_neg = env_pos.clone()
            self._firmas[nombre] = (edges_ref, env_pos, env_neg, nuc_pos, nuc_neg)

        # Trivial configuration 'a' in dim=2.
        if self.dim == 2 and self._firmas:
            edges_ref, env_pos_ejemplo, _, _, _ = next(iter(self._firmas.values()))
            zeros = torch.zeros_like(env_pos_ejemplo)
            self._firmas["a"] = (edges_ref, zeros, zeros, zeros, zeros)

    # ------------------------------------------------------------------
    @property
    def firmas(self):
        return dict(self._firmas)

    def identificar(self, edge_attr: torch.Tensor,
                    edge_index: torch.Tensor) -> ResultadoIdentificacion:
        mask = _mascara(edge_attr)
        edges = _edges(edge_index)

        # Special case: empty mask (edge_attr all zero) corresponds to
        # the trivial algebra. Labelled 'a' in dim=2, 'i' in dim=3 (the
        # empty trivial topology of the Bai-Meng catalogue).
        if int(mask.sum().item()) == 0:
            if self.dim == 2:
                return ResultadoIdentificacion("a", "positivo",
                                                0, 0,
                                                int(mask.sum().item()))
            else:
                return ResultadoIdentificacion("i", "positivo",
                                                0, 0,
                                                int(mask.sum().item()))

        candidatos = [(c, ep, en, np_, nn)
                      for c, (e, ep, en, np_, nn) in self._firmas.items()
                      if e == edges]

        if not candidatos:
            return ResultadoIdentificacion(None, "desconocido",
                                            0, 0, int(mask.sum().item()))

        # Positive: nucleo_pos subseteq mask subseteq envelope_pos
        pos_matches = []
        for c, ep, en, np_, nn in candidatos:
            dentro_env = not bool((mask & ~ep).any().item())
            contiene_nucleo = not bool((np_ & ~mask).any().item())
            if dentro_env and contiene_nucleo:
                pos_matches.append((c, int(ep.sum().item()),
                                     int(en.sum().item()),
                                     int(np_.sum().item())))

        if pos_matches:
            # Prefer the config with the largest kernel (most specific)
            pos_matches.sort(key=lambda x: (-x[3], x[1]))
            if (len(pos_matches) == 1
                or pos_matches[0][3] > pos_matches[1][3]
                or pos_matches[0][1] < pos_matches[1][1]):
                c, ps, ns, nps = pos_matches[0]
                return ResultadoIdentificacion(c, "positivo", ps, ns,
                                                int(mask.sum().item()),
                                                nps, -1)
            return ResultadoIdentificacion(None, "ambiguo",
                                            pos_matches[0][1], -1,
                                            int(mask.sum().item()),
                                            pos_matches[0][3], -1)

        # Negative: nucleo_neg subseteq mask subseteq envelope_neg
        # (and does not fit in env_pos)
        neg_matches = []
        for c, ep, en, np_, nn in candidatos:
            dentro_env = not bool((mask & ~en).any().item())
            contiene_nucleo = not bool((nn & ~mask).any().item())
            if dentro_env and contiene_nucleo:
                neg_matches.append((c, int(ep.sum().item()),
                                     int(en.sum().item()),
                                     int(nn.sum().item())))

        if neg_matches:
            # Prefer largest kernel, then smallest envelope
            neg_matches.sort(key=lambda x: (-x[3], x[2]))
            if (len(neg_matches) == 1
                or neg_matches[0][3] > neg_matches[1][3]
                or neg_matches[0][2] < neg_matches[1][2]):
                c, ps, ns, nns = neg_matches[0]
                return ResultadoIdentificacion(c, "negativo", ps, ns,
                                                int(mask.sum().item()),
                                                -1, nns)
            return ResultadoIdentificacion(None, "ambiguo",
                                            neg_matches[0][1],
                                            neg_matches[0][2],
                                            int(mask.sum().item()),
                                            -1, neg_matches[0][3])

        return ResultadoIdentificacion(None, "desconocido", 0, 0,
                                        int(mask.sum().item()))


# ------------------------------------------------------------------
if __name__ == "__main__":
    for dim in (2, 3):
        cat = CatalogoTopologias(dim)
        print(f"=== Signatures dim={dim} ===")
        for cfg, (edges, ep, en, np_, nn) in cat.firmas.items():
            print(f"  {cfg}: arcs={len(edges)}  "
                  f"env_pos={int(ep.sum().item()):2d}  "
                  f"nuc_pos={int(np_.sum().item()):2d}  "
                  f"env_neg={int(en.sum().item()):2d}  "
                  f"nuc_neg={int(nn.sum().item()):2d}")
