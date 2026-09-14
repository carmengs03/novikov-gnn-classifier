"""Orchestrator for the dim3 dataset.

Calls the 31 available configurations:
    - 18 disconnected pseudodigraphs (i - xviii)
    - 13 connected digraphs (1 - 13)

After the initial generation it applies a 50/50 balance between classes
(same scheme as main_generador.py for dim 2), discarding the excess of the
majority class. Output: dataset_novikov_dim3.pt
"""
import random
import torch
from torch_geometric.data import Data

from verificador import es_novikov

# --- 18 DISCONNECTED pseudodigraphs (i - xviii) ---
from generadores_dim3.generadores_digrafos_no_conexos.config_i     import generar_grafo_i
from generadores_dim3.generadores_digrafos_no_conexos.config_ii    import generar_grafo_ii
from generadores_dim3.generadores_digrafos_no_conexos.config_iii   import generar_grafo_iii
from generadores_dim3.generadores_digrafos_no_conexos.config_iv    import generar_grafo_iv
from generadores_dim3.generadores_digrafos_no_conexos.config_v     import generar_grafo_v
from generadores_dim3.generadores_digrafos_no_conexos.config_vi    import generar_grafo_vi
from generadores_dim3.generadores_digrafos_no_conexos.config_vii   import generar_grafo_vii
from generadores_dim3.generadores_digrafos_no_conexos.config_viii  import generar_grafo_viii
from generadores_dim3.generadores_digrafos_no_conexos.config_ix    import generar_grafo_ix
from generadores_dim3.generadores_digrafos_no_conexos.config_x     import generar_grafo_x
from generadores_dim3.generadores_digrafos_no_conexos.config_xi    import generar_grafo_xi
from generadores_dim3.generadores_digrafos_no_conexos.config_xii   import generar_grafo_xii
from generadores_dim3.generadores_digrafos_no_conexos.config_xiii  import generar_grafo_xiii
from generadores_dim3.generadores_digrafos_no_conexos.config_xiv   import generar_grafo_xiv
from generadores_dim3.generadores_digrafos_no_conexos.config_xv    import generar_grafo_xv
from generadores_dim3.generadores_digrafos_no_conexos.config_xvi   import generar_grafo_xvi
from generadores_dim3.generadores_digrafos_no_conexos.config_xvii  import generar_grafo_xvii
from generadores_dim3.generadores_digrafos_no_conexos.config_xviii import generar_grafo_xviii

# --- 13 CONNECTED digraphs (1 - 13) ---
from generadores_dim3.generadores_digrafos_conexos.config_1  import generar_grafo_1
from generadores_dim3.generadores_digrafos_conexos.config_2  import generar_grafo_2
from generadores_dim3.generadores_digrafos_conexos.config_3  import generar_grafo_3
from generadores_dim3.generadores_digrafos_conexos.config_4  import generar_grafo_4
from generadores_dim3.generadores_digrafos_conexos.config_5  import generar_grafo_5
from generadores_dim3.generadores_digrafos_conexos.config_6  import generar_grafo_6
from generadores_dim3.generadores_digrafos_conexos.config_7  import generar_grafo_7
from generadores_dim3.generadores_digrafos_conexos.config_8  import generar_grafo_8
from generadores_dim3.generadores_digrafos_conexos.config_9  import generar_grafo_9
from generadores_dim3.generadores_digrafos_conexos.config_10 import generar_grafo_10
from generadores_dim3.generadores_digrafos_conexos.config_11 import generar_grafo_11
from generadores_dim3.generadores_digrafos_conexos.config_12 import generar_grafo_12
from generadores_dim3.generadores_digrafos_conexos.config_13 import generar_grafo_13


DIM_ALGEBRA = 3

LISTA_CONFIGURACIONES = [
    # 18 disconnected
    generar_grafo_i, generar_grafo_ii, generar_grafo_iii, generar_grafo_iv,
    generar_grafo_v, generar_grafo_vi, generar_grafo_vii, generar_grafo_viii,
    generar_grafo_ix, generar_grafo_x, generar_grafo_xi, generar_grafo_xii,
    generar_grafo_xiii, generar_grafo_xiv, generar_grafo_xv, generar_grafo_xvi,
    generar_grafo_xvii, generar_grafo_xviii,
    # 13 connected
    generar_grafo_1, generar_grafo_2, generar_grafo_3, generar_grafo_4,
    generar_grafo_5, generar_grafo_6, generar_grafo_7, generar_grafo_8,
    generar_grafo_9, generar_grafo_10, generar_grafo_11, generar_grafo_12,
    generar_grafo_13,
]


def fijar_semilla(semilla=42):
    random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    print(f"Seed fixed at: {semilla}")


def generar_dataset_masivo(num_muestras_deseadas=300000, dim_algebra=DIM_ALGEBRA):
    """Generate num_muestras_deseadas graphs by sampling configs uniformly
    and label them with the verifier. Then balance 50/50 between classes
    by discarding the excess of the majority.

    With 300k of initial generation we expect approx 80-100k balanced
    (forbidden configs dominate in count and the balance discards
    near-miss).
    """
    print(f"Initial generation of {num_muestras_deseadas:,} graphs over "
          f"{len(LISTA_CONFIGURACIONES)} distinct configs...")

    grafos_clase_1 = []
    grafos_clase_0 = []
    x_base = torch.ones((dim_algebra, 1), dtype=torch.float)

    for i in range(num_muestras_deseadas):
        generador = random.choice(LISTA_CONFIGURACIONES)
        edge_index, edge_attr = generador()

        etiqueta = es_novikov(edge_index, edge_attr, dim_algebra)
        y = torch.tensor([etiqueta], dtype=torch.float)

        grafo = Data(x=x_base, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if etiqueta == 1.0:
            grafos_clase_1.append(grafo)
        else:
            grafos_clase_0.append(grafo)

        if (i + 1) % 25000 == 0:
            print(f"  generated {i + 1:,}  "
                  f"(N={len(grafos_clase_1):,}, nm={len(grafos_clase_0):,})")

    print("\n--- 50/50 balance ---")
    print(f"  class 1 (Novikov):   {len(grafos_clase_1):,}")
    print(f"  class 0 (near-miss): {len(grafos_clase_0):,}")

    n = min(len(grafos_clase_1), len(grafos_clase_0))
    grafos_clase_1 = random.sample(grafos_clase_1, n)
    grafos_clase_0 = random.sample(grafos_clase_0, n)
    dataset = grafos_clase_1 + grafos_clase_0
    random.shuffle(dataset)
    print(f"  balanced: {len(dataset):,} graphs ({n:,} per class)")
    return dataset


if __name__ == "__main__":
    import sys
    nombre = sys.argv[1] if len(sys.argv) > 1 else "dataset_novikov_dim3.pt"
    fijar_semilla(42)
    dataset = generar_dataset_masivo(num_muestras_deseadas=300_000)
    torch.save(dataset, nombre)
    print(f"\nSaved to {nombre}")
