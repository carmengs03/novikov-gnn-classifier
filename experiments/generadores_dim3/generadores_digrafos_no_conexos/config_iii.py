import torch
import random


def generar_grafo_iii(dim_algebra=3):
    """
    Configuration iii): loops on e1 and e3, everything else zero.
    Topology: loops on e1 and e3. e2 isolated.
    Novikov: c113 = 0 and c331 = 0.
    """
    es_caso_valido = random.choice([True, False])

    c111 = random.uniform(-5.0, 5.0)
    c112 = random.uniform(-5.0, 5.0)
    c332 = random.uniform(-5.0, 5.0)
    c333 = random.uniform(-5.0, 5.0)

    if es_caso_valido:
        c113 = 0.0
        c331 = 0.0
        # Ensure both loops exist.
        if abs(c111) < 0.1 and abs(c112) < 0.1:
            c111 = 1.0
        if abs(c332) < 0.1 and abs(c333) < 0.1:
            c332 = 1.0
    else:
        trampa = random.choice([1, 2, 3])
        c113 = random.uniform(0.005, 0.3) * random.choice([-1, 1]) if trampa in [1, 3] else 0.0
        c331 = random.uniform(0.005, 0.3) * random.choice([-1, 1]) if trampa in [2, 3] else 0.0
        if abs(c111) < 0.1 and abs(c112) < 0.1 and abs(c113) < 0.1:
            c111 = 1.0
        if abs(c331) < 0.1 and abs(c332) < 0.1 and abs(c333) < 0.1:
            c332 = 1.0

    edge_index = torch.tensor([[0, 2], [0, 2]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [c331, c332, c333]],
        dtype=torch.float
    )
    return edge_index, edge_attr
