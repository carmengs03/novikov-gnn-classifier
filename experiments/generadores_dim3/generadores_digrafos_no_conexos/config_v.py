import torch
import random


def generar_grafo_v(dim_algebra=3):
    """
    Configuration v): one edge e2->e3 with weights (c233, c323). e1 isolated.
    Topology: one edge e2->e3. e1 isolated.
    Novikov: c233 != 0 and c323 = 0.
    """
    es_caso_valido = random.choice([True, False])

    c233 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1:
        c233 = 1.0

    if es_caso_valido:
        c323 = 0.0
    else:
        c323 = random.uniform(0.005, 0.3) * random.choice([-1, 1])

    edge_index = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[0.0, 0.0, c233],
         [0.0, 0.0, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
