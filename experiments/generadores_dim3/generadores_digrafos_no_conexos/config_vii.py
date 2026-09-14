import torch
import random


def generar_grafo_vii(dim_algebra=3):
    """
    Configuration vii): loop on e2 and edge e2->e3 with weights (c233, c323). e1 isolated.
    Topology: loop on e2. Edge e2->e3. e1 isolated.
    Novikov: c222 = c323.
    """
    es_caso_valido = random.choice([True, False])

    c221 = random.uniform(-5.0, 5.0)
    c223 = random.uniform(-5.0, 5.0)
    c323 = random.uniform(-5.0, 5.0)

    c233 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1: c233 = 1.0

    if es_caso_valido:
        c222 = c323
        # Ensure (c221, c222, c223) != (0, 0, 0).
        if abs(c221) < 0.1 and abs(c222) < 0.1 and abs(c223) < 0.1:
            c221 = 1.0
    else:
        ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        c222 = c323 + ruido
        if abs(c221) < 0.1 and abs(c222) < 0.1 and abs(c223) < 0.1:
            c221 = 1.0

    edge_index = torch.tensor([[1, 1, 2], [1, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c221, c222, c223],
         [0.0,  0.0,  c233],
         [0.0,  0.0,  c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
