import torch
import random


def generar_grafo_xiii(dim_algebra=3):
    """
    Configuration xiii): arrows e2<->e3 with components in e2 and e3. e1 isolated.
    Topology: arrows e2->e3 and e3->e2 with weights (c233, c323) and (c232, c322).
    Novikov: never. Forbidden structure. Pure class-0 generator.
    """
    c232 = random.uniform(-5.0, 5.0)
    c233 = random.uniform(-5.0, 5.0)
    c322 = random.uniform(-5.0, 5.0)
    c323 = random.uniform(-5.0, 5.0)

    if abs(c232) < 0.1 and abs(c322) < 0.1:
        c232 = 1.0
    if abs(c233) < 0.1 and abs(c323) < 0.1:
        c233 = 1.0

    edge_index = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[0.0, c232, c233],
         [0.0, c322, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
