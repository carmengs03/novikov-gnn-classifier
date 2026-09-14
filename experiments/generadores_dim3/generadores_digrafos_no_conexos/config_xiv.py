import torch
import random


def generar_grafo_xiv(dim_algebra=3):
    """
    Configuration xiv): loop on e1, arrows e2<->e3 with components in e2 and e3.
    Topology: loop on e1. Arrows e2->e3 and e3->e2 with weights (c233, c323) and (c232, c322).
    Novikov: never. Forbidden structure. Pure class-0 generator.
    """
    c111 = random.uniform(-5.0, 5.0)
    c112 = random.uniform(-5.0, 5.0)
    c113 = random.uniform(-5.0, 5.0)
    if abs(c111) < 0.1 and abs(c112) < 0.1 and abs(c113) < 0.1:
        c111 = 1.0

    c232 = random.uniform(-5.0, 5.0)
    c233 = random.uniform(-5.0, 5.0)
    c322 = random.uniform(-5.0, 5.0)
    c323 = random.uniform(-5.0, 5.0)
    if abs(c232) < 0.1 and abs(c322) < 0.1:
        c232 = 1.0
    if abs(c233) < 0.1 and abs(c323) < 0.1:
        c233 = 1.0

    edge_index = torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [0.0,  c232, c233],
         [0.0,  c322, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
