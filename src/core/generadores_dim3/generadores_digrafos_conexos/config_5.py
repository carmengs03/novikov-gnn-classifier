import torch
import random


def generar_grafo_5(dim_algebra=3):
    """
    Configuration 5): arrows e1<->e2 with components in e1 and e2, arrows e2->e3 with components in e3.
    Topology: arrows e1->e2, e2->e1, e2->e3.
    Novikov: forbidden structure. Pure class-0 generator.
    """
    c121 = random.uniform(-5.0, 5.0)
    if abs(c121) < 0.1: c121 = 1.0
    c122 = random.uniform(-5.0, 5.0)
    if abs(c122) < 0.1: c122 = 1.0
    c211 = random.uniform(-5.0, 5.0)
    if abs(c211) < 0.1: c211 = 1.0
    c212 = random.uniform(-5.0, 5.0)
    if abs(c212) < 0.1: c212 = 1.0
    c233 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1: c233 = 1.0
    c323 = random.uniform(-5.0, 5.0)
    if abs(c323) < 0.1: c323 = 1.0

    edge_index = torch.tensor([[0, 1, 1, 2],
                                [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c121, c122, 0.0],
         [c211, c212, 0.0],
         [0.0,  0.0,  c233],
         [0.0,  0.0,  c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
