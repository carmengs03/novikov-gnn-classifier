import torch
import random


def generar_grafo_viii(dim_algebra=3):
    """
    Configuration viii): loop on e3 and edge e2->e3 with weights (c233, c323). e1 isolated.
    Topology: loop on e3. Edge e2->e3. e1 isolated.
    Novikov: never. Forbidden structure. Pure class-0 generator.
    """
    c331 = random.uniform(-5.0, 5.0)
    c332 = random.uniform(-5.0, 5.0)
    c333 = random.uniform(-5.0, 5.0)
    if abs(c331) < 0.1 and abs(c332) < 0.1 and abs(c333) < 0.1:
        c331 = 1.0

    c233 = random.uniform(-5.0, 5.0)
    c323 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1 and abs(c323) < 0.1:
        c233 = 1.0

    edge_index = torch.tensor([[2, 1, 2], [2, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c331, c332, c333],
         [0.0,  0.0,  c233],
         [0.0,  0.0,  c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
