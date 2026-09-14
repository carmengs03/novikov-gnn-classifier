import torch
import random


def generar_grafo_x(dim_algebra=3):
    """
    Configuration x): loops on e1 and e3, edge e2->e3 with weights (c233, c323).
    Topology: loops on e1 and e3. Edge e2->e3.
    Novikov: never. Forbidden structure. Pure class-0 generator.
    """
    c111 = random.uniform(-5.0, 5.0)
    c112 = random.uniform(-5.0, 5.0)
    c113 = random.uniform(-5.0, 5.0)
    if abs(c111) < 0.1 and abs(c112) < 0.1 and abs(c113) < 0.1:
        c111 = 1.0

    c331 = random.uniform(-5.0, 5.0)
    c332 = random.uniform(-5.0, 5.0)
    c333 = random.uniform(-5.0, 5.0)
    if abs(c331) < 0.1 and abs(c332) < 0.1 and abs(c333) < 0.1:
        c331 = 1.0

    c233 = random.uniform(-5.0, 5.0)
    c323 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1 and abs(c323) < 0.1:
        c233 = 1.0

    edge_index = torch.tensor([[0, 2, 1, 2], [0, 2, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [c331, c332, c333],
         [0.0,  0.0,  c233],
         [0.0,  0.0,  c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
