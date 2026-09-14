import torch
import random


def generar_grafo_11(dim_algebra=3):
    """
    Configuration 11): pair {e1,e2} in dir. e1; pair {e2,e3} in dir. e2 and e3; pair {e1,e3} in dir. e3.
    Topology: arrows e2->e1, e3->e2, e2->e3, e1->e3.
    Novikov: forbidden structure. Pure class-0 generator.
    """
    c121 = random.uniform(-5.0, 5.0)
    if abs(c121) < 0.1: c121 = 1.0
    c211 = random.uniform(-5.0, 5.0)
    if abs(c211) < 0.1: c211 = 1.0
    c232 = random.uniform(-5.0, 5.0)
    if abs(c232) < 0.1: c232 = 1.0
    c233 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1: c233 = 1.0
    c322 = random.uniform(-5.0, 5.0)
    if abs(c322) < 0.1: c322 = 1.0
    c323 = random.uniform(-5.0, 5.0)
    if abs(c323) < 0.1: c323 = 1.0
    c133 = random.uniform(-5.0, 5.0)
    if abs(c133) < 0.1: c133 = 1.0
    c313 = random.uniform(-5.0, 5.0)
    if abs(c313) < 0.1: c313 = 1.0

    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c121, 0.0,  0.0 ],   # e1·e2
         [c211, 0.0,  0.0 ],   # e2·e1
         [0.0,  c232, c233],   # e2·e3
         [0.0,  c322, c323],   # e3·e2
         [0.0,  0.0,  c133],   # e1·e3
         [0.0,  0.0,  c313]],  # e3·e1
        dtype=torch.float
    )
    return edge_index, edge_attr
