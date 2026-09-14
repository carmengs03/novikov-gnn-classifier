import torch
import random


def generar_grafo_8(dim_algebra=3):
    """
    Configuration 8): three pairs, components in e2 for {e1,e2} and {e2,e3}, in e3 for {e1,e3}.
    Topology: arrows e1->e2, e3->e2, e1->e3.
    Novikov: forbidden structure. Pure class-0 generator.
    """
    c122 = random.uniform(-5.0, 5.0)
    if abs(c122) < 0.1: c122 = 1.0
    c212 = random.uniform(-5.0, 5.0)
    if abs(c212) < 0.1: c212 = 1.0
    c232 = random.uniform(-5.0, 5.0)
    if abs(c232) < 0.1: c232 = 1.0
    c322 = random.uniform(-5.0, 5.0)
    if abs(c322) < 0.1: c322 = 1.0
    c133 = random.uniform(-5.0, 5.0)
    if abs(c133) < 0.1: c133 = 1.0
    c313 = random.uniform(-5.0, 5.0)
    if abs(c313) < 0.1: c313 = 1.0

    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[0.0, c122, 0.0 ],   # e1·e2
         [0.0, c212, 0.0 ],   # e2·e1
         [0.0, c232, 0.0 ],   # e2·e3
         [0.0, c322, 0.0 ],   # e3·e2
         [0.0, 0.0,  c133],   # e1·e3
         [0.0, 0.0,  c313]],  # e3·e1
        dtype=torch.float
    )
    return edge_index, edge_attr
