import torch
import random


def generar_grafo_ii(dim_algebra=3):
    """
    Configuration ii): loop on e1, everything else zero.
    Topology: one loop on e1. e2 and e3 isolated.
    Novikov: always, for any (c111, c112, c113) != (0, 0, 0).
    """
    c111 = random.uniform(-5.0, 5.0)
    c112 = random.uniform(-5.0, 5.0)
    c113 = random.uniform(-5.0, 5.0)

    # Ensure the loop exists (otherwise this degenerates to config_i).
    if abs(c111) < 0.1 and abs(c112) < 0.1 and abs(c113) < 0.1:
        c111 = 1.0

    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_attr = torch.tensor([[c111, c112, c113]], dtype=torch.float)
    return edge_index, edge_attr
