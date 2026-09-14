import torch
import random


def generar_grafo_xvi(dim_algebra=3):
    """
    Configuration xvi): loops on e1 and e2, arrows e2<->e3 with components in e2 and e3.
    Topology: loops on e1 and e2. Arrows e2->e3 and e3->e2.
    Novikov: c111 != 0, c112 = c113 = c221 = c232 = c233 = 0, c322 != 0, c323 != 0,
             c222 = -c323, c223 = -c323^2/c322.
    """
    es_caso_valido = random.choice([True, False])

    c111 = random.uniform(-5.0, 5.0)
    if abs(c111) < 0.1: c111 = 1.0
    c322 = random.uniform(-5.0, 5.0)
    if abs(c322) < 0.1: c322 = 1.0
    c323 = random.uniform(-5.0, 5.0)
    if abs(c323) < 0.1: c323 = 1.0

    c112 = c113 = c221 = c232 = c233 = 0.0
    c222 = -c323
    c223 = -(c323**2) / c322

    if not es_caso_valido:
        trampa = random.choice([1, 2, 3, 4, 5, 6, 7])
        ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        if trampa == 1:
            c112 = ruido
        elif trampa == 2:
            c113 = ruido
        elif trampa == 3:
            c221 = ruido
        elif trampa == 4:
            c232 = ruido
        elif trampa == 5:
            c233 = ruido
        elif trampa == 6:
            c222 += ruido
        else:
            c223 += ruido

    edge_index = torch.tensor([[0, 1, 1, 2], [0, 1, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [c221, c222, c223],
         [0.0,  c232, c233],
         [0.0,  c322, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
