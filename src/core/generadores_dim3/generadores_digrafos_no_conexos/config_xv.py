import torch
import random


def generar_grafo_xv(dim_algebra=3):
    """
    Configuration xv): loop on e2, arrows e2<->e3 with components in e2 and e3. e1 isolated.
    Topology: loop on e2. Arrows e2->e3 and e3->e2.
    Novikov: c221 = c232 = c233 = 0, c322 != 0, c323 != 0,
             c222 = -c323, c223 = -c323^2/c322.
    """
    es_caso_valido = random.choice([True, False])

    c323 = random.uniform(-5.0, 5.0)
    if abs(c323) < 0.1: c323 = 1.0
    c322 = random.uniform(-5.0, 5.0)
    if abs(c322) < 0.1: c322 = 1.0

    c221 = 0.0
    c232 = 0.0
    c233 = 0.0
    c222 = -c323
    c223 = -(c323**2) / c322

    if not es_caso_valido:
        trampa = random.choice([1, 2, 3, 4])
        ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        if trampa == 1:
            c222 += ruido
        elif trampa == 2:
            c223 += ruido
        elif trampa == 3:
            c221 = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        else:
            c232 = random.uniform(0.005, 0.3) * random.choice([-1, 1])

    edge_index = torch.tensor([[1, 1, 2], [1, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c221, c222, c223],
         [0.0,  c232, c233],
         [0.0,  c322, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
