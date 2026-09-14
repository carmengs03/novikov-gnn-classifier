import torch
import random


def generar_grafo_vi(dim_algebra=3):
    """
    Configuration vi): loop on e1 and edge e2->e3 with weights (c233, c323).
    Topology: loop on e1. Edge e2->e3. e2 and e3 isolated from each other except for that edge.
    Novikov: c111 != 0, c112 = c113 = c323 = 0, c233 != 0.
    """
    es_caso_valido = random.choice([True, False])

    c111 = random.uniform(-5.0, 5.0)
    if abs(c111) < 0.1: c111 = 1.0

    c233 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1: c233 = 1.0

    if es_caso_valido:
        c112 = c113 = c323 = 0.0
    else:
        opciones = ['c112', 'c113', 'c323']
        contaminados = random.sample(opciones, k=random.randint(1, 3))
        vals = {c: 0.0 for c in opciones}
        for c in contaminados:
            vals[c] = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        c112, c113, c323 = vals['c112'], vals['c113'], vals['c323']

    edge_index = torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [0.0,  0.0,  c233],
         [0.0,  0.0,  c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
