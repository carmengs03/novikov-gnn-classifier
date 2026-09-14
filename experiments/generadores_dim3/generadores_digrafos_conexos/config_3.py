import torch
import random


def generar_grafo_3(dim_algebra=3):
    """
    Configuration 3): arrows e1<-e2 with components in e1, arrows e2->e3 with components in e3.
    Topology: arrows e2->e1, e2->e3.
    Novikov: c211 != 0, c233 != 0, c121 = 0, c323 = 0.
    """
    es_caso_valido = random.choice([True, False])

    c211 = random.uniform(-5.0, 5.0)
    if abs(c211) < 0.1: c211 = 1.0
    c233 = random.uniform(-5.0, 5.0)
    if abs(c233) < 0.1: c233 = 1.0

    c121 = 0.0
    c323 = 0.0

    if not es_caso_valido:
        trampa = random.choice([1, 2])
        ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        if trampa == 1:
            c121 = ruido
        else:
            c323 = ruido

    edge_index = torch.tensor([[0, 1, 1, 2],
                                [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c121, 0.0, 0.0],
         [c211, 0.0, 0.0],
         [0.0,  0.0, c233],
         [0.0,  0.0, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
