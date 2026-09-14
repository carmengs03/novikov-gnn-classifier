import torch
import random


def generar_grafo_1(dim_algebra=3):
    """
    Configuration 1): arrows e1->e2 and e2<-e3, components in e2.
    Topology: arrows e1->e2, e3->e2.
    Novikov: c122 != 0, c322 != 0, c212 = 0, c232 = 0.
    """
    es_caso_valido = random.choice([True, False])

    c122 = random.uniform(-5.0, 5.0)
    if abs(c122) < 0.1: c122 = 1.0
    c322 = random.uniform(-5.0, 5.0)
    if abs(c322) < 0.1: c322 = 1.0

    c212 = 0.0
    c232 = 0.0

    if not es_caso_valido:
        trampa = random.choice([1, 2])
        ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        if trampa == 1:
            c212 = ruido
        else:
            c232 = ruido

    edge_index = torch.tensor([[0, 1, 1, 2],
                                [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[0.0, c122, 0.0],
         [0.0, c212, 0.0],
         [0.0, c232, 0.0],
         [0.0, c322, 0.0]],
        dtype=torch.float
    )
    return edge_index, edge_attr
