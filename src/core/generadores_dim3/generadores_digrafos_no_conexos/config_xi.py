import torch
import random


def generar_grafo_xi(dim_algebra=3):
    """
    Configuration xi): loops on e2 and e3, edge e2->e3 with weights (c233, c323). e1 isolated.
    Topology: loops on e2 and e3. Edge e2->e3.
    Novikov:
      Family 1: c223 = 0, c222 = c233 = c323 != 0, c331 = c221*c332/c323.
      Family 2: c331 = c332 = 0, c333 != 0, c233 = c323 != 0,
                c222 = (-c223*c333 + c323^2)/c323.
    """
    es_caso_valido = random.choice([True, False])
    familia = random.choice([1, 2])

    if familia == 1:
        a = random.uniform(-5.0, 5.0)
        if abs(a) < 0.1: a = 1.0

        c221 = random.uniform(-5.0, 5.0)
        c332 = random.uniform(-5.0, 5.0)
        c333 = random.uniform(-5.0, 5.0)
        if abs(c332) < 0.1 and abs(c333) < 0.1:
            c332 = 1.0

        c223 = 0.0
        c222 = a
        c233 = a
        c323 = a
        c331 = (c221 * c332) / a

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3])
            ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            if trampa == 1:
                c331 += ruido
            elif trampa == 2:
                c222 += ruido
            else:
                c223 = random.uniform(0.005, 0.3) * random.choice([-1, 1])

    else:
        b = random.uniform(-5.0, 5.0)
        if abs(b) < 0.1: b = 1.0

        c333 = random.uniform(-5.0, 5.0)
        if abs(c333) < 0.1: c333 = 1.0

        c221 = random.uniform(-5.0, 5.0)
        c223 = random.uniform(-5.0, 5.0)
        c331 = 0.0
        c332 = 0.0
        c233 = b
        c323 = b
        c222 = (-c223 * c333 + b**2) / b
        if abs(c221) < 0.1 and abs(c222) < 0.1 and abs(c223) < 0.1:
            c221 = 1.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3])
            ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            if trampa == 1:
                c222 += ruido
            elif trampa == 2:
                c233 += ruido
            else:
                c331 = random.uniform(0.005, 0.3) * random.choice([-1, 1])

    edge_index = torch.tensor([[1, 2, 1, 2], [1, 2, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c221, c222, c223],
         [c331, c332, c333],
         [0.0,  0.0,  c233],
         [0.0,  0.0,  c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
