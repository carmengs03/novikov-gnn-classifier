import torch
import random

def generar_grafo_c():
    """
    Configuration c) for dimension 2: both loops (E1 and E2) exist; crossed edges are strictly 0.
    Novikov condition: c111 != 0, c222 != 0, c112 = c221 = 0.
    """
    es_caso_valido = random.choice([True, False])

    if es_caso_valido:
        c111 = random.uniform(-5.0, 5.0)
        c222 = random.uniform(-5.0, 5.0)

        if abs(c111) < 0.1: c111 = 1.0
        if abs(c222) < 0.1: c222 = 1.0

        c112 = 0.0
        c221 = 0.0

    else:
        c111 = random.uniform(-5.0, 5.0)
        c222 = random.uniform(-5.0, 5.0)
        if abs(c111) < 0.1: c111 = 1.0
        if abs(c222) < 0.1: c222 = 1.0

        c112 = 0.0
        c221 = 0.0

        # Near-miss close to the boundary: small perturbation of the coefficients
        # that Novikov requires to be strictly zero.
        trampa = random.choice([1, 2, 3])
        if trampa == 1:
            c112 = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        elif trampa == 2:
            c221 = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        else:
            c112 = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            c221 = random.uniform(0.005, 0.3) * random.choice([-1, 1])

    pesos = [
        [c111, c112],
        [0.0,  0.0 ],
        [0.0,  0.0 ],
        [c221, c222]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
