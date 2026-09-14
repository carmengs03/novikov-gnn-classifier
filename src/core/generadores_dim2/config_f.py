import torch
import random

def generar_grafo_f():
    """
    Configuration f) for dimension 2: full E1 loop; crossed edges only produce components in e2;
    E2 loop does not exist.
    Novikov condition: c111 = c212.
    Existence constraints: (c111, c112) != (0, 0) and (c122, c212) != (0, 0).
    """
    es_caso_valido = random.choice([True, False])

    c111 = random.uniform(-5.0, 5.0)
    c112 = random.uniform(-5.0, 5.0)
    c122 = random.uniform(-5.0, 5.0)

    if abs(c111) < 0.1 and abs(c112) < 0.1:
        c111 = 1.0

    if es_caso_valido:
        c212 = c111

        if abs(c122) < 0.1 and abs(c212) < 0.1:
            c122 = 1.0

    else:
        # Near-miss close to the boundary: small perturbation on the equality
        # c212 = c111 required by Novikov.
        ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        c212 = c111 + ruido

        if abs(c122) < 0.1 and abs(c212) < 0.1:
            c122 = 1.0

    pesos = [
        [c111, c112],
        [0.0,  c122],
        [0.0,  c212],
        [0.0,  0.0 ]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
