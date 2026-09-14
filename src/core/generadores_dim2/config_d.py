import torch
import random

def generar_grafo_d():
    """
    Configuration d) for dimension 2: no loops; crossed edges only produce components in e2
    (c121 = 0 and c211 = 0 strictly).
    Novikov condition: c122 != 0 and c212 = 0.
    """
    es_caso_valido = random.choice([True, False])

    if es_caso_valido:
        c122 = random.uniform(-5.0, 5.0)
        if abs(c122) < 0.1:
            c122 = 1.0

        c212 = 0.0

    else:
        c122 = random.uniform(-5.0, 5.0)
        if abs(c122) < 0.1:
            c122 = 1.0

        # Near-miss close to the boundary: small perturbation on c212
        # (Novikov requires c212 = 0).
        c212 = random.uniform(-0.3, 0.3)
        if abs(c212) < 0.005:
            c212 = random.choice([-0.05, 0.05])

    pesos = [
        [0.0, 0.0 ],
        [0.0, c122],
        [0.0, c212],
        [0.0, 0.0 ]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
