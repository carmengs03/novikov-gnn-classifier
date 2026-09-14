import torch
import random

def generar_grafo_e():
    """
    Configuration e) for dimension 2: only the crossed edges exist; loops are strictly 0.
    Novikov condition: mathematically impossible. This factory only generates class 0.
    """
    c121 = random.uniform(-5.0, 5.0)
    c122 = random.uniform(-5.0, 5.0)
    c211 = random.uniform(-5.0, 5.0)
    c212 = random.uniform(-5.0, 5.0)

    # Existence constraint: (c122, c212) != (0, 0)
    if abs(c122) < 0.1 and abs(c212) < 0.1:
        c122 = 1.0

    # Existence constraint: (c121, c211) != (0, 0)
    if abs(c121) < 0.1 and abs(c211) < 0.1:
        c211 = 1.0

    pesos = [
        [0.0,  0.0 ],
        [c121, c122],
        [c211, c212],
        [0.0,  0.0 ]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
