import torch
import random

def generar_grafo_g():
    """
    Configuration g) for dimension 2: full E1 loop; crossed edges only produce components in e1;
    E2 loop does not exist.
    Novikov condition: mathematically impossible. This factory only generates class 0.
    """
    c111 = random.uniform(-5.0, 5.0)
    c112 = random.uniform(-5.0, 5.0)
    c121 = random.uniform(-5.0, 5.0)
    c211 = random.uniform(-5.0, 5.0)

    # Existence constraint: (c111, c112) != (0, 0)
    if abs(c111) < 0.1 and abs(c112) < 0.1:
        c111 = 1.0

    # Existence constraint: (c121, c211) != (0, 0)
    if abs(c121) < 0.1 and abs(c211) < 0.1:
        c121 = 1.0

    pesos = [
        [c111, c112],
        [c121, 0.0 ],
        [c211, 0.0 ],
        [0.0,  0.0 ]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
