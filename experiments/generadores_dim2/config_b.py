import torch
import random

def generar_grafo_b():
    """Configuration b) for dimension 2: only the loop on E1 exists; the rest is 0."""
    es_caso_valido = random.choice([True, False])

    if es_caso_valido:
        c111 = random.uniform(-5.0, 5.0)
        c112 = random.uniform(-5.0, 5.0)
        if abs(c111) < 0.1 and abs(c112) < 0.1:
            c111 = 1.0
    else:
        c111 = 0.0
        c112 = 0.0

    pesos = [
        [c111, c112],
        [0.0,  0.0],
        [0.0,  0.0],
        [0.0,  0.0]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
