import torch
import random

def generar_grafo_i():
    """
    Configuration i) for dimension 2: E1 loop and both crossed edges exist; E2 loop is strictly 0.
    Novikov condition: c121 = c122 = 0, c111 = -c212, c112 = -(c212^2)/c211.
    """
    es_caso_valido = random.choice([True, False])

    # c211 is a denominator and c212 is set equal to c111, so both must be nonzero
    # to preserve the existence constraints.
    c211 = random.uniform(-5.0, 5.0)
    if abs(c211) < 0.1: c211 = random.choice([-1.0, 1.0])

    c212 = random.uniform(-5.0, 5.0)
    if abs(c212) < 0.1: c212 = random.choice([-1.0, 1.0])

    if es_caso_valido:
        c121 = 0.0
        c122 = 0.0

        c111 = -c212
        c112 = -(c212**2) / c211

    else:
        trampa = random.choice(['romper_ceros', 'romper_ecuacion'])

        if trampa == 'romper_ceros':
            # Keep the quadratic equation and slightly perturb the edge
            # that Novikov requires to be zero.
            c111 = -c212
            c112 = -(c212**2) / c211
            c121 = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            c122 = random.uniform(0.005, 0.3) * random.choice([-1, 1])

        else:
            # Keep the null edge and slightly perturb the quadratic formula
            # for c112 that Novikov requires.
            c121 = 0.0
            c122 = 0.0
            c111 = -c212
            perturbacion = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            c112 = (-(c212**2) / c211) + perturbacion

    # Existence constraints of config i): (c111,c112) != (0,0), (c121,c211) != (0,0),
    # (c122,c212) != (0,0). Since c211 and c212 are forced nonzero above, all three
    # are satisfied automatically.

    pesos = [
        [c111, c112],
        [c121, c122],
        [c211, c212],
        [0.0,  0.0 ]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
