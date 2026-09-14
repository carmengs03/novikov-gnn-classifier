import torch
import random

def generar_grafo_j():
    """
    Configuration j) for dimension 2: dense graph. All edges and loops exist.
    Has 3 solution families.
    """
    es_caso_valido = random.choice([True, False])
    familia = random.choice([1, 2, 3])

    if familia == 1:
        # Family 1: c121 = c211
        c221 = random.uniform(-5.0, 5.0)
        if abs(c221) < 0.1: c221 = random.choice([-1.0, 1.0])  # denominator

        c211 = random.uniform(-5.0, 5.0)
        if abs(c211) < 0.1: c211 = random.choice([-1.0, 1.0])
        c121 = c211

        c122 = random.uniform(-5.0, 5.0)
        if abs(c122) < 0.1: c122 = random.choice([-1.0, 1.0])
        c212 = c122

        c222 = random.uniform(-5.0, 5.0)

        c111 = (c122 * c221 + c211**2 - c222 * c211) / c221
        c112 = (c122 * c211) / c221

    elif familia == 2:
        # Family 2: c221 = 0
        c221 = 0.0
        c122 = 0.0

        c121 = random.uniform(-5.0, 5.0)
        if abs(c121) < 0.1: c121 = random.choice([-1.0, 1.0])
        c222 = c121

        c212 = random.uniform(-5.0, 5.0)
        if abs(c212) < 0.1: c212 = random.choice([-1.0, 1.0])

        c211 = random.uniform(-5.0, 5.0)
        # c121 - c211 appears as denominator
        if abs(c121 - c211) < 0.1: c211 += 1.0

        c111 = (c212 * (c121 + c211)) / (c121 - c211)
        c112 = (- (c212**2) * c211) / ((c121 - c211)**2)

    else:
        # Family 3: c221 != 0 (general case)
        c221 = random.uniform(-5.0, 5.0)
        if abs(c221) < 0.1: c221 = random.choice([-1.0, 1.0])  # denominator

        c121 = random.uniform(-5.0, 5.0)
        c211 = random.uniform(-5.0, 5.0)
        c222 = random.uniform(-5.0, 5.0)

        # To avoid c122 and c212 both being 0 (general constraint), c121 != c222.
        if abs(c121 - c222) < 0.1: c222 += 1.0

        c122 = -((c121 - c222)**2) / (4 * c221)
        c112 = (c211 * c122) / c221
        c111 = (c121**2 + 2*c121*c211 - 2*c222*c211 - c222**2) / (4 * c221)
        c212 = (c121**2 - 2*c121*c211 + 2*c222*c211 - c222**2) / (4 * c221)

    if not es_caso_valido:
        # Near-miss: small perturbation on the E1 loop, enough to break the
        # equalities of the chosen family while keeping the algebra close
        # to a valid Novikov.
        c111 += random.uniform(0.005, 0.3) * random.choice([-1, 1])
        c112 += random.uniform(0.005, 0.3) * random.choice([-1, 1])

    pesos = [
        [c111, c112],
        [c121, c122],
        [c211, c212],
        [c221, c222]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
