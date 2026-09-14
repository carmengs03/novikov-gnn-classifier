import torch
import random

def generar_grafo_h():
    """
    Configuration h) for dimension 2: complete loops; crossed edges only produce components in e1.
    Has 2 solution families to be Novikov.
    """
    es_caso_valido = random.choice([True, False])
    familia = random.choice([1, 2])

    if familia == 1:
        # Family 1 (based on c112 = 0)
        c111 = random.uniform(-5.0, 5.0)
        if abs(c111) < 0.1: c111 = 1.0  # c111 appears as denominator below

        c112 = 0.0

        c121 = random.uniform(-5.0, 5.0)
        if abs(c121) < 0.1: c121 = 1.0  # c121 != 0 required

        c211 = c121
        c222 = random.uniform(-5.0, 5.0)

        c221 = (c121 * (c121 - c222)) / c111

        if not es_caso_valido:
            # Near-miss family 1: small perturbation on the predicted c221.
            perturbacion = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            c221 += perturbacion

    else:
        # Family 2 (based on c221 = 0)
        c121 = random.uniform(-5.0, 5.0)
        if abs(c121) < 0.1: c121 = 1.0  # c121 != 0 required

        # Strict triple equality
        c211 = c121
        c222 = c121
        c221 = 0.0

        c111 = random.uniform(-5.0, 5.0)
        c112 = random.uniform(-5.0, 5.0)
        if abs(c111) < 0.1 and abs(c112) < 0.1:
            c111 = 1.0

        if not es_caso_valido:
            # Near-miss family 2: small perturbation on the triple equality.
            trampa = random.choice(['romper_c211', 'romper_c222'])
            ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])

            if trampa == 'romper_c211':
                c211 = c121 + ruido
            else:
                c222 = c121 + ruido

    pesos = [
        [c111, c112],
        [c121, 0.0 ],
        [c211, 0.0 ],
        [c221, c222]
    ]

    edge_attr = torch.tensor(pesos, dtype=torch.float)

    origenes = [0, 0, 1, 1]
    destinos = [0, 1, 0, 1]
    edge_index = torch.tensor([origenes, destinos], dtype=torch.long)

    return edge_index, edge_attr
