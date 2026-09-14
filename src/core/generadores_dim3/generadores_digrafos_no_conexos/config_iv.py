import torch
import random


def generar_grafo_iv(dim_algebra=3):
    """
    Configuration iv): loops on e1, e2 and e3, everything else zero.
    Topology: three independent loops. No crossed edges.
    Novikov: c112 = c113 = c221 = c223 = c331 = c332 = 0, c111 != 0, c222 != 0, c333 != 0.
    """
    es_caso_valido = random.choice([True, False])

    c111 = random.uniform(-5.0, 5.0)
    c222 = random.uniform(-5.0, 5.0)
    c333 = random.uniform(-5.0, 5.0)

    if abs(c111) < 0.1: c111 = 1.0
    if abs(c222) < 0.1: c222 = 1.0
    if abs(c333) < 0.1: c333 = 1.0

    if es_caso_valido:
        c112 = c113 = c221 = c223 = c331 = c332 = 0.0
    else:
        # Contaminate one or several of the coefficients that must be zero.
        ceros = ['c112', 'c113', 'c221', 'c223', 'c331', 'c332']
        contaminados = random.sample(ceros, k=random.randint(1, 3))
        vals = {c: 0.0 for c in ceros}
        for c in contaminados:
            vals[c] = random.uniform(0.005, 0.3) * random.choice([-1, 1])
        c112, c113 = vals['c112'], vals['c113']
        c221, c223 = vals['c221'], vals['c223']
        c331, c332 = vals['c331'], vals['c332']

    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [c221, c222, c223],
         [c331, c332, c333]],
        dtype=torch.float
    )
    return edge_index, edge_attr
