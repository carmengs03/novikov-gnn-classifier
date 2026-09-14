import torch
import random


def generar_grafo_xvii(dim_algebra=3):
    """
    Configuration xvii): loops on e2 and e3, arrows e2<->e3 with components in e2 and e3. e1 isolated.
    Topology: loops on e2 and e3. Arrows e2->e3 and e3->e2.
    Novikov:
      Family 1: c221 = c331 = 0, c332 != 0,
                c222 = (c232^2 + 2 c232 c322 - 2 c333 c322 - c333^2)/(4 c332),
                c223 = -c322 (c232 - c333)^2/(4 c332^2),
                c233 = -(c232 - c333)^2/(4 c332),
                c323 = (c232^2 - 2 c232 c322 + 2 c333 c322 - c333^2)/(4 c332).
      Family 2: c332 != 0, c232 = c322 != 0, c233 = c323 != 0,
                c222 = (c322^2 - c333 c322 + c323 c332)/c332,
                c223 = c323 c322/c332, c221 = c323 c331/c332.
      Family 3: c221 = c233 = c331 = c332 = 0, c323 != 0, c333 = c232 != 0, c232 != c322,
                c222 = c323 (c232 + c322)/(c232 - c322),
                c223 = -c323^2 c322/(c232 - c322).
    """
    es_caso_valido = random.choice([True, False])
    familia = random.choice([1, 2, 3])

    if familia == 1:
        c332 = random.uniform(-5.0, 5.0)
        if abs(c332) < 0.1: c332 = 1.0
        c333 = random.uniform(-5.0, 5.0)
        c232 = random.uniform(-5.0, 5.0)
        # c232 != c333 required so that (c233, c323) != (0, 0)
        if abs(c232 - c333) < 0.2: c232 = c333 + 1.0
        c322 = random.uniform(-5.0, 5.0)
        if abs(c232) < 0.1 and abs(c322) < 0.1: c232 = 1.0

        c221 = c331 = 0.0
        d = c232 - c333
        c222 = (c232**2 + 2*c232*c322 - 2*c333*c322 - c333**2) / (4*c332)
        c223 = -c322 * d**2 / (4*c332**2)
        c233 = -d**2 / (4*c332)
        c323 = (c232**2 - 2*c232*c322 + 2*c333*c322 - c333**2) / (4*c332)

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            if trampa == 1:
                c222 += ruido
            elif trampa == 2:
                c233 += ruido
            elif trampa == 3:
                c323 += ruido
            elif trampa == 4:
                c221 = abs(ruido)
            else:
                c331 = abs(ruido)

    elif familia == 2:
        p = random.uniform(-5.0, 5.0)      # c232 = c322 = p
        if abs(p) < 0.1: p = 1.0
        q = random.uniform(-5.0, 5.0)      # c233 = c323 = q
        if abs(q) < 0.1: q = 1.0
        c332 = random.uniform(-5.0, 5.0)
        if abs(c332) < 0.1: c332 = 1.0
        c333 = random.uniform(-5.0, 5.0)
        c331 = random.uniform(-5.0, 5.0)

        c232 = c322 = p
        c233 = c323 = q
        c222 = (p**2 - c333*p + q*c332) / c332
        c223 = q*p / c332
        c221 = q*c331 / c332

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            if trampa == 1:
                c232 += ruido
            elif trampa == 2:
                c233 += ruido
            elif trampa == 3:
                c222 += ruido
            else:
                c223 += ruido

    else:
        c232 = random.uniform(-5.0, 5.0)
        if abs(c232) < 0.1: c232 = 1.0
        c322 = random.uniform(-5.0, 5.0)
        if abs(c232 - c322) < 0.2: c322 = c232 + 1.0   # c232 != c322
        c323 = random.uniform(-5.0, 5.0)
        if abs(c323) < 0.1: c323 = 1.0

        c221 = c233 = c331 = c332 = 0.0
        c333 = c232
        d = c232 - c322
        c222 = c323 * (c232 + c322) / d
        c223 = -c323**2 * c322 / d**2

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            ruido = random.uniform(0.005, 0.3) * random.choice([-1, 1])
            if trampa == 1:
                c333 += ruido
            elif trampa == 2:
                c222 += ruido
            elif trampa == 3:
                c223 += ruido
            else:
                c221 = abs(ruido)

    edge_index = torch.tensor([[1, 2, 1, 2], [1, 2, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c221, c222, c223],
         [c331, c332, c333],
         [0.0,  c232, c233],
         [0.0,  c322, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
