import torch
import random
import math


def generar_grafo_xviii(dim_algebra=3):
    """
    Configuration xviii): loops on e1, e2, e3 and arrows e2<->e3 with components in e2 and e3.
    Topology: loops on e1, e2, e3. Arrows e2->e3 and e3->e2.
    11 Novikov families.
    Note: families 3 and 7 require c233*c332 < 0 (discriminant is -c233*c332).
    """
    es_caso_valido = random.choice([True, False])
    familia = random.randint(1, 11)

    ruido = lambda: random.uniform(0.005, 0.3) * random.choice([-1, 1])
    rnd = lambda: random.uniform(-5.0, 5.0)
    rnd_nz = lambda: (lambda v: v if abs(v) >= 0.1 else 1.0)(rnd())

    if familia == 1:
        # c222 = c323, c232 = c333, c112 = c113 = c221 = c223 = c233 = c322 = c331 = c332 = 0
        c111 = rnd_nz()
        a = rnd_nz()   # c222 = c323
        b = rnd_nz()   # c232 = c333
        c112 = c113 = c221 = c223 = c233 = c322 = c331 = c332 = 0.0
        c222 = c323 = a
        c232 = c333 = b

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            if trampa == 1:
                c222 += ruido()
            elif trampa == 2:
                c232 += ruido()
            elif trampa == 3:
                c223 = abs(ruido())
            elif trampa == 4:
                c233 = abs(ruido())
            else:
                c322 = abs(ruido())

    elif familia == 2:
        # c333 = c232, c112 = c113 = c221 = c233 = c331 = c332 = 0
        # c222 = c323*(c322+c232)/(c232-c322), c223 = -c322*c323^2/(c232-c322)^2
        c111 = rnd_nz()
        c232 = rnd_nz()
        c322 = rnd()
        if abs(c232 - c322) < 0.3: c322 = c232 + 1.0
        c323 = rnd_nz()
        c333 = c232
        d = c232 - c322
        c222 = c323 * (c322 + c232) / d
        c223 = -c322 * c323**2 / d**2
        c112 = c113 = c221 = c233 = c331 = c332 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            if trampa == 1:
                c333 += ruido()
            elif trampa == 2:
                c222 += ruido()
            elif trampa == 3:
                c223 += ruido()
            elif trampa == 4:
                c233 = abs(ruido())
            else:
                c332 = abs(ruido())

    elif familia == 3:
        # c323 = 0, c112 = c113 = c221 = c331 = 0, c233 != 0, c332 != 0, c233*c332 < 0
        # c322 = c333 + sqrt(-c233*c332), c232 = 2*c322 - c333
        # c222 = 2*c322*(c322-c333)/c332, c223 = c233*c322/c332
        c111 = rnd_nz()
        c333 = rnd()
        c233 = rnd_nz()
        c332 = rnd_nz()
        if c233 * c332 > 0: c332 = -c332   # enforce c233*c332 < 0
        alpha = c333 + math.sqrt(-c233 * c332)
        c322 = alpha
        c232 = 2 * alpha - c333
        c222 = 2 * alpha * (alpha - c333) / c332
        c223 = c233 * alpha / c332
        c112 = c113 = c221 = c323 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            if trampa == 1:
                c222 += ruido()
            elif trampa == 2:
                c223 += ruido()
            elif trampa == 3:
                c323 = abs(ruido())
            else:
                c331 = abs(ruido())

    elif familia == 4:
        # c112 = c113 = c221 = c331 = 0, c332 != 0
        # c222 = (c232^2 + 2 c232 c322 - 2 c322 c333 - c333^2)/(4 c332)
        # c223 = -((c232 - c333)^2 c322)/(4 c332^2)
        # c233 = -(c232 - c333)^2/(4 c332)
        # c323 = (c232^2 - 2 c232 c322 + 2 c322 c333 - c333^2)/(4 c332)
        c111 = rnd_nz()
        c332 = rnd_nz()
        c333 = rnd()
        c232 = rnd()
        if abs(c232 - c333) < 0.2: c232 = c333 + 1.0
        c322 = rnd()
        c331 = rnd()
        d = c232 - c333
        c222 = (c232**2 + 2*c232*c322 - 2*c322*c333 - c333**2) / (4*c332)
        c223 = -(d**2 * c322) / (4*c332**2)
        c233 = -d**2 / (4*c332)
        c323 = (c232**2 - 2*c232*c322 + 2*c322*c333 - c333**2) / (4*c332)
        c112 = c113 = c221 = 0.0
        c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            if trampa == 1:
                c221 = abs(ruido())
            elif trampa == 2:
                c222 += ruido()
            elif trampa == 3:
                c233 += ruido()
            elif trampa == 4:
                c323 += ruido()
            else:
                c331 = abs(ruido())

    elif familia == 5:
        # c112 = c113 = c221 = c331 = 0, c232 = c322, c323 = c233, c332 != 0
        # c222 = (c233 c332 + c322^2 - c322 c333)/c332, c223 = c233 c322/c332
        c111 = rnd_nz()
        c332 = rnd_nz()
        c322 = rnd_nz()
        c233 = rnd()
        c333 = rnd()
        c232 = c322
        c323 = c233
        c222 = (c233*c332 + c322**2 - c322*c333) / c332
        c223 = c233 * c322 / c332
        c112 = c113 = c221 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            if trampa == 1:
                c232 += ruido()
            elif trampa == 2:
                c323 += ruido()
            elif trampa == 3:
                c222 += ruido()
            else:
                c223 += ruido()

    elif familia == 6:
        # c112 = c113 = c221 = c233 = c331 = c332 = 0, c333 = c232
        # c222 = c323*(c322+c232)/(c232-c322), c223 = -c322*c323^2/(c232-c322)^2
        c111 = rnd_nz()
        c232 = rnd_nz()
        c322 = rnd()
        if abs(c232 - c322) < 0.3: c322 = c232 + 1.0
        c323 = rnd_nz()
        c333 = c232
        d = c232 - c322
        c222 = c323 * (c322 + c232) / d
        c223 = -c322 * c323**2 / d**2
        c112 = c113 = c221 = c233 = c331 = c332 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            if trampa == 1:
                c333 += ruido()
            elif trampa == 2:
                c222 += ruido()
            elif trampa == 3:
                c223 += ruido()
            elif trampa == 4:
                c233 = abs(ruido())
            else:
                c332 = abs(ruido())

    elif familia == 7:
        # c323 = 0, c112 = c113 = c221 = c331 = 0, c233 != 0, c332 != 0, c233*c332 < 0
        # beta = c232 + sqrt(-c233*c332), c322 = beta, c333 = 2*beta - c232
        # c222 = -2*(beta*c232 - c232^2 - c233*c332)/c332, c223 = beta*c233/c332
        c111 = rnd_nz()
        c232 = rnd_nz()
        c233 = rnd_nz()
        c332 = rnd_nz()
        if c233 * c332 > 0: c332 = -c332   # enforce c233*c332 < 0
        beta = c232 + math.sqrt(-c233 * c332)
        c322 = beta
        c333 = 2 * beta - c232
        c222 = -2 * (beta * c232 - c232**2 - c233 * c332) / c332
        c223 = beta * c233 / c332
        c112 = c113 = c221 = c323 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            if trampa == 1:
                c222 += ruido()
            elif trampa == 2:
                c223 += ruido()
            elif trampa == 3:
                c323 = abs(ruido())
            else:
                c331 = abs(ruido())

    elif familia == 8:
        # c112 = c113 = c221 = c223 = c322 = c331 = 0, c222 = c323, c233 != c323, c232 != 0
        # c332 = -c232^2*c233/(c233-c323)^2, c333 = -(c233+c323)*c232/(c233-c323)
        c111 = rnd_nz()
        c232 = rnd_nz()
        c233 = rnd_nz()
        c323 = rnd_nz()
        if abs(c233 - c323) < 0.3: c323 = c233 + 1.0
        c222 = c323
        d = c233 - c323
        c332 = -c232**2 * c233 / d**2
        c333 = -(c233 + c323) * c232 / d
        c112 = c113 = c221 = c223 = c322 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            if trampa == 1:
                c222 += ruido()
            elif trampa == 2:
                c223 = abs(ruido())
            elif trampa == 3:
                c322 = abs(ruido())
            elif trampa == 4:
                c332 += ruido()
            else:
                c333 += ruido()

    elif familia == 9:
        # c112 = c113 = c221 = c331 = 0, c233 != 0, c232 != c333
        # c222 = (c223*(c232-c333)^2 - 2 c232 c233^2 - 2 c233^2 c333)/(2 c233 (c232-c333))
        # c323 = -(c223*(c232-c333)^2 + 2 c232 c233^2 + 2 c233^2 c333)/(2 c233 (c232-c333))
        # c322 = -(c232-c333)^2 c223/(4 c233^2)
        # c332 = -(c232-c333)^2/(4 c233)
        c111 = rnd_nz()
        c232 = rnd()
        c333 = rnd()
        if abs(c232 - c333) < 0.3: c333 = c232 + 1.0
        c233 = rnd_nz()
        c223 = rnd()
        d = c232 - c333
        c222 = (c223*d**2 - 2*c232*c233**2 - 2*c233**2*c333) / (2*c233*d)
        c323 = -(c223*d**2 + 2*c232*c233**2 + 2*c233**2*c333) / (2*c233*d)
        c322 = -c223 * d**2 / (4*c233**2)
        c332 = -d**2 / (4*c233)
        c112 = c113 = c221 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            if trampa == 1:
                c221 = abs(ruido())
            elif trampa == 2:
                c222 += ruido()
            elif trampa == 3:
                c323 += ruido()
            else:
                c322 += ruido()

    elif familia == 10:
        # c112 = c113 = c221 = c331 = 0, c233 = c323, c322 = c232, c223 != 0
        # c332 = c323*c232/c223, c222 = (c223 c232 - c223 c333 + c323^2)/c323
        c111 = rnd_nz()
        c232 = rnd()
        c323 = rnd_nz()
        c223 = rnd_nz()
        c333 = rnd()
        c233 = c323
        c322 = c232
        c332 = c323 * c232 / c223
        c222 = (c223*c232 - c223*c333 + c323**2) / c323
        c112 = c113 = c221 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4])
            if trampa == 1:
                c233 += ruido()
            elif trampa == 2:
                c322 += ruido()
            elif trampa == 3:
                c332 += ruido()
            else:
                c222 += ruido()

    else:  # familia == 11
        # c221 = c331 = 0, c112 = -c113*c333/c323, c232 = c322, c233 = c323, c323 != 0, c333 != 0
        # c222 = c323*c322/c333, c223 = c323^2/c333, c332 = c322*c333/c323
        c111 = rnd()
        c113 = rnd()
        c322 = rnd_nz()
        c323 = rnd_nz()
        c333 = rnd_nz()
        c232 = c322
        c233 = c323
        c112 = -c113 * c333 / c323
        c222 = c323 * c322 / c333
        c223 = c323**2 / c333
        c332 = c322 * c333 / c323
        if abs(c111) < 0.1 and abs(c112) < 0.1 and abs(c113) < 0.1:
            c111 = 1.0
        c221 = c331 = 0.0

        if not es_caso_valido:
            trampa = random.choice([1, 2, 3, 4, 5])
            if trampa == 1:
                c221 = abs(ruido())
            elif trampa == 2:
                c112 += ruido()
            elif trampa == 3:
                c232 += ruido()
            elif trampa == 4:
                c233 += ruido()
            else:
                c222 += ruido()

    edge_index = torch.tensor([[0, 1, 2, 1, 2],
                                [0, 1, 2, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[c111, c112, c113],
         [c221, c222, c223],
         [c331, c332, c333],
         [0.0,  c232, c233],
         [0.0,  c322, c323]],
        dtype=torch.float
    )
    return edge_index, edge_attr
