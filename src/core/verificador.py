import torch

def tensor_a_matriz_C(edge_index, edge_attr, dim_algebra):
    C = torch.zeros((dim_algebra, dim_algebra, dim_algebra), dtype=torch.float)
    for k in range(edge_index.shape[1]):
        origen = edge_index[0, k]
        destino = edge_index[1, k]
        C[origen, destino, :] = edge_attr[k]
    return C

def es_novikov(edge_index, edge_attr, dim_algebra, tolerancia=1e-4):
    C = tensor_a_matriz_C(edge_index, edge_attr, dim_algebra)

    # IDENTITY 2: (x*y)*z = (x*z)*y
    term2_izq = torch.einsum('ijm,mkh->ijkh', C, C)
    term2_der = torch.einsum('ikm,mjh->ijkh', C, C)

    if not torch.allclose(term2_izq, term2_der, atol=tolerancia):
        return 0.0

    # IDENTITY 1: x*(y*z) - (x*y)*z = y*(x*z) - (y*x)*z
    term1_izq = torch.einsum('jkm,imh->ijkh', C, C)
    term1_der = torch.einsum('ikm,jmh->ijkh', C, C)
    term2_der_id1 = torch.einsum('jim,mkh->ijkh', C, C)

    izq = term1_izq - term2_izq
    der = term1_der - term2_der_id1

    if not torch.allclose(izq, der, atol=tolerancia):
        return 0.0

    return 1.0
