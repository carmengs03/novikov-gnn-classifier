"""Exact Novikov verifier, plus two helpers (`residuos`,
`es_novikov_tensor`) used by the 04 scripts.
"""
import torch


def tensor_a_matriz_C(edge_index, edge_attr, dim_algebra):
    C = torch.zeros((dim_algebra, dim_algebra, dim_algebra), dtype=torch.float)
    for k in range(edge_index.shape[1]):
        origen = edge_index[0, k]
        destino = edge_index[1, k]
        C[origen, destino, :] = edge_attr[k]
    return C


def residuos(C):
    """Return the residual tensors R^(1) and R^(2) for a 3D tensor C.

    The expressions are exactly those used by `es_novikov()`: the two
    sides of identities (2) and (1) reformulated as contractions.
    """
    # Residual of IDENTITY 2: (x*y)*z = (x*z)*y
    R2 = (torch.einsum('ijm,mkh->ijkh', C, C)
          - torch.einsum('ikm,mjh->ijkh', C, C))
    # Residual of IDENTITY 1: x*(y*z) - (x*y)*z = y*(x*z) - (y*x)*z
    term_id2 = torch.einsum('ijm,mkh->ijkh', C, C)
    term1_izq = torch.einsum('jkm,imh->ijkh', C, C)
    term1_der = torch.einsum('ikm,jmh->ijkh', C, C)
    term2_der_id1 = torch.einsum('jim,mkh->ijkh', C, C)
    R1 = (term1_izq - term_id2) - (term1_der - term2_der_id1)
    return R1, R2


def es_novikov(edge_index, edge_attr, dim_algebra, tolerancia=1e-4):
    C = tensor_a_matriz_C(edge_index, edge_attr, dim_algebra)
    R1, R2 = residuos(C)
    if R2.abs().max().item() > tolerancia:
        return 0.0
    if R1.abs().max().item() > tolerancia:
        return 0.0
    return 1.0


def es_novikov_tensor(C, tolerancia=1e-4):
    """Verify directly on the tensor C (useful without edge_index)."""
    R1, R2 = residuos(C)
    return (R2.abs().max().item() <= tolerancia
            and R1.abs().max().item() <= tolerancia)
