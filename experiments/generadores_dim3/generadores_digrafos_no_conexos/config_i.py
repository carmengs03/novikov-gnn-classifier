import torch


def generar_grafo_i(dim_algebra=3):
    """
    Configuration i) of the NON-CONNECTED pseudo-digraphs with 3 vertices.
    Topology: three fully isolated vertices. No loops. No directed edges.
    """
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, dim_algebra), dtype=torch.float)
    return edge_index, edge_attr
