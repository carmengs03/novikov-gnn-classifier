import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class CapaAlgebraica(MessagePassing):
    def __init__(self, dim_in, dim_aristas, dim_oculta):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + dim_aristas, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))


class RedNovikov(nn.Module):
    """MAX aggregation, MEAN pooling, 3 propagation layers with residual connections."""

    def __init__(self, dim_nodos=6, dim_aristas=2, dim_oculta=32, n_capas=3, residual=True):
        super().__init__()
        self.residual = residual

        # Input projection maps node features (dim_nodos) to the shared latent
        # space dim_oculta. Required so residual connections are valid
        # (all message-passing layers operate dim_oculta -> dim_oculta).
        self.proyeccion_entrada = nn.Linear(dim_nodos, dim_oculta)

        self.capas_mp = nn.ModuleList(
            [CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta) for _ in range(n_capas)]
        )

        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, datos):
        x, edge_index, edge_attr, batch = datos.x, datos.edge_index, datos.edge_attr, datos.batch

        x = F.relu(self.proyeccion_entrada(x))

        for capa in self.capas_mp:
            h = capa(x, edge_index, edge_attr)
            x = F.relu(x + h) if self.residual else F.relu(h)

        x_grafo = global_mean_pool(x, batch)
        return self.clasificador(x_grafo)
