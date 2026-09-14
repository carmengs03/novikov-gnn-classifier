import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class CapaAlgebraica(MessagePassing):
    def __init__(self, dim_nodos_entrada, dim_aristas, dim_oculta):
        # aggr='add' sums the incoming messages
        super(CapaAlgebraica, self).__init__(aggr='add')

        self.transformador_mensajes = nn.Sequential(
            nn.Linear(dim_nodos_entrada + dim_aristas, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: sender node features; edge_attr: structure constants (c_ijk)
        info_combinada = torch.cat([x_j, edge_attr], dim=1)
        mensaje_transformado = self.transformador_mensajes(info_combinada)
        return mensaje_transformado


class RedNovikov(nn.Module):
    def __init__(self, dim_nodos=1, dim_aristas=2, dim_oculta=32):
        super(RedNovikov, self).__init__()

        # Two message-passing hops. The first reads the raw node features
        # (dim=1); the second reads the already-enriched hidden state (dim=32).
        self.capa_mensaje_1 = CapaAlgebraica(dim_nodos, dim_aristas, dim_oculta)
        self.capa_mensaje_2 = CapaAlgebraica(dim_oculta, dim_aristas, dim_oculta)

        self.clasificador = nn.Sequential(
            nn.Linear(dim_oculta, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, datos):
        x, edge_index, edge_attr, batch = datos.x, datos.edge_index, datos.edge_attr, datos.batch

        x = self.capa_mensaje_1(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Second hop covers identities involving products of three constants
        x = self.capa_mensaje_2(x, edge_index, edge_attr)
        x = torch.relu(x)

        x_grafo = global_mean_pool(x, batch)
        prediccion = self.clasificador(x_grafo)

        return prediccion
