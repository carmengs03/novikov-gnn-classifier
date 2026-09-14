import torch
from torch_geometric.data import Data

def inyectar_caracteristicas(grafo, dim_algebra=2):
    """
    Replace the blind [1.0] vector with a 6-feature node vector:
    [in_degree, out_degree, in_mass, out_mass, self_loop_mass, centralidad]
    """
    in_degree = torch.zeros(dim_algebra)
    out_degree = torch.zeros(dim_algebra)
    in_mass = torch.zeros(dim_algebra)
    out_mass = torch.zeros(dim_algebra)
    self_loop_mass = torch.zeros(dim_algebra)
    centralidad = torch.zeros(dim_algebra)

    origenes = grafo.edge_index[0]
    destinos = grafo.edge_index[1]
    pesos = grafo.edge_attr

    # Masses, degrees and self-loops
    for i in range(len(origenes)):
        origen = origenes[i]
        destino = destinos[i]
        masa_arista = torch.sum(torch.abs(pesos[i])).item()

        if masa_arista > 1e-5:
            out_degree[origen] += 1
            out_mass[origen] += masa_arista
            in_degree[destino] += 1
            in_mass[destino] += masa_arista

            # Detect self-products (e.g. e1 * e1)
            if origen == destino:
                self_loop_mass[origen] += masa_arista

    # One-hop influence centrality: "I receive prestige proportional to the
    # total outgoing mass of the nodes pointing at me."
    for i in range(len(origenes)):
        origen = origenes[i]
        destino = destinos[i]
        masa_arista = torch.sum(torch.abs(pesos[i])).item()

        if masa_arista > 1e-5:
            centralidad[destino] += out_mass[origen] * masa_arista

    # Normalise centrality to [0, 1] so the network does not saturate
    suma_cent = torch.sum(centralidad)
    if suma_cent > 1e-5:
        centralidad = centralidad / suma_cent

    # Stack the 6 features
    nuevo_x = torch.stack([
        in_degree, out_degree, in_mass, out_mass, self_loop_mass, centralidad
    ], dim=1)

    return Data(x=nuevo_x, edge_index=grafo.edge_index, edge_attr=grafo.edge_attr, y=grafo.y)

if __name__ == "__main__":
    nombre_archivo = "dataset_novikov_dim2.pt"
    print(f"Loading {nombre_archivo}...")

    dataset_original = torch.load(nombre_archivo, weights_only=False)
    dataset_enriquecido = []

    print("Injecting 6 topological features per node...")
    for grafo in dataset_original:
        grafo_inteligente = inyectar_caracteristicas(grafo)
        dataset_enriquecido.append(grafo_inteligente)

    nombre_salida = "dataset_enriquecido_dim2.pt"
    torch.save(dataset_enriquecido, nombre_salida)
    print(f"Done. Saved as {nombre_salida}")
