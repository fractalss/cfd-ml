import torch
from torch_geometric.nn import radius_graph, knn_graph

def build_radius_graph_with_attr(pos, batch=None, radius=0.1, loop=False):
    """
    Constructs a radius-based graph with edge attributes based on inter-particle distances.

    Args:
        pos (Tensor): [N, 3] particle positions
        batch (Tensor, optional): [N] batch indices for each node
        radius (float): connection radius
        loop (bool): whether to include self-loops

    Returns:
        edge_index (LongTensor): [2, E] edge indices
        edge_attr (FloatTensor): [E, 1] edge distances
    """
    edge_index = radius_graph(pos, r=radius, batch=batch, loop=loop)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]  # [E, 3]
    edge_attr = torch.norm(edge_vec, dim=1, keepdim=True)  # [E, 1]
    return edge_index, edge_attr

def build_knn_graph_with_attr(pos, batch=None, k=8):
    """
    Constructs a k-NN graph with edge attributes based on inter-particle distances.

    Args:
        pos (Tensor): [N, 3] particle positions
        batch (Tensor, optional): [N] batch indices for each node
        k (int): number of nearest neighbors

    Returns:
        edge_index (LongTensor): [2, E] edge indices
        edge_attr (FloatTensor): [E, 1] edge distances
    """
    edge_index = knn_graph(pos, k=k, batch=batch, loop=False)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]  # [E, 3]
    edge_attr = torch.norm(edge_vec, dim=1, keepdim=True)  # [E, 1]
    return edge_index, edge_attr