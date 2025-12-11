from torch.utils.data import Dataset

class GraphSnapshotDataset(Dataset):
    """
    Dataset wrapper for a list of PyG Data objects.
    Each item is a single torch_geometric.data.Data graph.
    Assumes any parameters are already attached to the Data object (e.g., data.params).
    """
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]
