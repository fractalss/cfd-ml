import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GraphSnapshotDataset(Dataset):
    """
    Dataset wrapper for a list of PyG Data objects.
    Ensures each graph has a `.y` field representing ground truth [x, y, z, field].
    Assumes normalization has already been applied in data_loader.
    """

    def __init__(self, graph_list):
        self.graph_list = self._validate_graphs(graph_list)

    def _validate_graphs(self, graph_list):
        processed = []
        for data in graph_list:
            assert isinstance(data, Data), "Each element must be a PyG Data object"
            assert hasattr(data, "x"), "Data missing input features 'x'"
            assert hasattr(data, "y"), "Data missing ground truth 'y'"
            assert data.x.shape[1] == 4, "Expected input x to have 4 dimensions: [x, y, z, field]"
            assert data.y.shape[1] == 4, "Expected ground truth y to have 4 dimensions: [x, y, z, field]"
            processed.append(data)
        return processed

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]
