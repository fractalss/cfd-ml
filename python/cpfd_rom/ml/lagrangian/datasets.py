import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GraphSnapshotDataset(Dataset):
    """
    Dataset wrapper for a list of PyG Data objects.

    Expected graph contract for raw-particle Lagrangian ROM:
      - data.x        : [N, 4]  normalized [x, y, z, field]
      - data.y        : [N, 4]  normalized reconstruction target
      - data.pos      : [N, 3]  physical xyz
      - data.params   : [1, P_aug]
      - data.time     : [1]
      - data.cloud_id : [N]

    Notes:
      - No cloud_id_base is required in the new raw-particle path.
      - Normalization is assumed to be handled in data_loader.py.
    """

    def __init__(self, graph_list):
        self.graph_list = self._validate_graphs(graph_list)

    def _validate_graphs(self, graph_list):
        processed = []

        for i, data in enumerate(graph_list):
            if not isinstance(data, Data):
                raise TypeError(
                    f"Graph {i}: expected torch_geometric.data.Data, got {type(data)}"
                )

            if not hasattr(data, "x"):
                raise AttributeError(f"Graph {i}: missing input features 'x'")
            if not hasattr(data, "y"):
                raise AttributeError(f"Graph {i}: missing ground truth 'y'")
            if not hasattr(data, "pos"):
                raise AttributeError(f"Graph {i}: missing physical positions 'pos'")
            if not hasattr(data, "params"):
                raise AttributeError(f"Graph {i}: missing graph-level params 'params'")
            if not hasattr(data, "time"):
                raise AttributeError(f"Graph {i}: missing snapshot time 'time'")
            if not hasattr(data, "cloud_id"):
                raise AttributeError(f"Graph {i}: missing particle identifier 'cloud_id'")

            if data.x.ndim != 2 or data.x.shape[1] != 4:
                raise ValueError(
                    f"Graph {i}: expected x shape [N,4] for [x,y,z,field], got {tuple(data.x.shape)}"
                )

            if data.y.ndim != 2 or data.y.shape[1] != 4:
                raise ValueError(
                    f"Graph {i}: expected y shape [N,4] for [x,y,z,field], got {tuple(data.y.shape)}"
                )

            if data.pos.ndim != 2 or data.pos.shape[1] != 3:
                raise ValueError(
                    f"Graph {i}: expected pos shape [N,3], got {tuple(data.pos.shape)}"
                )

            if data.x.shape[0] != data.y.shape[0] or data.x.shape[0] != data.pos.shape[0]:
                raise ValueError(
                    f"Graph {i}: x/y/pos must have same number of particles, got "
                    f"x={data.x.shape[0]}, y={data.y.shape[0]}, pos={data.pos.shape[0]}"
                )

            if data.params.ndim != 2 or data.params.shape[0] != 1:
                raise ValueError(
                    f"Graph {i}: expected params shape [1,P_aug], got {tuple(data.params.shape)}"
                )

            if data.cloud_id.ndim != 1:
                raise ValueError(
                    f"Graph {i}: expected cloud_id shape [N], got {tuple(data.cloud_id.shape)}"
                )

            if data.cloud_id.shape[0] != data.x.shape[0]:
                raise ValueError(
                    f"Graph {i}: cloud_id length must match particle count N={data.x.shape[0]}, "
                    f"got {data.cloud_id.shape[0]}"
                )

            processed.append(data)

        return processed

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]