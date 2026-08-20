# cpfd_rom/ml/lagrangian/model_pointnet_gnn.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool

from cpfd_rom.ml.lagrangian.mlp_encoder import _make_mlp, EdgeConvBlock


class FourierTimeEncoding(nn.Module):
    def __init__(
        self,
        num_bands: int = 8,
        max_freq: float = 10.0,
        include_raw_time: bool = True,
    ):
        super().__init__()

        if num_bands < 0:
            raise ValueError(f"num_bands must be >= 0, got {num_bands}")
        if num_bands > 0 and max_freq <= 0.0:
            raise ValueError(f"max_freq must be > 0 when num_bands > 0, got {max_freq}")

        self.num_bands = int(num_bands)
        self.max_freq = float(max_freq)
        self.include_raw_time = bool(include_raw_time)

        if self.num_bands > 0:
            freqs = torch.logspace(
                start=0.0,
                end=torch.log10(torch.tensor(self.max_freq)).item(),
                steps=self.num_bands,
            )
        else:
            freqs = torch.empty(0)

        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        dim = 2 * self.num_bands
        if self.include_raw_time:
            dim += 1
        return dim

    def _normalize_time_shape(
        self,
        time: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if time is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)

        if not torch.is_tensor(time):
            time = torch.tensor(time, device=device, dtype=dtype)
        else:
            time = time.to(device=device, dtype=dtype)

        if time.dim() == 0:
            time = time.view(1, 1)
        elif time.dim() == 1:
            time = time.view(-1, 1)
        elif time.dim() == 2:
            if time.shape[1] != 1:
                raise ValueError(f"time must have shape [B,1] when 2D, got {tuple(time.shape)}")
        else:
            raise ValueError(f"time must be scalar, [B], or [B,1], got {tuple(time.shape)}")

        if time.shape[0] == 1 and batch_size > 1:
            time = time.expand(batch_size, 1)

        if time.shape[0] != batch_size:
            raise ValueError(
                f"time batch mismatch: got {time.shape[0]} graph times for batch_size={batch_size}"
            )

        return time

    def forward(
        self,
        time: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = self._normalize_time_shape(time, batch_size, device, dtype)

        feats = []
        if self.include_raw_time:
            feats.append(t)

        if self.num_bands > 0:
            freqs = self.freqs.to(device=device, dtype=dtype).view(1, -1)
            angles = 2.0 * torch.pi * t * freqs
            feats.append(torch.sin(angles))
            feats.append(torch.cos(angles))

        if not feats:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)

        return torch.cat(feats, dim=1)


class PointNetGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        time_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_gnn_layers: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        enc_in = in_dim + time_dim

        self.encoder_mlp = _make_mlp([enc_in, 64, 128, hidden_dim], use_bn=use_bn)

        self.gnn_layers = nn.ModuleList(
            [EdgeConvBlock(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )

        self.post_gnn = _make_mlp([hidden_dim, hidden_dim, hidden_dim], use_bn=use_bn)

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        if node_time_feat.shape[0] != x.shape[0]:
            raise ValueError(
                f"node_time_feat row mismatch: {node_time_feat.shape[0]} vs {x.shape[0]}"
            )

        h = torch.cat([x, node_time_feat], dim=1)
        h = self.encoder_mlp(h)

        for gnn in self.gnn_layers:
            h = h + gnn(h, edge_index)

        h = self.post_gnn(h)
        pooled = global_max_pool(h, batch)
        graph_latent = self.to_latent(pooled)
        return graph_latent


class TemplateBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()

        self.template_encoder = _make_mlp([in_dim, 64, 128, hidden_dim], use_bn=use_bn)
        self.template_gnn_layers = nn.ModuleList(
            [EdgeConvBlock(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )

    def forward(
        self,
        template_x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.template_encoder(template_x)
        for gnn in self.template_gnn_layers:
            h = h + gnn(h, edge_index)
        return h


class ConditionalDecoder(nn.Module):
    """
    Predicts raw decoder output:
      [dx, dy, dz, field_raw, ...]
    """
    def __init__(
        self,
        template_hidden_dim: int,
        latent_dim: int,
        cond_dim: int,
        out_dim: int = 4,
        use_bn: bool = True,
    ):
        super().__init__()

        if out_dim < 4:
            raise ValueError(f"Expected out_dim >= 4, got {out_dim}")

        self.decoder_mlp = _make_mlp(
            [template_hidden_dim + latent_dim + cond_dim, 128, 64, out_dim],
            use_bn=use_bn,
        )

    def forward(
        self,
        template_feat: torch.Tensor,
        fused_latent: torch.Tensor,
        batch: torch.Tensor,
        graph_cond: torch.Tensor,
    ) -> torch.Tensor:
        fused_broadcast = fused_latent[batch]
        cond_broadcast = graph_cond[batch]
        dec_in = torch.cat([template_feat, fused_broadcast, cond_broadcast], dim=1)
        return self.decoder_mlp(dec_in)


class PointNetGNNAutoencoder(nn.Module):
    """
    Contract:
      - forward(x, edge_index, batch, params, time=None) -> (recon_x, graph_latent)
      - decode(latent_z, template_x, edge_index, batch, params, time=None) -> recon_x

    Behavior:
      - encoder learns PRE-fuse graph_latent
      - decoder predicts [dx, dy, dz, field_raw]
      - final output is [x, y, z, field]
    """
    def __init__(
        self,
        in_dim: int = 4,
        param_dim: int = 0,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        out_dim: int = 4,
        num_gnn_layers: int = 3,
        num_time_bands: int = 8,
        max_time_freq: float = 10.0,
        include_raw_time: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()

        if in_dim < 4:
            raise ValueError(f"Expected in_dim >= 4 for [x,y,z,field], got {in_dim}")
        if out_dim < 4:
            raise ValueError(f"Expected out_dim >= 4 for [x,y,z,field], got {out_dim}")

        self.in_dim = in_dim
        self.param_dim = param_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.time_encoder = FourierTimeEncoding(
            num_bands=num_time_bands,
            max_freq=max_time_freq,
            include_raw_time=include_raw_time,
        )
        self.time_dim = self.time_encoder.out_dim
        self.cond_dim = self.param_dim + self.time_dim

        self.encoder = PointNetGNNEncoder(
            in_dim=in_dim,
            time_dim=self.time_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_gnn_layers=num_gnn_layers,
            use_bn=use_bn,
        )

        self.template_branch = TemplateBranch(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            use_bn=use_bn,
        )

        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + self.cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

        self.decoder = ConditionalDecoder(
            template_hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            cond_dim=self.cond_dim,
            out_dim=out_dim,
            use_bn=use_bn,
        )

    def _num_graphs_from_batch(self, batch: torch.Tensor) -> int:
        if batch.numel() == 0:
            return 0
        return int(batch.max().item()) + 1

    def _normalize_params(
        self,
        params: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.param_dim == 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)

        if params is None:
            raise ValueError(
                f"params cannot be None because model was initialized with param_dim={self.param_dim}"
            )

        if not torch.is_tensor(params):
            params = torch.tensor(params, device=device, dtype=dtype)
        else:
            params = params.to(device=device, dtype=dtype)

        if params.dim() == 1:
            if params.shape[0] != self.param_dim:
                raise ValueError(
                    f"1D params must have length {self.param_dim}, got {tuple(params.shape)}"
                )
            params = params.view(1, self.param_dim)
        elif params.dim() == 2:
            if params.shape[1] != self.param_dim:
                raise ValueError(
                    f"params second dim must equal {self.param_dim}, got {tuple(params.shape)}"
                )
        else:
            raise ValueError(f"params must have shape [P] or [B,P], got {tuple(params.shape)}")

        if params.shape[0] == 1 and batch_size > 1:
            params = params.expand(batch_size, self.param_dim)

        if params.shape[0] != batch_size:
            raise ValueError(
                f"params batch mismatch: got {params.shape[0]} rows for batch_size={batch_size}"
            )

        return params

    def _build_graph_condition(
        self,
        batch: torch.Tensor,
        params: Optional[torch.Tensor],
        time: Optional[torch.Tensor],
        ref_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._num_graphs_from_batch(batch)
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        graph_params = self._normalize_params(
            params=params,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        graph_time_feat = self.time_encoder(
            time=time,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        node_time_feat = graph_time_feat[batch]
        graph_cond = torch.cat([graph_params, graph_time_feat], dim=1)
        return graph_cond, node_time_feat

    def _compose_output_from_template(
        self,
        raw_out: torch.Tensor,
        template_x: torch.Tensor,
    ) -> torch.Tensor:
        if raw_out.shape[-1] < 4:
            return raw_out

        template_xyz = template_x[..., :3]
        delta_xyz = raw_out[..., :3]
        pred_xyz = template_xyz + delta_xyz

        field_raw = raw_out[..., 3:4]
        pred_field = torch.sigmoid(field_raw)

        if raw_out.shape[-1] == 4:
            return torch.cat([pred_xyz, pred_field], dim=-1)

        rest = raw_out[..., 4:]
        return torch.cat([pred_xyz, pred_field, rest], dim=-1)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = self._num_graphs_from_batch(batch)
        node_time_feat = self.time_encoder(
            time=time,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )[batch]
        return self.encoder(x, edge_index, batch, node_time_feat)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        params: Optional[torch.Tensor],
        time: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_cond, node_time_feat = self._build_graph_condition(
            batch=batch,
            params=params,
            time=time,
            ref_tensor=x,
        )

        graph_latent = self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            node_time_feat=node_time_feat,
        )

        fused_latent = self.fuse(torch.cat([graph_latent, graph_cond], dim=1))

        template_feat = self.template_branch(
            template_x=x[:, :self.in_dim],
            edge_index=edge_index,
        )

        raw_out = self.decoder(
            template_feat=template_feat,
            fused_latent=fused_latent,
            batch=batch,
            graph_cond=graph_cond,
        )

        recon_x = self._compose_output_from_template(
            raw_out=raw_out,
            template_x=x[:, :self.in_dim],
        )
        return recon_x, graph_latent

    def decode(
        self,
        latent_z: torch.Tensor,
        template_x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        params: Optional[torch.Tensor],
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        graph_cond, _ = self._build_graph_condition(
            batch=batch,
            params=params,
            time=time,
            ref_tensor=template_x,
        )

        fused_latent = self.fuse(torch.cat([latent_z, graph_cond], dim=1))

        template_feat = self.template_branch(
            template_x=template_x[:, :self.in_dim],
            edge_index=edge_index,
        )

        raw_out = self.decoder(
            template_feat=template_feat,
            fused_latent=fused_latent,
            batch=batch,
            graph_cond=graph_cond,
        )

        recon_x = self._compose_output_from_template(
            raw_out=raw_out,
            template_x=template_x[:, :self.in_dim],
        )
        return recon_x


__all__ = ["PointNetGNNAutoencoder"]