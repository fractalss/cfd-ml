from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TimeConfig:
    add_time: bool
    time_mode: str
    fourier_m: int
    time_gain: float
    t_feat_dim: int
    conv_type: str
    gat_heads: int
    attn_drop: float
    use_baseline_as_feature: bool
    loss_weighting: str


def resolve_time_features(cfg, times_train) -> TimeConfig:
    time_mode = str(getattr(cfg, "time_mode", "none")).strip().lower()
    fourier_m = int(getattr(cfg, "fourier_m", 4))
    _add_time_attr = getattr(cfg, "add_time", None)
    add_time = bool(_add_time_attr) if _add_time_attr is not None else (time_mode != "none")

    # gain
    time_gain = float(getattr(cfg, "time_gain", 1.0))
    # conv extras
    conv_type = str(getattr(cfg, "conv_type", "sage")).lower()
    gat_heads = int(getattr(cfg, "gat_heads", 4))
    attn_drop = float(getattr(cfg, "attn_dropout", 0.1))

    # baseline/weighting extras
    use_baseline_as_feature = bool(getattr(cfg, "use_baseline_as_feature", False))
    loss_weighting = str(getattr(cfg, "loss_weighting", "none")).strip().lower()

    try:
        from ..model_gnn import _time_feature_dim
        t_feat_dim = _time_feature_dim(time_mode, fourier_m) if add_time else 0
    except Exception:
        t_feat_dim = 0

    print(
        f"[CFG] conv_type={conv_type}  add_time={add_time}  time_mode={time_mode}  "
        f"fourier_m={fourier_m}  t_feat_dim={t_feat_dim}  time_gain={time_gain}"
    )
    if conv_type == "gat":
        print(f"[CFG] GAT extras: heads={gat_heads} attn_dropout={attn_drop}")
    if use_baseline_as_feature:
        print("[CFG] Baseline channel will be concatenated as feature")
    if loss_weighting != "none":
        print(f"[CFG] Loss weighting strategy: {loss_weighting}")

    return TimeConfig(
        add_time=add_time,
        time_mode=time_mode,
        fourier_m=fourier_m,
        time_gain=time_gain,
        t_feat_dim=t_feat_dim,
        conv_type=conv_type,
        gat_heads=gat_heads,
        attn_drop=attn_drop,
        use_baseline_as_feature=use_baseline_as_feature,
        loss_weighting=loss_weighting,
    )
