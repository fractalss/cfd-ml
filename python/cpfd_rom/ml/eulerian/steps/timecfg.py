"""Resolve temporal-feature and graph-convolution configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)

__all__ = ["TimeConfig", "resolve_time_features"]


@dataclass
class TimeConfig:
    """Resolved time-feature and graph-model configuration."""

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
    """Resolve temporal-feature settings from the runtime configuration.

    ``times_train`` is retained in the public signature for compatibility with
    existing pipeline callers. The feature dimension depends on the configured
    encoding rather than on the number of training times.
    """
    time_mode = str(getattr(cfg, "time_mode", "none")).strip().lower()
    fourier_m = int(getattr(cfg, "fourier_m", 4))

    add_time_attr = getattr(cfg, "add_time", None)
    add_time = (
        bool(add_time_attr)
        if add_time_attr is not None
        else time_mode != "none"
    )

    time_gain = float(getattr(cfg, "time_gain", 1.0))

    conv_type = str(getattr(cfg, "conv_type", "sage")).strip().lower()
    gat_heads = int(getattr(cfg, "gat_heads", 4))
    attn_drop = float(getattr(cfg, "attn_dropout", 0.1))

    use_baseline_as_feature = bool(
        getattr(cfg, "use_baseline_as_feature", False)
    )
    loss_weighting = str(
        getattr(cfg, "loss_weighting", "none")
    ).strip().lower()

    try:
        from ..model_gnn import _time_feature_dim

        t_feat_dim = _time_feature_dim(time_mode, fourier_m) if add_time else 0
    except Exception:
        # Preserve the existing fallback while making its cause available only
        # in debug mode. exc_info is evaluated by logging without changing the
        # numerical path.
        t_feat_dim = 0
        logger.debug(
            "Could not determine the time-feature dimension; using 0",
            exc_info=True,
        )

    detail(
        logger,
        "Time configuration: conv_type=%s, add_time=%s, time_mode=%s, "
        "fourier_m=%d, t_feat_dim=%d, time_gain=%g",
        conv_type,
        add_time,
        time_mode,
        fourier_m,
        t_feat_dim,
        time_gain,
    )

    if conv_type == "gat":
        detail(
            logger,
            "GAT configuration: heads=%d, attention_dropout=%g",
            gat_heads,
            attn_drop,
        )

    if use_baseline_as_feature:
        detail(logger, "Using baseline prediction as an input feature")

    if loss_weighting != "none":
        detail(logger, "Using loss-weighting strategy: %s", loss_weighting)

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
