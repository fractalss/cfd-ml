from cpfd_rom.util import config
# Expose configuration and model interface
from .shared.rom_base import PCARBF_ROM_Generic

# Eulerian ROM utilities
from .rom_eulerian_pca_rbf.rom_analysis import evaluate_and_collect_snapshots
from .rom_eulerian_pca_rbf.visualization import (
    plot_explained_variance,
    plot_rmse_vs_time,
    plot_comparisons,
)

# Lagrangian ROM utilities
from .rom_lagrangian_pca_rbf.analysis import evaluate_lagrangian_rom
from .rom_lagrangian_pca_rbf.visualization_lagrangian import (
    plot_lagrangian_rmse,
    plot_snapshot_comparison,
)

