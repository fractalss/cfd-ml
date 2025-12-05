import numpy as np

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots

rev_dirs = [
    "/data3/sauravmitra/ML_CFD_PINN/BVR-Kuipers-ROM/Rev1_npy",
    "/data3/sauravmitra/ML_CFD_PINN/BVR-Kuipers-ROM/Rev2_npy",
    "/data3/sauravmitra/ML_CFD_PINN/BVR-Kuipers-ROM/Rev3_npy",
    "/data3/sauravmitra/ML_CFD_PINN/BVR-Kuipers-ROM/Rev4_npy",
    "/data3/sauravmitra/ML_CFD_PINN/BVR-Kuipers-ROM/Rev5_npy",
]

train_times, train_data = load_lagrangian_snapshots(rev_dirs)

# find unique times for each rev
n_revs = len(rev_dirs)
snaps_per_rev = len(train_times) // n_revs

for r in range(n_revs):
    times_r = train_times[r*snaps_per_rev:(r+1)*snaps_per_rev]
    print(f"Rev{r+1} first 10 times:", times_r[:10])


# compute bed heights
from cpfd_rom.ml_rom.rom_lagrangian_ml.baseline_all4 import _compute_bed_height_scalar

S, N, F = train_data.shape
bed_heights = _compute_bed_height_scalar(train_data)

print("\nBed heights at t=0 for each rev:")
for r in range(n_revs):
    idx = r * snaps_per_rev
    print(f"Rev{r+1}: H = {bed_heights[idx]:.4f}")
