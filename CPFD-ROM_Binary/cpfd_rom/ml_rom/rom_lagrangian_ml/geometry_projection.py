# cpfd_rom/ml_rom/rom_lagrangian_ml/geometry_projection.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh


class GeometryProjector:
    """
    Geometry-aware projector using an STL volume.

    Assumes the STL is a closed, watertight mesh describing the *valid* region
    for particles (usually the fluid region of the reactor).
    """

    def __init__(self, stl_path: str | Path):
        stl_path = Path(stl_path)
        if not stl_path.exists():
            raise FileNotFoundError(f"[Lagrangian] STL file not found: {stl_path}")

        print(f"[INFO] Loading geometry STL: {stl_path}")
        self.mesh = trimesh.load_mesh(stl_path, process=True)

        if not self.mesh.is_watertight:
            print("[WARN] STL mesh is not watertight; contains() may be unreliable.")

    def project_points_inside(self, xyz: np.ndarray) -> np.ndarray:
        """
        Project points into the valid volume defined by the STL mesh.

        Args:
            xyz: array of shape (M, 3) with point coordinates.

        Returns:
            xyz_proj: array (M, 3) where any point that was outside has been
                      snapped onto (or slightly into) the mesh surface.
        """
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (M, 3), got {xyz.shape}")

        # 1) Boolean mask: which points are inside?
        inside = self.mesh.contains(xyz)

        # 2) For outside points, find nearest point on the surface
        outside_idx = np.where(~inside)[0]
        xyz_proj = xyz.copy()

        if len(outside_idx) > 0:
            pts_out = xyz[outside_idx]

            # nearest.on_surface returns (points, distances, triangle_index)
            nearest_pts, _, tri_idx = self.mesh.nearest.on_surface(pts_out)

            # Optional: nudge slightly inward along -normal to avoid being just outside
            normals = self.mesh.face_normals[tri_idx]   # outward normals
            eps = 1e-4

            # Try moving slightly opposite the normal; if that is still outside,
            # just use the nearest point itself.
            candidate_inside = self.mesh.contains(nearest_pts - eps * normals)
            nearest_pts_adj = nearest_pts.copy()
            nearest_pts_adj[candidate_inside] -= eps * normals[candidate_inside]

            xyz_proj[outside_idx] = nearest_pts_adj

        return xyz_proj


    def project(self, p: np.ndarray) -> np.ndarray:
        """
        Project a single point into the STL-defined volume.
        Thin wrapper over project_points_inside for a single (3,) vector.
        """
        p = np.asarray(p, dtype=float).reshape(1, 3)
        return self.project_points_inside(p)[0]

    def project_points(self, xyz: np.ndarray) -> np.ndarray:
        """Vectorised projection for an array of points.

        This is a thin wrapper around :meth:`project_points_inside` and
        does **not** loop per-point in Python. It expects ``xyz`` to be
        of shape (M, 3) and returns an array of the same shape.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (M, 3), got {xyz.shape}")
        return self.project_points_inside(xyz)
