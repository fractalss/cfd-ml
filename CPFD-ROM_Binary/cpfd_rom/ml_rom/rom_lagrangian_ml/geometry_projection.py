"""Geometry-aware projection utilities for Lagrangian particle positions."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)


class GeometryProjector:
    """Project particle positions into a valid volume defined by an STL mesh.

    The STL is expected to be a closed, watertight mesh describing the valid
    particle region, which is typically the reactor's fluid volume.
    """

    def __init__(self, stl_path: str | Path) -> None:
        stl_path = Path(stl_path)
        if not stl_path.exists():
            raise FileNotFoundError(f"[Lagrangian] STL file not found: {stl_path}")

        detail(logger, "[Lagrangian] Loading geometry STL: %s", stl_path)
        self.mesh = trimesh.load_mesh(stl_path, process=True)

        if not self.mesh.is_watertight:
            logger.warning(
                "[Lagrangian] STL mesh is not watertight; point-containment "
                "checks may be unreliable."
            )

    def project_points_inside(self, xyz: np.ndarray) -> np.ndarray:
        """Project points into the valid volume defined by the STL mesh.

        Parameters
        ----------
        xyz:
            Point coordinates with shape ``(M, 3)``.

        Returns
        -------
        np.ndarray
            An array with shape ``(M, 3)`` in which outside points have been
            moved onto, or slightly inside, the mesh surface.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (M, 3), got {xyz.shape}")

        inside = self.mesh.contains(xyz)
        outside_idx = np.flatnonzero(~inside)
        xyz_proj = xyz.copy()

        if outside_idx.size == 0:
            return xyz_proj

        points_outside = xyz[outside_idx]

        # ``on_surface`` returns surface points, distances, and triangle IDs.
        nearest_points, _, triangle_idx = self.mesh.nearest.on_surface(
            points_outside
        )

        # Move surface points slightly opposite their outward face normal. If
        # that candidate is not inside, retain the nearest surface point.
        normals = self.mesh.face_normals[triangle_idx]
        epsilon = 1.0e-4
        inward_candidates = nearest_points - epsilon * normals
        candidate_inside = self.mesh.contains(inward_candidates)

        adjusted_points = nearest_points.copy()
        adjusted_points[candidate_inside] = inward_candidates[candidate_inside]
        xyz_proj[outside_idx] = adjusted_points

        return xyz_proj

    def project(self, point: np.ndarray) -> np.ndarray:
        """Project one point with shape ``(3,)`` into the STL volume."""
        point = np.asarray(point, dtype=float).reshape(1, 3)
        return self.project_points_inside(point)[0]

    def project_points(self, xyz: np.ndarray) -> np.ndarray:
        """Project an array of points without a per-point Python loop."""
        return self.project_points_inside(xyz)


__all__ = ["GeometryProjector"]
