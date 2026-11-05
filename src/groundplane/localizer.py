"""Turn a pixel into a survey-frame coordinate, using only the camera.

After the pose is known, a foot pixel is back-projected onto the floor plane.
No LiDAR and no depth network at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .extrinsics import CameraPose


class GroundLocalizer:
    """Intersects camera rays with a horizontal plane in world coordinates."""

    def __init__(
        self,
        pose: CameraPose,
        *,
        ground_z: float = 0.0,
        ground_xy_offset: Tuple[float, float] = (0.0, 0.0),
        origin_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.pose = pose
        self.ground_z = ground_z
        self.ground_xy_offset = (float(ground_xy_offset[0]), float(ground_xy_offset[1]))
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self._R = pose.rotation
        self._R_inv = self._R.T
        self._camera_center = pose.camera_center_world
        self._K = pose.intrinsics.camera_matrix
        self._dist = pose.intrinsics.dist_coeffs

    @classmethod
    def from_file(cls, path: Path | str, *, ground_z: Optional[float] = None) -> "GroundLocalizer":
        path = Path(path)
        extra = json.loads(path.read_text(encoding="utf-8"))
        z = extra.get("ground_z", 0.0) if ground_z is None else ground_z
        offset = extra.get("ground_xy_offset", [0.0, 0.0])
        origin = extra.get("origin_xy", [0.0, 0.0])
        return cls(
            CameraPose.load(path),
            ground_z=float(z),
            ground_xy_offset=(float(offset[0]), float(offset[1])),
            origin_xy=(float(origin[0]), float(origin[1])),
        )

    def ray_direction(self, u: float, v: float) -> np.ndarray:
        """Unit ray through pixel (u, v), expressed in world coordinates."""
        undistorted = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float64), self._K, self._dist
        ).reshape(2)
        ray_cam = np.array([undistorted[0], undistorted[1], 1.0])
        ray_world = self._R_inv @ ray_cam
        return ray_world / np.linalg.norm(ray_world)

    def pixel_to_ground(
        self, u: float, v: float, *, plane_z: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """World point where the ray through (u, v) meets the plane.

        Returns ``None`` when the ray is parallel to the plane or points away
        from it.
        """
        z = self.ground_z if plane_z is None else plane_z
        ray = self.ray_direction(u, v)
        if abs(ray[2]) < 1e-9:
            return None
        scale = (z - self._camera_center[2]) / ray[2]
        if scale <= 0:
            return None
        point = self._camera_center + scale * ray
        dx, dy = self.ground_xy_offset
        ox, oy = self.origin_xy
        if dx or dy or ox or oy:
            point = np.asarray(point, dtype=np.float64).copy()
            point[0] += dx - ox
            point[1] += dy - oy
        return point

    def distance_to_camera(
        self, u: float, v: float, *, plane_z: Optional[float] = None
    ) -> Optional[float]:
        """Straight-line distance from the camera to the ground point, in metres."""
        point = self.pixel_to_ground(u, v, plane_z=plane_z)
        if point is None:
            return None
        return float(np.linalg.norm(point - self._camera_center))

    def ground_distance(
        self, u: float, v: float, *, plane_z: Optional[float] = None
    ) -> Optional[float]:
        """Horizontal distance from the camera, in metres."""
        point = self.pixel_to_ground(u, v, plane_z=plane_z)
        if point is None:
            return None
        return float(np.linalg.norm(point[:2] - self._camera_center[:2]))

    def world_to_pixel(self, point_world: np.ndarray) -> Tuple[float, float]:
        point = np.asarray(point_world, dtype=np.float64).reshape(3).copy()
        point[0] += self.origin_xy[0] - self.ground_xy_offset[0]
        point[1] += self.origin_xy[1] - self.ground_xy_offset[1]
        projected, _ = cv2.projectPoints(
            point.reshape(1, 3),
            self.pose.rvec,
            self.pose.tvec,
            self._K,
            self._dist,
        )
        u, v = projected.reshape(2)
        return float(u), float(v)

    def horizon_v(self, u: Optional[float] = None) -> Optional[float]:
        """Image row where the ground plane vanishes."""
        u = self.pose.intrinsics.cx if u is None else u
        lo, hi = 0.0, float(self.pose.intrinsics.image_size[1])
        if self.pixel_to_ground(u, hi - 1) is None:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.pixel_to_ground(u, mid) is None:
                lo = mid
            else:
                hi = mid
        return hi

    def meters_per_pixel(self, u: float, v: float, *, delta: float = 1.0) -> Optional[float]:
        """Local ground scale at a given image row."""
        a = self.pixel_to_ground(u, v)
        b = self.pixel_to_ground(u, v + delta)
        if a is None or b is None:
            return None
        return float(np.linalg.norm(b[:2] - a[:2]) / delta)
