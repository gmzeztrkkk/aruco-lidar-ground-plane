"""Per-station marker poses and the ground plane they imply.

The marker rides the same stand at every station, so its centres all sit at one
height. Fitting a plane through those centres recovers the floor orientation in
camera coordinates - the camera's tilt and roll - from video alone. The LiDAR
survey is then only needed to pin down the remaining horizontal rotation and
origin that tie this plane to the survey frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .intrinsics import CameraIntrinsics


@dataclass
class MarkerPose:
    """Marker pose in camera coordinates, from a single known-size square."""

    station_index: int
    rvec: np.ndarray
    tvec: np.ndarray
    reproj_err_px: float

    @property
    def position_cam(self) -> np.ndarray:
        return self.tvec.ravel()

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.tvec))

    @property
    def normal_cam(self) -> np.ndarray:
        """Marker face normal (its local +Z) in camera coordinates."""
        return cv2.Rodrigues(self.rvec)[0][:, 2]


class MarkerPoseEstimator:
    """Solves a planar square's pose; IPPE_SQUARE is the right tool for 4 corners."""

    def __init__(self, intrinsics: CameraIntrinsics, marker_length_m: float) -> None:
        self.intrinsics = intrinsics
        self.marker_length_m = marker_length_m
        h = marker_length_m / 2.0
        self._object_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64
        )

    def estimate(self, corners: np.ndarray, station_index: int = -1) -> MarkerPose:
        image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        K = self.intrinsics.camera_matrix
        dist = self.intrinsics.dist_coeffs

        # IPPE_SQUARE returns two mirrored solutions; keep the better-fitting one.
        retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            self._object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not retval:
            raise RuntimeError(f"Pose estimation failed for station {station_index}")

        best = min(range(len(rvecs)), key=lambda i: float(errors[i]))
        rvec, tvec = rvecs[best], tvecs[best]
        projected, _ = cv2.projectPoints(self._object_points, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1))))

        return MarkerPose(
            station_index=station_index,
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            reproj_err_px=err,
        )


@dataclass
class PlaneFit:
    """A plane ``normal . x + offset = 0`` fitted to 3D points."""

    normal: np.ndarray
    offset: float
    rms_m: float
    max_dev_m: float
    n_points: int

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.dot(self.normal, np.asarray(point).ravel()) + self.offset)

    def tilt_vs(self, reference: Sequence[float] = (0.0, -1.0, 0.0)) -> float:
        """Angle in degrees between this plane's normal and a reference direction."""
        ref = np.asarray(reference, dtype=float)
        ref = ref / np.linalg.norm(ref)
        cos = float(np.clip(abs(np.dot(self.normal, ref)), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))


class GroundPlaneEstimator:
    """Least-squares plane through the marker centres (SVD, no RANSAC needed at n~10)."""

    @staticmethod
    def fit(points: Sequence[np.ndarray]) -> PlaneFit:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 3:
            raise ValueError(f"Need at least 3 points to fit a plane, got {len(pts)}")
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid)
        normal = vt[-1]
        normal = normal / np.linalg.norm(normal)
        if normal[1] > 0:  # keep "up" pointing towards -Y in camera coordinates
            normal = -normal
        offset = float(-np.dot(normal, centroid))
        deviations = np.abs(pts @ normal + offset)
        return PlaneFit(
            normal=normal,
            offset=offset,
            rms_m=float(np.sqrt(np.mean(deviations**2))),
            max_dev_m=float(deviations.max()),
            n_points=len(pts),
        )

    @staticmethod
    def camera_height_above(plane: PlaneFit, drop_m: float = 0.0) -> float:
        """Camera height above a plane parallel to ``plane`` but ``drop_m`` below it.

        With ``drop_m`` set to the stand height, this is the camera's height
        above the actual floor.
        """
        return float(abs(plane.offset) + drop_m)

    @staticmethod
    def orientation_angles(plane: PlaneFit) -> Tuple[float, float]:
        """(pitch, roll) of the camera relative to the plane, in degrees."""
        n = plane.normal
        pitch = float(np.degrees(np.arctan2(-n[2], -n[1])))
        roll = float(np.degrees(np.arctan2(n[0], -n[1])))
        return pitch, roll
