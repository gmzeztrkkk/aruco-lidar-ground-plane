"""Camera intrinsics from repeated views of one known-size ArUco marker.

Zhang's method needs many views of a planar target at varied orientations. A
calibration recording where the marker is carried around the room supplies
exactly that, so no separate checkerboard session is required.

Caveat: four corners per view is thin. Views are therefore chosen for pose
diversity, the principal point is fixed at the image centre by default, and the
distortion model is kept short. Treat the result as a solid starting estimate
and prefer a checkerboard run when the highest accuracy matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .track import MarkerTrack, TrackSample


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics plus the quality metrics that earned them."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]  # (width, height)
    rms_reproj_px: float
    n_views: int
    method: str = "aruco_zhang"

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.image_size[0] / (2 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.image_size[1] / (2 * self.fy))))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "method": self.method,
                    "image_size": list(self.image_size),
                    "camera_matrix": self.camera_matrix.tolist(),
                    "dist_coeffs": self.dist_coeffs.ravel().tolist(),
                    "rms_reproj_px": self.rms_reproj_px,
                    "n_views": self.n_views,
                    "fx": self.fx,
                    "fy": self.fy,
                    "cx": self.cx,
                    "cy": self.cy,
                    "hfov_deg": self.hfov_deg,
                    "vfov_deg": self.vfov_deg,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CameraIntrinsics":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            image_size=tuple(data["image_size"]),
            rms_reproj_px=float(data["rms_reproj_px"]),
            n_views=int(data["n_views"]),
            method=data.get("method", "unknown"),
        )

    def summary(self) -> str:
        return (
            f"fx={self.fx:.1f}  fy={self.fy:.1f}  cx={self.cx:.1f}  cy={self.cy:.1f}\n"
            f"dist={np.array2string(self.dist_coeffs.ravel(), precision=5)}\n"
            f"FOV: {self.hfov_deg:.1f}deg horizontal, {self.vfov_deg:.1f}deg vertical\n"
            f"RMS reprojection: {self.rms_reproj_px:.3f} px over {self.n_views} views"
        )


class ArucoIntrinsicsEstimator:
    """Selects diverse marker views and runs Zhang calibration on them."""

    def __init__(
        self,
        marker_length_m: float,
        image_size: Tuple[int, int],
        *,
        max_views: int = 80,
        min_area_px: float = 300.0,
        min_squareness: float = 0.25,
        fix_principal_point: bool = True,
        fix_aspect_ratio: bool = True,
        n_dist_coeffs: int = 2,
    ) -> None:
        self.marker_length_m = marker_length_m
        self.image_size = image_size
        self.max_views = max_views
        self.min_area_px = min_area_px
        self.min_squareness = min_squareness
        self.fix_principal_point = fix_principal_point
        self.fix_aspect_ratio = fix_aspect_ratio
        self.n_dist_coeffs = n_dist_coeffs

    @property
    def object_points(self) -> np.ndarray:
        """Marker corners in its own frame, matching cv2.aruco corner order."""
        h = self.marker_length_m / 2.0
        return np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float32
        )

    # ---- view selection ----------------------------------------------

    @staticmethod
    def _descriptor(sample: TrackSample, image_size: Tuple[int, int]) -> np.ndarray:
        """Where the marker is, how big it is, and how it is tilted."""
        corners = sample.corners
        center = corners.mean(axis=0)
        scale = np.sqrt(max(_quad_area(corners), 1e-6))
        shape = ((corners - center) / scale).reshape(-1)
        position = center / np.array(image_size, dtype=float)
        return np.concatenate([position * 2.0, shape, [np.log(scale)]])

    def select_views(self, track: MarkerTrack) -> List[TrackSample]:
        """Farthest-point sampling so the views are not all near-duplicates."""
        candidates = [
            s
            for s in track.samples
            if _quad_area(s.corners) >= self.min_area_px
            and _squareness(s.corners) >= self.min_squareness
        ]
        if len(candidates) <= self.max_views:
            return candidates

        descriptors = np.array([self._descriptor(s, self.image_size) for s in candidates])
        chosen = [int(np.argmax(descriptors[:, -1]))]  # start from the largest view
        distances = np.linalg.norm(descriptors - descriptors[chosen[0]], axis=1)
        while len(chosen) < self.max_views:
            nxt = int(np.argmax(distances))
            if distances[nxt] <= 0:
                break
            chosen.append(nxt)
            distances = np.minimum(
                distances, np.linalg.norm(descriptors - descriptors[nxt], axis=1)
            )
        return [candidates[i] for i in sorted(chosen)]

    # ---- calibration --------------------------------------------------

    def estimate(
        self, track: MarkerTrack, views: Optional[Sequence[TrackSample]] = None
    ) -> Tuple[CameraIntrinsics, List[TrackSample]]:
        views = list(views) if views is not None else self.select_views(track)
        if len(views) < 8:
            raise RuntimeError(
                f"Need at least 8 usable marker views, got {len(views)}. "
                "Lower min_area_px or scan with a smaller stride."
            )

        object_points = [self.object_points for _ in views]
        image_points = [s.corners.astype(np.float32).reshape(4, 1, 2) for s in views]

        width, height = self.image_size
        init_f = 0.9 * width  # starting guess near a 60 degree horizontal field of view
        camera_matrix = np.array(
            [[init_f, 0, width / 2.0], [0, init_f, height / 2.0], [0, 0, 1]], dtype=np.float64
        )
        dist = np.zeros((5, 1), dtype=np.float64)

        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if self.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if self.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        for idx, flag in enumerate(
            (cv2.CALIB_FIX_K1, cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K3), start=1
        ):
            if idx > self.n_dist_coeffs:
                flags |= flag

        rms, camera_matrix, dist, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            (width, height),
            camera_matrix,
            dist,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
        )
        return (
            CameraIntrinsics(
                camera_matrix=camera_matrix,
                dist_coeffs=dist,
                image_size=self.image_size,
                rms_reproj_px=float(rms),
                n_views=len(views),
            ),
            views,
        )


def _quad_area(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _squareness(corners: np.ndarray) -> float:
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return float(edges.min() / edges.max()) if edges.max() > 0 else 0.0
