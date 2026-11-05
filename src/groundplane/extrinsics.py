"""Camera pose in the survey frame, from marker-station correspondences.

Once this pose exists, later frames are localised from pixels alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .intrinsics import CameraIntrinsics


@dataclass
class Correspondence:
    """One surveyed point and where the camera saw it."""

    name: str
    world_xyz: Tuple[float, float, float]
    pixel_uv: Tuple[float, float]


@dataclass
class CameraPose:
    """Rigid transform from world to camera, plus fit quality."""

    rvec: np.ndarray  # (3, 1) Rodrigues, world -> camera
    tvec: np.ndarray  # (3, 1) metres
    intrinsics: CameraIntrinsics
    rms_reproj_px: float
    per_point_error_px: List[float]
    inlier_names: List[str]
    outlier_names: List[str]

    @property
    def rotation(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def T_world_cam(self) -> np.ndarray:
        """4x4 world -> camera."""
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.tvec.ravel()
        return transform

    @property
    def T_cam_world(self) -> np.ndarray:
        """4x4 camera -> world."""
        return np.linalg.inv(self.T_world_cam)

    @property
    def camera_center_world(self) -> np.ndarray:
        """Where the camera sits, in world coordinates."""
        return (-self.rotation.T @ self.tvec).ravel()

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "rvec": self.rvec.ravel().tolist(),
                    "tvec": self.tvec.ravel().tolist(),
                    "T_world_cam": self.T_world_cam.tolist(),
                    "camera_center_world": self.camera_center_world.tolist(),
                    "camera_matrix": self.intrinsics.camera_matrix.tolist(),
                    "dist_coeffs": self.intrinsics.dist_coeffs.ravel().tolist(),
                    "image_size": list(self.intrinsics.image_size),
                    "rms_reproj_px": self.rms_reproj_px,
                    "per_point_error_px": self.per_point_error_px,
                    "inliers": self.inlier_names,
                    "outliers": self.outlier_names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CameraPose":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        intrinsics = CameraIntrinsics(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            image_size=tuple(data["image_size"]),
            rms_reproj_px=0.0,
            n_views=0,
            method="loaded",
        )
        return cls(
            rvec=np.array(data["rvec"], dtype=np.float64).reshape(3, 1),
            tvec=np.array(data["tvec"], dtype=np.float64).reshape(3, 1),
            intrinsics=intrinsics,
            rms_reproj_px=float(data["rms_reproj_px"]),
            per_point_error_px=list(data["per_point_error_px"]),
            inlier_names=list(data.get("inliers", [])),
            outlier_names=list(data.get("outliers", [])),
        )

    def summary(self) -> str:
        center = self.camera_center_world
        return (
            f"camera centre (world): x={center[0]:.3f}  y={center[1]:.3f}  z={center[2]:.3f} m\n"
            f"RMS reprojection: {self.rms_reproj_px:.2f} px  "
            f"({len(self.inlier_names)} inliers, {len(self.outlier_names)} rejected)"
        )


class ExtrinsicSolver:
    """Solves camera pose from at least four world-pixel correspondences.

    Stations surveyed on one plane are coplanar. That is fine for PnP when the
    intrinsics are already known.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        ransac: bool = True,
        reproj_threshold_px: float = 4.0,
        refine: bool = True,
    ) -> None:
        self.intrinsics = intrinsics
        self.ransac = ransac
        self.reproj_threshold_px = reproj_threshold_px
        self.refine = refine

    def solve(self, correspondences: Sequence[Correspondence]) -> CameraPose:
        if len(correspondences) < 4:
            raise ValueError(f"Need at least 4 correspondences, got {len(correspondences)}")

        object_points = np.array([c.world_xyz for c in correspondences], dtype=np.float64)
        image_points = np.array([c.pixel_uv for c in correspondences], dtype=np.float64)
        camera_matrix = self.intrinsics.camera_matrix
        dist = self.intrinsics.dist_coeffs

        flag_order = [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP]
        if self._is_coplanar(object_points):
            flag_order.insert(0, cv2.SOLVEPNP_IPPE)

        ok, rvec, tvec, inliers = False, None, None, None
        for flags in flag_order:
            if self.ransac and len(correspondences) >= 6 and flags != cv2.SOLVEPNP_IPPE:
                try:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist,
                        reprojectionError=self.reproj_threshold_px,
                        iterationsCount=5000,
                        confidence=0.999,
                        flags=flags,
                    )
                except cv2.error:
                    ok = False
                if ok and inliers is not None and len(inliers) >= 4:
                    break
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist, flags=flags
                )
                inliers = np.arange(len(correspondences)).reshape(-1, 1)
            except cv2.error:
                ok = False
            if ok:
                break

        if not ok or rvec is None:
            raise RuntimeError("solvePnP failed to find a pose")
        inlier_idx = (
            inliers.ravel().tolist() if inliers is not None else list(range(len(correspondences)))
        )

        if self.refine and len(inlier_idx) >= 4:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_idx],
                image_points[inlier_idx],
                camera_matrix,
                dist,
                rvec,
                tvec,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10),
            )

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        inlier_set = set(inlier_idx)
        rms = float(np.sqrt(np.mean(errors[list(inlier_idx)] ** 2)))

        return CameraPose(
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            intrinsics=self.intrinsics,
            rms_reproj_px=rms,
            per_point_error_px=[round(float(e), 3) for e in errors],
            inlier_names=[correspondences[i].name for i in sorted(inlier_set)],
            outlier_names=[c.name for i, c in enumerate(correspondences) if i not in inlier_set],
        )

    @staticmethod
    def _is_coplanar(points: np.ndarray, tol: float = 1e-3) -> bool:
        centered = points - points.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        return bool(singular_values[-1] < tol * max(singular_values[0], 1e-9))
