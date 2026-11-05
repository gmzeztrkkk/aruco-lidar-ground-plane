"""Solve the synthetic pose and drop one pixel onto the floor.

    python examples/solve_and_localize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from groundplane import (  # noqa: E402
    CameraIntrinsics,
    Correspondence,
    ExtrinsicSolver,
    GroundLocalizer,
)
from scene import IMAGE_SIZE, WORLD_XYZ, camera_matrix  # noqa: E402


def main() -> None:
    payload = json.loads((ROOT / "examples" / "synthetic_correspondences.json").read_text())
    intrinsics = CameraIntrinsics(
        camera_matrix=camera_matrix(),
        dist_coeffs=np_zeros(),
        image_size=IMAGE_SIZE,
        rms_reproj_px=0.0,
        n_views=0,
        method="synthetic",
    )
    correspondences = [
        Correspondence(row["name"], tuple(row["world_xyz"]), tuple(row["pixel_uv"]))
        for row in payload["correspondences"]
    ]
    pose = ExtrinsicSolver(intrinsics, ransac=False).solve(correspondences)
    localizer = GroundLocalizer(pose)
    u, v = correspondences[0].pixel_uv
    point = localizer.pixel_to_ground(u, v)
    center = pose.camera_center_world
    print(f"camera centre: {center[0]:.3f} {center[1]:.3f} {center[2]:.3f} m")
    print(f"rms reprojection: {pose.rms_reproj_px:.4f} px")
    if point is None:
        raise SystemExit("pixel missed the ground plane")
    expected = WORLD_XYZ[0]
    print(f"pixel ({u:.1f}, {v:.1f}) -> {point[0]:.3f} {point[1]:.3f} {point[2]:.3f} m")
    print(f"survey point:          {expected[0]:.3f} {expected[1]:.3f} {expected[2]:.3f} m")


def np_zeros():
    import numpy as np

    return np.zeros((5, 1))


if __name__ == "__main__":
    main()
