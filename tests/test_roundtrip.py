"""Synthetic room: pose solve, pixel-to-ground, UTC match, LiDAR cluster."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from groundplane import (  # noqa: E402
    CameraIntrinsics,
    Correspondence,
    ExtrinsicSolver,
    GroundLocalizer,
    LidarStop,
    find_marker_candidates,
    match_stop,
    parse_utc,
)
from scene import CAMERA_CENTER, IMAGE_SIZE, WORLD_XYZ, camera_matrix, project  # noqa: E402


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_matrix=camera_matrix(),
        dist_coeffs=np.zeros((5, 1)),
        image_size=IMAGE_SIZE,
        rms_reproj_px=0.0,
        n_views=0,
        method="synthetic",
    )


def _correspondences() -> list[Correspondence]:
    _, _, pixels = project()
    return [
        Correspondence(f"p{i}", tuple(float(v) for v in xyz), (float(uv[0]), float(uv[1])))
        for i, (xyz, uv) in enumerate(zip(WORLD_XYZ, pixels))
    ]


def test_solve_recovers_camera_center() -> None:
    pose = ExtrinsicSolver(_intrinsics(), ransac=False).solve(_correspondences())
    assert pose.rms_reproj_px < 0.05
    np.testing.assert_allclose(pose.camera_center_world, CAMERA_CENTER, atol=1e-3)


def test_pixel_roundtrip_on_the_floor() -> None:
    pose = ExtrinsicSolver(_intrinsics(), ransac=False).solve(_correspondences())
    localizer = GroundLocalizer(pose)
    _, _, pixels = project()
    for xyz, uv in zip(WORLD_XYZ, pixels):
        point = localizer.pixel_to_ground(float(uv[0]), float(uv[1]))
        assert point is not None
        np.testing.assert_allclose(point, xyz, atol=1e-2)


def test_origin_shift_is_applied_after_the_hit() -> None:
    pose = ExtrinsicSolver(_intrinsics(), ransac=False).solve(_correspondences())
    shifted = GroundLocalizer(pose, origin_xy=(1.0, 2.0))
    _, _, pixels = project()
    point = shifted.pixel_to_ground(float(pixels[0, 0]), float(pixels[0, 1]))
    assert point is not None
    np.testing.assert_allclose(point[:2], WORLD_XYZ[0, :2] - np.array([1.0, 2.0]), atol=1e-2)


def test_match_stop_uses_the_time_window() -> None:
    t0 = parse_utc("2024-01-01T00:00:00Z")
    stops = [
        LidarStop(t0, t0 + 5.0, (1.0, 2.0, 0.0), name="a"),
        LidarStop(t0 + 20.0, t0 + 25.0, (3.0, 4.0, 0.0), name="b"),
    ]
    hit = match_stop(stops, t0 + 21.0)
    assert hit is not None and hit.name == "b"
    assert match_stop(stops, t0 + 12.0) is None


def test_marker_cluster_sits_above_the_floor() -> None:
    rng = np.random.default_rng(0)
    floor = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, 500),
            rng.uniform(-3.0, 3.0, 500),
            rng.normal(0.0, 0.01, 500),
        ]
    )
    stand = np.column_stack(
        [
            rng.normal(0.4, 0.03, 60),
            rng.normal(-0.2, 0.03, 60),
            rng.normal(0.75, 0.02, 60),
        ]
    )
    _, candidates = find_marker_candidates(np.vstack([floor, stand]), min_points=15)
    assert candidates
    xyz = np.array(candidates[0].xyz)
    assert abs(xyz[0] - 0.4) < 0.15
    assert abs(xyz[1] + 0.2) < 0.15
    assert 0.4 < candidates[0].height_above_ground_m < 1.1
