"""A synthetic room. Numbers are invented so the solver can be checked."""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = (1280, 720)
CAMERA_CENTER = np.array([0.0, -6.0, 2.5], dtype=np.float64)
WORLD_XYZ = np.array(
    [
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-1.5, 3.0, 0.0],
        [1.5, 3.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 4.0, 0.0],
    ],
    dtype=np.float64,
)


def camera_matrix() -> np.ndarray:
    width, height = IMAGE_SIZE
    focal = 900.0
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def look_at_rotation(center: np.ndarray) -> np.ndarray:
    """World-to-camera rotation. The camera looks toward the origin, tilted down."""
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = target - np.array([center[0], center[1], 0.0])
    forward = forward / np.linalg.norm(forward)
    forward = forward + np.array([0.0, 0.0, -0.28])
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)


def project() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rvec, tvec, pixels)`` for ``WORLD_XYZ``."""
    rotation = look_at_rotation(CAMERA_CENTER)
    tvec = (-rotation @ CAMERA_CENTER).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    pixels, _ = cv2.projectPoints(
        WORLD_XYZ,
        rvec,
        tvec,
        camera_matrix(),
        np.zeros((5, 1)),
    )
    return rvec, tvec, pixels.reshape(-1, 2)
