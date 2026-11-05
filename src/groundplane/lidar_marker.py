"""Find a marker stand in a LiDAR scan: ground plane, then a compact cluster."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PlaneModel:
    normal: np.ndarray
    offset: float
    inlier_ratio: float

    def height(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self.normal + self.offset


@dataclass
class MarkerCandidate:
    xyz: List[float]
    n_points: int
    size_xy_m: List[float]
    height_above_ground_m: float
    score: float
    ground_z: float

    def to_dict(self) -> dict:
        return asdict(self)


def ransac_plane(
    xyz: np.ndarray,
    *,
    thresh_m: float = 0.05,
    iterations: int = 250,
    rng: Optional[np.random.Generator] = None,
) -> PlaneModel:
    """Fit a plane; bias samples toward the lower half so we get the floor."""
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) < 50:
        raise ValueError("Not enough points to fit a ground plane")

    z_cut = np.quantile(pts[:, 2], 0.35)
    pool = pts[pts[:, 2] <= z_cut]
    if len(pool) < 20:
        pool = pts

    best_inliers = -1
    best_n = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    for _ in range(iterations):
        sample = pool[rng.choice(len(pool), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        offset = -float(np.dot(normal, sample[0]))
        n_in = int(np.count_nonzero(np.abs(pts @ normal + offset) < thresh_m))
        if n_in > best_inliers:
            best_inliers = n_in
            best_n, best_d = normal, offset

    if best_inliers < 10:
        raise RuntimeError("Ground-plane RANSAC failed")
    return PlaneModel(best_n, best_d, best_inliers / len(pts))


def _voxel_downsample(xyz: np.ndarray, leaf_m: float) -> np.ndarray:
    keys = np.floor(xyz / leaf_m).astype(np.int32)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return xyz[unique]


def _cluster_xy(points: np.ndarray, radius_m: float) -> List[np.ndarray]:
    """Greedy radius clustering in XY (KD-tree)."""
    if len(points) == 0:
        return []
    tree = cKDTree(points[:, :2])
    unused = np.ones(len(points), dtype=bool)
    clusters: List[np.ndarray] = []
    for seed in range(len(points)):
        if not unused[seed]:
            continue
        members: List[int] = []
        queue = [seed]
        unused[seed] = False
        while queue:
            i = queue.pop()
            members.append(i)
            for j in tree.query_ball_point(points[i, :2], radius_m):
                if unused[j]:
                    unused[j] = False
                    queue.append(j)
        clusters.append(points[np.array(members)])
    return clusters


def find_marker_candidates(
    xyz: np.ndarray,
    *,
    marker_length_m: float = 0.20,
    height_min_m: float = 0.25,
    height_max_m: float = 1.20,
    cluster_radius_m: float = 0.12,
    min_points: int = 20,
    max_candidates: int = 5,
) -> Tuple[PlaneModel, List[MarkerCandidate]]:
    plane = ransac_plane(xyz)
    height = plane.height(xyz)
    band = xyz[(height >= height_min_m) & (height <= height_max_m)]
    if len(band) < min_points:
        return plane, []
    band = _voxel_downsample(band, leaf_m=0.04)
    if len(band) > 8000:
        rng = np.random.default_rng(0)
        band = band[rng.choice(len(band), 8000, replace=False)]

    clusters = [
        c for c in _cluster_xy(band, cluster_radius_m) if len(c) >= min_points
    ]
    candidates: List[MarkerCandidate] = []
    for cluster in clusters:
        xy = cluster[:, :2]
        span = xy.max(axis=0) - xy.min(axis=0)
        extent = float(np.max(span))
        # Prefer a compact patch near the printed marker size.
        size_err = abs(extent - marker_length_m * 1.8)
        score = float(len(cluster)) / (1.0 + 8.0 * size_err)
        if extent > 1.2:
            continue
        center = cluster.mean(axis=0)
        candidates.append(
            MarkerCandidate(
                xyz=[round(float(v), 4) for v in center],
                n_points=int(len(cluster)),
                size_xy_m=[round(float(span[0]), 3), round(float(span[1]), 3)],
                height_above_ground_m=round(float(plane.height(center.reshape(1, 3))[0]), 3),
                score=round(score, 3),
                ground_z=round(float(center[2] - plane.height(center.reshape(1, 3))[0]), 4),
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return plane, candidates[:max_candidates]


def _rigid_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    u, _, vt = np.linalg.svd(src_c.T @ dst_c)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = dst.mean(axis=0) - r @ src.mean(axis=0)
    aligned = (r @ src.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return r, t, rms


def match_to_camera_poses(
    cam_xyz: dict,
    candidates_by_station: dict,
    *,
    inlier_m: float = 0.45,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Pick one LiDAR candidate per station so pairwise geometry matches camera-frame 3D.

    ``cam_xyz`` and ``candidates_by_station`` are keyed by station index.
    Returns ``{station: MarkerCandidate}`` for the best consistent set.
    """
    rng = rng or np.random.default_rng(1)
    stations = [s for s in cam_xyz if s in candidates_by_station and candidates_by_station[s]]
    if len(stations) < 3:
        return {s: candidates_by_station[s][0] for s in stations}

    best_score = (-1, 1e9)
    best_picks: dict = {}

    for _ in range(400):
        seed_ids = [int(i) for i in rng.choice(stations, size=3, replace=False)]
        for _combo in range(8):
            chosen = []
            src, dst = [], []
            ok = True
            for sid in seed_ids:
                opts = candidates_by_station[sid]
                cand = opts[int(rng.integers(0, min(3, len(opts))))]
                chosen.append((sid, cand))
                src.append(cam_xyz[sid])
                dst.append(cand.xyz)
            try:
                r, t, _ = _rigid_align(np.asarray(src, float), np.asarray(dst, float))
            except np.linalg.LinAlgError:
                ok = False
            if not ok:
                continue
            picks = {sid: cand for sid, cand in chosen}
            inliers = 3
            resid = []
            for sid in stations:
                predicted = r @ np.asarray(cam_xyz[sid], float) + t
                dists = [
                    float(np.linalg.norm(np.asarray(c.xyz, float) - predicted))
                    for c in candidates_by_station[sid]
                ]
                j = int(np.argmin(dists))
                if dists[j] <= inlier_m:
                    picks[sid] = candidates_by_station[sid][j]
                    inliers += 0 if sid in seed_ids else 1
                    resid.append(dists[j])
            score = (inliers, -float(np.mean(resid) if resid else 9))
            if score > best_score:
                best_score = score
                best_picks = picks

    return best_picks
