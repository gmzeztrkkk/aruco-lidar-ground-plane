# aruco-lidar-ground-plane

Solve a fixed camera from a walked ArUco board and a LiDAR survey. After that, a pixel on the floor becomes metres. Runtime does not need the LiDAR.

This repository is the method only. It does not contain recordings, floor plans, photographs, or site coordinates. Bring your own survey.

## Pipeline

1. Detect an ArUco marker (`DICT_4X4_50` by default) and keep the frames where it sits still.
2. In the LiDAR scan, fit the floor, then look for a compact cluster about the size of the board.
3. Match each still station to a LiDAR stop by UTC. The stop CSV uses `start_utc`, `end_utc`, `x_m`, `y_m`, `z_m`.
4. `solvePnP` with known intrinsics turns those pairs into the camera pose. RANSAC drops a bad station.
5. A pixel is a ray. Intersect it with `z = 0`. Optional `origin_xy` / `ground_xy_offset` only shift the output frame.

## Try the synthetic room

```bash
pip install -e ".[dev]"
python examples/solve_and_localize.py
pytest
```

`examples/synthetic_correspondences.json` is an invented room: camera at `(0, -6, 2.5)` m, six floor points. It is not a real floor.

## Layout

```text
src/groundplane/    detector, still stations, intrinsics, LiDAR cluster, UTC match, PnP, localizer
examples/           synthetic correspondences and a short script
tests/              round-trip on that room
```

Do not commit bags, video, images, DXF, or point clouds. `.gitignore` already refuses those extensions.
