"""Find the intervals where the marker sat still.

During a calibration recording the marker is carried to a known spot, left there
for a while, then moved to the next spot. Each of those rest periods is one
usable 2D observation to pair with a surveyed 3D coordinate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .track import MarkerTrack, TrackSample


@dataclass
class StationarySegment:
    """A stretch of frames where the marker centre barely moved."""

    index: int
    start_frame: int
    end_frame: int
    start_utc: Optional[str]
    end_utc: Optional[str]
    duration_s: float
    n_samples: int
    center_px: List[float]
    corners_px: List[List[float]]  # median corner positions, (4, 2)
    jitter_px: float               # max distance of any sample centre from the median
    std_px: float
    mean_area_px: float

    def to_dict(self) -> dict:
        return asdict(self)


class StationarySegmentFinder:
    """Greedy scan that cuts the track wherever the marker moves or disappears."""

    def __init__(
        self,
        *,
        max_jitter_px: float = 6.0,
        min_duration_s: float = 3.0,
        min_samples: int = 10,
        max_gap_s: float = 1.5,
    ) -> None:
        self.max_jitter_px = max_jitter_px
        self.min_duration_s = min_duration_s
        self.min_samples = min_samples
        self.max_gap_s = max_gap_s

    def find(self, track: MarkerTrack) -> List[StationarySegment]:
        if not track.samples:
            return []

        fps = track.fps if track.fps and track.fps > 0 else 30.0
        max_gap_frames = self.max_gap_s * fps
        samples = sorted(track.samples, key=lambda s: s.frame_no)

        segments: List[StationarySegment] = []
        current: List[TrackSample] = [samples[0]]

        for sample in samples[1:]:
            gap = sample.frame_no - current[-1].frame_no
            anchor = np.median(np.array([s.center for s in current]), axis=0)
            drift = float(np.linalg.norm(sample.center - anchor))

            if gap > max_gap_frames or drift > self.max_jitter_px:
                self._flush(segments, current, fps)
                current = [sample]
            else:
                current.append(sample)

        self._flush(segments, current, fps)
        for i, segment in enumerate(segments):
            segment.index = i
        return segments

    def _flush(self, out: List[StationarySegment], group: List[TrackSample], fps: float) -> None:
        if len(group) < self.min_samples:
            return
        duration = (group[-1].frame_no - group[0].frame_no) / fps
        if duration < self.min_duration_s:
            return

        centers = np.array([s.center for s in group])
        corners = np.array([s.corners for s in group])  # (n, 4, 2)
        median_center = np.median(centers, axis=0)
        median_corners = np.median(corners, axis=0)
        areas = [
            0.5
            * abs(
                float(
                    np.dot(c[:, 0], np.roll(c[:, 1], -1))
                    - np.dot(c[:, 1], np.roll(c[:, 0], -1))
                )
            )
            for c in corners
        ]

        out.append(
            StationarySegment(
                index=len(out),
                start_frame=group[0].frame_no,
                end_frame=group[-1].frame_no,
                start_utc=group[0].timestamp_utc,
                end_utc=group[-1].timestamp_utc,
                duration_s=round(duration, 3),
                n_samples=len(group),
                center_px=[round(float(v), 3) for v in median_center],
                corners_px=[[round(float(v), 3) for v in pt] for pt in median_corners],
                jitter_px=round(float(np.linalg.norm(centers - median_center, axis=1).max()), 3),
                std_px=round(float(np.linalg.norm(centers - median_center, axis=1).std()), 3),
                mean_area_px=round(float(np.mean(areas)), 1),
            )
        )

    @staticmethod
    def save(
        segments: List[StationarySegment],
        path: Path | str,
        *,
        meta: Optional[dict] = None,
        stations: Optional[List["MarkerStation"]] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": meta or {},
            "stations": [s.to_dict() for s in stations] if stations is not None else [],
            "segments": [s.to_dict() for s in segments],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


@dataclass
class MarkerStation:
    """One physical marker placement, possibly seen across several segments.

    Occlusion (someone walking past) splits a single placement into multiple
    segments; merging by pixel proximity recovers the real station list.
    """

    index: int
    center_px: List[float]
    corners_px: List[List[float]]
    segment_indices: List[int]
    n_samples: int
    total_duration_s: float
    first_utc: Optional[str]
    last_utc: Optional[str]
    frame_ranges: List[List[int]]
    spread_px: float
    mean_area_px: float
    world_xyz: Optional[List[float]] = None  # filled in from LiDAR later

    def to_dict(self) -> dict:
        return asdict(self)


class StationGrouper:
    """Merges segments whose median centre lands within ``merge_radius_px``."""

    def __init__(self, merge_radius_px: float = 25.0) -> None:
        self.merge_radius_px = merge_radius_px

    def group(self, segments: List[StationarySegment]) -> List[MarkerStation]:
        groups: List[List[StationarySegment]] = []
        for segment in segments:
            center = np.array(segment.center_px)
            for group in groups:
                anchor = np.median(np.array([s.center_px for s in group]), axis=0)
                if np.linalg.norm(center - anchor) <= self.merge_radius_px:
                    group.append(segment)
                    break
            else:
                groups.append([segment])

        stations: List[MarkerStation] = []
        for i, group in enumerate(groups):
            centers = np.array([s.center_px for s in group])
            weights = np.array([s.n_samples for s in group], dtype=float)
            corners = np.array([s.corners_px for s in group])  # (g, 4, 2)
            center = np.average(centers, axis=0, weights=weights)
            mean_corners = np.average(corners, axis=0, weights=weights)
            stations.append(
                MarkerStation(
                    index=i,
                    center_px=[round(float(v), 3) for v in center],
                    corners_px=[[round(float(v), 3) for v in pt] for pt in mean_corners],
                    segment_indices=[s.index for s in group],
                    n_samples=int(sum(s.n_samples for s in group)),
                    total_duration_s=round(sum(s.duration_s for s in group), 2),
                    first_utc=min((s.start_utc or "" for s in group), default=None) or None,
                    last_utc=max((s.end_utc or "" for s in group), default=None) or None,
                    frame_ranges=[[s.start_frame, s.end_frame] for s in group],
                    spread_px=round(float(np.linalg.norm(centers - center, axis=1).max()), 2),
                    mean_area_px=round(
                        float(np.average([s.mean_area_px for s in group], weights=weights)), 1
                    ),
                )
            )
        stations.sort(key=lambda s: s.first_utc or "")
        for i, station in enumerate(stations):
            station.index = i
        return stations
