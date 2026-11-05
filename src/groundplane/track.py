"""Per-frame marker observations collected from one video."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .detector import MarkerDetection


@dataclass
class TrackSample:
    """One accepted detection, enriched with wall-clock time."""

    frame_no: int
    marker_id: int
    corners: np.ndarray  # (4, 2)
    timestamp_utc: Optional[str] = None
    epoch_sec: Optional[float] = None

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


@dataclass
class MarkerTrack:
    """All samples of a single marker ID through one video."""

    video_path: str
    marker_id: int
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int
    stride: int = 1
    samples: List[TrackSample] = field(default_factory=list)
    frames_scanned: int = 0

    def add(self, detection: MarkerDetection, utc: Optional[str], epoch: Optional[float]) -> None:
        self.samples.append(
            TrackSample(
                frame_no=detection.frame_no,
                marker_id=detection.marker_id,
                corners=detection.corners,
                timestamp_utc=utc,
                epoch_sec=epoch,
            )
        )

    @property
    def detection_rate(self) -> float:
        return len(self.samples) / self.frames_scanned if self.frames_scanned else 0.0

    def centers(self) -> np.ndarray:
        return np.array([s.center for s in self.samples]) if self.samples else np.empty((0, 2))

    def frame_numbers(self) -> np.ndarray:
        return np.array([s.frame_no for s in self.samples], dtype=int)

    def __iter__(self) -> Iterator[TrackSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    # ---- persistence -------------------------------------------------

    FIELDNAMES = [
        "frame_no",
        "timestamp_utc",
        "epoch_sec",
        "marker_id",
        "cx",
        "cy",
        "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3",
    ]

    @classmethod
    def from_csv(cls, path: Path | str, **meta) -> "MarkerTrack":
        """Reload a previously scanned track, so grouping can be re-tuned cheaply."""
        path = Path(path)
        track = cls(
            video_path=meta.get("video_path", ""),
            marker_id=meta.get("marker_id", 0),
            fps=meta.get("fps", 0.0),
            frame_width=meta.get("frame_width", 0),
            frame_height=meta.get("frame_height", 0),
            total_frames=meta.get("total_frames", 0),
            stride=meta.get("stride", 1),
        )
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                corners = np.array(
                    [[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(4)]
                )
                track.samples.append(
                    TrackSample(
                        frame_no=int(row["frame_no"]),
                        marker_id=int(row["marker_id"]),
                        corners=corners,
                        timestamp_utc=row["timestamp_utc"] or None,
                        epoch_sec=float(row["epoch_sec"]) if row["epoch_sec"] else None,
                    )
                )
        track.frames_scanned = meta.get("frames_scanned", len(track.samples))
        return track

    def to_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for s in self.samples:
                cx, cy = s.center
                flat = s.corners.reshape(-1)
                row: Dict[str, object] = {
                    "frame_no": s.frame_no,
                    "timestamp_utc": s.timestamp_utc or "",
                    "epoch_sec": f"{s.epoch_sec:.6f}" if s.epoch_sec is not None else "",
                    "marker_id": s.marker_id,
                    "cx": f"{cx:.3f}",
                    "cy": f"{cy:.3f}",
                }
                for i in range(4):
                    row[f"x{i}"] = f"{flat[2 * i]:.3f}"
                    row[f"y{i}"] = f"{flat[2 * i + 1]:.3f}"
                writer.writerow(row)
        return path
