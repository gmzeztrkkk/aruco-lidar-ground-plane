"""Frame number to UTC, from a sidecar CSV next to a recording."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


class FrameTimestamps:
    """Lookup table built from ``<video_stem>_timestamps.csv``.

    Expected columns: ``frame_no``, ``timestamp_utc``, ``epoch_sec``.
    """

    def __init__(self, utc: Dict[int, str], epoch: Dict[int, float]) -> None:
        self._utc = utc
        self._epoch = epoch

    @classmethod
    def from_csv(cls, path: Path | str) -> "FrameTimestamps":
        path = Path(path)
        utc: Dict[int, str] = {}
        epoch: Dict[int, float] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame_no = int(row["frame_no"])
                utc[frame_no] = row["timestamp_utc"]
                epoch[frame_no] = float(row["epoch_sec"])
        if not utc:
            raise ValueError(f"No timestamp rows in {path}")
        return cls(utc, epoch)

    @classmethod
    def sidecar_for(cls, video_path: Path | str) -> Optional["FrameTimestamps"]:
        """Load ``<stem>_timestamps.csv`` next to the video, if present."""
        video_path = Path(video_path)
        candidate = video_path.with_name(f"{video_path.stem}_timestamps.csv")
        if candidate.is_file():
            return cls.from_csv(candidate)
        return None

    def utc(self, frame_no: int) -> Optional[str]:
        return self._utc.get(frame_no)

    def epoch(self, frame_no: int) -> Optional[float]:
        return self._epoch.get(frame_no)

    def __len__(self) -> int:
        return len(self._utc)

    def __contains__(self, frame_no: int) -> bool:
        return frame_no in self._utc
