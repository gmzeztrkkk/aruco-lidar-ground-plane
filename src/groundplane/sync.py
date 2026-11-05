"""Match a camera timestamp to a LiDAR stop window."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def parse_utc(value: str) -> float:
    """ISO-8601 timestamp to Unix seconds. A trailing ``Z`` is UTC."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@dataclass(frozen=True)
class LidarStop:
    """One stationary LiDAR pose, in metres, over a UTC window."""

    t0: float
    t1: float
    xyz_m: tuple[float, float, float]
    name: str = ""


def match_stop(
    stops: list[LidarStop],
    epoch: float,
    *,
    slop_s: float = 0.75,
) -> LidarStop | None:
    """Return the stop whose ``[t0, t1]`` contains ``epoch``, within ``slop_s``."""
    for stop in stops:
        if stop.t0 - slop_s <= epoch <= stop.t1 + slop_s:
            return stop
    return None


def load_stops_csv(path: Path | str) -> list[LidarStop]:
    """Load stops from CSV.

    Required columns: ``start_utc``, ``end_utc``, ``x_m``, ``y_m``, ``z_m``.
    Optional column: ``name``.
    """
    path = Path(path)
    stops: list[LidarStop] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("start_utc") or not row.get("end_utc"):
                continue
            stops.append(
                LidarStop(
                    t0=parse_utc(row["start_utc"]),
                    t1=parse_utc(row["end_utc"]),
                    xyz_m=(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])),
                    name=row.get("name") or "",
                )
            )
    if not stops:
        raise ValueError(f"No LiDAR stops in {path}")
    return stops
