"""ArUco marker detection with sub-pixel corner refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import cv2
import cv2.aruco as aruco
import numpy as np

# cv2.aruco returns corners in this order for an upright marker.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class MarkerDetection:
    """One marker seen in one frame."""

    frame_no: int
    marker_id: int
    corners: np.ndarray  # (4, 2) float32, in CORNER_ORDER

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.corners.astype(np.float32)))

    @property
    def edge_lengths(self) -> np.ndarray:
        rolled = np.roll(self.corners, -1, axis=0)
        return np.linalg.norm(rolled - self.corners, axis=1)

    @property
    def squareness(self) -> float:
        """1.0 for a perfect square; drops as the quad gets more distorted.

        A very low value usually means a false positive or an extreme viewing angle.
        """
        edges = self.edge_lengths
        return float(edges.min() / edges.max()) if edges.max() > 0 else 0.0


class ArucoMarkerDetector:
    """Detects a known set of ArUco marker IDs in BGR frames."""

    def __init__(
        self,
        dictionary: str = "DICT_4X4_50",
        marker_ids: Optional[Iterable[int]] = (0,),
        *,
        refine_corners: bool = True,
        min_squareness: float = 0.0,
        min_area_px: float = 0.0,
    ) -> None:
        if not hasattr(aruco, dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary}")
        self.dictionary_name = dictionary
        self.marker_ids = set(marker_ids) if marker_ids is not None else None
        self.min_squareness = min_squareness
        self.min_area_px = min_area_px

        params = aruco.DetectorParameters()
        if refine_corners:
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 50
            params.cornerRefinementMinAccuracy = 0.01
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 43
        params.adaptiveThreshWinSizeStep = 8
        params.minMarkerPerimeterRate = 0.01

        self._detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(getattr(aruco, dictionary)), params
        )

    def detect(self, frame: np.ndarray, frame_no: int = -1) -> List[MarkerDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        out: List[MarkerDetection] = []
        for quad, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if self.marker_ids is not None and marker_id not in self.marker_ids:
                continue
            det = MarkerDetection(
                frame_no=frame_no,
                marker_id=marker_id,
                corners=quad.reshape(4, 2).astype(np.float64),
            )
            if det.area < self.min_area_px or det.squareness < self.min_squareness:
                continue
            out.append(det)
        return out

    @staticmethod
    def draw(frame: np.ndarray, detections: Sequence[MarkerDetection]) -> np.ndarray:
        """Return an annotated copy. Do not commit the result."""
        vis = frame.copy()
        for det in detections:
            pts = det.corners.astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
            for label, point in zip(CORNER_ORDER, pts):
                cv2.circle(vis, tuple(point), 5, (0, 0, 255), -1)
            cx, cy = det.center.astype(int)
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(
                vis,
                f"id={det.marker_id}",
                (pts[0][0], pts[0][1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return vis
