import math
import time
from typing import Dict, List, Optional

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _v3(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


class HandPoseTracker:
    """
    Perception layer for Web Teleop V3.

    Inputs:
    - MediaPipe pose landmarks
    - MediaPipe hand landmarks (single selected hand)

    Outputs:
    - Stable palm-centered control point in camera coordinates
    - Tracking confidence
    - Gripper pinch ratio
    - Human arm points for the digital twin ghost line
    """

    def __init__(self, config: dict):
        tracker_cfg = config.get("hand_tracker", {})
        self.pose_vis_min = float(config.get("pose_visibility_threshold", 0.55))
        self.position_alpha = _clamp(tracker_cfg.get("position_alpha", 0.3), 0.01, 1.0)
        self.confidence_alpha = _clamp(tracker_cfg.get("confidence_alpha", 0.45), 0.01, 1.0)
        self.min_hand_span = float(tracker_cfg.get("min_hand_span", 0.02))
        self.fallback_wrist_confidence = _clamp(tracker_cfg.get("fallback_wrist_confidence", 0.35), 0.0, 1.0)
        self._smoothed_control: Optional[np.ndarray] = None
        self._smoothed_conf = 0.0

    def _compute_pose_state(self, pose_landmarks, arm_side: str) -> Dict:
        if pose_landmarks is None:
            return {
                "pose_ok": False,
                "pose_vis_score": 0.0,
                "shoulder": None,
                "elbow": None,
                "wrist": None,
                "human_arm_world": None,
            }

        if arm_side.lower() == "left":
            shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
        else:
            shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16

        sl = pose_landmarks[shoulder_idx]
        el = pose_landmarks[elbow_idx]
        wl = pose_landmarks[wrist_idx]

        vis = [
            float(getattr(sl, "visibility", 1.0)),
            float(getattr(el, "visibility", 1.0)),
            float(getattr(wl, "visibility", 1.0)),
        ]
        pose_ok = min(vis) >= self.pose_vis_min
        pose_vis_score = _clamp(sum(vis) / 3.0, 0.0, 1.0)

        s = _v3(sl)
        e = _v3(el)
        w = _v3(wl)

        def world(pt: np.ndarray) -> List[float]:
            return [float(pt[0]), float(-pt[1]), float(-pt[2])]

        human_arm_world = [
            [0.0, 0.0, 0.0],
            world(e - s),
            world(w - s),
        ]
        return {
            "pose_ok": bool(pose_ok),
            "pose_vis_score": float(pose_vis_score),
            "shoulder": s,
            "elbow": e,
            "wrist": w,
            "human_arm_world": human_arm_world,
        }

    def _compute_hand_state(self, hand_landmarks) -> Dict:
        if hand_landmarks is None:
            return {
                "hand_ok": False,
                "control_raw": None,
                "palm_span": 0.0,
                "pinch_ratio": None,
                "hand_score": 0.0,
            }

        palm_ids = [0, 5, 9, 13, 17]
        palm_pts = [_v3(hand_landmarks[idx]) for idx in palm_ids]
        control_raw = np.mean(palm_pts, axis=0)

        thumb_tip = _v3(hand_landmarks[4])
        index_tip = _v3(hand_landmarks[8])
        index_mcp = _v3(hand_landmarks[5])
        pinky_mcp = _v3(hand_landmarks[17])

        pinch = float(np.linalg.norm(thumb_tip - index_tip))
        palm_span = float(np.linalg.norm(index_mcp - pinky_mcp))
        pinch_ratio = pinch / max(1e-6, palm_span)

        hand_score = _clamp((palm_span - self.min_hand_span) / max(1e-6, self.min_hand_span), 0.0, 1.0)
        return {
            "hand_ok": True,
            "control_raw": control_raw,
            "palm_span": palm_span,
            "pinch_ratio": pinch_ratio,
            "hand_score": hand_score,
        }

    def process(self, pose_landmarks, hand_landmarks, arm_side: str) -> Dict:
        pose = self._compute_pose_state(pose_landmarks=pose_landmarks, arm_side=arm_side)
        hand = self._compute_hand_state(hand_landmarks=hand_landmarks)

        control_raw = hand["control_raw"]
        if control_raw is None and pose["wrist"] is not None:
            control_raw = pose["wrist"].copy()
        raw_control = control_raw.copy() if control_raw is not None else None

        if control_raw is not None:
            if self._smoothed_control is None:
                self._smoothed_control = control_raw.copy()
            else:
                self._smoothed_control = _lerp_vec(
                    self._smoothed_control,
                    control_raw,
                    self.position_alpha,
                )

        if pose["pose_ok"] and hand["hand_ok"]:
            conf_raw = 0.5 * pose["pose_vis_score"] + 0.5 * hand["hand_score"]
        elif pose["pose_ok"] and control_raw is not None:
            conf_raw = pose["pose_vis_score"] * self.fallback_wrist_confidence
        else:
            conf_raw = 0.0

        self._smoothed_conf = (
            self.confidence_alpha * conf_raw + (1.0 - self.confidence_alpha) * self._smoothed_conf
        )

        out = {
            "timestamp": time.time(),
            "pose_detected": bool(pose["pose_ok"]),
            "hand_detected": bool(hand["hand_ok"]),
            "tracking_confidence": float(_clamp(self._smoothed_conf, 0.0, 1.0)),
            "raw_control_point_camera_xyz": (
                [float(v) for v in raw_control.tolist()]
                if raw_control is not None
                else None
            ),
            "control_point_camera_xyz": (
                [float(v) for v in self._smoothed_control.tolist()]
                if self._smoothed_control is not None
                else None
            ),
            "wrist_camera_xyz": (
                [float(v) for v in pose["wrist"].tolist()]
                if pose["wrist"] is not None
                else None
            ),
            "pinch_ratio": (
                float(hand["pinch_ratio"]) if hand["pinch_ratio"] is not None else None
            ),
            "human_arm_world": pose["human_arm_world"],
        }
        return out
