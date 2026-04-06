import time
from typing import Optional, Tuple

import numpy as np


def _v3(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def detect_hand_gesture(hand_landmarks) -> Tuple[Optional[str], float]:
    if hand_landmarks is None:
        return None, 0.0

    wrist = _v3(hand_landmarks[0])
    idx_mcp = _v3(hand_landmarks[5])
    pinky_mcp = _v3(hand_landmarks[17])
    palm_span = np.linalg.norm(idx_mcp - pinky_mcp)
    if palm_span < 1e-6:
        return None, 0.0

    tips = [8, 12, 16, 20]
    avg_tip_dist = 0.0
    for tid in tips:
        avg_tip_dist += np.linalg.norm(_v3(hand_landmarks[tid]) - wrist)
    avg_tip_dist /= len(tips)
    openness = avg_tip_dist / palm_span

    thumb_tip = _v3(hand_landmarks[4])
    index_tip = _v3(hand_landmarks[8])
    pinch = np.linalg.norm(thumb_tip - index_tip) / palm_span

    if pinch < 0.35:
        return "pinch", float(1.0 - pinch)
    if openness < 1.05:
        return "fist", float(1.2 - openness)
    if openness > 1.85:
        return "open_palm", float(openness - 1.5)
    return None, 0.0


class GestureMacroEngine:
    def __init__(self, config: dict):
        self.enabled = bool(config.get("gesture_macros_enabled", True))
        self.hold_s = float(config.get("gesture_hold_seconds", 0.9))
        self.cooldown_s = float(config.get("gesture_cooldown_seconds", 2.0))
        self.current = None
        self.current_since = 0.0
        self.cooldown_until = 0.0

    def process(self, hand_landmarks) -> Optional[str]:
        if not self.enabled:
            return None
        now = time.time()
        if now < self.cooldown_until:
            return None

        gesture, _score = detect_hand_gesture(hand_landmarks)
        if gesture is None:
            self.current = None
            self.current_since = 0.0
            return None

        if gesture != self.current:
            self.current = gesture
            self.current_since = now
            return None

        if now - self.current_since < self.hold_s:
            return None

        self.current = None
        self.current_since = 0.0
        self.cooldown_until = now + self.cooldown_s
        if gesture == "fist":
            return "toggle_freeze"
        if gesture == "open_palm":
            return "home"
        if gesture == "pinch":
            return "toggle_estop"
        return None
