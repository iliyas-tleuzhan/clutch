import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = clamp(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def joint_angle(a, b, c):
    # Angle ABC in degrees
    return angle_between(a - b, c - b)


def _v3(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def extract_arm_features(
    pose_landmarks, hand_landmarks, arm_side: str = "right", pose_vis_min: float = 0.55
) -> Tuple[Dict[str, float], bool, bool, Optional[List[List[float]]]]:
    """
    Extract full-arm-centric features:
    - base_yaw (wrist horizontal movement relative to shoulder)
    - shoulder_pitch (upper-arm lift)
    - elbow_bend
    - forearm_roll
    - wrist_pitch
    - gripper (single openness metric)
    """
    if pose_landmarks is None:
        return {}, False, hand_landmarks is not None, None

    if arm_side.lower() == "left":
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
    else:
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16

    sl = pose_landmarks[shoulder_idx]
    el = pose_landmarks[elbow_idx]
    wl = pose_landmarks[wrist_idx]

    vis_min = float(pose_vis_min)
    pose_ok = (
        getattr(sl, "visibility", 1.0) >= vis_min
        and getattr(el, "visibility", 1.0) >= vis_min
        and getattr(wl, "visibility", 1.0) >= vis_min
    )
    if not pose_ok:
        return {}, False, hand_landmarks is not None, None

    s = _v3(sl)
    e = _v3(el)
    w = _v3(wl)

    upper = e - s
    fore = w - e

    features: Dict[str, float] = {}
    features["base_yaw"] = float(math.degrees(math.atan2((w[0] - s[0]), -(w[2] - s[2]) + 1e-6)))
    features["shoulder_pitch"] = float(
        math.degrees(
            math.atan2(-(e[1] - s[1]), math.sqrt((e[0] - s[0]) ** 2 + (e[2] - s[2]) ** 2) + 1e-6)
        )
    )
    features["elbow_bend"] = float(180.0 - joint_angle(s, e, w))
    features["forearm_roll"] = float(math.degrees(math.atan2(fore[2], fore[0] + 1e-6)))
    features["wrist_pitch"] = float(
        math.degrees(math.atan2(-(w[1] - e[1]), abs(w[0] - e[0]) + 1e-6))
    )

    hand_ok = hand_landmarks is not None
    if hand_ok:
        hw = _v3(hand_landmarks[0])
        idx_mcp = _v3(hand_landmarks[5])
        mid_mcp = _v3(hand_landmarks[9])
        pinky_mcp = _v3(hand_landmarks[17])

        palm_vec = pinky_mcp[:2] - idx_mcp[:2]
        features["forearm_roll"] = float(
            math.degrees(math.atan2(palm_vec[1], palm_vec[0] + 1e-6))
        )

        wrist_to_mid = mid_mcp[:2] - hw[:2]
        features["wrist_pitch"] = float(
            math.degrees(math.atan2(-wrist_to_mid[1], abs(wrist_to_mid[0]) + 1e-6))
        )

        palm_span = np.linalg.norm(idx_mcp - pinky_mcp)
        tip_ids = [8, 12, 16, 20]
        tip_dist = 0.0
        for tid in tip_ids:
            tip_dist += np.linalg.norm(_v3(hand_landmarks[tid]) - hw)
        tip_dist /= len(tip_ids)
        features["gripper"] = float(tip_dist / max(1e-6, palm_span))
    else:
        features["gripper"] = 1.2

    def world(pt):
        return [float(pt[0]), float(-pt[1]), float(-pt[2])]

    human_arm_world = [
        [0.0, 0.0, 0.0],
        world(e - s),
        world(w - s),
    ]
    return features, True, hand_ok, human_arm_world


class ArmFollowMapper:
    def __init__(self, config: dict):
        self.config = config
        self.motors = config["motors"]
        self.feature_ranges = config["feature_ranges"]
        self.alpha = float(config.get("smoothing_alpha", 0.25))
        self.alpha = clamp(self.alpha, 0.01, 1.0)
        self.feature_alpha = float(config.get("feature_smoothing_alpha", 0.2))
        self.feature_alpha = clamp(self.feature_alpha, 0.01, 1.0)
        self.deadzone_deg = float(config.get("deadzone_deg", 1.0))
        self.feature_deadzone = float(config.get("feature_deadzone", 0.0))
        self.lost_pose_hold_frames = int(config.get("lost_pose_hold_frames", 8))
        self.lost_mode = config.get("lost_mode", "hold")

        self.home = [float(m.get("home_deg", 0.0)) for m in self.motors]
        self.prev = self.home[:]
        ms = config.get("max_step_deg", 4.0)
        if isinstance(ms, list) and len(ms) == 6:
            self.max_step_deg = [float(v) for v in ms]
        else:
            self.max_step_deg = [float(ms)] * 6
        self.center_offsets = {}
        for feat_name, fr in self.feature_ranges.items():
            self.center_offsets[feat_name] = float(fr.get("center_offset", 0.0))
        self.feature_prev: Dict[str, float] = {}
        self.lost_count = 0

    def calibrate(self, features: Dict[str, float]):
        for feat_name, fr in self.feature_ranges.items():
            if fr.get("center_calibrated", False) and feat_name in features:
                self.center_offsets[feat_name] = float(features[feat_name])

    def _feature_to_norm(self, feat_name: str, raw_value: float) -> float:
        fr = self.feature_ranges[feat_name]
        lo = float(fr["min"])
        hi = float(fr["max"])
        v = float(raw_value)
        if fr.get("center_calibrated", False):
            v -= self.center_offsets.get(feat_name, 0.0)
        if abs(hi - lo) < 1e-9:
            return 0.5
        return clamp((v - lo) / (hi - lo), 0.0, 1.0)

    def _smooth_features(self, features: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in features.items():
            fv = float(v)
            if k in self.feature_prev:
                sv = lerp(self.feature_prev[k], fv, self.feature_alpha)
                if abs(sv - self.feature_prev[k]) < self.feature_deadzone:
                    sv = self.feature_prev[k]
            else:
                sv = fv
            self.feature_prev[k] = sv
            out[k] = sv
        return out

    def _map_motor(self, idx: int, features: Dict[str, float], runtime_trim: float) -> float:
        m = self.motors[idx]
        feat_name = m["feature_source"]
        if feat_name not in features:
            return self.prev[idx]

        lo = float(m["min_deg"])
        hi = float(m["max_deg"])
        home = float(m.get("home_deg", (lo + hi) * 0.5))
        gain = float(m.get("gain", 1.0))
        offset = float(m.get("offset", 0.0))
        mapping_mode = m.get("mapping", "range")

        sign = -1.0 if bool(m.get("invert", False)) else 1.0
        if mapping_mode == "delta":
            fr = self.feature_ranges[feat_name]
            center = self.center_offsets.get(feat_name, 0.0) if fr.get("center_calibrated", False) else 0.0
            delta = float(features[feat_name]) - center
            mapped = home + sign * gain * delta + offset
        else:
            norm = self._feature_to_norm(feat_name, features[feat_name])
            if bool(m.get("invert", False)):
                norm = 1.0 - norm
            mapped = lerp(lo, hi, norm)
            mapped = home + gain * (mapped - home) + offset

        mapped += runtime_trim
        return clamp(mapped, lo, hi)

    def map(self, features: Dict[str, float], pose_ok: bool, runtime_trims: Optional[List[float]] = None) -> List[float]:
        trims = runtime_trims if runtime_trims and len(runtime_trims) == 6 else [0.0] * 6
        if not pose_ok:
            self.lost_count += 1
            if self.lost_count <= self.lost_pose_hold_frames:
                target = self.prev[:]
            elif self.lost_mode == "home":
                target = self.home[:]
            else:
                target = self.prev[:]
        else:
            self.lost_count = 0
            features = self._smooth_features(features)
            target = [self._map_motor(i, features, trims[i]) for i in range(6)]

        out = []
        for i, (p, t) in enumerate(zip(self.prev, target)):
            t_limited = clamp(t, p - self.max_step_deg[i], p + self.max_step_deg[i])
            s = lerp(p, t_limited, self.alpha)
            if abs(s - p) < self.deadzone_deg:
                s = p
            out.append(s)
        self.prev = out
        return out
