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


def _augment_features_from_hand(
    features: Dict[str, float], hand_landmarks, shoulder_x: Optional[float] = None
) -> Dict[str, float]:
    hw = _v3(hand_landmarks[0])
    idx_mcp = _v3(hand_landmarks[5])
    mid_mcp = _v3(hand_landmarks[9])
    pinky_mcp = _v3(hand_landmarks[17])

    palm_vec = pinky_mcp[:2] - idx_mcp[:2]
    features["forearm_roll"] = float(math.degrees(math.atan2(palm_vec[1], palm_vec[0] + 1e-6)))

    wrist_to_mid = mid_mcp[:2] - hw[:2]
    features["wrist_pitch"] = float(math.degrees(math.atan2(-wrist_to_mid[1], abs(wrist_to_mid[0]) + 1e-6)))

    if shoulder_x is not None:
        features["palm_lateral"] = float(hw[0] - shoulder_x)
        features["wrist_lateral"] = float(hw[0] - shoulder_x)
    else:
        lateral = float(hw[0] - 0.5)
        features["palm_lateral"] = lateral
        features["wrist_lateral"] = lateral
        features["base_yaw"] = float(clamp((lateral / 0.25) * 90.0, -160.0, 160.0))
        features["palm_yaw"] = float(features["base_yaw"])
        features["wrist_reach"] = float(clamp(0.24 + (-hw[2]) * 0.28, 0.08, 0.62))
        features["wrist_height"] = float(clamp((0.60 - hw[1]) * 0.85, -0.45, 0.45))

    palm_span = np.linalg.norm(idx_mcp - pinky_mcp)
    tip_ids = [4, 8, 12, 16, 20]
    tip_dist = 0.0
    for tid in tip_ids:
        tip_dist += np.linalg.norm(_v3(hand_landmarks[tid]) - hw)
    tip_dist /= len(tip_ids)
    thumb_tip = _v3(hand_landmarks[4])
    index_tip = _v3(hand_landmarks[8])
    thumb_index = np.linalg.norm(thumb_tip - index_tip)
    open_metric = (0.75 * tip_dist + 0.25 * thumb_index) / max(1e-6, palm_span)
    # 0=open, 1=closed
    features["gripper"] = float(1.0 - clamp((open_metric - 0.45) / (1.55 - 0.45), 0.0, 1.0))
    return features


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
    hand_ok = hand_landmarks is not None
    if pose_landmarks is None:
        if hand_ok:
            return _augment_features_from_hand({}, hand_landmarks, shoulder_x=None), False, True, None
        return {}, False, False, None

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
        if hand_ok:
            return _augment_features_from_hand({}, hand_landmarks, shoulder_x=None), False, True, None
        return {}, False, False, None

    s = _v3(sl)
    e = _v3(el)
    w = _v3(wl)

    fore = w - e

    features: Dict[str, float] = {}
    se = e - s  # shoulder->elbow
    ew = w - e  # elbow->wrist

    # Shoulder segment controls base+shoulder:
    # - horizontal sweep (M1/base yaw)
    # - vertical lift (M2/shoulder)
    features["base_yaw"] = float(clamp((se[0] / 0.28) * 70.0, -100.0, 100.0))
    features["palm_yaw"] = float(features["base_yaw"])
    features["shoulder_pitch"] = float(clamp((-(se[1]) / 0.24) * 80.0, -100.0, 100.0))

    # Elbow segment vertical motion (M3) follows elbow->wrist vertical movement.
    features["elbow_bend"] = float(clamp((-(ew[1]) / 0.22) * 110.0, -130.0, 130.0))

    sw = w - s
    features["wrist_lateral"] = float(sw[0])
    features["wrist_reach"] = float(math.sqrt(sw[0] * sw[0] + sw[2] * sw[2]))
    features["wrist_height"] = float(-sw[1])
    features["forearm_roll"] = float(math.degrees(math.atan2(fore[2], fore[0] + 1e-6)))
    features["wrist_pitch"] = float(math.degrees(math.atan2(-ew[1], abs(ew[0]) + 1e-6)))

    if hand_ok:
        features = _augment_features_from_hand(features, hand_landmarks, shoulder_x=float(s[0]))
        # Stronger wrist flex/roll and gripper from hand landmarks.
        hw = _v3(hand_landmarks[0])
        idx_mcp = _v3(hand_landmarks[5])
        mid_mcp = _v3(hand_landmarks[9])
        pinky_mcp = _v3(hand_landmarks[17])

        fore2 = np.array([ew[0], ew[1]], dtype=float)
        palm2 = np.array([mid_mcp[0] - hw[0], mid_mcp[1] - hw[1]], dtype=float)
        n_f = np.linalg.norm(fore2)
        n_p = np.linalg.norm(palm2)
        if n_f > 1e-6 and n_p > 1e-6:
            f = fore2 / n_f
            p = palm2 / n_p
            det = f[0] * p[1] - f[1] * p[0]
            dot = clamp(float(np.dot(f, p)), -1.0, 1.0)
            flex = math.degrees(math.atan2(det, dot))
            features["wrist_pitch"] = float(clamp(flex, -95.0, 95.0))

        palm_vec = pinky_mcp[:2] - idx_mcp[:2]
        features["forearm_roll"] = float(
            clamp(math.degrees(math.atan2(palm_vec[1], palm_vec[0] + 1e-6)), -130.0, 130.0)
        )

        tip_ids = [8, 12, 16, 20]
        palm_span = max(1e-6, np.linalg.norm(idx_mcp - pinky_mcp))
        mean_tip = 0.0
        for tid in tip_ids:
            mean_tip += np.linalg.norm(_v3(hand_landmarks[tid]) - hw) / palm_span
        mean_tip /= len(tip_ids)
        thumb_index = np.linalg.norm(_v3(hand_landmarks[4]) - _v3(hand_landmarks[8])) / palm_span
        close_from_spread = 1.0 - clamp((mean_tip - 1.10) / (2.20 - 1.10), 0.0, 1.0)
        close_from_pinch = 1.0 - clamp((thumb_index - 0.20) / (1.00 - 0.20), 0.0, 1.0)
        features["gripper"] = float(clamp(0.75 * close_from_spread + 0.25 * close_from_pinch, 0.0, 1.0))
    else:
        features["gripper"] = 0.0

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
        self.control_mode = str(config.get("control_mode", "palm_follow")).strip().lower()
        palm_follow = config.get("palm_follow", {}) or {}
        self.ik_upper_len = float(palm_follow.get("ik_upper_len", 0.34))
        self.ik_fore_len = float(palm_follow.get("ik_fore_len", 0.33))
        self.ik_blend = clamp(float(palm_follow.get("ik_blend", 0.9)), 0.0, 1.0)
        self.wrist_comp_shoulder = float(palm_follow.get("wrist_comp_shoulder", 0.35))
        self.wrist_comp_elbow = float(palm_follow.get("wrist_comp_elbow", 0.22))
        self.base_lateral_norm = max(1e-3, float(palm_follow.get("base_lateral_norm", 0.18)))
        self.base_blend = clamp(float(palm_follow.get("base_blend", 0.9)), 0.0, 1.0)
        self.height_scale = float(palm_follow.get("height_scale", 1.0))
        self.reach_scale = float(palm_follow.get("reach_scale", 1.0))
        self.shoulder_blend = clamp(float(palm_follow.get("shoulder_blend", 0.9)), 0.0, 1.0)
        self.elbow_blend = clamp(float(palm_follow.get("elbow_blend", 0.9)), 0.0, 1.0)
        self.roll_blend = clamp(float(palm_follow.get("roll_blend", 0.85)), 0.0, 1.0)
        self.pitch_blend = clamp(float(palm_follow.get("pitch_blend", 0.85)), 0.0, 1.0)
        self.base_sign = float(palm_follow.get("base_sign", 1.0))
        self.shoulder_sign = float(palm_follow.get("shoulder_sign", 1.0))
        self.elbow_sign = float(palm_follow.get("elbow_sign", 1.0))
        self.roll_sign = float(palm_follow.get("roll_sign", 1.0))
        self.pitch_sign = float(palm_follow.get("pitch_sign", 1.0))

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

    def _motor_bounds(self, idx: int) -> Tuple[float, float]:
        m = self.motors[idx]
        lo = float(m["min_deg"])
        hi = float(m["max_deg"])
        return (lo, hi) if lo <= hi else (hi, lo)

    def _apply_direct_motor(self, idx: int, direct_cmd: float, runtime_trim: float) -> float:
        """
        Apply per-motor post transforms for direct (palm-follow) commands so
        inversion/gain/offset behave consistently with config expectations.
        """
        m = self.motors[idx]
        lo, hi = self._motor_bounds(idx)
        home = float(m.get("home_deg", (lo + hi) * 0.5))
        gain = float(m.get("gain", 1.0))
        offset = float(m.get("offset", 0.0))
        invert = bool(m.get("invert", False))

        delta = float(direct_cmd) - home
        if invert:
            delta = -delta
        mapped = home + gain * delta + offset + runtime_trim
        return clamp(mapped, lo, hi)

    def _map_palm_follow(self, features: Dict[str, float], trims: List[float]) -> List[float]:
        f = dict(features)
        home1 = float(self.motors[0].get("home_deg", 0.0))
        home2 = float(self.motors[1].get("home_deg", 0.0))
        home3 = float(self.motors[2].get("home_deg", 0.0))
        home4 = float(self.motors[3].get("home_deg", 0.0))
        home5 = float(self.motors[4].get("home_deg", 0.0))
        lo1, hi1 = self._motor_bounds(0)
        lo2, hi2 = self._motor_bounds(1)
        lo3, hi3 = self._motor_bounds(2)
        lo4, hi4 = self._motor_bounds(3)
        lo5, hi5 = self._motor_bounds(4)

        # Base from palm lateral displacement (stable on monocular webcam).
        lat = float(f.get("palm_lateral", f.get("wrist_lateral", 0.0)))
        lat_u = clamp(lat / self.base_lateral_norm, -1.0, 1.0)
        base_span_neg = abs(home1 - lo1)
        base_span_pos = abs(hi1 - home1)
        base_direct = home1 + self.base_sign * (base_span_pos if lat_u >= 0 else base_span_neg) * lat_u
        base_from_yaw = home1 + self.base_sign * float(f.get("base_yaw", 0.0))
        base_cmd = lerp(base_from_yaw, base_direct, self.base_blend)

        # Shoulder + elbow follow palm through a 2-link IK solve,
        # so moving hand up/down drives both joints together.
        x = max(0.01, float(f.get("wrist_reach", 0.25)) * self.reach_scale)
        y = float(f.get("wrist_height", 0.0)) * self.height_scale
        l1 = max(0.05, self.ik_upper_len)
        l2 = max(0.05, self.ik_fore_len)
        d = clamp(math.hypot(x, y), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
        cos_el = clamp((d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        el_rad = math.acos(cos_el)
        sh_rad = math.atan2(y, x) - math.atan2(l2 * math.sin(el_rad), l1 + l2 * math.cos(el_rad))
        shoulder_ik = math.degrees(sh_rad)
        elbow_ik = math.degrees(el_rad)

        sh_from_feat = float(f.get("shoulder_pitch", shoulder_ik))
        el_from_feat = float(f.get("elbow_bend", elbow_ik))
        shoulder_cmd = home2 + self.shoulder_sign * lerp(sh_from_feat, shoulder_ik, self.shoulder_blend * self.ik_blend)
        elbow_cmd = home3 + self.elbow_sign * lerp(el_from_feat, elbow_ik, self.elbow_blend * self.ik_blend)

        # Wrist compensation for arm lift.
        wrist_pitch_cmd = float(f.get("wrist_pitch", 0.0)) - (
            self.wrist_comp_shoulder * shoulder_cmd - self.wrist_comp_elbow * elbow_cmd
        )
        roll_cmd = home4 + self.roll_sign * float(f.get("forearm_roll", 0.0))
        roll_cmd = lerp(self.prev[3], roll_cmd, self.roll_blend)
        wrist_pitch_cmd = home5 + self.pitch_sign * wrist_pitch_cmd
        wrist_pitch_cmd = lerp(self.prev[4], wrist_pitch_cmd, self.pitch_blend)

        # Use direct arm-follow commands for first 5 joints, and keep config-driven
        # mapping for gripper (motor 6) from hand openness.
        mapped = [0.0] * 6
        mapped[0] = self._apply_direct_motor(0, base_cmd, trims[0])
        mapped[1] = self._apply_direct_motor(1, shoulder_cmd, trims[1])
        mapped[2] = self._apply_direct_motor(2, elbow_cmd, trims[2])
        mapped[3] = self._apply_direct_motor(3, roll_cmd, trims[3])
        mapped[4] = self._apply_direct_motor(4, wrist_pitch_cmd, trims[4])
        mapped[5] = self._map_motor(5, f, trims[5])
        return mapped

    def map(self, features: Dict[str, float], pose_ok: bool, runtime_trims: Optional[List[float]] = None) -> List[float]:
        trims = runtime_trims if runtime_trims and len(runtime_trims) == 6 else [0.0] * 6
        if not pose_ok:
            # Fallback: if hand features exist, still run palm-follow mapping
            # so robot does not freeze when body pose confidence dips.
            if self.control_mode == "palm_follow" and len(features) > 0:
                self.lost_count = 0
                features = self._smooth_features(features)
                target = self._map_palm_follow(features, trims)
            else:
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
            if self.control_mode == "palm_follow":
                target = self._map_palm_follow(features, trims)
            else:
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
