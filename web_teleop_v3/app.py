import argparse
import asyncio
import base64
import json
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_teleop_v3.arm_follow_mapper import ArmFollowMapper, extract_arm_features
from web_teleop_v3.camera_quality import evaluate_frame_quality
from web_teleop_v3.gesture_macros import GestureMacroEngine
from web_teleop_v3.ros2_bridge_stub import ROS2BridgeStub
from web_teleop_v3.runtime_metrics import RuntimeMetrics
from web_teleop_v3.safety_supervisor import SafetySupervisor
from web_teleop_v3.trajectory_manager import TrajectoryManager
from web_teleop_v3.validation_report import generate_sim_real_report
from web_teleop_v3.voice_commands import VoiceCommandEngine


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


ROOT = THIS_DIR
DEFAULT_CONFIG = ROOT / "config.web_demo.json"
INDEX_HTML = ROOT / "static" / "index.html"
MAIN_HTML = ROOT / "static" / "main.html"
AUKI_INDEX_HTML = ROOT / "static" / "auki_bundle" / "index.html"
AUKI_DEMO_HTML = ROOT / "static" / "auki_bundle" / "demo.html"
DEFAULT_SO101_GROUP_FOLLOWER = Path(
    "C:/Users/user/.cache/huggingface/lerobot/calibration/robots/so101_follower/Group_Follower.json"
)

SO101_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_roll",
    "wrist_flex",
    "gripper",
]

GUIDE = [
    {"motor": "motor_1", "name": "Base (Shoulder-Elbow Horizontal)", "how": "Shoulder->elbow line moves left/right (horizontal)"},
    {"motor": "motor_2", "name": "Shoulder (Shoulder-Elbow Vertical)", "how": "Shoulder->elbow line moves up/down (vertical)"},
    {"motor": "motor_3", "name": "Elbow (Elbow-Wrist Vertical)", "how": "Elbow->wrist line moves up/down (vertical)"},
    {"motor": "motor_4", "name": "Wrist Flex", "how": "Human wrist flex up/down controls robot wrist flex"},
    {"motor": "motor_5", "name": "Wrist Roll", "how": "Human wrist roll controls robot wrist roll"},
    {"motor": "motor_6", "name": "Gripper", "how": "Close thumb + fingers to close gripper, open hand to open gripper"},
]

runtime_state = {
    "selected_arm": "right",
    "trims_deg": [0.0] * 6,
    "motor_ranges": [[-160.0, 160.0], [-90.0, 90.0], [-120.0, 120.0], [-180.0, 180.0], [-90.0, 90.0], [0.0, 90.0]],
    "center_offsets": {},
    "calibration_file": str(ROOT / "calibration.runtime.json"),
    "wizard_step": 1,
    "wizard_steps": [
        "Step 1: Choose tracked arm and ensure shoulder-elbow-wrist are visible.",
        "Step 2: Hold neutral pose for 2 seconds and click Calibrate Neutral.",
        "Step 3: Move each limb and verify joint direction/alignment.",
        "Step 4: Test gestures (fist/open/pinch) and safety controls.",
        "Step 5: Record a short trajectory and replay it.",
        "Step 6: Save calibration JSON.",
    ],
    "depth_calibration": {
        "right": {"neutral": None, "near": None, "far": None},
        "left": {"neutral": None, "near": None, "far": None},
    },
}


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {
            "timestamp": time.time(),
            "frame_b64": "",
            "motors_deg": [0.0] * 6,
            "motors_right_deg": [0.0] * 6,
            "motors_left_deg": [0.0] * 6,
            "motor_1": 0.0,
            "motor_2": 0.0,
            "motor_3": 0.0,
            "motor_4": 0.0,
            "motor_5": 0.0,
            "motor_6": 0.0,
            "pose_detected": False,
            "hand_detected": False,
            "pose_detected_right": False,
            "pose_detected_left": False,
            "hand_detected_right": False,
            "hand_detected_left": False,
            "selected_arm": runtime_state["selected_arm"],
            "trims_deg": runtime_state["trims_deg"],
            "center_offsets": runtime_state["center_offsets"],
            "human_arm_world": None,
            "human_arm_world_right": None,
            "human_arm_world_left": None,
            "guide": GUIDE,
            "status": "initializing",
            "macro_event": "",
            "quality": {},
            "metrics": {},
            "safety": {},
            "safety_right": {},
            "safety_left": {},
            "trajectory": {},
            "wizard_step": runtime_state["wizard_step"],
            "collision_blocked": False,
            "depth_right": {},
            "depth_left": {},
        }
        self.running = True
        self.calibrate_request = False
        self.reset_request = False
        self.depth_capture_request = {"right": "", "left": ""}


state = SharedState()
worker_thread: Optional[threading.Thread] = None
trajectory_manager = TrajectoryManager(ROOT / "trajectories")
voice_engine = VoiceCommandEngine()
ros2_bridge = ROS2BridgeStub()
runtime_settings = {
    "camera": 0,
    "config_path": str(DEFAULT_CONFIG),
    "arm_side": "right",
    "host": "127.0.0.1",
    "port": 8010,
}
safety_supervisor: Optional[SafetySupervisor] = None
right_safety_supervisor: Optional[SafetySupervisor] = None
left_safety_supervisor: Optional[SafetySupervisor] = None
gesture_engine: Optional[GestureMacroEngine] = None


def resolve_mediapipe_modules():
    if hasattr(mp, "solutions"):
        return (
            mp.solutions.pose,
            mp.solutions.hands,
            mp.solutions.drawing_utils,
            mp.solutions.drawing_styles,
        )
    try:
        from mediapipe.python.solutions import drawing_styles, drawing_utils, hands, pose

        return pose, hands, drawing_utils, drawing_styles
    except Exception as exc:
        raise RuntimeError(
            "Unsupported mediapipe installation. Reinstall with "
            "`python -m pip install --force-reinstall mediapipe==0.10.14`."
        ) from exc


def load_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "motors" not in cfg or len(cfg["motors"]) != 6:
        raise ValueError("Config must define 6 motors.")
    if "feature_ranges" not in cfg:
        raise ValueError("Config must define feature_ranges.")
    apply_group_follower_calibration(cfg)
    return cfg


def _ticks_to_deg(raw_tick: float) -> float:
    # STS-style 12-bit servo ticks (0..4095) mapped to approximately [-180, +180] degrees.
    return (float(raw_tick) / 4095.0) * 360.0 - 180.0


def apply_group_follower_calibration(cfg: dict) -> None:
    calib_path_raw = cfg.get("robot_calibration_file")
    if calib_path_raw:
        calib_path = Path(str(calib_path_raw))
    else:
        calib_path = DEFAULT_SO101_GROUP_FOLLOWER
    if not calib_path.exists():
        return

    try:
        data = json.loads(calib_path.read_text(encoding="utf-8"))
    except Exception:
        return

    motors = cfg.get("motors")
    if not isinstance(motors, list) or len(motors) != 6:
        return

    for i, joint_name in enumerate(SO101_JOINT_ORDER):
        joint_cfg = data.get(joint_name)
        if not isinstance(joint_cfg, dict):
            continue

        mn = joint_cfg.get("range_min")
        mx = joint_cfg.get("range_max")
        if mn is None or mx is None:
            continue
        try:
            min_deg = _ticks_to_deg(float(mn))
            max_deg = _ticks_to_deg(float(mx))
        except Exception:
            continue
        if not math.isfinite(min_deg) or not math.isfinite(max_deg):
            continue

        lo, hi = (min_deg, max_deg) if min_deg <= max_deg else (max_deg, min_deg)
        motors[i]["min_deg"] = float(round(lo, 3))
        motors[i]["max_deg"] = float(round(hi, 3))

        home = float(motors[i].get("home_deg", (lo + hi) * 0.5))
        motors[i]["home_deg"] = float(max(lo, min(hi, home)))
        motors[i]["calib_joint"] = joint_name
        motors[i]["calib_servo_id"] = int(joint_cfg.get("id", i + 1))
        motors[i]["calib_homing_offset"] = float(joint_cfg.get("homing_offset", 0.0))


def _safe_float_list(values, n=6, default=0.0):
    out = []
    if not isinstance(values, list):
        return [default] * n
    for i in range(n):
        try:
            out.append(float(values[i]))
        except Exception:
            out.append(default)
    return out


def load_saved_calibration(calib_path: str):
    p = Path(calib_path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    with state.lock:
        arm = data.get("selected_arm")
        if arm in ("left", "right"):
            runtime_state["selected_arm"] = arm
        runtime_state["trims_deg"] = _safe_float_list(data.get("trims_deg"), n=6, default=0.0)
        center = data.get("center_offsets")
        if isinstance(center, dict):
            safe_center = {}
            for k, v in center.items():
                try:
                    safe_center[str(k)] = float(v)
                except Exception:
                    continue
            runtime_state["center_offsets"] = safe_center
        depth_cal = data.get("depth_calibration")
        if isinstance(depth_cal, dict):
            for side in ("right", "left"):
                d = depth_cal.get(side)
                if not isinstance(d, dict):
                    continue
                out = {}
                for k in ("neutral", "near", "far"):
                    v = d.get(k)
                    if v is None:
                        out[k] = None
                    else:
                        try:
                            out[k] = float(v)
                        except Exception:
                            out[k] = None
                runtime_state["depth_calibration"][side] = out


def save_calibration_to_file(calib_path: str):
    with state.lock:
        payload = {
            "saved_at": time.time(),
            "selected_arm": runtime_state["selected_arm"],
            "trims_deg": runtime_state["trims_deg"],
            "center_offsets": runtime_state["center_offsets"],
            "depth_calibration": runtime_state.get("depth_calibration", {}),
        }
    p = Path(calib_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def pick_hand_landmarks(hands_result, requested_side: str):
    if not hands_result.multi_hand_landmarks:
        return None
    if not hands_result.multi_handedness:
        return hands_result.multi_hand_landmarks[0].landmark

    req = requested_side.lower()
    for h_lm, h_side in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
        lbl = h_side.classification[0].label.lower()
        if lbl == req:
            return h_lm.landmark
    return hands_result.multi_hand_landmarks[0].landmark


def split_hand_landmarks_by_side(hands_result, pose_landmarks=None):
    out = {"left": None, "right": None}
    if not hands_result.multi_hand_landmarks:
        return out

    hands = list(hands_result.multi_hand_landmarks)
    handed = list(hands_result.multi_handedness or [])

    # 1) Try MediaPipe handedness labels first.
    if handed:
        for h_lm, h_side in zip(hands, handed):
            lbl = str(h_side.classification[0].label).strip().lower()
            if lbl in out and out[lbl] is None:
                out[lbl] = h_lm.landmark

    # 2) Fallback: assign by nearest pose wrist in image space.
    if pose_landmarks is not None and (out["left"] is None or out["right"] is None):
        candidates = []
        for h_lm in hands:
            w = h_lm.landmark[0]
            candidates.append((h_lm.landmark, float(w.x), float(w.y)))

        if candidates:
            lw = pose_landmarks[15]
            rw = pose_landmarks[16]
            left_w = np.array([float(lw.x), float(lw.y)], dtype=float)
            right_w = np.array([float(rw.x), float(rw.y)], dtype=float)

            # Greedy nearest assignment without duplicating one hand to both sides.
            used = set()
            for side_name, side_w in (("left", left_w), ("right", right_w)):
                if out[side_name] is not None:
                    continue
                best_idx = None
                best_d = 1e9
                for i, (_, x, y) in enumerate(candidates):
                    if i in used:
                        continue
                    d = float(np.linalg.norm(np.array([x, y]) - side_w))
                    if d < best_d:
                        best_d = d
                        best_idx = i
                if best_idx is not None:
                    used.add(best_idx)
                    out[side_name] = candidates[best_idx][0]

    # 3) Last resort: fill any missing side with the first available hand.
    if out["left"] is None and out["right"] is not None:
        out["left"] = out["right"]
    if out["right"] is None and out["left"] is not None:
        out["right"] = out["left"]
    if out["left"] is None and hands:
        out["left"] = hands[0].landmark
    if out["right"] is None and hands:
        out["right"] = hands[0].landmark
    return out


def remap_side_for_mirrored_input(side: str, mirrored_input: bool = True) -> str:
    s = str(side).strip().lower()
    if s not in ("left", "right"):
        return "right"
    if not mirrored_input:
        return s
    return "left" if s == "right" else "right"


def _lm_xy(frame, lm):
    h, w = frame.shape[:2]
    return int(lm.x * w), int(lm.y * h), float(getattr(lm, "visibility", 1.0))


def draw_arm_guides(frame, pose_landmarks, hand_landmarks, arm_side: str):
    if pose_landmarks is None:
        return
    side = arm_side.lower()
    shoulder_idx, elbow_idx, wrist_idx = (11, 13, 15) if side == "left" else (12, 14, 16)
    shoulder_l = pose_landmarks[11]
    shoulder_r = pose_landmarks[12]
    elbow = pose_landmarks[elbow_idx]
    wrist_pose = pose_landmarks[wrist_idx]

    sh_l = _lm_xy(frame, shoulder_l)
    sh_r = _lm_xy(frame, shoulder_r)
    el = _lm_xy(frame, elbow)
    wr = _lm_xy(frame, wrist_pose)

    # 1) Required arm lines: shoulder->elbow and elbow->wrist.
    if el[2] > 0.3 and wr[2] > 0.3:
        cv2.line(frame, (el[0], el[1]), (wr[0], wr[1]), (0, 220, 255), 4, cv2.LINE_AA)
    src_sh = sh_l if side == "left" else sh_r
    if src_sh[2] > 0.3 and el[2] > 0.3:
        cv2.line(frame, (src_sh[0], src_sh[1]), (el[0], el[1]), (0, 220, 255), 4, cv2.LINE_AA)

    # 2) Required POIs: 2 shoulders, 1 elbow, 2 wrists, 1 thumb.
    poi = [
        (sh_l[0], sh_l[1]),  # shoulder 1
        (sh_r[0], sh_r[1]),  # shoulder 2
        (el[0], el[1]),      # elbow
        (wr[0], wr[1]),      # wrist from pose
    ]
    if hand_landmarks is not None:
        h, w = frame.shape[:2]
        wrist_hand = hand_landmarks[0]
        thumb_tip = hand_landmarks[4]
        poi.append((int(wrist_hand.x * w), int(wrist_hand.y * h)))  # wrist from hand model
        poi.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))    # thumb
    else:
        # Keep consistent point count even when hand temporarily drops.
        poi.append((wr[0] + 8, wr[1] + 8))
        poi.append((wr[0] + 16, wr[1] + 16))

    for i, p in enumerate(poi):
        color = (255, 255, 255) if i < 4 else (80, 255, 160)
        radius = 8 if i == (0 if side == "left" else 1) else 6
        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)

    # Mapping annotation requested by user:
    # Shoulder->Elbow line controls M1 (horizontal) and M2 (vertical).
    # Elbow->Wrist line controls M3 (vertical).
    m_se = ((src_sh[0] + el[0]) // 2, (src_sh[1] + el[1]) // 2)
    m_ew = ((el[0] + wr[0]) // 2, (el[1] + wr[1]) // 2)
    cv2.putText(frame, "M1 horiz / M2 vert", m_se, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "M3 vert", m_ew, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "BASE",
        (src_sh[0] + 10, src_sh[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_palm_only(frame, hand_landmarks):
    if hand_landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = {}
    ids = [0, 5, 9, 13, 17]
    for idx in ids:
        lm = hand_landmarks[idx]
        pts[idx] = (int(lm.x * w), int(lm.y * h))

    palm_edges = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 0), (0, 9), (5, 17)]
    for a, b in palm_edges:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 235, 180), 3, cv2.LINE_AA)
    for idx in ids:
        if idx in pts:
            cv2.circle(frame, pts[idx], 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_overlay(frame, motors, pose_ok, hand_ok, selected_arm, safety_snapshot=None, quality=None):
    safety_snapshot = safety_snapshot or {}
    quality = quality or {}
    lines = [
        f"arm_side: {selected_arm}",
        f"pose_detected: {pose_ok}",
        f"hand_detected: {hand_ok}",
        f"freeze: {bool(safety_snapshot.get('freeze', False))}",
        f"estop: {bool(safety_snapshot.get('estop', False))}",
    ]
    if quality.get("low_light", False):
        lines.append("warning: low_light")
    if quality.get("blurry", False):
        lines.append("warning: blurry")
    lines += [f"m{i+1}: {motors[i]:+06.1f} deg" for i in range(6)]
    for i, txt in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 255, 70),
            2,
            cv2.LINE_AA,
        )


def _to_rad(d: float) -> float:
    return float(d) * math.pi / 180.0


def _arm_key_points(motors_deg, base_xyz):
    # Kinematic proxy used for self-collision checks.
    m1, m2, m3 = [float(v) for v in motors_deg[:3]]
    yaw = _to_rad(m1)
    sh = _to_rad(m2)
    el = _to_rad(m3)
    l1 = 0.34
    l2 = 0.33
    l3 = 0.16
    sx, sy, sz = base_xyz

    d1 = (
        math.cos(yaw) * math.cos(sh),
        math.sin(sh),
        math.sin(yaw) * math.cos(sh),
    )
    ex = sx + l1 * d1[0]
    ey = sy + l1 * d1[1]
    ez = sz + l1 * d1[2]

    fore_pitch = sh - el
    d2 = (
        math.cos(yaw) * math.cos(fore_pitch),
        math.sin(fore_pitch),
        math.sin(yaw) * math.cos(fore_pitch),
    )
    wx = ex + l2 * d2[0]
    wy = ey + l2 * d2[1]
    wz = ez + l2 * d2[2]
    tx = wx + l3 * d2[0]
    ty = wy + l3 * d2[1]
    tz = wz + l3 * d2[2]
    return {
        "base": (sx, sy, sz),
        "elbow": (ex, ey, ez),
        "wrist": (wx, wy, wz),
        "tool": (tx, ty, tz),
    }


def _dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _segment_distance(a0, a1, b0, b1):
    # Closest distance between 3D line segments.
    p1 = np.array(a0, dtype=float)
    q1 = np.array(a1, dtype=float)
    p2 = np.array(b0, dtype=float)
    q2 = np.array(b1, dtype=float)
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    den = max(1e-9, a * c - b * b)

    s = clamp((b * e - c * d) / den, 0.0, 1.0)
    t = clamp((a * e - b * d) / den, 0.0, 1.0)

    cp1 = p1 + s * u
    cp2 = p2 + t * v
    return float(np.linalg.norm(cp1 - cp2))


def _arms_min_distance(right_motors, left_motors, right_base, left_base):
    rp = _arm_key_points(right_motors, right_base)
    lp = _arm_key_points(left_motors, left_base)
    rseg = [
        (rp["base"], rp["elbow"]),
        (rp["elbow"], rp["wrist"]),
        (rp["wrist"], rp["tool"]),
    ]
    lseg = [
        (lp["base"], lp["elbow"]),
        (lp["elbow"], lp["wrist"]),
        (lp["wrist"], lp["tool"]),
    ]
    d_min = 1e9
    for sa0, sa1 in rseg:
        for sb0, sb1 in lseg:
            d_min = min(d_min, _segment_distance(sa0, sa1, sb0, sb1))
    return float(d_min)


def _motor_delta_norm(a, b):
    return float(math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(min(len(a), len(b), 6)))))


def collision_limit_guard(
    right_motors,
    left_motors,
    prev_right,
    prev_left,
    min_sep_m=0.16,
):
    # Two-arm anti-collision with segment distances and directional blocking.
    right_base = (0.24, 0.06, 0.0)
    left_base = (-0.24, 0.06, 0.0)
    sep_limit = float(min_sep_m)
    hysteresis = 0.01

    d_new = _arms_min_distance(right_motors, left_motors, right_base, left_base)
    d_prev = _arms_min_distance(prev_right, prev_left, right_base, left_base)

    if d_new >= (sep_limit + hysteresis):
        return right_motors, left_motors, False

    # Evaluate candidate responses and choose the safest valid one.
    candidates = [
        ("keep", right_motors, left_motors, d_new),
        ("freeze_right", prev_right, left_motors, _arms_min_distance(prev_right, left_motors, right_base, left_base)),
        ("freeze_left", right_motors, prev_left, _arms_min_distance(right_motors, prev_left, right_base, left_base)),
        ("freeze_both", prev_right, prev_left, _arms_min_distance(prev_right, prev_left, right_base, left_base)),
    ]

    # Prefer solutions that clear separation; tie-break by larger final separation.
    viable = [c for c in candidates if c[3] >= sep_limit]
    if viable:
        best = max(viable, key=lambda x: x[3])
        blocked = best[0] != "keep"
        return best[1][:], best[2][:], blocked

    # If none fully clear, avoid motions that continue reducing separation.
    moving_r = _motor_delta_norm(right_motors, prev_right)
    moving_l = _motor_delta_norm(left_motors, prev_left)
    if d_new < d_prev:
        if moving_r > moving_l * 1.15:
            return prev_right[:], left_motors[:], True
        if moving_l > moving_r * 1.15:
            return right_motors[:], prev_left[:], True
        return prev_right[:], prev_left[:], True

    # Already close but not approaching further: keep current command.
    return right_motors, left_motors, False


class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / max(1e-6, dt))

    def filter(self, x: float, t: float) -> float:
        x = float(x)
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = float(t)
            return x
        dt = max(1e-6, float(t) - float(self.t_prev))
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat


def depth_components(pose_landmarks, hand_landmarks, arm_side: str):
    if pose_landmarks is None:
        return None
    side = arm_side.lower()
    shoulder_idx, wrist_idx = (11, 15) if side == "left" else (12, 16)
    try:
        shoulder = pose_landmarks[shoulder_idx]
        wrist_pose = pose_landmarks[wrist_idx]
    except Exception:
        return None
    if getattr(shoulder, "visibility", 0.0) < 0.25 or getattr(wrist_pose, "visibility", 0.0) < 0.25:
        return None

    sw_scale = math.hypot(float(wrist_pose.x) - float(shoulder.x), float(wrist_pose.y) - float(shoulder.y))
    mp_z = -float(wrist_pose.z)

    return {
        "mediapipe_z": float(mp_z),
        "shoulder_wrist_scale": float(sw_scale),
    }


def normalize_depth_with_calibration(depth_raw: float, depth_calib: dict):
    neutral = depth_calib.get("neutral")
    near = depth_calib.get("near")
    far = depth_calib.get("far")
    if near is not None and far is not None and abs(float(near) - float(far)) > 1e-6:
        num = float(depth_raw) - float(far)
        den = float(near) - float(far)
        norm = clamp(num / den, 0.0, 1.0)
    elif neutral is not None:
        # fallback symmetric normalization around neutral
        norm = clamp(0.5 + (float(depth_raw) - float(neutral)) * 1.8, 0.0, 1.0)
    else:
        norm = 0.5
    forward = clamp((norm - 0.5) * 2.0, -1.0, 1.0)
    return float(norm), float(forward)


def camera_worker(camera_idx: int, config_path: str):
    global safety_supervisor, right_safety_supervisor, left_safety_supervisor, gesture_engine
    cfg = load_config(config_path)
    mapper_right = ArmFollowMapper(cfg)
    mapper_left = ArmFollowMapper(cfg)
    right_safety = SafetySupervisor(cfg)
    left_safety = SafetySupervisor(cfg)
    right_safety_supervisor = right_safety
    left_safety_supervisor = left_safety
    safety_supervisor = right_safety
    gesture_engine = GestureMacroEngine(cfg)
    metrics = RuntimeMetrics(window=int(cfg.get("metrics_window", 120)))
    draw_camera_overlay = bool(cfg.get("draw_camera_overlay", False))
    draw_pose_overlay = bool(cfg.get("draw_pose_overlay", draw_camera_overlay))
    draw_hand_landmarks = bool(cfg.get("draw_hand_landmarks", True))
    draw_text_overlay = bool(cfg.get("draw_text_overlay", draw_camera_overlay))
    mirror_side_labels = bool(cfg.get("mirror_side_labels", False))
    min_collision_sep_m = float(cfg.get("collision_min_sep_m", 0.16))
    depth_cfg = cfg.get("depth_fusion", {}) or {}
    depth_enabled = bool(depth_cfg.get("enabled", True))
    depth_w = depth_cfg.get("weights", {}) or {}
    w_mp = float(depth_w.get("mediapipe_z", 0.56))
    w_sw = float(depth_w.get("shoulder_wrist_scale", 0.44))
    ws = max(1e-6, w_mp + w_sw)
    w_mp, w_sw = w_mp / ws, w_sw / ws
    oe = depth_cfg.get("one_euro", {}) or {}
    depth_filter_right = OneEuroFilter(
        min_cutoff=float(oe.get("min_cutoff", 1.2)),
        beta=float(oe.get("beta", 0.03)),
        d_cutoff=float(oe.get("d_cutoff", 1.0)),
    )
    depth_filter_left = OneEuroFilter(
        min_cutoff=float(oe.get("min_cutoff", 1.2)),
        beta=float(oe.get("beta", 0.03)),
        d_cutoff=float(oe.get("d_cutoff", 1.0)),
    )
    depth_shoulder_gain = float(depth_cfg.get("shoulder_gain_deg", 8.0))
    depth_elbow_gain = float(depth_cfg.get("elbow_gain_deg", -20.0))
    depth_apply_joint_coupling = bool(
        depth_cfg.get(
            "apply_to_shoulder_elbow",
            (mapper_right.control_mode == "palm_follow" or mapper_left.control_mode == "palm_follow"),
        )
    )

    pose_vis_min = float(cfg.get("pose_visibility_threshold", 0.55))
    quality_low_light = float(cfg.get("quality_low_light_threshold", 45.0))
    quality_blur = float(cfg.get("quality_blur_threshold", 35.0))
    with state.lock:
        for k, v in runtime_state["center_offsets"].items():
            if k in mapper_right.center_offsets:
                mapper_right.center_offsets[k] = float(v)
            if k in mapper_left.center_offsets:
                mapper_left.center_offsets[k] = float(v)
    mp_pose, mp_hands, mp_draw, mp_style = resolve_mediapipe_modules()

    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        with state.lock:
            state.latest["status"] = f"camera_open_failed(index={camera_idx})"
        return

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        prev_loop = time.time()
        prev_right = mapper_right.home[:]
        prev_left = mapper_left.home[:]
        while state.running:
            t_loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)
            quality = evaluate_frame_quality(
                frame,
                low_light_thresh=quality_low_light,
                blur_thresh=quality_blur,
            )

            with state.lock:
                selected_arm = runtime_state["selected_arm"]
                trims = runtime_state["trims_deg"][:]
                wizard_step = runtime_state["wizard_step"]
                reset_request = state.reset_request
                depth_req = dict(state.depth_capture_request)
                depth_cal = {
                    "right": dict(runtime_state.get("depth_calibration", {}).get("right", {})),
                    "left": dict(runtime_state.get("depth_calibration", {}).get("left", {})),
                }

            model_right_side = remap_side_for_mirrored_input("right", mirrored_input=mirror_side_labels)
            model_left_side = remap_side_for_mirrored_input("left", mirrored_input=mirror_side_labels)

            pose_landmarks = (
                pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
            )
            hands_by_side = split_hand_landmarks_by_side(hands_res, pose_landmarks=pose_landmarks)
            hand_right = hands_by_side.get(model_right_side)
            hand_left = hands_by_side.get(model_left_side)

            features_right, pose_ok_right, hand_ok_right, human_arm_world_right = extract_arm_features(
                pose_landmarks=pose_landmarks,
                hand_landmarks=hand_right,
                arm_side=model_right_side,
                pose_vis_min=pose_vis_min,
            )
            features_left, pose_ok_left, hand_ok_left, human_arm_world_left = extract_arm_features(
                pose_landmarks=pose_landmarks,
                hand_landmarks=hand_left,
                arm_side=model_left_side,
                pose_vis_min=pose_vis_min,
            )

            depth_debug_right = {}
            depth_debug_left = {}
            now_ts = time.time()
            if depth_enabled:
                comps_r = depth_components(pose_landmarks, hand_right, model_right_side)
                comps_l = depth_components(pose_landmarks, hand_left, model_left_side)

                if comps_r is not None:
                    raw_r = (
                        w_mp * comps_r["mediapipe_z"]
                        + w_sw * comps_r["shoulder_wrist_scale"]
                    )
                    filt_r = depth_filter_right.filter(raw_r, now_ts)
                    if depth_req.get("right") in ("neutral", "near", "far") and pose_ok_right:
                        with state.lock:
                            runtime_state["depth_calibration"]["right"][depth_req["right"]] = float(filt_r)
                            state.depth_capture_request["right"] = ""
                            depth_cal["right"] = dict(runtime_state["depth_calibration"]["right"])
                    norm_r, forward_r = normalize_depth_with_calibration(filt_r, depth_cal["right"])
                    features_right["depth_norm"] = float(norm_r)
                    features_right["depth_forward"] = float(forward_r)
                    if depth_apply_joint_coupling and pose_ok_right:
                        features_right["shoulder_pitch"] = float(features_right.get("shoulder_pitch", 0.0)) + (
                            depth_shoulder_gain * forward_r
                        )
                        features_right["elbow_bend"] = float(features_right.get("elbow_bend", 0.0)) + (
                            depth_elbow_gain * forward_r
                        )
                        fr = cfg.get("feature_ranges", {})
                        if "shoulder_pitch" in fr:
                            features_right["shoulder_pitch"] = clamp(
                                features_right["shoulder_pitch"],
                                float(fr["shoulder_pitch"]["min"]),
                                float(fr["shoulder_pitch"]["max"]),
                            )
                        if "elbow_bend" in fr:
                            features_right["elbow_bend"] = clamp(
                                features_right["elbow_bend"],
                                float(fr["elbow_bend"]["min"]),
                                float(fr["elbow_bend"]["max"]),
                            )
                    depth_debug_right = {
                        "raw": float(raw_r),
                        "filtered": float(filt_r),
                        "norm": float(norm_r),
                        "forward": float(forward_r),
                        "components": comps_r,
                        "calibration": depth_cal["right"],
                    }

                if comps_l is not None:
                    raw_l = (
                        w_mp * comps_l["mediapipe_z"]
                        + w_sw * comps_l["shoulder_wrist_scale"]
                    )
                    filt_l = depth_filter_left.filter(raw_l, now_ts)
                    if depth_req.get("left") in ("neutral", "near", "far") and pose_ok_left:
                        with state.lock:
                            runtime_state["depth_calibration"]["left"][depth_req["left"]] = float(filt_l)
                            state.depth_capture_request["left"] = ""
                            depth_cal["left"] = dict(runtime_state["depth_calibration"]["left"])
                    norm_l, forward_l = normalize_depth_with_calibration(filt_l, depth_cal["left"])
                    features_left["depth_norm"] = float(norm_l)
                    features_left["depth_forward"] = float(forward_l)
                    if depth_apply_joint_coupling and pose_ok_left:
                        features_left["shoulder_pitch"] = float(features_left.get("shoulder_pitch", 0.0)) + (
                            depth_shoulder_gain * forward_l
                        )
                        features_left["elbow_bend"] = float(features_left.get("elbow_bend", 0.0)) + (
                            depth_elbow_gain * forward_l
                        )
                        fr = cfg.get("feature_ranges", {})
                        if "shoulder_pitch" in fr:
                            features_left["shoulder_pitch"] = clamp(
                                features_left["shoulder_pitch"],
                                float(fr["shoulder_pitch"]["min"]),
                                float(fr["shoulder_pitch"]["max"]),
                            )
                        if "elbow_bend" in fr:
                            features_left["elbow_bend"] = clamp(
                                features_left["elbow_bend"],
                                float(fr["elbow_bend"]["min"]),
                                float(fr["elbow_bend"]["max"]),
                            )
                    depth_debug_left = {
                        "raw": float(raw_l),
                        "filtered": float(filt_l),
                        "norm": float(norm_l),
                        "forward": float(forward_l),
                        "components": comps_l,
                        "calibration": depth_cal["left"],
                    }

            macro_event = ""
            if gesture_engine is not None:
                selected_hand_landmarks = hand_right if selected_arm == "right" else hand_left
                macro = gesture_engine.process(selected_hand_landmarks)
                if macro:
                    right_safety.trigger_macro(macro)
                    left_safety.trigger_macro(macro)
                    macro_event = macro

            with state.lock:
                if state.calibrate_request:
                    if selected_arm == "right" and pose_ok_right:
                        mapper_right.calibrate(features_right)
                        runtime_state["center_offsets"] = dict(mapper_right.center_offsets)
                        state.calibrate_request = False
                    elif selected_arm == "left" and pose_ok_left:
                        mapper_left.calibrate(features_left)
                        runtime_state["center_offsets"] = dict(mapper_left.center_offsets)
                        state.calibrate_request = False
                if reset_request:
                    state.reset_request = False

            if reset_request:
                mapper_right.prev = mapper_right.home[:]
                mapper_left.prev = mapper_left.home[:]
                right_safety.last_safe = mapper_right.home[:]
                left_safety.last_safe = mapper_left.home[:]
                right_safety.lost_count = 0
                left_safety.lost_count = 0

            mapped_right = mapper_right.map(
                features=features_right,
                pose_ok=pose_ok_right,
                runtime_trims=trims,
            )
            mapped_left = mapper_left.map(
                features=features_left,
                pose_ok=pose_ok_left,
                runtime_trims=trims,
            )

            playback = trajectory_manager.sample_playback()
            if playback is not None:
                if selected_arm == "right":
                    mapped_right = playback
                else:
                    mapped_left = playback

            safe_right = right_safety.process(
                target_motors=mapped_right,
                pose_ok=pose_ok_right,
                hand_ok=hand_ok_right,
                quality=quality,
            )
            safe_left = left_safety.process(
                target_motors=mapped_left,
                pose_ok=pose_ok_left,
                hand_ok=hand_ok_left,
                quality=quality,
            )
            safe_right, safe_left, collision_blocked = collision_limit_guard(
                safe_right,
                safe_left,
                prev_right=prev_right,
                prev_left=prev_left,
                min_sep_m=min_collision_sep_m,
            )
            prev_right = safe_right[:]
            prev_left = safe_left[:]

            selected_safe = safe_right if selected_arm == "right" else safe_left
            selected_pose_ok = pose_ok_right if selected_arm == "right" else pose_ok_left
            selected_hand_ok = hand_ok_right if selected_arm == "right" else hand_ok_left

            if trajectory_manager.recording:
                trajectory_manager.append(selected_safe)

            if ros2_bridge.connected:
                ros2_bridge.publish_joint_targets(selected_safe)

            if draw_pose_overlay:
                draw_arm_guides(frame, pose_landmarks, hand_right, model_right_side)
                draw_arm_guides(frame, pose_landmarks, hand_left, model_left_side)
            if draw_hand_landmarks and hands_res.multi_hand_landmarks:
                for h_lm in hands_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        h_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )
            if draw_text_overlay:
                _draw_overlay(
                    frame,
                    selected_safe,
                    selected_pose_ok,
                    selected_hand_ok,
                    selected_arm,
                    safety_snapshot=(
                        right_safety.snapshot() if selected_arm == "right" else left_safety.snapshot()
                    ),
                    quality=quality,
                )

            ok_jpg, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok_jpg:
                continue
            frame_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

            now = time.time()
            dt_s = max(1e-6, now - prev_loop)
            prev_loop = now
            proc_ms = (now - t_loop_start) * 1000.0
            metrics.update(
                dt_s=dt_s,
                proc_ms=proc_ms,
                motors=selected_safe,
                pose_ok=selected_pose_ok,
                quality_score=quality["quality_score"],
            )

            payload = {
                "timestamp": time.time(),
                "frame_b64": frame_b64,
                "motors_deg": [round(v, 3) for v in selected_safe],
                "motors_right_deg": [round(v, 3) for v in safe_right],
                "motors_left_deg": [round(v, 3) for v in safe_left],
                "pose_detected": bool(selected_pose_ok),
                "hand_detected": bool(selected_hand_ok),
                "pose_detected_right": bool(pose_ok_right),
                "pose_detected_left": bool(pose_ok_left),
                "hand_detected_right": bool(hand_ok_right),
                "hand_detected_left": bool(hand_ok_left),
                "selected_arm": selected_arm,
                "trims_deg": trims,
                "center_offsets": dict(
                    mapper_right.center_offsets if selected_arm == "right" else mapper_left.center_offsets
                ),
                "human_arm_world": (
                    human_arm_world_right if selected_arm == "right" else human_arm_world_left
                ),
                "human_arm_world_right": human_arm_world_right,
                "human_arm_world_left": human_arm_world_left,
                "guide": GUIDE,
                "status": "running",
                "macro_event": macro_event
                or (
                    right_safety.snapshot().get("last_macro", "")
                    if selected_arm == "right"
                    else left_safety.snapshot().get("last_macro", "")
                ),
                "quality": quality,
                "metrics": metrics.snapshot(),
                "safety": right_safety.snapshot() if selected_arm == "right" else left_safety.snapshot(),
                "safety_right": right_safety.snapshot(),
                "safety_left": left_safety.snapshot(),
                "trajectory": trajectory_manager.snapshot(),
                "wizard_step": wizard_step,
                "collision_blocked": bool(collision_blocked),
                "depth_right": depth_debug_right,
                "depth_left": depth_debug_left,
            }
            for i, v in enumerate(payload["motors_deg"], start=1):
                payload[f"motor_{i}"] = v

            with state.lock:
                state.latest = payload

            time.sleep(0.001)

    cap.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_thread
    state.running = True
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(
            target=camera_worker,
            args=(
                int(runtime_settings["camera"]),
                runtime_settings["config_path"],
            ),
            daemon=True,
        )
        worker_thread.start()
    try:
        yield
    finally:
        state.running = False


app = FastAPI(title="Clutch Web Teleop V3", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/main", response_class=HTMLResponse)
async def landing_page():
    return HTMLResponse(MAIN_HTML.read_text(encoding="utf-8"))


@app.get("/hand-tracking-manipulation", response_class=HTMLResponse)
@app.get("/teleop", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/auki", response_class=HTMLResponse)
async def auki_index():
    return HTMLResponse(AUKI_INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/auki-demo", response_class=HTMLResponse)
async def auki_demo():
    return HTMLResponse(AUKI_DEMO_HTML.read_text(encoding="utf-8"))


@app.get("/api/guide")
async def api_guide():
    return JSONResponse({"guide": GUIDE})


@app.get("/api/health")
async def api_health():
    with state.lock:
        latest = dict(state.latest)
    safety = safety_supervisor.snapshot() if safety_supervisor else {}
    safety_r = right_safety_supervisor.snapshot() if right_safety_supervisor else {}
    safety_l = left_safety_supervisor.snapshot() if left_safety_supervisor else {}
    return JSONResponse(
        {
            "ok": True,
            "status": latest.get("status", "unknown"),
            "pose_detected": bool(latest.get("pose_detected", False)),
            "hand_detected": bool(latest.get("hand_detected", False)),
            "pose_detected_right": bool(latest.get("pose_detected_right", False)),
            "pose_detected_left": bool(latest.get("pose_detected_left", False)),
            "hand_detected_right": bool(latest.get("hand_detected_right", False)),
            "hand_detected_left": bool(latest.get("hand_detected_left", False)),
            "selected_arm": runtime_state.get("selected_arm", "right"),
            "safety": safety,
            "safety_right": safety_r,
            "safety_left": safety_l,
        }
    )


@app.get("/api/settings")
async def api_settings_get():
    with state.lock:
        sel = runtime_state["selected_arm"]
        payload = {
            "selected_arm": sel,
            "trims_deg": runtime_state["trims_deg"],
            "motor_ranges": runtime_state["motor_ranges"],
            "center_offsets": runtime_state["center_offsets"],
            "depth_calibration": runtime_state.get("depth_calibration", {}),
            "calibration_file": runtime_state["calibration_file"],
            "wizard_step": runtime_state["wizard_step"],
            "wizard_steps": runtime_state["wizard_steps"],
        }
    payload["trajectory"] = trajectory_manager.snapshot()
    payload["safety"] = (
        right_safety_supervisor.snapshot() if sel == "right" else left_safety_supervisor.snapshot()
    ) if (right_safety_supervisor and left_safety_supervisor) else {}
    payload["safety_right"] = right_safety_supervisor.snapshot() if right_safety_supervisor else {}
    payload["safety_left"] = left_safety_supervisor.snapshot() if left_safety_supervisor else {}
    return JSONResponse(payload)


@app.post("/api/settings")
async def api_settings_set(payload: dict):
    arm = payload.get("selected_arm")
    with state.lock:
        if arm in ("left", "right"):
            runtime_state["selected_arm"] = arm
    return JSONResponse({"ok": True, "selected_arm": runtime_state["selected_arm"]})


@app.post("/api/trims")
async def api_trims_set(payload: dict):
    trims = payload.get("trims_deg")
    if not isinstance(trims, list) or len(trims) != 6:
        return JSONResponse({"ok": False, "error": "trims_deg must be 6 values"}, status_code=400)
    safe = []
    for v in trims:
        try:
            safe.append(float(v))
        except Exception:
            safe.append(0.0)
    with state.lock:
        runtime_state["trims_deg"] = safe
    return JSONResponse({"ok": True, "trims_deg": safe})


@app.get("/api/safety")
async def api_safety_get():
    snap = safety_supervisor.snapshot() if safety_supervisor is not None else {}
    snap_r = right_safety_supervisor.snapshot() if right_safety_supervisor is not None else {}
    snap_l = left_safety_supervisor.snapshot() if left_safety_supervisor is not None else {}
    return JSONResponse({"ok": True, "safety": snap, "safety_right": snap_r, "safety_left": snap_l})


@app.post("/api/safety")
async def api_safety_set(payload: dict):
    if right_safety_supervisor is None or left_safety_supervisor is None:
        return JSONResponse({"ok": False, "error": "safety_not_ready"}, status_code=503)
    action = str(payload.get("action", "")).strip().lower()
    safety_targets = [right_safety_supervisor, left_safety_supervisor]
    if action == "toggle_freeze":
        for s in safety_targets:
            s.trigger_macro("toggle_freeze")
    elif action == "toggle_estop":
        for s in safety_targets:
            s.trigger_macro("toggle_estop")
    elif action == "clear_estop":
        for s in safety_targets:
            s.clear_estop()
    elif action == "home":
        for s in safety_targets:
            s.trigger_macro("home")
    elif action == "freeze_on":
        for s in safety_targets:
            s.set_freeze(True)
    elif action == "freeze_off":
        for s in safety_targets:
            s.set_freeze(False)
    else:
        return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)
    sel = runtime_state.get("selected_arm", "right")
    selected_snap = (
        right_safety_supervisor.snapshot() if sel == "right" else left_safety_supervisor.snapshot()
    )
    return JSONResponse(
        {
            "ok": True,
            "safety": selected_snap,
            "safety_right": right_safety_supervisor.snapshot(),
            "safety_left": left_safety_supervisor.snapshot(),
        }
    )


@app.post("/api/reset")
async def api_reset():
    if right_safety_supervisor is not None:
        right_safety_supervisor.set_freeze(False)
        right_safety_supervisor.clear_estop()
        right_safety_supervisor.trigger_macro("home")
    if left_safety_supervisor is not None:
        left_safety_supervisor.set_freeze(False)
        left_safety_supervisor.clear_estop()
        left_safety_supervisor.trigger_macro("home")
    with state.lock:
        state.reset_request = True
    return JSONResponse({"ok": True})


@app.get("/api/trajectory")
async def api_trajectory_status():
    return JSONResponse({"ok": True, "trajectory": trajectory_manager.snapshot()})


@app.post("/api/trajectory/start")
async def api_trajectory_start(payload: dict):
    name = str(payload.get("name", "")).strip()
    if not name:
        name = f"traj_{int(time.time())}"
    trajectory_manager.start_record(name)
    return JSONResponse({"ok": True, "trajectory": trajectory_manager.snapshot()})


@app.post("/api/trajectory/stop")
async def api_trajectory_stop():
    out = trajectory_manager.stop_record()
    return JSONResponse({"ok": True, "saved": str(out) if out else None, "trajectory": trajectory_manager.snapshot()})


@app.post("/api/trajectory/play")
async def api_trajectory_play(payload: dict):
    name = str(payload.get("name", "")).strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name_required"}, status_code=400)
    ok = trajectory_manager.start_playback(name)
    return JSONResponse({"ok": ok, "trajectory": trajectory_manager.snapshot()})


@app.post("/api/trajectory/stop_playback")
async def api_trajectory_stop_playback():
    trajectory_manager.stop_playback()
    return JSONResponse({"ok": True, "trajectory": trajectory_manager.snapshot()})


@app.post("/api/wizard")
async def api_wizard(payload: dict):
    action = str(payload.get("action", "")).strip().lower()
    with state.lock:
        if action == "next":
            runtime_state["wizard_step"] = min(len(runtime_state["wizard_steps"]), runtime_state["wizard_step"] + 1)
        elif action == "prev":
            runtime_state["wizard_step"] = max(1, runtime_state["wizard_step"] - 1)
        elif action == "reset":
            runtime_state["wizard_step"] = 1
        elif action.startswith("goto:"):
            try:
                idx = int(action.split(":", 1)[1])
                runtime_state["wizard_step"] = max(1, min(len(runtime_state["wizard_steps"]), idx))
            except Exception:
                pass
        payload_out = {
            "step": runtime_state["wizard_step"],
            "total": len(runtime_state["wizard_steps"]),
            "text": runtime_state["wizard_steps"][runtime_state["wizard_step"] - 1],
        }
    return JSONResponse({"ok": True, "wizard": payload_out})


@app.get("/api/wizard")
async def api_wizard_get():
    with state.lock:
        payload_out = {
            "step": runtime_state["wizard_step"],
            "total": len(runtime_state["wizard_steps"]),
            "text": runtime_state["wizard_steps"][runtime_state["wizard_step"] - 1],
        }
    return JSONResponse({"ok": True, "wizard": payload_out})


@app.post("/api/voice")
async def api_voice(payload: dict):
    action = str(payload.get("action", "")).strip().lower()
    if action == "start":
        voice_engine.start()
    elif action == "stop":
        voice_engine.stop()
    elif action == "command":
        macro = voice_engine.push_text_command(str(payload.get("text", "")))
        if macro and safety_supervisor is not None:
            safety_supervisor.trigger_macro(macro)
    else:
        return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)
    return JSONResponse({"ok": True, "enabled": voice_engine.enabled, "last_command": voice_engine.last_command})


@app.post("/api/ros2")
async def api_ros2(payload: dict):
    action = str(payload.get("action", "")).strip().lower()
    if action == "connect":
        ros2_bridge.connect()
    elif action == "disconnect":
        ros2_bridge.disconnect()
    else:
        return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)
    return JSONResponse({"ok": True, "connected": ros2_bridge.connected})


@app.post("/api/validation/run")
async def api_validation_run(payload: dict):
    sim_name = str(payload.get("sim_name", "")).strip()
    real_name = str(payload.get("real_name", "")).strip()
    if not sim_name or not real_name:
        return JSONResponse({"ok": False, "error": "sim_name_and_real_name_required"}, status_code=400)
    sim_path = ROOT / "trajectories" / f"{sim_name}.json"
    real_path = ROOT / "trajectories" / f"{real_name}.json"
    out_path = ROOT / "reports" / f"validation_{sim_name}_vs_{real_name}.json"
    if not sim_path.exists() or not real_path.exists():
        return JSONResponse({"ok": False, "error": "trajectory_file_missing"}, status_code=404)
    report = generate_sim_real_report(str(sim_path), str(real_path), str(out_path))
    return JSONResponse({"ok": True, "report_path": str(out_path), "report": report})


@app.post("/api/calibration/save")
async def api_calibration_save():
    with state.lock:
        calib_file = runtime_state["calibration_file"]
    try:
        payload = save_calibration_to_file(calib_file)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "calibration_file": calib_file, "data": payload})


@app.post("/api/calibrate")
async def api_calibrate():
    with state.lock:
        state.calibrate_request = True
    return JSONResponse({"ok": True, "message": "Calibration requested."})


@app.get("/api/depth_calibration")
async def api_depth_calibration_get():
    with state.lock:
        out = {
            "selected_arm": runtime_state["selected_arm"],
            "depth_calibration": runtime_state.get("depth_calibration", {}),
            "pending_capture": dict(state.depth_capture_request),
        }
    return JSONResponse({"ok": True, **out})


@app.post("/api/depth_calibration")
async def api_depth_calibration_set(payload: dict):
    action = str(payload.get("action", "")).strip().lower()
    side = str(payload.get("side", "")).strip().lower()
    with state.lock:
        if side not in ("left", "right"):
            side = runtime_state["selected_arm"]
        if action in ("capture_neutral", "capture_near", "capture_far"):
            key = action.split("_", 1)[1]
            state.depth_capture_request[side] = key
            out = {
                "depth_calibration": runtime_state.get("depth_calibration", {}),
                "pending_capture": dict(state.depth_capture_request),
            }
            return JSONResponse({"ok": True, **out})
        if action == "reset":
            runtime_state["depth_calibration"][side] = {"neutral": None, "near": None, "far": None}
            state.depth_capture_request[side] = ""
            out = {
                "depth_calibration": runtime_state.get("depth_calibration", {}),
                "pending_capture": dict(state.depth_capture_request),
            }
            return JSONResponse({"ok": True, **out})
    return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)


@app.websocket("/ws")
async def ws_feed(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(1 / 30)
            with state.lock:
                payload = state.latest
            await ws.send_json(payload)
    except WebSocketDisconnect:
        return


def parse_cli_args():
    p = argparse.ArgumentParser(description="Run web teleop v3 server.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index.")
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config.web_demo.json",
    )
    p.add_argument(
        "--arm-side",
        choices=["left", "right"],
        default="right",
        help="Which human arm to map.",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    return p.parse_args()


def main():
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install with: python -m pip install uvicorn[standard]"
        ) from exc

    args = parse_cli_args()
    runtime_settings["camera"] = args.camera
    runtime_settings["config_path"] = str(Path(args.config).resolve())
    runtime_settings["arm_side"] = args.arm_side
    runtime_settings["host"] = args.host
    runtime_settings["port"] = args.port
    cfg = load_config(runtime_settings["config_path"])
    runtime_state["selected_arm"] = args.arm_side
    runtime_state["motor_ranges"] = [
        [float(m.get("min_deg", -180.0)), float(m.get("max_deg", 180.0))]
        for m in cfg.get("motors", [])
    ]
    calib_rel = cfg.get("calibration_file", str(ROOT / "calibration.runtime.json"))
    runtime_state["calibration_file"] = str((Path(calib_rel) if Path(calib_rel).is_absolute() else (PROJECT_ROOT / calib_rel)).resolve())
    load_saved_calibration(runtime_state["calibration_file"])

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
