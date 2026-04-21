import argparse
import asyncio
import base64
import json
import sys
import threading
import time
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

from web_teleop_v3.camera_quality import evaluate_frame_quality
from web_teleop_v3.ee_target_mapper import EETargetMapper
from web_teleop_v3.gesture_macros import GestureMacroEngine
from web_teleop_v3.hand_pose_tracker import HandPoseTracker
from web_teleop_v3.ik_solver import DampedLeastSquaresIKSolver
from web_teleop_v3.ros2_bridge_stub import ROS2BridgeStub
from web_teleop_v3.runtime_metrics import RuntimeMetrics
from web_teleop_v3.safety_supervisor import SafetySupervisor
from web_teleop_v3.session_logger import SessionLogger
from web_teleop_v3.simulation_modes import SimulationModes
from web_teleop_v3.target_visualizer import build_debug_lines
from web_teleop_v3.trajectory_manager import TrajectoryManager
from web_teleop_v3.validation_report import generate_sim_real_report
from web_teleop_v3.voice_commands import VoiceCommandEngine

ROOT = THIS_DIR
DEFAULT_CONFIG = ROOT / "config.web_demo.json"
INDEX_HTML = ROOT / "static" / "index.html"
MAIN_HTML = ROOT / "static" / "main.html"

GUIDE = [
    {"motor": "ee_xyz", "name": "End-Effector Position", "how": "Move your palm in 3D space"},
    {"motor": "motor_1..motor_5", "name": "Arm Joints", "how": "Auto-solved by IK from EE target"},
    {"motor": "motor_6", "name": "Gripper", "how": "Pinch to close, release to open"},
    {"motor": "calibrate", "name": "Neutral Calibration", "how": "Set current palm center as workspace center"},
]

runtime_state = {
    "selected_arm": "right",
    "trims_deg": [0.0] * 6,
    "motor_ranges": [[-160.0, 160.0], [-90.0, 90.0], [-120.0, 120.0], [-180.0, 180.0], [-90.0, 90.0], [0.0, 90.0]],
    "center_offsets": {},
    "depth_calibration": {
        "right": {"neutral": None, "near": None, "far": None},
        "left": {"neutral": None, "near": None, "far": None},
    },
    "depth_live": {"right": {}, "left": {}},
    "calibration_file": str(ROOT / "calibration.runtime.json"),
    "debug_mode": False,
    "input_mode": "camera",
    "wizard_step": 1,
    "wizard_steps": [
        "Step 1: Choose tracked arm and ensure shoulder-elbow-wrist and palm are visible.",
        "Step 2: Hold neutral palm pose and click Calibrate Neutral for EE center.",
        "Step 3: Move palm in XYZ and verify the twin follows end-effector target.",
        "Step 4: Test pinch for gripper and safety controls (freeze/estop/home).",
        "Step 5: Record a short trajectory and replay it.",
        "Step 6: Save calibration JSON.",
    ],
    "enable_motion": False,
    "hold_to_enable": True,
    "hold_active": False,
    "hold_last_ts": 0.0,
    "one_axis_mode": False,
    "active_axis": 0,
    "calibration_workflow": {
        "active": False,
        "step": 0,
        "steps": [
            "Move robot to neutral/home pose",
            "Capture robot neutral FK EE reference",
            "Capture hand neutral pose",
            "Run +X hand motion test",
            "Run +Y hand motion test",
            "Run +Z hand motion test",
        ],
        "robot_neutral_ee_xyz": None,
        "hand_neutral_camera_xyz": None,
        "axis_tests": {
            "x": {"pass": None, "message": "", "delta_target": None, "delta_fk": None},
            "y": {"pass": None, "message": "", "delta_target": None, "delta_fk": None},
            "z": {"pass": None, "message": "", "delta_target": None, "delta_fk": None},
        },
        "axis_sign_correction": [1.0, 1.0, 1.0],
        "axis_scale_correction": [1.0, 1.0, 1.0],
        "axis_offset_correction_m": [0.0, 0.0, 0.0],
        "status": "idle",
    },
    "pending_calibration_capture": "",
    "pending_neutral_ee_reference": None,
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
            "selected_arm": runtime_state["selected_arm"],
            "trims_deg": runtime_state["trims_deg"],
            "center_offsets": runtime_state["center_offsets"],
            "human_arm_world": None,
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
            "ee_target": {},
            "tracking_confidence": 0.0,
            "ik_ok": False,
            "ik_error_m": None,
            "joint_limit_hit": False,
            "singularity_warning": False,
            "ee_fk_xyz": None,
            "ee_error_norm": None,
            "input_mode": runtime_state["input_mode"],
            "debug_mode": runtime_state["debug_mode"],
            "target_reachable": False,
            "target_clamped": False,
            "clamp_delta_xyz": [0.0, 0.0, 0.0],
            "workspace_violation_axes": [],
            "ik_fail_reason": "",
            "safety_suppressed": False,
            "simulation_parity": {},
            "motion_armed": False,
            "enable_motion": runtime_state["enable_motion"],
            "one_axis_mode": runtime_state["one_axis_mode"],
            "active_axis": runtime_state["active_axis"],
            "robot_command_suppressed": True,
            "collision_blocked": False,
            "depth_right": {},
            "depth_left": {},
            "calibration_workflow": runtime_state["calibration_workflow"],
        }
        self.running = True
        self.calibrate_request = False


state = SharedState()
worker_thread: Optional[threading.Thread] = None
trajectory_manager = TrajectoryManager(ROOT / "trajectories")
voice_engine = VoiceCommandEngine()
ros2_bridge = ROS2BridgeStub()
runtime_settings = {
    "camera": 0,
    "config_path": str(DEFAULT_CONFIG),
    "arm_side": "right",
    "lerobot_calibration_file": "",
    "host": "127.0.0.1",
    "port": 8010,
}
safety_supervisor: Optional[SafetySupervisor] = None
gesture_engine: Optional[GestureMacroEngine] = None
session_logger: Optional[SessionLogger] = None


def _log_event(event_type: str, fields: Optional[dict] = None):
    if session_logger is None:
        return
    try:
        session_logger.log_event(event_type=event_type, fields=fields or {})
    except Exception:
        return


def _xyz_or_none(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    out = []
    for i in range(3):
        try:
            out.append(float(values[i]))
        except Exception:
            return None
    return out


def _float_list(values, n=3, default=0.0):
    out = []
    if not isinstance(values, (list, tuple)):
        return [float(default)] * n
    for i in range(n):
        try:
            out.append(float(values[i]))
        except Exception:
            out.append(float(default))
    return out


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
    return cfg


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


def _motor_ranges_from_config(cfg: dict) -> list:
    out = []
    for m in cfg.get("motors", []):
        lo = float(m.get("min_deg", -180.0))
        hi = float(m.get("max_deg", 180.0))
        out.append([min(lo, hi), max(lo, hi)])
    while len(out) < 6:
        out.append([-180.0, 180.0])
    return out[:6]


def _home_motors_from_config(cfg: dict) -> list:
    out = []
    for m in cfg.get("motors", []):
        out.append(float(m.get("home_deg", 0.0)))
    while len(out) < 6:
        out.append(0.0)
    return out[:6]


def _default_depth_calibration() -> dict:
    return {
        "right": {"neutral": None, "near": None, "far": None},
        "left": {"neutral": None, "near": None, "far": None},
    }


def _depth_debug_from_raw(raw_z: Optional[float], calib: dict) -> dict:
    if raw_z is None:
        return {}
    out = {"forward": float(raw_z), "norm": None}
    near_v = calib.get("near")
    far_v = calib.get("far")
    try:
        near_f = float(near_v)
        far_f = float(far_v)
    except Exception:
        return out
    den = near_f - far_f
    if abs(den) < 1e-6:
        return out
    out["norm"] = float(_clamp((float(raw_z) - far_f) / den, 0.0, 1.0))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _map_gripper_to_motor(cfg: dict, grip_closedness: float) -> float:
    """
    Convert grip scalar (0=open, 1=closed) into motor_6 angle.
    """
    motors = cfg.get("motors", [])
    m6 = motors[5] if len(motors) >= 6 else {}
    ee_cfg = cfg.get("ee_target_mapper", {})
    grip_cfg = ee_cfg.get("gripper_motor", {})

    lo = float(m6.get("min_deg", 0.0))
    hi = float(m6.get("max_deg", 90.0))
    open_deg = float(grip_cfg.get("open_deg", m6.get("home_deg", lo)))
    closed_deg = float(grip_cfg.get("closed_deg", hi))

    g = _clamp(grip_closedness, 0.0, 1.0)
    deg = open_deg + (closed_deg - open_deg) * g
    return _clamp(deg, min(lo, hi), max(lo, hi))


def _apply_runtime_trims(cfg: dict, motors: list, trims: list) -> list:
    out = []
    limits = []
    for m in cfg.get("motors", []):
        limits.append((float(m.get("min_deg", -180.0)), float(m.get("max_deg", 180.0))))
    while len(limits) < 6:
        limits.append((-180.0, 180.0))

    for i in range(6):
        base = float(motors[i]) if i < len(motors) else 0.0
        trim = float(trims[i]) if i < len(trims) else 0.0
        lo, hi = limits[i]
        out.append(_clamp(base + trim, lo, hi))
    return out


def _new_calibration_workflow() -> dict:
    return {
        "active": True,
        "step": 1,
        "steps": [
            "Move robot to neutral/home pose",
            "Capture robot neutral FK EE reference",
            "Capture hand neutral pose",
            "Run +X hand motion test",
            "Run +Y hand motion test",
            "Run +Z hand motion test",
        ],
        "robot_neutral_ee_xyz": None,
        "hand_neutral_camera_xyz": None,
        "axis_tests": {
            "x": {"pass": None, "message": "", "delta_target": None, "delta_fk": None},
            "y": {"pass": None, "message": "", "delta_target": None, "delta_fk": None},
            "z": {"pass": None, "message": "", "delta_target": None, "delta_fk": None},
        },
        "axis_sign_correction": [1.0, 1.0, 1.0],
        "axis_scale_correction": [1.0, 1.0, 1.0],
        "axis_offset_correction_m": [0.0, 0.0, 0.0],
        "status": "step_1_home_robot",
    }


def _evaluate_axis_test(workflow: dict, latest_payload: dict, axis_name: str, min_delta_m: float = 0.008):
    idx_map = {"x": 0, "y": 1, "z": 2}
    if axis_name not in idx_map:
        return
    idx = idx_map[axis_name]

    robot_neutral = workflow.get("robot_neutral_ee_xyz")
    ee_target = latest_payload.get("ee_target_xyz")
    ee_fk = latest_payload.get("ee_fk_xyz")
    if not (
        isinstance(robot_neutral, list)
        and len(robot_neutral) == 3
        and isinstance(ee_target, list)
        and len(ee_target) == 3
        and isinstance(ee_fk, list)
        and len(ee_fk) == 3
    ):
        workflow["axis_tests"][axis_name] = {
            "pass": False,
            "message": "missing_reference_or_live_data",
            "delta_target": None,
            "delta_fk": None,
        }
        return

    delta_target = [float(ee_target[i]) - float(robot_neutral[i]) for i in range(3)]
    delta_fk = [float(ee_fk[i]) - float(robot_neutral[i]) for i in range(3)]
    target_axis_delta = float(delta_target[idx])
    fk_axis_delta = float(delta_fk[idx])

    moved_target = abs(target_axis_delta) >= min_delta_m
    moved_fk = abs(fk_axis_delta) >= min_delta_m
    target_positive = target_axis_delta > 0.0
    fk_positive = fk_axis_delta > 0.0
    same_sign = (target_axis_delta == 0.0 and fk_axis_delta == 0.0) or ((target_axis_delta > 0) == (fk_axis_delta > 0))
    passed = bool(moved_target and moved_fk and target_positive and fk_positive and same_sign)

    msg = "ok"
    if not moved_target:
        msg = "insufficient_target_axis_motion"
    elif not moved_fk:
        msg = "insufficient_fk_axis_motion"
    elif not target_positive:
        msg = "target_not_positive_axis"
    elif not same_sign:
        msg = "fk_direction_mismatch"
    elif not fk_positive:
        msg = "fk_not_positive_axis"

    workflow["axis_tests"][axis_name] = {
        "pass": passed,
        "message": msg,
        "delta_target": delta_target,
        "delta_fk": delta_fk,
    }

    if moved_fk and moved_target:
        if fk_axis_delta < 0.0:
            workflow["axis_sign_correction"][idx] *= -1.0
        ratio = abs(target_axis_delta) / max(1e-6, abs(fk_axis_delta))
        ratio = _clamp(ratio, 0.25, 4.0)
        workflow["axis_scale_correction"][idx] = float(
            _clamp(workflow["axis_scale_correction"][idx] * ratio, 0.1, 10.0)
        )


def _limit_command_step(prev_cmd: list, target_cmd: list, max_step: list) -> list:
    out = []
    for i in range(6):
        p = float(prev_cmd[i]) if i < len(prev_cmd) else 0.0
        t = float(target_cmd[i]) if i < len(target_cmd) else 0.0
        s = float(max_step[i]) if i < len(max_step) else float(max_step[-1] if max_step else 1.0)
        out.append(_clamp(t, p - s, p + s))
    return out


def _sync_workflow_corrections_to_center_offsets(workflow: dict, center_offsets: dict) -> dict:
    out = dict(center_offsets or {})
    sign = workflow.get("axis_sign_correction", [1.0, 1.0, 1.0])
    scale = workflow.get("axis_scale_correction", [1.0, 1.0, 1.0])
    offs = workflow.get("axis_offset_correction_m", [0.0, 0.0, 0.0])
    try:
        out["axis_sign_x"] = float(sign[0])
        out["axis_sign_y"] = float(sign[1])
        out["axis_sign_z"] = float(sign[2])
    except Exception:
        pass
    try:
        out["axis_scale_x"] = float(scale[0])
        out["axis_scale_y"] = float(scale[1])
        out["axis_scale_z"] = float(scale[2])
    except Exception:
        pass
    try:
        out["axis_offset_x_m"] = float(offs[0])
        out["axis_offset_y_m"] = float(offs[1])
        out["axis_offset_z_m"] = float(offs[2])
    except Exception:
        pass

    robot_neutral = workflow.get("robot_neutral_ee_xyz")
    if isinstance(robot_neutral, list) and len(robot_neutral) == 3:
        try:
            out["workspace_center_x"] = float(robot_neutral[0])
            out["workspace_center_y"] = float(robot_neutral[1])
            out["workspace_center_z"] = float(robot_neutral[2])
        except Exception:
            pass
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
            wf = runtime_state.get("calibration_workflow", {})
            if isinstance(wf, dict):
                wf["axis_sign_correction"] = [
                    float(safe_center.get("axis_sign_x", 1.0)),
                    float(safe_center.get("axis_sign_y", 1.0)),
                    float(safe_center.get("axis_sign_z", 1.0)),
                ]
                wf["axis_scale_correction"] = [
                    float(safe_center.get("axis_scale_x", 1.0)),
                    float(safe_center.get("axis_scale_y", 1.0)),
                    float(safe_center.get("axis_scale_z", 1.0)),
                ]
                wf["axis_offset_correction_m"] = [
                    float(safe_center.get("axis_offset_x_m", 0.0)),
                    float(safe_center.get("axis_offset_y_m", 0.0)),
                    float(safe_center.get("axis_offset_z_m", 0.0)),
                ]
                runtime_state["calibration_workflow"] = wf


def _save_calibration_into_config(config_path: str, calibration_payload: dict):
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found for calibration save: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ee_cfg = cfg.setdefault("ee_target_mapper", {})
    center = calibration_payload.get("center_offsets", {})
    if isinstance(center, dict):
        try:
            ee_cfg["workspace_center_m"] = [
                float(center["workspace_center_x"]),
                float(center["workspace_center_y"]),
                float(center["workspace_center_z"]),
            ]
        except Exception:
            pass
        try:
            ee_cfg["axis_sign_correction"] = [
                float(center["axis_sign_x"]),
                float(center["axis_sign_y"]),
                float(center["axis_sign_z"]),
            ]
        except Exception:
            pass
        try:
            ee_cfg["axis_scale_correction"] = [
                float(center["axis_scale_x"]),
                float(center["axis_scale_y"]),
                float(center["axis_scale_z"]),
            ]
        except Exception:
            pass
        try:
            ee_cfg["axis_offset_correction_m"] = [
                float(center["axis_offset_x_m"]),
                float(center["axis_offset_y_m"]),
                float(center["axis_offset_z_m"]),
            ]
        except Exception:
            pass
    cfg["calibration_runtime"] = {
        "saved_at": float(calibration_payload.get("saved_at", time.time())),
        "selected_arm": calibration_payload.get("selected_arm", "right"),
        "center_offsets": center,
    }

    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def save_calibration_to_file(calib_path: str, config_path: Optional[str] = None):
    with state.lock:
        payload = {
            "saved_at": time.time(),
            "selected_arm": runtime_state["selected_arm"],
            "trims_deg": runtime_state["trims_deg"],
            "center_offsets": runtime_state["center_offsets"],
        }
    p = Path(calib_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if config_path:
        _save_calibration_into_config(config_path=config_path, calibration_payload=payload)
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


def draw_selected_arm(frame, pose_landmarks, arm_side: str):
    if pose_landmarks is None:
        return
    ids = (11, 13, 15) if arm_side.lower() == "left" else (12, 14, 16)
    h, w = frame.shape[:2]
    pts = []
    for idx in ids:
        lm = pose_landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h), float(getattr(lm, "visibility", 1.0))))

    for i in range(2):
        p0 = pts[i]
        p1 = pts[i + 1]
        if p0[2] > 0.35 and p1[2] > 0.35:
            cv2.line(frame, (p0[0], p0[1]), (p1[0], p1[1]), (0, 210, 255), 4, cv2.LINE_AA)
    for p in pts:
        if p[2] > 0.35:
            cv2.circle(frame, (p[0], p[1]), 7, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_overlay(
    frame,
    motors,
    pose_ok,
    hand_ok,
    selected_arm,
    safety_snapshot=None,
    quality=None,
    debug_lines=None,
):
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
    if debug_lines:
        lines += list(debug_lines)
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


def _default_quality() -> dict:
    return {
        "quality_score": 100.0,
        "low_light": False,
        "critical_low_light": False,
        "blurry": False,
        "brightness": 100.0,
        "blur_var": 100.0,
    }


def _build_sim_frame(width: int, height: int, mode_label: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"SIMULATION MODE: {mode_label}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def camera_worker(camera_idx: int, config_path: str):
    global safety_supervisor, gesture_engine, session_logger
    cfg = load_config(config_path)
    if session_logger is not None:
        try:
            session_logger.close()
        except Exception:
            pass
    session_logger = SessionLogger.from_config(cfg, project_root=PROJECT_ROOT)
    _log_event(
        "session_started",
        {
            "config_path": str(config_path),
            "camera_idx": int(camera_idx),
            "log_path": str(session_logger.log_path) if session_logger and session_logger.log_path else None,
            "logging_enabled": bool(session_logger.enabled) if session_logger else False,
            "lerobot_calibration_file": str(runtime_settings.get("lerobot_calibration_file", "")),
            "lerobot_calibration_exists": bool(
                runtime_settings.get("lerobot_calibration_file")
                and Path(str(runtime_settings.get("lerobot_calibration_file"))).exists()
            ),
        },
    )
    tracker = HandPoseTracker(cfg)
    ee_mapper = EETargetMapper(cfg)
    ik_solver = DampedLeastSquaresIKSolver(cfg)
    sim_modes = SimulationModes(cfg)
    safety_supervisor = SafetySupervisor(cfg)
    gesture_engine = GestureMacroEngine(cfg)
    metrics = RuntimeMetrics(window=int(cfg.get("metrics_window", 120)))

    debug_default = bool(cfg.get("debug_mode", False))
    input_mode = sim_modes.mode if sim_modes.enabled else "camera"
    _log_event(
        "simulation_mode",
        {
            "input_mode": input_mode,
            "simulation_enabled": bool(sim_modes.enabled),
        },
    )
    sim_cfg = cfg.get("simulation", {})
    sim_frame_w = int(sim_cfg.get("frame_width", 1280))
    sim_frame_h = int(sim_cfg.get("frame_height", 720))
    hw_cfg = cfg.get("hardware_validation", {})
    hw_enabled = bool(hw_cfg.get("enabled", True))
    hw_hold_to_enable = bool(hw_cfg.get("hold_to_enable", True))
    hw_hold_timeout_s = max(0.1, float(hw_cfg.get("hold_timeout_sec", 0.45)))
    hw_one_axis_mode_default = bool(hw_cfg.get("one_axis_mode", False))
    hw_active_axis_default = int(hw_cfg.get("active_axis", 0))
    hw_enable_default = bool(hw_cfg.get("enable_motion_default", False))
    hw_step = hw_cfg.get("max_step_deg_per_frame", [1.0] * 6)
    if not isinstance(hw_step, list) or len(hw_step) != 6:
        hw_step = [1.0] * 6
    hw_step = [max(0.05, float(v)) for v in hw_step]
    quality_low_light = float(cfg.get("quality_low_light_threshold", 45.0))
    quality_blur = float(cfg.get("quality_blur_threshold", 35.0))
    home_motors = _home_motors_from_config(cfg)
    axis_validation_report = ik_solver.axis_validation_report(
        step_deg=float(sim_cfg.get("joint_diag_step_deg", 5.0))
    )
    with state.lock:
        ee_mapper.set_calibration_offsets(runtime_state["center_offsets"])
        runtime_state["motor_ranges"] = _motor_ranges_from_config(cfg)
        runtime_state["debug_mode"] = bool(runtime_state.get("debug_mode", debug_default))
        runtime_state["input_mode"] = input_mode
        runtime_state["enable_motion"] = bool(runtime_state.get("enable_motion", hw_enable_default))
        runtime_state["hold_to_enable"] = bool(runtime_state.get("hold_to_enable", hw_hold_to_enable))
        runtime_state["one_axis_mode"] = bool(runtime_state.get("one_axis_mode", hw_one_axis_mode_default))
        runtime_state["active_axis"] = int(runtime_state.get("active_axis", hw_active_axis_default))
        if not isinstance(runtime_state.get("depth_calibration"), dict):
            runtime_state["depth_calibration"] = _default_depth_calibration()
        if not isinstance(runtime_state.get("depth_live"), dict):
            runtime_state["depth_live"] = {"right": {}, "left": {}}
        if not isinstance(runtime_state.get("calibration_workflow"), dict):
            runtime_state["calibration_workflow"] = _new_calibration_workflow()

    use_camera = input_mode == "camera"
    last_robot_cmd = home_motors[:]
    last_visual_right = home_motors[:]
    last_visual_left = home_motors[:]
    last_motion_armed_state = None
    cap = None
    pose = None
    hands = None
    mp_pose = mp_hands = mp_draw = mp_style = None

    if use_camera:
        mp_pose, mp_hands, mp_draw, mp_style = resolve_mediapipe_modules()
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            with state.lock:
                state.latest["status"] = f"camera_open_failed(index={camera_idx})"
            return
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    try:
        prev_loop = time.time()
        while state.running:
            t_loop_start = time.time()
            now = time.time()

            with state.lock:
                selected_arm = runtime_state["selected_arm"]
                trims = runtime_state["trims_deg"][:]
                wizard_step = runtime_state["wizard_step"]
                debug_mode = bool(runtime_state["debug_mode"])
                enable_motion = bool(runtime_state.get("enable_motion", False))
                hold_to_enable = bool(runtime_state.get("hold_to_enable", hw_hold_to_enable))
                hold_active = bool(runtime_state.get("hold_active", False))
                hold_last_ts = float(runtime_state.get("hold_last_ts", 0.0))
                one_axis_mode = bool(runtime_state.get("one_axis_mode", hw_one_axis_mode_default))
                active_axis = int(runtime_state.get("active_axis", hw_active_axis_default))
                pending_capture = str(runtime_state.get("pending_calibration_capture", "")).strip()
                pending_neutral_ee = runtime_state.get("pending_neutral_ee_reference")
                workflow_snapshot = dict(runtime_state.get("calibration_workflow", {}))
                depth_calib_snapshot = dict(runtime_state.get("depth_calibration", _default_depth_calibration()))
                depth_live_snapshot = dict(runtime_state.get("depth_live", {"right": {}, "left": {}}))

            if use_camera:
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

                pose_landmarks = (
                    pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
                )
                hand_landmarks = pick_hand_landmarks(hands_res, requested_side=selected_arm)
                observation = tracker.process(
                    pose_landmarks=pose_landmarks,
                    hand_landmarks=hand_landmarks,
                    arm_side=selected_arm,
                )
                pose_ok = bool(observation.get("pose_detected", False))
                hand_ok = bool(observation.get("hand_detected", False))
                human_arm_world = observation.get("human_arm_world")
                macro_event = ""
                if gesture_engine is not None:
                    macro = gesture_engine.process(hand_landmarks)
                    if macro:
                        safety_supervisor.trigger_macro(macro)
                        macro_event = macro
                ee_target = ee_mapper.map(observation=observation, arm_side=selected_arm)
            else:
                frame = _build_sim_frame(sim_frame_w, sim_frame_h, input_mode)
                quality = _default_quality()
                pose_landmarks = None
                hand_landmarks = None
                macro_event = ""
                scripted_xyz = sim_modes.scripted_target(now_ts=now)
                scripted_grip = 0.45 + 0.35 * np.sin(2.0 * np.pi * 0.09 * (now - sim_modes.start_ts))
                projection = ee_mapper.project_robot_target([float(scripted_xyz[0]), float(scripted_xyz[1]), float(scripted_xyz[2])])
                observation = {
                    "timestamp": now,
                    "pose_detected": True,
                    "hand_detected": True,
                    "tracking_confidence": 1.0,
                    "raw_control_point_camera_xyz": None,
                    "control_point_camera_xyz": None,
                    "human_arm_world": None,
                    "pinch_ratio": None,
                }
                ee_target = {
                    "position_xyz": projection["clamped_target_xyz"],
                    "orientation_rpy": [0.0, 0.0, 0.0],
                    "grip": float(_clamp(scripted_grip, 0.0, 1.0)),
                    "valid": True,
                    "confidence_gated": False,
                    "workspace_clamped": bool(projection["target_clamped"]),
                    "tracking_confidence": 1.0,
                    "raw_control_camera_xyz": None,
                    "mapped_control_camera_xyz": None,
                    "neutral_control_camera_xyz": None,
                    "raw_target_xyz": projection["raw_target_xyz"],
                    "clamped_target_xyz": projection["clamped_target_xyz"],
                    "target_reachable": bool(projection["target_reachable"]),
                    "target_clamped": bool(projection["target_clamped"]),
                    "clamp_delta_xyz": projection["clamp_delta_xyz"],
                    "workspace_violation_axes": projection["workspace_violation_axes"],
                }
                pose_ok = True
                hand_ok = True
                human_arm_world = None

            raw_ctrl_xyz = (
                _xyz_or_none(observation.get("control_point_camera_xyz"))
                or _xyz_or_none(observation.get("raw_control_point_camera_xyz"))
                or _xyz_or_none(ee_target.get("raw_control_camera_xyz"))
            )
            raw_depth_z = float(raw_ctrl_xyz[2]) if raw_ctrl_xyz is not None else None
            selected_depth_cal = depth_calib_snapshot.get(selected_arm, {}) if isinstance(depth_calib_snapshot, dict) else {}
            depth_debug_selected = _depth_debug_from_raw(raw_depth_z, selected_depth_cal)
            if selected_arm == "right":
                depth_debug_right = depth_debug_selected
                depth_debug_left = depth_live_snapshot.get("left", {}) if isinstance(depth_live_snapshot, dict) else {}
            else:
                depth_debug_left = depth_debug_selected
                depth_debug_right = depth_live_snapshot.get("right", {}) if isinstance(depth_live_snapshot, dict) else {}

            if isinstance(workflow_snapshot.get("axis_sign_correction"), list):
                ee_mapper.set_axis_corrections(
                    sign_xyz=workflow_snapshot.get("axis_sign_correction"),
                    scale_xyz=workflow_snapshot.get("axis_scale_correction"),
                    offset_xyz_m=workflow_snapshot.get("axis_offset_correction_m"),
                )

            if use_camera:
                persist_calibration = False
                calib_file_snapshot = None
                with state.lock:
                    if state.calibrate_request and observation.get("control_point_camera_xyz") is not None and pose_ok:
                        neutral_fk = ik_solver.current_fk_xyz()
                        ee_mapper.calibrate(
                            observation,
                            neutral_ee_reference_xyz=neutral_fk,
                        )
                        runtime_state["center_offsets"] = ee_mapper.export_calibration_offsets()
                        state.calibrate_request = False
                        persist_calibration = True
                        calib_file_snapshot = runtime_state["calibration_file"]
                        _log_event(
                            "neutral_capture",
                            {
                                "source": "api_calibrate",
                                "hand_neutral_camera_xyz": observation.get("control_point_camera_xyz"),
                                "robot_neutral_fk_xyz": neutral_fk,
                                "center_offsets": dict(runtime_state["center_offsets"]),
                            },
                        )
                    if pending_capture == "hand_neutral" and observation.get("control_point_camera_xyz") is not None and pose_ok:
                        ee_mapper.calibrate(
                            observation,
                            neutral_ee_reference_xyz=pending_neutral_ee if isinstance(pending_neutral_ee, list) and len(pending_neutral_ee) == 3 else None,
                        )
                        runtime_state["center_offsets"] = ee_mapper.export_calibration_offsets()
                        flow = runtime_state.get("calibration_workflow", {})
                        flow["hand_neutral_camera_xyz"] = observation.get("control_point_camera_xyz")
                        flow["status"] = "step_3_hand_neutral_captured"
                        flow["step"] = max(int(flow.get("step", 1)), 3)
                        runtime_state["calibration_workflow"] = flow
                        runtime_state["pending_calibration_capture"] = ""
                        runtime_state["pending_neutral_ee_reference"] = None
                        persist_calibration = True
                        calib_file_snapshot = runtime_state["calibration_file"]
                        _log_event(
                            "neutral_capture",
                            {
                                "source": "workflow_hand_neutral",
                                "hand_neutral_camera_xyz": observation.get("control_point_camera_xyz"),
                                "robot_neutral_fk_xyz": pending_neutral_ee,
                                "workflow_step": flow.get("step"),
                                "workflow_status": flow.get("status"),
                            },
                        )
                if persist_calibration and calib_file_snapshot:
                    try:
                        save_calibration_to_file(calib_file_snapshot, config_path=config_path)
                    except Exception:
                        pass
            else:
                with state.lock:
                    state.calibrate_request = False

            ik_result = ik_solver.solve(
                target_position_xyz=ee_target["position_xyz"],
                tracking_confidence=float(ee_target.get("tracking_confidence", 0.0)),
            )
            if (not bool(ee_target.get("target_reachable", True))) and (not bool(ik_result.get("ik_ok", False))):
                if bool(ee_target.get("target_clamped", False)):
                    ik_result["ik_fail_reason"] = "target_clamped"
            ik_output_motors = ik_result["motors_deg"][:]
            if use_camera:
                ik_output_motors[5] = _map_gripper_to_motor(cfg, ee_target.get("grip", 0.0))

            if input_mode == "joint_axis_diagnostic":
                joint_diag = sim_modes.apply_joint_diagnostic(
                    base_motors=ik_output_motors,
                    pos_joint_count=ik_solver.n_pos_joints,
                    now_ts=now,
                )
                ik_output_motors = joint_diag["motors_deg"][:]
                fk_diag = ik_solver.forward_kinematics(ik_output_motors)
                err = float(np.linalg.norm(np.array(ee_target["position_xyz"], dtype=float) - np.array(fk_diag, dtype=float)))
                ik_result["ee_fk_xyz"] = fk_diag
                ik_result["ik_error_m"] = err
                ik_result["ik_ok"] = bool(err <= ik_solver.max_error_m)
                ik_result["ik_fail_reason"] = "" if ik_result["ik_ok"] else "max_error_exceeded"
                ik_result["joint_diagnostic"] = joint_diag

            mapped_motors = _apply_runtime_trims(cfg, ik_output_motors, trims)

            playback = trajectory_manager.sample_playback()
            if playback is not None:
                mapped_motors = playback

            safe_motors = safety_supervisor.process(
                target_motors=mapped_motors,
                pose_ok=pose_ok,
                hand_ok=hand_ok,
                quality=quality,
            )

            if trajectory_manager.recording:
                trajectory_manager.append(safe_motors)

            safety_snapshot = safety_supervisor.snapshot()
            safety_suppressed = bool(
                safety_snapshot.get("freeze", False)
                or safety_snapshot.get("estop", False)
                or (safety_supervisor.pose_required and not pose_ok)
                or (safety_supervisor.hand_required and not hand_ok)
            )
            hold_fresh = bool(hold_active and ((now - hold_last_ts) <= hw_hold_timeout_s))
            motion_armed = bool(enable_motion and (not hold_to_enable or hold_fresh))
            if not hw_enabled:
                motion_armed = bool(enable_motion)

            robot_cmd = safe_motors[:]
            if hw_enabled:
                robot_cmd = _limit_command_step(last_robot_cmd, robot_cmd, hw_step)
                if one_axis_mode:
                    ax = int(_clamp(active_axis, 0, 5))
                    for i in range(6):
                        if i != ax:
                            robot_cmd[i] = last_robot_cmd[i]

            if ros2_bridge.connected and motion_armed:
                ros2_bridge.publish_joint_targets(robot_cmd)
                last_robot_cmd = robot_cmd[:]
            robot_command_suppressed = bool(ros2_bridge.connected and not motion_armed)
            if last_motion_armed_state is None or bool(last_motion_armed_state) != bool(motion_armed):
                _log_event(
                    "motion_arming_changed",
                    {
                        "motion_armed": bool(motion_armed),
                        "enable_motion": bool(enable_motion),
                        "hold_to_enable": bool(hold_to_enable),
                        "hold_to_enable_active": bool(hold_fresh),
                        "safety_suppressed": bool(safety_suppressed),
                    },
                )
                last_motion_armed_state = bool(motion_armed)

            if use_camera:
                draw_selected_arm(frame, pose_landmarks, selected_arm)
                if hand_landmarks is not None:
                    class _HL:
                        def __init__(self, landmark):
                            self.landmark = landmark

                    mp_draw.draw_landmarks(
                        frame,
                        _HL(hand_landmarks),
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )

            if debug_mode:
                debug_lines = build_debug_lines(observation, ee_target, ik_result)
                _draw_overlay(
                    frame,
                    safe_motors,
                    pose_ok,
                    hand_ok,
                    selected_arm,
                    safety_snapshot=safety_supervisor.snapshot(),
                    quality=quality,
                    debug_lines=debug_lines,
                )

            ok_jpg, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok_jpg:
                continue
            frame_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

            dt_s = max(1e-6, now - prev_loop)
            prev_loop = now
            proc_ms = (now - t_loop_start) * 1000.0
            metrics.update(
                dt_s=dt_s,
                proc_ms=proc_ms,
                motors=safe_motors,
                pose_ok=pose_ok,
                quality_score=float(quality.get("quality_score", 0.0)),
            )

            ee_fk_xyz = ik_result.get("ee_fk_xyz")
            ee_target_xyz = ee_target.get("position_xyz")
            ee_error_norm = ik_result.get("ik_error_m")
            center_offsets_payload = ee_mapper.export_calibration_offsets()
            simulation_parity = {}
            with state.lock:
                workflow_snapshot = dict(runtime_state.get("calibration_workflow", {}))
            if input_mode in ("scripted_target", "joint_axis_diagnostic"):
                simulation_parity = {
                    "raw_target_xyz": ee_target.get("raw_target_xyz"),
                    "clamped_target_xyz": ee_target.get("clamped_target_xyz", ee_target.get("position_xyz")),
                    "fk_xyz": ee_fk_xyz,
                    "target_fk_error": ee_error_norm,
                }
            if selected_arm == "right":
                last_visual_right = safe_motors[:]
            else:
                last_visual_left = safe_motors[:]
            payload = {
                "timestamp": now,
                "frame_b64": frame_b64,
                "motors_deg": [round(v, 3) for v in safe_motors],
                "motors_right_deg": [round(v, 3) for v in last_visual_right],
                "motors_left_deg": [round(v, 3) for v in last_visual_left],
                "robot_cmd_deg": [round(v, 3) for v in robot_cmd],
                "ik_output_joints_deg": [round(float(v), 3) for v in ik_output_motors],
                "pose_detected": bool(pose_ok),
                "hand_detected": bool(hand_ok),
                "tracking_confidence": float(observation.get("tracking_confidence", 0.0)),
                "selected_arm": selected_arm,
                "trims_deg": trims,
                "center_offsets": center_offsets_payload,
                "human_arm_world": human_arm_world,
                "guide": GUIDE,
                "status": "running",
                "input_mode": input_mode,
                "debug_mode": bool(debug_mode),
                "macro_event": macro_event or safety_snapshot.get("last_macro", ""),
                "quality": quality,
                "metrics": metrics.snapshot(),
                "safety": safety_snapshot,
                "safety_right": safety_snapshot if selected_arm == "right" else {},
                "safety_left": safety_snapshot if selected_arm == "left" else {},
                "trajectory": trajectory_manager.snapshot(),
                "wizard_step": wizard_step,
                "ee_target": ee_target,
                "ee_target_xyz": ee_target_xyz,
                "ee_fk_xyz": ee_fk_xyz,
                "ee_error_norm": ee_error_norm,
                "ik_ok": bool(ik_result.get("ik_ok", False)),
                "ik_error_m": ee_error_norm,
                "ik_iterations": int(ik_result.get("ik_iterations", 0)),
                "joint_limit_hit": bool(ik_result.get("joint_limit_hit", False)),
                "singularity_warning": bool(ik_result.get("singularity_warning", False)),
                "ik_fail_reason": str(ik_result.get("ik_fail_reason", "")),
                "target_reachable": bool(ee_target.get("target_reachable", False)),
                "target_clamped": bool(ee_target.get("target_clamped", False)),
                "clamp_delta_xyz": ee_target.get("clamp_delta_xyz", [0.0, 0.0, 0.0]),
                "workspace_violation_axes": ee_target.get("workspace_violation_axes", []),
                "safety_suppressed": bool(safety_suppressed),
                "axis_validation_report": axis_validation_report,
                "sigma_min": ik_result.get("sigma_min"),
                "condition_estimate": ik_result.get("condition_estimate"),
                "simulation_parity": simulation_parity,
                "motion_armed": bool(motion_armed),
                "enable_motion": bool(enable_motion),
                "hold_to_enable": bool(hold_to_enable),
                "hold_active": bool(hold_fresh),
                "one_axis_mode": bool(one_axis_mode),
                "active_axis": int(_clamp(active_axis, 0, 5)),
                "robot_command_suppressed": bool(robot_command_suppressed),
                "collision_blocked": False,
                "depth_right": depth_debug_right,
                "depth_left": depth_debug_left,
                "calibration_workflow": workflow_snapshot,
            }
            if "joint_diagnostic" in ik_result:
                payload["joint_diagnostic"] = ik_result["joint_diagnostic"]
            for i, v in enumerate(payload["motors_deg"], start=1):
                payload[f"motor_{i}"] = v

            if session_logger is not None:
                try:
                    raw_hand_target_xyz = _xyz_or_none(ee_target.get("raw_control_camera_xyz"))
                    if raw_hand_target_xyz is None:
                        raw_hand_target_xyz = _xyz_or_none(observation.get("raw_control_point_camera_xyz"))
                    session_logger.log_frame(
                        {
                            "timestamp": float(now),
                            "input_mode": input_mode,
                            "hand_detected": bool(hand_ok),
                            "tracking_confidence": float(observation.get("tracking_confidence", 0.0)),
                            "raw_hand_target_xyz": raw_hand_target_xyz,
                            "mapped_target_xyz": _xyz_or_none(ee_target.get("raw_target_xyz"))
                            or _xyz_or_none(ee_target.get("position_xyz")),
                            "clamped_target_xyz": _xyz_or_none(ee_target.get("clamped_target_xyz"))
                            or _xyz_or_none(ee_target.get("position_xyz")),
                            "ee_fk_xyz": _xyz_or_none(ee_fk_xyz),
                            "ee_error_norm": float(ee_error_norm) if ee_error_norm is not None else None,
                            "ik_ok": bool(ik_result.get("ik_ok", False)),
                            "ik_fail_reason": str(ik_result.get("ik_fail_reason", "")),
                            "joint_limit_hit": bool(ik_result.get("joint_limit_hit", False)),
                            "singularity_warning": bool(ik_result.get("singularity_warning", False)),
                            "target_reachable": bool(ee_target.get("target_reachable", False)),
                            "target_clamped": bool(ee_target.get("target_clamped", False)),
                            "clamp_delta_xyz": _float_list(ee_target.get("clamp_delta_xyz", [0.0, 0.0, 0.0]), n=3, default=0.0),
                            "workspace_violation_axes": list(ee_target.get("workspace_violation_axes", [])),
                            "safety_suppressed": bool(safety_suppressed),
                            "motion_armed": bool(motion_armed),
                            "hold_to_enable_active": bool(hold_fresh),
                            "motors_deg": [float(v) for v in safe_motors],
                        }
                    )
                except Exception:
                    pass

            with state.lock:
                state.latest = payload
                runtime_state["center_offsets"] = payload["center_offsets"]
                runtime_state["hold_active"] = bool(hold_fresh)
                runtime_state["depth_live"] = {"right": depth_debug_right, "left": depth_debug_left}

            time.sleep(0.001)
    finally:
        if pose is not None:
            pose.close()
        if hands is not None:
            hands.close()
        if cap is not None:
            cap.release()
        _log_event("session_ended", {"reason": "camera_worker_exit"})
        if session_logger is not None:
            try:
                session_logger.close()
            except Exception:
                pass
            session_logger = None


app = FastAPI(title="Clutch Web Teleop V3")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.on_event("startup")
async def on_startup():
    global worker_thread
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


@app.on_event("shutdown")
async def on_shutdown():
    state.running = False


@app.get("/", response_class=HTMLResponse)
@app.get("/main", response_class=HTMLResponse)
async def landing_page():
    if MAIN_HTML.exists():
        return HTMLResponse(MAIN_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/hand-tracking-manipulation", response_class=HTMLResponse)
@app.get("/teleop", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/guide")
async def api_guide():
    return JSONResponse({"guide": GUIDE})


@app.get("/api/settings")
async def api_settings_get():
    with state.lock:
        payload = {
            "selected_arm": runtime_state["selected_arm"],
            "trims_deg": runtime_state["trims_deg"],
            "motor_ranges": runtime_state.get("motor_ranges", [[-180.0, 180.0]] * 6),
            "center_offsets": runtime_state["center_offsets"],
            "depth_calibration": runtime_state.get("depth_calibration", _default_depth_calibration()),
            "calibration_file": runtime_state["calibration_file"],
            "lerobot_calibration_file": runtime_settings.get("lerobot_calibration_file", ""),
            "wizard_step": runtime_state["wizard_step"],
            "wizard_steps": runtime_state["wizard_steps"],
            "control_mode": "ee_ik",
            "debug_mode": bool(runtime_state["debug_mode"]),
            "input_mode": runtime_state["input_mode"],
            "enable_motion": bool(runtime_state.get("enable_motion", False)),
            "hold_to_enable": bool(runtime_state.get("hold_to_enable", True)),
            "hold_active": bool(runtime_state.get("hold_active", False)),
            "one_axis_mode": bool(runtime_state.get("one_axis_mode", False)),
            "active_axis": int(runtime_state.get("active_axis", 0)),
            "calibration_workflow": runtime_state.get("calibration_workflow", {}),
        }
    payload["trajectory"] = trajectory_manager.snapshot()
    payload["safety"] = safety_supervisor.snapshot() if safety_supervisor else {}
    return JSONResponse(payload)


@app.post("/api/settings")
async def api_settings_set(payload: dict):
    arm = payload.get("selected_arm")
    debug_mode = payload.get("debug_mode")
    enable_motion = payload.get("enable_motion")
    hold_to_enable = payload.get("hold_to_enable")
    one_axis_mode = payload.get("one_axis_mode")
    active_axis = payload.get("active_axis")
    requested_input_mode = payload.get("input_mode")
    changed = {}
    with state.lock:
        prev_enable_motion = bool(runtime_state.get("enable_motion", False))
        if arm in ("left", "right"):
            runtime_state["selected_arm"] = arm
            changed["selected_arm"] = arm
        if isinstance(debug_mode, bool):
            runtime_state["debug_mode"] = bool(debug_mode)
            changed["debug_mode"] = bool(debug_mode)
        if isinstance(enable_motion, bool):
            runtime_state["enable_motion"] = bool(enable_motion)
            if prev_enable_motion != bool(enable_motion):
                changed["enable_motion"] = bool(enable_motion)
        if isinstance(hold_to_enable, bool):
            runtime_state["hold_to_enable"] = bool(hold_to_enable)
            changed["hold_to_enable"] = bool(hold_to_enable)
        if isinstance(one_axis_mode, bool):
            runtime_state["one_axis_mode"] = bool(one_axis_mode)
            changed["one_axis_mode"] = bool(one_axis_mode)
        if active_axis is not None:
            try:
                runtime_state["active_axis"] = int(_clamp(int(active_axis), 0, 5))
                changed["active_axis"] = int(runtime_state["active_axis"])
            except Exception:
                pass
        active_input_mode = str(runtime_state.get("input_mode", "camera"))
    if requested_input_mode is not None:
        requested_mode = str(requested_input_mode).strip().lower()
        if requested_mode and requested_mode != active_input_mode:
            _log_event(
                "simulation_mode_change_requested",
                {
                    "requested_mode": requested_mode,
                    "active_mode": active_input_mode,
                    "applied": False,
                    "requires_restart": True,
                },
            )
    if changed:
        _log_event("settings_updated", changed)
    return JSONResponse(
        {
            "ok": True,
            "selected_arm": runtime_state["selected_arm"],
            "debug_mode": bool(runtime_state["debug_mode"]),
            "enable_motion": bool(runtime_state.get("enable_motion", False)),
            "hold_to_enable": bool(runtime_state.get("hold_to_enable", True)),
            "one_axis_mode": bool(runtime_state.get("one_axis_mode", False)),
            "active_axis": int(runtime_state.get("active_axis", 0)),
        }
    )


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
    return JSONResponse({"ok": True, "safety": snap})


@app.post("/api/safety")
async def api_safety_set(payload: dict):
    if safety_supervisor is None:
        return JSONResponse({"ok": False, "error": "safety_not_ready"}, status_code=503)
    action = str(payload.get("action", "")).strip().lower()
    if action == "toggle_freeze":
        safety_supervisor.trigger_macro("toggle_freeze")
    elif action == "toggle_estop":
        safety_supervisor.trigger_macro("toggle_estop")
    elif action == "clear_estop":
        safety_supervisor.clear_estop()
    elif action == "home":
        safety_supervisor.trigger_macro("home")
    elif action == "freeze_on":
        safety_supervisor.set_freeze(True)
    elif action == "freeze_off":
        safety_supervisor.set_freeze(False)
    else:
        return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)
    return JSONResponse({"ok": True, "safety": safety_supervisor.snapshot()})


@app.post("/api/reset")
async def api_reset():
    if safety_supervisor is not None:
        safety_supervisor.set_freeze(False)
        safety_supervisor.clear_estop()
        safety_supervisor.trigger_macro("home")
    try:
        cfg = load_config(runtime_settings["config_path"])
        home = _home_motors_from_config(cfg)
    except Exception:
        home = [0.0] * 6
    with state.lock:
        state.latest["motors_deg"] = [float(v) for v in home]
        state.latest["motors_right_deg"] = [float(v) for v in home]
        state.latest["motors_left_deg"] = [float(v) for v in home]
        for i, v in enumerate(home, start=1):
            state.latest[f"motor_{i}"] = float(v)
    _log_event("reset_requested", {"home_motors_deg": home})
    return JSONResponse({"ok": True})


@app.get("/api/depth_calibration")
async def api_depth_calibration_get():
    with state.lock:
        payload = runtime_state.get("depth_calibration", _default_depth_calibration())
    return JSONResponse({"ok": True, "depth_calibration": payload})


@app.post("/api/depth_calibration")
async def api_depth_calibration_set(payload: dict):
    action = str(payload.get("action", "")).strip().lower()
    side = str(payload.get("side", runtime_state.get("selected_arm", "right"))).strip().lower()
    if side not in ("left", "right"):
        side = "right"

    with state.lock:
        depth_cfg = runtime_state.get("depth_calibration")
        if not isinstance(depth_cfg, dict):
            depth_cfg = _default_depth_calibration()
        if side not in depth_cfg or not isinstance(depth_cfg[side], dict):
            depth_cfg[side] = {"neutral": None, "near": None, "far": None}

        latest_ee = state.latest.get("ee_target", {})
        raw_xyz = (
            _xyz_or_none(latest_ee.get("raw_control_camera_xyz"))
            or _xyz_or_none(latest_ee.get("mapped_control_camera_xyz"))
            or _xyz_or_none(state.latest.get("ee_target_xyz"))
        )
        raw_z = float(raw_xyz[2]) if raw_xyz is not None else None

        if action == "reset":
            depth_cfg[side] = {"neutral": None, "near": None, "far": None}
        elif action in ("capture_neutral", "capture_near", "capture_far"):
            if raw_z is None:
                return JSONResponse({"ok": False, "error": "raw_depth_not_ready"}, status_code=503)
            key = action.split("_", 1)[1]
            depth_cfg[side][key] = float(raw_z)
        else:
            return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)

        runtime_state["depth_calibration"] = depth_cfg
        depth_debug = _depth_debug_from_raw(raw_z, depth_cfg.get(side, {}))
        depth_live = runtime_state.get("depth_live")
        if not isinstance(depth_live, dict):
            depth_live = {"right": {}, "left": {}}
        depth_live[side] = depth_debug
        runtime_state["depth_live"] = depth_live
        result_cfg = dict(depth_cfg)

    _log_event(
        "depth_calibration_updated",
        {
            "action": action,
            "side": side,
            "raw_depth_z": raw_z,
            "depth_calibration": result_cfg.get(side, {}),
        },
    )
    return JSONResponse({"ok": True, "depth_calibration": result_cfg, "side": side})


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
        config_path = runtime_settings["config_path"]
    try:
        payload = save_calibration_to_file(calib_file, config_path=config_path)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    _log_event(
        "config_corrections_applied",
        {
            "source": "api_calibration_save",
            "calibration_file": str(calib_file),
            "config_path": str(config_path),
            "center_offsets": payload.get("center_offsets", {}),
        },
    )
    return JSONResponse(
        {
            "ok": True,
            "calibration_file": calib_file,
            "config_path": config_path,
            "data": payload,
        }
    )


@app.post("/api/calibrate")
async def api_calibrate():
    with state.lock:
        state.calibrate_request = True
    _log_event("neutral_capture_requested", {"source": "api_calibrate"})
    return JSONResponse({"ok": True, "message": "EE neutral calibration requested."})


@app.get("/api/calibration/workflow")
async def api_calibration_workflow_get():
    with state.lock:
        wf = runtime_state.get("calibration_workflow", {})
    return JSONResponse({"ok": True, "workflow": wf})


@app.post("/api/calibration/workflow")
async def api_calibration_workflow_post(payload: dict):
    action = str(payload.get("action", "")).strip().lower()
    axis_name = str(payload.get("axis", "")).strip().lower()
    event_fields = {}
    event_type = ""
    with state.lock:
        wf = runtime_state.get("calibration_workflow")
        if not isinstance(wf, dict):
            wf = _new_calibration_workflow()

        if action == "start":
            wf = _new_calibration_workflow()
            runtime_state["calibration_workflow"] = wf
            runtime_state["pending_calibration_capture"] = ""
            runtime_state["pending_neutral_ee_reference"] = None
            event_type = "calibration_workflow_step"
            event_fields = {"action": action, "step": int(wf.get("step", 1)), "status": str(wf.get("status", ""))}
        elif action == "step1_done":
            wf["step"] = max(int(wf.get("step", 1)), 1)
            wf["status"] = "step_1_home_confirmed"
            if safety_supervisor is not None:
                safety_supervisor.trigger_macro("home")
            runtime_state["calibration_workflow"] = wf
            event_type = "calibration_workflow_step"
            event_fields = {"action": action, "step": int(wf.get("step", 1)), "status": str(wf.get("status", ""))}
        elif action == "capture_robot_neutral":
            ee_fk = state.latest.get("ee_fk_xyz")
            if not (isinstance(ee_fk, list) and len(ee_fk) == 3):
                return JSONResponse({"ok": False, "error": "ee_fk_not_ready"}, status_code=503)
            wf["robot_neutral_ee_xyz"] = [float(ee_fk[0]), float(ee_fk[1]), float(ee_fk[2])]
            cx = float(runtime_state.get("center_offsets", {}).get("workspace_center_x", 0.0))
            cy = float(runtime_state.get("center_offsets", {}).get("workspace_center_y", 0.0))
            cz = float(runtime_state.get("center_offsets", {}).get("workspace_center_z", 0.0))
            wf["axis_offset_correction_m"] = [float(ee_fk[0]) - cx, float(ee_fk[1]) - cy, float(ee_fk[2]) - cz]
            wf["step"] = max(int(wf.get("step", 1)), 2)
            wf["status"] = "step_2_robot_neutral_captured"
            runtime_state["center_offsets"] = _sync_workflow_corrections_to_center_offsets(wf, runtime_state.get("center_offsets", {}))
            runtime_state["calibration_workflow"] = wf
            event_type = "robot_neutral_capture"
            event_fields = {
                "action": action,
                "step": int(wf.get("step", 1)),
                "status": str(wf.get("status", "")),
                "robot_neutral_ee_xyz": wf.get("robot_neutral_ee_xyz"),
                "axis_offset_correction_m": wf.get("axis_offset_correction_m"),
            }
        elif action == "capture_hand_neutral":
            robot_neutral = wf.get("robot_neutral_ee_xyz")
            if not (isinstance(robot_neutral, list) and len(robot_neutral) == 3):
                return JSONResponse({"ok": False, "error": "capture_robot_neutral_first"}, status_code=400)
            runtime_state["pending_calibration_capture"] = "hand_neutral"
            runtime_state["pending_neutral_ee_reference"] = [float(robot_neutral[0]), float(robot_neutral[1]), float(robot_neutral[2])]
            wf["status"] = "step_3_pending_hand_neutral_capture"
            runtime_state["calibration_workflow"] = wf
            event_type = "calibration_workflow_step"
            event_fields = {
                "action": action,
                "step": int(wf.get("step", 1)),
                "status": str(wf.get("status", "")),
                "robot_neutral_ee_xyz": robot_neutral,
            }
        elif action == "test_axis":
            if axis_name not in ("x", "y", "z"):
                return JSONResponse({"ok": False, "error": "axis_required_x_y_z"}, status_code=400)
            _evaluate_axis_test(wf, state.latest, axis_name=axis_name)
            wf["step"] = max(int(wf.get("step", 1)), 3 + {"x": 1, "y": 2, "z": 3}[axis_name])
            wf["status"] = f"step_test_{axis_name}_completed"
            runtime_state["center_offsets"] = _sync_workflow_corrections_to_center_offsets(wf, runtime_state.get("center_offsets", {}))
            runtime_state["calibration_workflow"] = wf
            event_type = "calibration_workflow_step"
            event_fields = {
                "action": action,
                "axis": axis_name,
                "step": int(wf.get("step", 1)),
                "status": str(wf.get("status", "")),
                "axis_test": wf.get("axis_tests", {}).get(axis_name),
            }
        elif action == "save":
            runtime_state["center_offsets"] = _sync_workflow_corrections_to_center_offsets(wf, runtime_state.get("center_offsets", {}))
            wf["active"] = False
            wf["status"] = "saved"
            runtime_state["calibration_workflow"] = wf
            calib_file = runtime_state["calibration_file"]
            cfg_path = runtime_settings["config_path"]
            try:
                saved = save_calibration_to_file(calib_file, config_path=cfg_path)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
            _log_event(
                "calibration_workflow_step",
                {
                    "action": action,
                    "step": int(wf.get("step", 0)),
                    "status": str(wf.get("status", "")),
                },
            )
            _log_event(
                "config_corrections_applied",
                {
                    "source": "calibration_workflow_save",
                    "step": int(wf.get("step", 0)),
                    "status": str(wf.get("status", "")),
                    "axis_sign_correction": wf.get("axis_sign_correction"),
                    "axis_scale_correction": wf.get("axis_scale_correction"),
                    "axis_offset_correction_m": wf.get("axis_offset_correction_m"),
                    "calibration_file": str(calib_file),
                    "config_path": str(cfg_path),
                    "center_offsets": saved.get("center_offsets", {}),
                },
            )
            return JSONResponse({"ok": True, "workflow": wf, "saved": saved})
        else:
            return JSONResponse({"ok": False, "error": "unknown_action"}, status_code=400)
    if event_type:
        _log_event(event_type, event_fields)
    return JSONResponse({"ok": True, "workflow": runtime_state.get("calibration_workflow", {})})


@app.get("/api/motion")
async def api_motion_get():
    with state.lock:
        payload = {
            "ok": True,
            "enable_motion": bool(runtime_state.get("enable_motion", False)),
            "hold_to_enable": bool(runtime_state.get("hold_to_enable", True)),
            "hold_active": bool(runtime_state.get("hold_active", False)),
            "one_axis_mode": bool(runtime_state.get("one_axis_mode", False)),
            "active_axis": int(runtime_state.get("active_axis", 0)),
        }
    return JSONResponse(payload)


@app.post("/api/motion")
async def api_motion_set(payload: dict):
    motion_event = {}
    with state.lock:
        prev_enable = bool(runtime_state.get("enable_motion", False))
        prev_hold_to_enable = bool(runtime_state.get("hold_to_enable", True))
        prev_one_axis = bool(runtime_state.get("one_axis_mode", False))
        prev_active_axis = int(runtime_state.get("active_axis", 0))
        if "enable_motion" in payload:
            runtime_state["enable_motion"] = bool(payload.get("enable_motion"))
        if "hold_to_enable" in payload:
            runtime_state["hold_to_enable"] = bool(payload.get("hold_to_enable"))
        if "one_axis_mode" in payload:
            runtime_state["one_axis_mode"] = bool(payload.get("one_axis_mode"))
        if "active_axis" in payload:
            try:
                runtime_state["active_axis"] = int(_clamp(int(payload.get("active_axis")), 0, 5))
            except Exception:
                pass
        if "hold_active" in payload:
            runtime_state["hold_active"] = bool(payload.get("hold_active"))
            runtime_state["hold_last_ts"] = time.time() if runtime_state["hold_active"] else 0.0
        out = {
            "ok": True,
            "enable_motion": bool(runtime_state.get("enable_motion", False)),
            "hold_to_enable": bool(runtime_state.get("hold_to_enable", True)),
            "hold_active": bool(runtime_state.get("hold_active", False)),
            "one_axis_mode": bool(runtime_state.get("one_axis_mode", False)),
            "active_axis": int(runtime_state.get("active_axis", 0)),
        }
        if prev_enable != out["enable_motion"]:
            motion_event["enable_motion"] = out["enable_motion"]
        if prev_hold_to_enable != out["hold_to_enable"]:
            motion_event["hold_to_enable"] = out["hold_to_enable"]
        if prev_one_axis != out["one_axis_mode"]:
            motion_event["one_axis_mode"] = out["one_axis_mode"]
        if prev_active_axis != out["active_axis"]:
            motion_event["active_axis"] = out["active_axis"]
    if motion_event:
        _log_event("motion_settings_updated", motion_event)
    return JSONResponse(out)


@app.get("/api/debug")
async def api_debug_get():
    with state.lock:
        payload = {"ok": True, "debug_mode": bool(runtime_state["debug_mode"])}
    return JSONResponse(payload)


@app.post("/api/debug")
async def api_debug_set(payload: dict):
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        return JSONResponse({"ok": False, "error": "enabled_bool_required"}, status_code=400)
    with state.lock:
        runtime_state["debug_mode"] = bool(enabled)
    return JSONResponse({"ok": True, "debug_mode": bool(enabled)})


@app.get("/api/diagnostics/joint_axes")
async def api_diagnostics_joint_axes():
    with state.lock:
        report = state.latest.get("axis_validation_report", None)
    if report is None:
        return JSONResponse({"ok": False, "error": "diagnostics_not_ready"}, status_code=503)
    return JSONResponse({"ok": True, "report": report})


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
    runtime_settings["lerobot_calibration_file"] = str(cfg.get("lerobot_calibration_file", "")).strip()
    runtime_state["selected_arm"] = args.arm_side
    runtime_state["motor_ranges"] = _motor_ranges_from_config(cfg)
    runtime_state["debug_mode"] = bool(cfg.get("debug_mode", runtime_state.get("debug_mode", False)))
    hw_cfg = cfg.get("hardware_validation", {})
    runtime_state["enable_motion"] = bool(hw_cfg.get("enable_motion_default", False))
    runtime_state["hold_to_enable"] = bool(hw_cfg.get("hold_to_enable", True))
    runtime_state["one_axis_mode"] = bool(hw_cfg.get("one_axis_mode", False))
    runtime_state["active_axis"] = int(_clamp(int(hw_cfg.get("active_axis", 0)), 0, 5))
    runtime_state["calibration_workflow"] = _new_calibration_workflow()
    runtime_state["calibration_workflow"]["active"] = False
    runtime_state["calibration_workflow"]["step"] = 0
    runtime_state["calibration_workflow"]["status"] = "idle"
    runtime_state["depth_calibration"] = _default_depth_calibration()
    runtime_state["depth_live"] = {"right": {}, "left": {}}
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
