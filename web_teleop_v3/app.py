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

ROOT = THIS_DIR
DEFAULT_CONFIG = ROOT / "config.web_demo.json"
INDEX_HTML = ROOT / "static" / "index.html"
MAIN_HTML = ROOT / "static" / "main.html"

GUIDE = [
    {"motor": "motor_1", "name": "Base Yaw", "how": "Move whole arm left/right from shoulder"},
    {"motor": "motor_2", "name": "Shoulder", "how": "Raise/lower your elbow"},
    {"motor": "motor_3", "name": "Elbow", "how": "Bend/extend your elbow"},
    {"motor": "motor_4", "name": "Forearm Roll", "how": "Rotate forearm (palm turn)"},
    {"motor": "motor_5", "name": "Wrist Pitch", "how": "Tilt your wrist up/down"},
    {"motor": "motor_6", "name": "Gripper", "how": "Open/close your hand"},
]

runtime_state = {
    "selected_arm": "right",
    "trims_deg": [0.0] * 6,
    "center_offsets": {},
    "calibration_file": str(ROOT / "calibration.runtime.json"),
    "wizard_step": 1,
    "wizard_steps": [
        "Step 1: Choose tracked arm and ensure shoulder-elbow-wrist are visible.",
        "Step 2: Hold neutral pose for 2 seconds and click Calibrate Neutral.",
        "Step 3: Move each limb and tune motor trim sliders for alignment.",
        "Step 4: Test gestures (fist/open/pinch) and safety controls.",
        "Step 5: Record a short trajectory and replay it.",
        "Step 6: Save calibration JSON.",
    ],
}


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {
            "timestamp": time.time(),
            "frame_b64": "",
            "motors_deg": [0.0] * 6,
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
            "trajectory": {},
            "wizard_step": runtime_state["wizard_step"],
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
    "host": "127.0.0.1",
    "port": 8010,
}
safety_supervisor: Optional[SafetySupervisor] = None
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


def save_calibration_to_file(calib_path: str):
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


def remap_side_for_mirrored_input(side: str, mirrored_input: bool = True) -> str:
    s = str(side).strip().lower()
    if s not in ("left", "right"):
        return "right"
    if not mirrored_input:
        return s
    return "left" if s == "right" else "right"


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


def camera_worker(camera_idx: int, config_path: str):
    global safety_supervisor, gesture_engine
    cfg = load_config(config_path)
    mapper = ArmFollowMapper(cfg)
    safety_supervisor = SafetySupervisor(cfg)
    gesture_engine = GestureMacroEngine(cfg)
    metrics = RuntimeMetrics(window=int(cfg.get("metrics_window", 120)))

    pose_vis_min = float(cfg.get("pose_visibility_threshold", 0.55))
    quality_low_light = float(cfg.get("quality_low_light_threshold", 45.0))
    quality_blur = float(cfg.get("quality_blur_threshold", 35.0))
    with state.lock:
        for k, v in runtime_state["center_offsets"].items():
            if k in mapper.center_offsets:
                mapper.center_offsets[k] = float(v)
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
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        prev_loop = time.time()
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

            # MediaPipe runs on a mirrored selfie frame, so semantic left/right
            # can be inverted relative to the user's physical left/right.
            model_arm_side = remap_side_for_mirrored_input(selected_arm, mirrored_input=True)

            pose_landmarks = (
                pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
            )
            hand_landmarks = pick_hand_landmarks(hands_res, requested_side=model_arm_side)

            features, pose_ok, hand_ok, human_arm_world = extract_arm_features(
                pose_landmarks=pose_landmarks,
                hand_landmarks=hand_landmarks,
                arm_side=model_arm_side,
                pose_vis_min=pose_vis_min,
            )

            macro_event = ""
            if gesture_engine is not None:
                macro = gesture_engine.process(hand_landmarks)
                if macro:
                    safety_supervisor.trigger_macro(macro)
                    macro_event = macro

            with state.lock:
                if state.calibrate_request and pose_ok:
                    mapper.calibrate(features)
                    runtime_state["center_offsets"] = dict(mapper.center_offsets)
                    state.calibrate_request = False

            mapped_motors = mapper.map(features=features, pose_ok=pose_ok, runtime_trims=trims)

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

            if ros2_bridge.connected:
                ros2_bridge.publish_joint_targets(safe_motors)

            draw_selected_arm(frame, pose_landmarks, model_arm_side)
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
            _draw_overlay(
                frame,
                safe_motors,
                pose_ok,
                hand_ok,
                selected_arm,
                safety_snapshot=safety_supervisor.snapshot(),
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
                motors=safe_motors,
                pose_ok=pose_ok,
                quality_score=quality["quality_score"],
            )

            payload = {
                "timestamp": time.time(),
                "frame_b64": frame_b64,
                "motors_deg": [round(v, 3) for v in safe_motors],
                "pose_detected": bool(pose_ok),
                "hand_detected": bool(hand_ok),
                "selected_arm": selected_arm,
                "trims_deg": trims,
                "center_offsets": dict(mapper.center_offsets),
                "human_arm_world": human_arm_world,
                "guide": GUIDE,
                "status": "running",
                "macro_event": macro_event or safety_supervisor.snapshot().get("last_macro", ""),
                "quality": quality,
                "metrics": metrics.snapshot(),
                "safety": safety_supervisor.snapshot(),
                "trajectory": trajectory_manager.snapshot(),
                "wizard_step": wizard_step,
            }
            for i, v in enumerate(payload["motors_deg"], start=1):
                payload[f"motor_{i}"] = v

            with state.lock:
                state.latest = payload

            time.sleep(0.001)

    cap.release()


app = FastAPI(title="Clutch Web Teleop V3")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")


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
    return HTMLResponse(MAIN_HTML.read_text(encoding="utf-8"))


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
            "center_offsets": runtime_state["center_offsets"],
            "calibration_file": runtime_state["calibration_file"],
            "wizard_step": runtime_state["wizard_step"],
            "wizard_steps": runtime_state["wizard_steps"],
        }
    payload["trajectory"] = trajectory_manager.snapshot()
    payload["safety"] = safety_supervisor.snapshot() if safety_supervisor else {}
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
