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

ROOT = THIS_DIR
DEFAULT_CONFIG = ROOT / "config.web_demo.json"
INDEX_HTML = ROOT / "static" / "index.html"

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
        }
        self.running = True
        self.calibrate_request = False


state = SharedState()
worker_thread: Optional[threading.Thread] = None
runtime_settings = {
    "camera": 0,
    "config_path": str(DEFAULT_CONFIG),
    "arm_side": "right",
    "host": "127.0.0.1",
    "port": 8010,
}


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


def _draw_overlay(frame, motors, pose_ok, hand_ok, selected_arm):
    lines = [
        f"arm_side: {selected_arm}",
        f"pose_detected: {pose_ok}",
        f"hand_detected: {hand_ok}",
    ]
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
    cfg = load_config(config_path)
    mapper = ArmFollowMapper(cfg)
    pose_vis_min = float(cfg.get("pose_visibility_threshold", 0.55))
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
        while state.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)

            with state.lock:
                selected_arm = runtime_state["selected_arm"]
                trims = runtime_state["trims_deg"][:]

            pose_landmarks = (
                pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
            )
            hand_landmarks = pick_hand_landmarks(hands_res, requested_side=selected_arm)

            features, pose_ok, hand_ok, human_arm_world = extract_arm_features(
                pose_landmarks=pose_landmarks,
                hand_landmarks=hand_landmarks,
                arm_side=selected_arm,
                pose_vis_min=pose_vis_min,
            )

            with state.lock:
                if state.calibrate_request and pose_ok:
                    mapper.calibrate(features)
                    runtime_state["center_offsets"] = dict(mapper.center_offsets)
                    state.calibrate_request = False

            motors = mapper.map(features=features, pose_ok=pose_ok, runtime_trims=trims)

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
            _draw_overlay(frame, motors, pose_ok, hand_ok, selected_arm)

            ok_jpg, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok_jpg:
                continue
            frame_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

            payload = {
                "timestamp": time.time(),
                "frame_b64": frame_b64,
                "motors_deg": [round(v, 3) for v in motors],
                "pose_detected": bool(pose_ok),
                "hand_detected": bool(hand_ok),
                "selected_arm": selected_arm,
                "trims_deg": trims,
                "center_offsets": dict(mapper.center_offsets),
                "human_arm_world": human_arm_world,
                "guide": GUIDE,
                "status": "running",
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
        }
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
