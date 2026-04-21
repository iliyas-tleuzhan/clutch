import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from vpython import rate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_to_so101_positions import TeleopMapper, draw_overlay, load_config  # noqa: E402
from so101_digital_twin_vpython import ArmTwinView, load_config_data  # noqa: E402


def resolve_mediapipe_modules():
    """
    Support both:
    - mediapipe.solutions.*
    - mediapipe.python.solutions.*
    """
    if hasattr(mp, "solutions"):
        return (
            mp.solutions.hands,
            mp.solutions.drawing_utils,
            mp.solutions.drawing_styles,
        )

    try:
        from mediapipe.python.solutions import drawing_styles, drawing_utils, hands

        return hands, drawing_utils, drawing_styles
    except Exception as exc:
        raise RuntimeError(
            "Unsupported mediapipe installation. Reinstall with "
            "`python -m pip install --force-reinstall mediapipe==0.10.14`."
        ) from exc


def resolve_config_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    candidate = REPO_ROOT / path_str
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {path_str}")


def load_home_pose(config_path: Path, fallback_limits):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    motors = cfg.get("motors", [])
    if isinstance(motors, list) and len(motors) == 6:
        out = []
        for i, m in enumerate(motors):
            lo, hi = fallback_limits[i]
            if isinstance(m, dict) and "home_deg" in m:
                out.append(max(lo, min(hi, float(m["home_deg"]))))
            else:
                out.append((lo + hi) * 0.5)
        return out
    return [(lo + hi) * 0.5 for lo, hi in fallback_limits]


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulation-only version: hand tracking + 3D twin, no physical arm control."
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index.")
    p.add_argument(
        "--config",
        type=str,
        default="sim_only_v2/config.sim_only.json",
        help="Config path (supports motors + digital_twin settings).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Render/update rate target.",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV hand preview window.",
    )
    p.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.6,
        help="MediaPipe detector confidence.",
    )
    p.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.6,
        help="MediaPipe tracker confidence.",
    )
    p.add_argument(
        "--lost-mode",
        choices=["hold", "home"],
        default="hold",
        help='When hand is lost: "hold" last pose or go to "home" pose.',
    )
    return p.parse_args()


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)

    tele_cfg = load_config(str(config_path))
    twin_cfg = load_config_data(str(config_path))

    mapper = TeleopMapper(
        limits_deg=tele_cfg["motor_limits_deg"],
        smooth_alpha=tele_cfg["smooth_alpha"],
    )
    viewer = ArmTwinView(link_lengths=twin_cfg["link_lengths_m"])
    home_motors = load_home_pose(config_path, tele_cfg["motor_limits_deg"])
    current_motors = home_motors[:]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    mp_hands, mp_draw, mp_style = resolve_mediapipe_modules()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            rate(args.fps)
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            hand_detected = False
            if res.multi_hand_landmarks:
                hand_detected = True
                hl = res.multi_hand_landmarks[0]
                lm = [(l.x, l.y, l.z) for l in hl.landmark]
                current_motors = mapper.map_landmarks(lm)

                if not args.no_preview:
                    mp_draw.draw_landmarks(
                        frame,
                        hl,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )
                    draw_overlay(frame, current_motors)
            else:
                if args.lost_mode == "home":
                    current_motors = home_motors[:]
                if not args.no_preview:
                    cv2.putText(
                        frame,
                        "NO HAND DETECTED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 60, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    draw_overlay(frame, current_motors)

            viewer.update(motors=current_motors, source_name="sim-only-direct")

            payload = {
                "timestamp": time.time(),
                "hand_detected": hand_detected,
                "motors_deg": [round(v, 3) for v in current_motors],
            }
            for i, v in enumerate(payload["motors_deg"], start=1):
                payload[f"motor_{i}"] = v
            print(json.dumps(payload, separators=(",", ":")), flush=True)

            if not args.no_preview:
                cv2.imshow("Sim-Only Hand Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
