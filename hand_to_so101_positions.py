import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


DEFAULT_LIMITS_DEG = [
    [-160.0, 160.0],  # m1 base yaw
    [-90.0, 90.0],  # m2 shoulder
    [-120.0, 120.0],  # m3 elbow
    [-180.0, 180.0],  # m4 wrist roll
    [-90.0, 90.0],  # m5 wrist pitch
    [0.0, 90.0],  # m6 gripper
]


def limits_from_motor_config(motors):
    if not isinstance(motors, list) or len(motors) != 6:
        raise ValueError("motors must be a list of 6 motor config objects.")
    limits = []
    for idx, m in enumerate(motors, start=1):
        if not isinstance(m, dict):
            raise ValueError(f"motor_{idx} config must be an object.")
        if "min_deg" not in m or "max_deg" not in m:
            raise ValueError(f"motor_{idx} config must include min_deg and max_deg.")
        limits.append([float(m["min_deg"]), float(m["max_deg"])])
    return limits


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def map_unit_to_range(u, lo, hi):
    return lerp(lo, hi, clamp(u, 0.0, 1.0))


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = clamp(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def joint_angle(a, b, c):
    # Angle ABC in degrees
    return angle_between(a - b, c - b)


def normalize01(value, lo, hi):
    if hi - lo < 1e-8:
        return 0.5
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


class TeleopMapper:
    """
    Converts hand landmarks to six joint commands.
    Mapping is heuristic but stable and calibration-friendly.
    """

    def __init__(self, limits_deg, smooth_alpha=0.25):
        self.limits_deg = limits_deg
        self.smooth_alpha = smooth_alpha
        self.prev = None

    def _flex_from_finger(self, lm, mcp, pip, dip):
        # 0=open, 1=closed
        a = np.array(lm[mcp])
        b = np.array(lm[pip])
        c = np.array(lm[dip])
        ang = joint_angle(a, b, c)  # ~180 open, smaller when bent
        return normalize01(180.0 - ang, 0.0, 100.0)

    def compute_unit(self, lm):
        wrist = np.array(lm[0])
        index_mcp = np.array(lm[5])
        middle_mcp = np.array(lm[9])
        pinky_mcp = np.array(lm[17])

        palm_center = (wrist + index_mcp + middle_mcp + pinky_mcp) / 4.0

        # m1 base yaw: horizontal hand displacement
        m1 = clamp(palm_center[0], 0.0, 1.0)

        # m2 shoulder: vertical hand displacement (invert image y)
        m2 = clamp(1.0 - palm_center[1], 0.0, 1.0)

        # m3 elbow: depth proxy from MediaPipe z (negative is closer)
        z_close = -middle_mcp[2]
        m3 = normalize01(z_close, -0.10, 0.20)

        # m4 wrist roll from palm line (index->pinky)
        palm_vec = pinky_mcp[:2] - index_mcp[:2]
        roll = math.degrees(math.atan2(palm_vec[1], palm_vec[0]))  # [-180, 180]
        m4 = normalize01(roll, -90.0, 90.0)

        # m5 wrist pitch from wrist->middle direction
        hand_vec = middle_mcp[:2] - wrist[:2]
        pitch = math.degrees(math.atan2(-hand_vec[1], hand_vec[0]))  # image-space proxy
        m5 = normalize01(pitch, -120.0, 60.0)

        # m6 gripper from averaged finger flexion
        i_flex = self._flex_from_finger(lm, 5, 6, 7)
        m_flex = self._flex_from_finger(lm, 9, 10, 11)
        r_flex = self._flex_from_finger(lm, 13, 14, 15)
        p_flex = self._flex_from_finger(lm, 17, 18, 19)
        m6 = clamp((i_flex + m_flex + r_flex + p_flex) / 4.0, 0.0, 1.0)

        return [m1, m2, m3, m4, m5, m6]

    def unit_to_degrees(self, u):
        out = []
        for idx, value in enumerate(u):
            lo, hi = self.limits_deg[idx]
            out.append(map_unit_to_range(value, lo, hi))
        return out

    def smooth(self, values):
        if self.prev is None:
            self.prev = values
            return values
        smoothed = []
        for p, v in zip(self.prev, values):
            smoothed.append(lerp(p, v, self.smooth_alpha))
        self.prev = smoothed
        return smoothed

    def map_landmarks(self, lm):
        unit = self.compute_unit(lm)
        deg = self.unit_to_degrees(unit)
        return self.smooth(deg)


def draw_overlay(frame, motors):
    y0 = 28
    for i, v in enumerate(motors, start=1):
        cv2.putText(
            frame,
            f"m{i}: {v:+06.1f} deg",
            (10, y0 + (i - 1) * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Hand tracking to 6 SO101-like motor positions."
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index.")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON config with motor_limits_deg and smooth_alpha.",
    )
    p.add_argument(
        "--udp",
        type=str,
        default="",
        help='Optional target "host:port" to stream JSON packets.',
    )
    p.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview.")
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
    return p.parse_args()


def load_config(path):
    if not path:
        return {
            "motor_limits_deg": DEFAULT_LIMITS_DEG,
            "smooth_alpha": 0.25,
        }
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "motors" in cfg:
        limits = limits_from_motor_config(cfg["motors"])
    else:
        limits = cfg.get("motor_limits_deg", DEFAULT_LIMITS_DEG)
    if len(limits) != 6:
        raise ValueError("motor_limits_deg must contain exactly 6 [min,max] pairs.")
    alpha = float(cfg.get("smooth_alpha", 0.25))
    alpha = clamp(alpha, 0.01, 1.0)
    return {
        "motor_limits_deg": limits,
        "smooth_alpha": alpha,
    }


def build_udp_sender(udp_target):
    if not udp_target:
        return None, None
    host, port_str = udp_target.split(":")
    port = int(port_str)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock, (host, port)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapper = TeleopMapper(
        limits_deg=cfg["motor_limits_deg"], smooth_alpha=cfg["smooth_alpha"]
    )

    udp_sock, udp_addr = build_udp_sender(args.udp)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            payload = {
                "timestamp": time.time(),
                "motors_deg": None,
                "motor_1": None,
                "motor_2": None,
                "motor_3": None,
                "motor_4": None,
                "motor_5": None,
                "motor_6": None,
                "hand_detected": False,
            }

            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                lm = [(l.x, l.y, l.z) for l in hl.landmark]
                motors = mapper.map_landmarks(lm)
                motors_rounded = [round(v, 3) for v in motors]
                payload["motors_deg"] = motors_rounded
                payload["hand_detected"] = True
                for idx, value in enumerate(motors_rounded, start=1):
                    payload[f"motor_{idx}"] = value

                if not args.no_preview:
                    mp_draw.draw_landmarks(
                        frame,
                        hl,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )
                    draw_overlay(frame, motors)

            line = json.dumps(payload, separators=(",", ":"))
            print(line, flush=True)

            if udp_sock and payload["motors_deg"] is not None:
                udp_sock.sendto(line.encode("utf-8"), udp_addr)

            if not args.no_preview:
                cv2.imshow("Hand -> SO101 Motors", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    cap.release()
    cv2.destroyAllWindows()
    if udp_sock:
        udp_sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
