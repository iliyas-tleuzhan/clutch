import argparse
import json
import math
import socket
import time
from pathlib import Path

import numpy as np

try:
    from vpython import (
        arrow,
        box,
        canvas,
        color,
        cylinder,
        label,
        rate,
        sphere,
        vector,
        wtext,
    )
except ImportError as exc:
    raise ImportError(
        "vpython is required. Install it with: pip install vpython"
    ) from exc


DEFAULT_LINK_LENGTHS_M = [0.10, 0.12, 0.10, 0.08, 0.06, 0.05]
DEFAULT_JOINT_AXES = [
    [0.0, 0.0, 1.0],  # base yaw
    [0.0, 1.0, 0.0],  # shoulder
    [0.0, 1.0, 0.0],  # elbow
    [1.0, 0.0, 0.0],  # wrist roll
    [0.0, 1.0, 0.0],  # wrist pitch
    [1.0, 0.0, 0.0],  # tool roll / gripper proxy
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rot_axis_angle(axis, angle_rad):
    axis = np.array(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    x, y, z = axis / n
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


def load_config_data(config_path):
    default_limits = [
        [-160.0, 160.0],
        [-90.0, 90.0],
        [-120.0, 120.0],
        [-180.0, 180.0],
        [-90.0, 90.0],
        [0.0, 90.0],
    ]
    if not config_path:
        return {"limits": default_limits, "link_lengths_m": DEFAULT_LINK_LENGTHS_M}

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "motors" in cfg:
        motors = cfg["motors"]
        if len(motors) != 6:
            raise ValueError("motors must have 6 entries.")
        limits = [[float(m["min_deg"]), float(m["max_deg"])] for m in motors]
    else:
        limits = cfg.get("motor_limits_deg", default_limits)
        if len(limits) != 6:
            raise ValueError("Config must include motors[] or motor_limits_deg (6 entries).")

    link_lengths = DEFAULT_LINK_LENGTHS_M
    dt_cfg = cfg.get("digital_twin", {})
    if isinstance(dt_cfg, dict) and "link_lengths_m" in dt_cfg:
        candidate = dt_cfg["link_lengths_m"]
        if isinstance(candidate, list) and len(candidate) == 6:
            link_lengths = [float(v) for v in candidate]

    return {"limits": limits, "link_lengths_m": link_lengths}


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple VPython digital twin for SO101-like 6-DOF arm."
    )
    p.add_argument(
        "--source",
        choices=["udp", "synthetic"],
        default="udp",
        help="Motor input source.",
    )
    p.add_argument("--udp-bind", type=str, default="127.0.0.1", help="UDP bind host.")
    p.add_argument("--udp-port", type=int, default=5005, help="UDP bind port.")
    p.add_argument(
        "--config",
        type=str,
        default="so101_config.example.json",
        help="Config containing motor limits.",
    )
    p.add_argument(
        "--link-lengths",
        type=str,
        default="",
        help='Comma-separated 6 link lengths in meters, e.g. "0.10,0.12,0.10,0.08,0.06,0.05"',
    )
    p.add_argument("--fps", type=int, default=60, help="Render loop target fps.")
    return p.parse_args()


def parse_link_lengths(raw):
    if not raw:
        return DEFAULT_LINK_LENGTHS_M
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("link-lengths must contain 6 values.")
    return vals


class UdpMotorSource:
    def __init__(self, bind_host, bind_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_host, bind_port))
        self.sock.setblocking(False)

    def get_latest(self):
        latest = None
        while True:
            try:
                data, _addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                payload = json.loads(data.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            motors = payload.get("motors_deg")
            if isinstance(motors, list) and len(motors) == 6:
                latest = [float(v) for v in motors]
        return latest

    def close(self):
        self.sock.close()


class SyntheticMotorSource:
    def __init__(self, limits):
        self.t0 = time.time()
        self.limits = limits

    def get_latest(self):
        t = time.time() - self.t0
        out = []
        for idx, (lo, hi) in enumerate(self.limits):
            mid = (lo + hi) * 0.5
            amp = (hi - lo) * 0.35
            w = 0.7 + idx * 0.19
            out.append(mid + amp * math.sin(t * w))
        return out

    def close(self):
        pass


def fk_positions(joint_deg, link_lengths):
    """
    Returns 7 points (base + 6 joints/end) in world frame.
    """
    p = np.array([0.0, 0.0, 0.0], dtype=float)
    R = np.eye(3, dtype=float)
    pts = [p.copy()]
    for i in range(6):
        R = R @ rot_axis_angle(DEFAULT_JOINT_AXES[i], math.radians(joint_deg[i]))
        p = p + R @ np.array([link_lengths[i], 0.0, 0.0], dtype=float)
        pts.append(p.copy())
    return pts


class ArmTwinView:
    def __init__(self, link_lengths):
        self.link_lengths = link_lengths
        self.scene = canvas(
            title="SO101 Digital Twin (VPython)",
            width=1100,
            height=700,
            background=vector(0.06, 0.06, 0.08),
        )
        self.scene.forward = vector(-0.8, -0.4, -1.0)
        self.scene.up = vector(0, 0, 1)

        box(pos=vector(0, 0, -0.01), size=vector(0.5, 0.5, 0.02), color=color.gray(0.3))
        arrow(pos=vector(0, 0, 0), axis=vector(0.15, 0, 0), color=color.red, shaftwidth=0.004)
        arrow(
            pos=vector(0, 0, 0), axis=vector(0, 0.15, 0), color=color.green, shaftwidth=0.004
        )
        arrow(
            pos=vector(0, 0, 0), axis=vector(0, 0, 0.15), color=color.blue, shaftwidth=0.004
        )

        self.joints = []
        self.links = []
        for i in range(7):
            j = sphere(radius=0.012 if i == 0 else 0.009, color=color.white, opacity=0.9)
            self.joints.append(j)
        for _ in range(6):
            c = cylinder(radius=0.006, color=vector(0.2, 0.7, 1.0))
            self.links.append(c)

        self.info = wtext(text="waiting for motor data...")
        self.status = label(
            pos=vector(0, 0, 0.22),
            text="source: waiting",
            box=False,
            opacity=0,
            color=color.white,
        )

    def update(self, motors, source_name):
        pts = fk_positions(motors, self.link_lengths)
        for i, p in enumerate(pts):
            self.joints[i].pos = vector(float(p[0]), float(p[1]), float(p[2]))
        for i in range(6):
            p0 = pts[i]
            p1 = pts[i + 1]
            self.links[i].pos = vector(float(p0[0]), float(p0[1]), float(p0[2]))
            self.links[i].axis = vector(
                float(p1[0] - p0[0]), float(p1[1] - p0[1]), float(p1[2] - p0[2])
            )
        motor_line = " | ".join([f"m{i + 1}:{motors[i]:+06.1f}" for i in range(6)])
        self.info.text = f"{motor_line}\n"
        self.status.text = f"source: {source_name}"


def main():
    args = parse_args()
    cfg = load_config_data(args.config)
    limits = cfg["limits"]
    links = parse_link_lengths(args.link_lengths) if args.link_lengths else cfg["link_lengths_m"]
    viewer = ArmTwinView(link_lengths=links)

    if args.source == "udp":
        src = UdpMotorSource(args.udp_bind, args.udp_port)
        source_name = f"udp://{args.udp_bind}:{args.udp_port}"
        motors = [0.0] * 6
    else:
        src = SyntheticMotorSource(limits=limits)
        source_name = "synthetic"
        motors = src.get_latest()

    try:
        while True:
            rate(args.fps)
            latest = src.get_latest()
            if latest is not None:
                motors = latest
            motors = [
                clamp(float(m), float(limits[i][0]), float(limits[i][1]))
                for i, m in enumerate(motors)
            ]
            viewer.update(motors=motors, source_name=source_name)
    finally:
        src.close()


if __name__ == "__main__":
    main()
