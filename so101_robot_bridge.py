import argparse
import importlib
import json
import socket
import time
from pathlib import Path


DEFAULT_LIMITS_DEG = [
    [-160.0, 160.0],
    [-90.0, 90.0],
    [-120.0, 120.0],
    [-180.0, 180.0],
    [-90.0, 90.0],
    [0.0, 90.0],
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_limits(config_path):
    if not config_path:
        return DEFAULT_LIMITS_DEG
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "motors" in cfg:
        motors = cfg["motors"]
        if len(motors) != 6:
            raise ValueError("motors config must contain 6 entries.")
        return [[float(m["min_deg"]), float(m["max_deg"])] for m in motors]
    limits = cfg.get("motor_limits_deg", DEFAULT_LIMITS_DEG)
    if len(limits) != 6:
        raise ValueError("motor_limits_deg must contain 6 entries.")
    return limits


class DryRunDriver:
    def __init__(self, config):
        self.config = config

    def send_targets(self, motors_deg):
        print(
            f"[DRY-RUN] send -> "
            + " ".join([f"m{i+1}:{motors_deg[i]:+06.1f}" for i in range(6)]),
            flush=True,
        )

    def close(self):
        return None


def load_driver(enable_robot, driver_module, driver_class, config_path):
    if not enable_robot:
        return DryRunDriver(config_path)
    if not driver_module or not driver_class:
        raise ValueError(
            "--enable-robot requires --driver-module and --driver-class."
        )
    mod = importlib.import_module(driver_module)
    cls = getattr(mod, driver_class)
    return cls(config_path=config_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Laptop-side bridge: receives motor targets and sends to SO101 driver."
    )
    p.add_argument("--udp-bind", type=str, default="127.0.0.1")
    p.add_argument("--udp-port", type=int, default=5005)
    p.add_argument("--config", type=str, default="so101_config.example.json")
    p.add_argument(
        "--enable-robot",
        action="store_true",
        help="Enable physical robot driver. Off => dry-run only.",
    )
    p.add_argument(
        "--driver-module",
        type=str,
        default="",
        help="Python module path for your SO101 hardware driver adapter.",
    )
    p.add_argument(
        "--driver-class",
        type=str,
        default="",
        help="Class name implementing send_targets(motors_deg).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    limits = load_limits(args.config)
    driver = load_driver(
        enable_robot=args.enable_robot,
        driver_module=args.driver_module,
        driver_class=args.driver_class,
        config_path=args.config,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.udp_bind, args.udp_port))
    sock.settimeout(0.2)

    print(
        f"Robot bridge listening on udp://{args.udp_bind}:{args.udp_port} "
        f"(robot_enabled={args.enable_robot})",
        flush=True,
    )

    last_sent = [0.0] * 6
    try:
        while True:
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                payload = json.loads(data.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            motors = payload.get("motors_deg")
            if not isinstance(motors, list) or len(motors) != 6:
                continue
            safe = [
                clamp(float(motors[i]), float(limits[i][0]), float(limits[i][1]))
                for i in range(6)
            ]
            if safe != last_sent:
                driver.send_targets(safe)
                last_sent = safe
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Stopping robot bridge...", flush=True)
    finally:
        sock.close()
        driver.close()


if __name__ == "__main__":
    main()
