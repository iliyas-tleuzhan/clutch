import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _urlopen_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float = 5.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))


def run_self_test(camera: int, host: str, port: int, config: str):
    python_exe = sys.executable
    app_path = ROOT / "web_teleop_v3" / "app.py"
    cfg_path = (ROOT / config).resolve() if not Path(config).is_absolute() else Path(config)
    base = f"http://{host}:{port}"

    cmd = [
        python_exe,
        str(app_path),
        "--camera",
        str(camera),
        "--config",
        str(cfg_path),
        "--arm-side",
        "right",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"[self-test] starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )

    try:
        deadline = time.time() + 20.0
        started = False
        while time.time() < deadline:
            try:
                code, health = _urlopen_json(f"{base}/api/health", timeout=1.0)
                if code == 200 and health.get("ok", False):
                    started = True
                    print("[self-test] api health ok")
                    status = str(health.get("status", "unknown"))
                    if "camera_open_failed" in status:
                        raise RuntimeError(f"camera failed to open: {status}")
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not started:
            raise RuntimeError("server did not become healthy in time")

        code, settings = _urlopen_json(f"{base}/api/settings")
        if code != 200 or "trims_deg" not in settings:
            raise RuntimeError("settings endpoint malformed")
        print("[self-test] settings endpoint ok")

        target_trims = [3.0, -2.0, 1.0, 0.5, -1.5, 6.0]
        code, post_resp = _post_json(f"{base}/api/trims", {"trims_deg": target_trims})
        if code != 200 or not post_resp.get("ok", False):
            raise RuntimeError("trims POST failed")
        print("[self-test] trims post ok")

        _, settings2 = _urlopen_json(f"{base}/api/settings")
        got = [float(v) for v in settings2.get("trims_deg", [])]
        if len(got) != 6:
            raise RuntimeError("trims not persisted in settings")
        if any(abs(a - b) > 1e-6 for a, b in zip(target_trims, got)):
            raise RuntimeError(f"trims mismatch: expected={target_trims}, got={got}")
        print("[self-test] trims round-trip ok")

        code, _ = _urlopen_json(f"{base}/api/trajectory")
        if code != 200:
            raise RuntimeError("trajectory endpoint failed")
        print("[self-test] trajectory endpoint ok")

        code, depth = _urlopen_json(f"{base}/api/depth_calibration")
        if code != 200 or not depth.get("ok", False):
            raise RuntimeError("depth calibration endpoint failed")
        print("[self-test] depth calibration endpoint ok")

        print("[self-test] PASS")
        return 0
    except (urllib.error.URLError, RuntimeError, ValueError) as exc:
        print(f"[self-test] FAIL: {exc}")
        if proc.poll() is None and proc.stdout is not None:
            try:
                out = proc.stdout.read()
                if out:
                    print("[self-test] server output:")
                    print(out[-4000:])
            except Exception:
                pass
        return 1
    finally:
        if proc.poll() is None:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.3)
            proc.terminate()
            try:
                proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


def main():
    p = argparse.ArgumentParser(description="Run web_teleop_v3 local self-tests.")
    p.add_argument("--camera", type=int, default=0, help="Camera index.")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    p.add_argument("--port", type=int, default=8011, help="Test port.")
    p.add_argument(
        "--config",
        type=str,
        default="web_teleop_v3/config.web_demo.json",
        help="Path to config file (absolute or relative to project root).",
    )
    args = p.parse_args()
    raise SystemExit(run_self_test(args.camera, args.host, args.port, args.config))


if __name__ == "__main__":
    main()
