import json
import time
from pathlib import Path


class SafetySupervisor:
    def __init__(self, config):
        self.limits = [(m["min"], m["max"]) for m in config["motors"]]
        self.home = [m["home"] for m in config["motors"]]
        s = config.get("safety") or {}
        self.max_vel_deg_s = s.get("max_vel_deg_s", [180, 180, 220, 260, 260, 240])
        self.max_accel_deg_s2 = s.get("max_accel_deg_s2", [800, 800, 1000, 1200, 1200, 900])
        self.hand_command_timeout_s = float(s.get("hand_command_timeout_ms", 400)) / 1000.0
        self.dt_cap_s = float(s.get("dt_cap_s", 0.12))

        self._vel = [0.0] * 6
        self._prev_m = None
        self._prev_t = None
        self._hand_filter_reset()

    def _hand_filter_reset(self) -> None:
        self._vel = [0.0] * 6
        self._prev_m = None
        self._prev_t = None

    def clamp(self, motors: list) -> list:
        return [max(lo, min(hi, v)) for v, (lo, hi) in zip(motors, self.limits)]

    def reset_hand_rate_state(self) -> None:
        """Call when switching away from hand teleop or after long gaps."""
        self._hand_filter_reset()

    def filter_hand_command(self, motors: list[float], t_mono: float | None = None) -> list:
        """
        Joint limits + per-joint velocity/acceleration limits for streamed absolute targets.
        """
        t = time.monotonic() if t_mono is None else t_mono
        m = self.clamp([float(x) for x in motors])
        if self._prev_t is None:
            self._prev_m = m[:]
            self._prev_t = t
            self._vel = [0.0] * 6
            return m

        dt = t - self._prev_t
        dt = max(1e-4, min(dt, self.dt_cap_s))
        out = [0.0] * 6
        for i in range(6):
            vmax = float(self.max_vel_deg_s[i])
            amax = float(self.max_accel_deg_s2[i])
            v_des = (m[i] - self._prev_m[i]) / dt
            v_des = max(-vmax, min(vmax, v_des))
            v_old = self._vel[i]
            v_new = max(v_old - amax * dt, min(v_old + amax * dt, v_des))
            v_new = max(-vmax, min(vmax, v_new))
            self._vel[i] = v_new
            out[i] = self._prev_m[i] + v_new * dt
        out = self.clamp(out)
        self._prev_m = out
        self._prev_t = t
        return out


def append_teleop_metric(base: Path, row: dict) -> None:
    log = base / "teleop_metrics.jsonl"
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        pass
