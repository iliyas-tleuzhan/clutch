import math
import time
from typing import Dict, List, Optional


def _safe_vec3(values, default: List[float]) -> List[float]:
    if isinstance(values, list) and len(values) == 3:
        try:
            return [float(values[0]), float(values[1]), float(values[2])]
        except Exception:
            return default[:]
    return default[:]


class SimulationModes:
    """
    Simulation-only input modes for correctness validation.
    """

    def __init__(self, config: dict):
        sim_cfg = config.get("simulation", {})
        self.mode = str(sim_cfg.get("mode", "camera")).strip().lower()
        self.enabled = self.mode in ("scripted_target", "joint_axis_diagnostic")
        self.start_ts = time.time()

        self.center = _safe_vec3(
            sim_cfg.get("target_center_m"),
            [0.22, 0.0, 0.14],
        )
        self.amp = _safe_vec3(
            sim_cfg.get("target_amplitude_m"),
            [0.08, 0.08, 0.05],
        )
        self.freq = _safe_vec3(
            sim_cfg.get("target_frequency_hz"),
            [0.13, 0.17, 0.11],
        )
        self.phase = _safe_vec3(
            sim_cfg.get("target_phase_rad"),
            [0.0, 1.1, 2.2],
        )

        self.joint_wave_step_deg = float(sim_cfg.get("joint_diag_step_deg", 5.0))
        self.joint_hold_seconds = max(0.5, float(sim_cfg.get("joint_diag_hold_seconds", 2.2)))

    def scripted_target(self, now_ts: float) -> List[float]:
        t = max(0.0, float(now_ts - self.start_ts))
        out = []
        for i in range(3):
            omega = 2.0 * math.pi * self.freq[i]
            val = self.center[i] + self.amp[i] * math.sin(omega * t + self.phase[i])
            out.append(float(val))
        return out

    def apply_joint_diagnostic(self, base_motors: List[float], pos_joint_count: int, now_ts: float) -> Dict:
        if not isinstance(base_motors, list) or len(base_motors) < 6:
            base_motors = [0.0] * 6
        motors = [float(v) for v in base_motors[:6]]

        t = max(0.0, float(now_ts - self.start_ts))
        n = max(1, int(pos_joint_count))
        active_joint = int(t / self.joint_hold_seconds) % n
        phase_t = (t % self.joint_hold_seconds) / self.joint_hold_seconds
        # Smooth oscillation on one joint at a time.
        delta = self.joint_wave_step_deg * math.sin(2.0 * math.pi * phase_t)
        motors[active_joint] += float(delta)
        return {
            "motors_deg": motors,
            "active_joint": int(active_joint),
            "delta_deg": float(delta),
            "mode": "joint_axis_diagnostic",
        }
