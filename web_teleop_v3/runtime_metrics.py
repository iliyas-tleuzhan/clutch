from collections import deque
from typing import Dict, List


class RuntimeMetrics:
    def __init__(self, window=120):
        self.window = int(window)
        self.dt_hist = deque(maxlen=self.window)
        self.proc_ms_hist = deque(maxlen=self.window)
        self.pose_hist = deque(maxlen=self.window)
        self.quality_hist = deque(maxlen=self.window)
        self.jitter_hist = deque(maxlen=self.window)
        self.prev_motors = None

    def update(self, dt_s: float, proc_ms: float, motors: List[float], pose_ok: bool, quality_score: float):
        self.dt_hist.append(max(1e-6, float(dt_s)))
        self.proc_ms_hist.append(float(proc_ms))
        self.pose_hist.append(1.0 if pose_ok else 0.0)
        self.quality_hist.append(float(quality_score))

        if self.prev_motors is not None:
            d = [abs(float(a) - float(b)) for a, b in zip(motors, self.prev_motors)]
            self.jitter_hist.append(sum(d) / len(d))
        self.prev_motors = motors[:]

    def snapshot(self) -> Dict:
        if not self.dt_hist:
            return {
                "fps": 0.0,
                "proc_ms_avg": 0.0,
                "pose_valid_ratio": 0.0,
                "quality_score_avg": 0.0,
                "motor_jitter_avg_deg": 0.0,
            }
        fps = 1.0 / (sum(self.dt_hist) / len(self.dt_hist))
        proc = sum(self.proc_ms_hist) / len(self.proc_ms_hist) if self.proc_ms_hist else 0.0
        pose_ratio = sum(self.pose_hist) / len(self.pose_hist) if self.pose_hist else 0.0
        q = sum(self.quality_hist) / len(self.quality_hist) if self.quality_hist else 0.0
        jit = sum(self.jitter_hist) / len(self.jitter_hist) if self.jitter_hist else 0.0
        return {
            "fps": round(fps, 2),
            "proc_ms_avg": round(proc, 2),
            "pose_valid_ratio": round(pose_ratio, 3),
            "quality_score_avg": round(q, 3),
            "motor_jitter_avg_deg": round(jit, 3),
        }
