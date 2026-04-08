import time
from typing import Dict, List, Optional


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SafetySupervisor:
    def __init__(self, config: dict):
        motors = config.get("motors", [])
        self.limits = []
        self.home = []
        for m in motors:
            self.limits.append((float(m["min_deg"]), float(m["max_deg"])))
            self.home.append(float(m.get("home_deg", 0.0)))

        self.max_step = config.get("safety_max_step_deg", [8.0] * 6)
        if not isinstance(self.max_step, list) or len(self.max_step) != 6:
            self.max_step = [8.0] * 6
        self.max_step = [float(v) for v in self.max_step]

        self.pose_required = bool(config.get("safety_require_pose", True))
        self.hand_required = bool(config.get("safety_require_hand", False))
        self.pose_hold_frames = int(config.get("safety_lost_hold_frames", 10))
        self.freeze = False
        self.estop = False
        self.last_safe = self.home[:]
        self.lost_count = 0
        self.last_macro = ""
        self.last_macro_ts = 0.0
        self.home_request_until = 0.0
        self._freeze_before_estop = False

    def trigger_macro(self, macro_name: str):
        now = time.time()
        self.last_macro = macro_name
        self.last_macro_ts = now
        if macro_name == "toggle_freeze":
            self.freeze = not self.freeze
        elif macro_name == "toggle_estop":
            if not self.estop:
                # Entering E-Stop: latch current freeze state and force freeze.
                self._freeze_before_estop = bool(self.freeze)
                self.estop = True
                self.freeze = True
            else:
                # Leaving E-Stop: restore the operator's previous freeze intent.
                self.estop = False
                self.freeze = bool(self._freeze_before_estop)
                self.lost_count = 0
        elif macro_name == "home":
            self.home_request_until = now + 1.0

    def clear_estop(self):
        self.estop = False
        self.freeze = bool(self._freeze_before_estop)
        self.lost_count = 0

    def set_freeze(self, freeze: bool):
        self.freeze = bool(freeze)

    def process(
        self,
        target_motors: List[float],
        pose_ok: bool,
        hand_ok: bool,
        quality: Optional[Dict[str, bool]] = None,
    ) -> List[float]:
        now = time.time()
        quality = quality or {}

        if self.estop:
            return self.last_safe[:]

        if now < self.home_request_until:
            target = self.home[:]
        else:
            target = target_motors[:]

        valid = True
        if self.pose_required and not pose_ok:
            valid = False
        if self.hand_required and not hand_ok:
            valid = False
        if quality.get("critical_low_light", False):
            valid = False

        if not valid:
            self.lost_count += 1
        else:
            self.lost_count = 0

        if self.freeze or self.lost_count > self.pose_hold_frames:
            return self.last_safe[:]

        safe = []
        for i in range(6):
            lo, hi = self.limits[i]
            t = clamp(float(target[i]), lo, hi)
            p = self.last_safe[i]
            t = clamp(t, p - self.max_step[i], p + self.max_step[i])
            safe.append(t)
        self.last_safe = safe
        return safe

    def snapshot(self) -> Dict:
        return {
            "freeze": self.freeze,
            "estop": self.estop,
            "lost_count": self.lost_count,
            "last_macro": self.last_macro,
            "last_macro_ts": self.last_macro_ts,
            "home_request_active": time.time() < self.home_request_until,
        }
