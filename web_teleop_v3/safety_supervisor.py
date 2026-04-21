import time
from typing import Dict, List, Optional


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _float_list_or_default(value, n: int, default: float) -> List[float]:
    if not isinstance(value, list) or len(value) != n:
        return [float(default)] * n
    out = []
    for i in range(n):
        try:
            out.append(float(value[i]))
        except Exception:
            out.append(float(default))
    return out


class SafetySupervisor:
    def __init__(self, config: dict):
        motors = config.get("motors", [])
        self.limits = []
        self.home = []
        for m in motors:
            self.limits.append((float(m["min_deg"]), float(m["max_deg"])))
            self.home.append(float(m.get("home_deg", 0.0)))

        self.max_step = _float_list_or_default(config.get("safety_max_step_deg", [8.0] * 6), 6, 8.0)
        self.max_velocity = _float_list_or_default(
            config.get("safety_max_velocity_deg_s", [120.0, 120.0, 140.0, 180.0, 180.0, 220.0]),
            6,
            150.0,
        )
        self.max_accel = _float_list_or_default(
            config.get("safety_max_accel_deg_s2", [600.0, 600.0, 700.0, 900.0, 900.0, 1200.0]),
            6,
            800.0,
        )
        self.soft_limit_margin = _float_list_or_default(config.get("safety_soft_limit_margin_deg", [8.0] * 6), 6, 8.0)
        self.soft_limit_pushback = _float_list_or_default(config.get("safety_soft_limit_pushback", [0.45] * 6), 6, 0.45)

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
        self.prev_vel = [0.0] * 6
        self.prev_t = time.time()

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
            self.prev_vel = [0.0] * 6
            self.prev_t = now
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
            self.prev_vel = [0.0] * 6
            self.prev_t = now
            return self.last_safe[:]

        dt = clamp(now - self.prev_t, 1.0 / 240.0, 0.25)
        safe = []
        next_vel = []
        for i in range(6):
            lo, hi = self.limits[i]
            t = clamp(float(target[i]), lo, hi)
            p = self.last_safe[i]
            step = clamp(t - p, -self.max_step[i], self.max_step[i])

            # Soft braking near joint limits to avoid slamming hard stops.
            margin = max(0.0, float(self.soft_limit_margin[i]))
            pushback = clamp(float(self.soft_limit_pushback[i]), 0.05, 1.0)
            dist_hi = hi - p
            dist_lo = p - lo
            if step > 0.0 and dist_hi < margin:
                step = min(step, max(0.0, dist_hi * pushback))
            elif step < 0.0 and dist_lo < margin:
                step = max(step, -max(0.0, dist_lo * pushback))

            desired_vel = clamp(step / dt, -self.max_velocity[i], self.max_velocity[i])
            max_dv = max(1e-6, self.max_accel[i] * dt)
            vel = clamp(desired_vel, self.prev_vel[i] - max_dv, self.prev_vel[i] + max_dv)
            t_next = clamp(p + vel * dt, lo, hi)

            safe.append(t_next)
            next_vel.append((t_next - p) / dt)

        self.last_safe = safe
        self.prev_vel = next_vel
        self.prev_t = now
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
