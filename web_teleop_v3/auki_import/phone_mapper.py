def _wrap(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

class PhoneMapper:
    def __init__(self, config):
        self.alpha = config["smoothing_alpha"]
        self.ref = None
        self.motors = [m["home"] for m in config["motors"]]

    def compute(self, alpha, beta, gamma, gripper_open=False, clutch=False):
        if clutch or self.ref is None:
            self.ref = (alpha, beta, gamma)
            return list(self.motors)

        a0, b0, g0 = self.ref
        da = _wrap(alpha - a0)
        db = beta - b0
        dg = gamma - g0

        raw = [
            da  * 0.50,           # base yaw
            db  * 0.70,           # shoulder
            abs(db) * 0.50,       # elbow
            dg  * 1.00,           # forearm roll
            db  * 0.30,           # wrist pitch
            0.0 if gripper_open else 90.0,
        ]

        a = self.alpha
        self.motors = [a * r + (1 - a) * p for r, p in zip(raw, self.motors)]
        return list(self.motors)
