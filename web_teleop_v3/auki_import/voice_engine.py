import re

_RULES = [
    (r'rotate left(?:\s+(\d+))?',   lambda g: ('delta', 0, -float(g or 20))),
    (r'rotate right(?:\s+(\d+))?',  lambda g: ('delta', 0, +float(g or 20))),
    (r'move up(?:\s+(\d+))?',       lambda g: ('delta', 1, +float(g or 20))),
    (r'move down(?:\s+(\d+))?',     lambda g: ('delta', 1, -float(g or 20))),
    (r'elbow up(?:\s+(\d+))?',      lambda g: ('delta', 2, +float(g or 20))),
    (r'elbow down(?:\s+(\d+))?',    lambda g: ('delta', 2, -float(g or 20))),
    (r'open gripper',               lambda g: ('abs',   5,  0.0)),
    (r'close gripper',              lambda g: ('abs',   5, 90.0)),
    (r'go home|home|reset',         lambda g: ('home',  None, None)),
    (r'stop|estop|emergency',       lambda g: ('estop', None, None)),
]

class VoiceEngine:
    def __init__(self, config):
        self.home = [m["home"] for m in config["motors"]]

    def process(self, text: str, current: list):
        t = text.strip().lower()
        motors = list(current)

        for pattern, handler in _RULES:
            m = re.search(pattern, t)
            if m:
                g = m.group(1) if m.lastindex else None
                action, idx, val = handler(g)
                if action == 'delta':
                    motors[idx] += val
                elif action == 'abs':
                    motors[idx] = val
                elif action == 'home':
                    return list(self.home)
                elif action == 'estop':
                    return list(current)
                return motors

        return None
