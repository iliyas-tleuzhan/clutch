"""
Starter hook for item #10 (voice commands).
This is intentionally lightweight; real microphone streaming can be integrated later.
"""


class VoiceCommandEngine:
    def __init__(self):
        self.enabled = False
        self.last_command = ""

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False

    def push_text_command(self, text: str):
        text = (text or "").strip().lower()
        self.last_command = text
        if text in {"freeze", "hold"}:
            return "toggle_freeze"
        if text in {"home", "reset pose"}:
            return "home"
        if text in {"estop", "stop", "emergency"}:
            return "toggle_estop"
        return None
