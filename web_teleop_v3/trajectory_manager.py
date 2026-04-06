import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


class TrajectoryManager:
    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

        self.recording = False
        self.record_name = ""
        self.record_t0 = 0.0
        self.frames: List[Dict] = []

        self.playing = False
        self.play_name = ""
        self.play_t0 = 0.0
        self.play_frames: List[Dict] = []

    def _safe_name(self, name: str) -> str:
        out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
        return out or f"traj_{int(time.time())}"

    def list(self) -> List[str]:
        return sorted([p.stem for p in self.root.glob("*.json")])

    def start_record(self, name: str):
        with self.lock:
            self.recording = True
            self.record_name = self._safe_name(name)
            self.record_t0 = time.time()
            self.frames = []

    def append(self, motors_deg: List[float]):
        with self.lock:
            if not self.recording:
                return
            self.frames.append(
                {
                    "t": time.time() - self.record_t0,
                    "motors_deg": [float(v) for v in motors_deg],
                }
            )

    def stop_record(self) -> Optional[Path]:
        with self.lock:
            if not self.recording:
                return None
            payload = {
                "name": self.record_name,
                "recorded_at": time.time(),
                "num_frames": len(self.frames),
                "frames": self.frames,
            }
            out_path = self.root / f"{self.record_name}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.recording = False
            self.record_name = ""
            self.frames = []
            return out_path

    def start_playback(self, name: str) -> bool:
        path = self.root / f"{self._safe_name(name)}.json"
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if not isinstance(frames, list) or not frames:
            return False
        with self.lock:
            self.playing = True
            self.play_name = path.stem
            self.play_t0 = time.time()
            self.play_frames = frames
        return True

    def stop_playback(self):
        with self.lock:
            self.playing = False
            self.play_name = ""
            self.play_frames = []
            self.play_t0 = 0.0

    def sample_playback(self) -> Optional[List[float]]:
        with self.lock:
            if not self.playing or not self.play_frames:
                return None
            t = time.time() - self.play_t0
            frames = self.play_frames
            if t >= float(frames[-1]["t"]):
                self.playing = False
                return [float(v) for v in frames[-1]["motors_deg"]]
            idx = 0
            while idx + 1 < len(frames) and float(frames[idx + 1]["t"]) < t:
                idx += 1
            f0 = frames[idx]
            f1 = frames[idx + 1]
            t0 = float(f0["t"])
            t1 = float(f1["t"])
            u = 0.0 if abs(t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)
            out = []
            for a, b in zip(f0["motors_deg"], f1["motors_deg"]):
                out.append(float(a) + (float(b) - float(a)) * u)
            return out

    def snapshot(self) -> Dict:
        with self.lock:
            return {
                "recording": self.recording,
                "record_name": self.record_name,
                "playing": self.playing,
                "play_name": self.play_name,
                "available": self.list(),
            }
