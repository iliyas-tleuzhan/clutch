import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional


class SessionLogger:
    """
    Single-file JSONL session logger.

    Each line is a JSON object with:
    - record_type: "frame" | "event"
    - run_id
    - timestamp
    - payload fields
    """

    def __init__(self, enabled: bool, log_path: Optional[Path], run_id: str):
        self.enabled = bool(enabled)
        self.log_path = log_path
        self.run_id = run_id
        self._lock = threading.Lock()
        self._fh = None
        if self.enabled and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open("a", encoding="utf-8")

    @classmethod
    def from_config(cls, config: Dict, project_root: Path):
        cfg = config.get("session_logging", {})
        enabled = bool(cfg.get("enabled", True))
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_name = str(cfg.get("file_name_template", "session_{run_id}.jsonl")).replace("{run_id}", run_id)
        logs_dir_cfg = str(cfg.get("logs_dir", "logs")).strip() or "logs"
        logs_dir = Path(logs_dir_cfg)
        if not logs_dir.is_absolute():
            logs_dir = (project_root / logs_dir).resolve()
        log_path = (logs_dir / log_name).resolve()
        return cls(enabled=enabled, log_path=log_path, run_id=run_id)

    def _write(self, obj: Dict):
        if not self.enabled or self._fh is None:
            return
        with self._lock:
            self._fh.write(json.dumps(obj, ensure_ascii=True) + "\n")
            self._fh.flush()

    def log_frame(self, fields: Dict):
        payload = {
            "record_type": "frame",
            "run_id": self.run_id,
            "timestamp": float(fields.get("timestamp", time.time())),
        }
        payload.update(fields)
        self._write(payload)

    def log_event(self, event_type: str, fields: Optional[Dict] = None):
        payload = {
            "record_type": "event",
            "run_id": self.run_id,
            "timestamp": time.time(),
            "event_type": str(event_type),
        }
        if isinstance(fields, dict):
            payload.update(fields)
        self._write(payload)

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
