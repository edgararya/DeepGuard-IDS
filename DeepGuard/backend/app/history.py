import json
import os
import threading
import time

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_HISTORY_FILE = os.path.join(_BACKEND_DIR, "data", "history.json")


class HistoryStore:
    """Thread-safe detection history with best-effort JSON persistence."""

    def __init__(self, file_path=None):
        self._lock = threading.Lock()
        self._file_path = file_path or os.environ.get("HISTORY_FILE") or _DEFAULT_HISTORY_FILE
        self._entries = []
        self._counter = 0
        self._load()

    def _load(self):
        self._entries = []
        self._counter = 0
        try:
            with open(self._file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            entries = data.get("entries", []) if isinstance(data, dict) else []
            for e in entries:
                if isinstance(e, dict) and isinstance(e.get("id"), str):
                    self._entries.append(e)
                    suffix = e["id"].split("_")[-1]
                    if suffix.isdigit():
                        self._counter = max(self._counter, int(suffix))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._entries = []
            self._counter = 0

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
            with open(self._file_path, "w", encoding="utf-8") as fh:
                json.dump({"entries": self._entries}, fh)
        except OSError:
            pass

    def add(self, source: str, label: str, confidence: float, timestamp: int = None) -> dict:
        with self._lock:
            self._counter += 1
            entry = {
                "id": f"h_{self._counter:03d}",
                "source": source,
                "label": label,
                "confidence": round(float(confidence), 4),
                "timestamp": int(timestamp) if timestamp is not None else int(time.time() * 1000),
            }
            self._entries.append(entry)
            self._save()
            return entry

    def list(self) -> list:
        with self._lock:
            return list(self._entries)

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            for i, e in enumerate(self._entries):
                if e.get("id") == entry_id:
                    del self._entries[i]
                    self._save()
                    return True
            return False


history_store = HistoryStore()
