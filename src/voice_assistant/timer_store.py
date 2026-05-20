from __future__ import annotations

import json
import logging
import os
import time
from typing import List

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TIMERS_FILE = os.path.join(_PROJECT_ROOT, "timers.json")


def save(active_timers: list) -> None:
    """Persist active (non-cancelled) timers to disk."""
    data = [
        {"label": t["label"], "fire_at": t["fire_at"]}
        for t in active_timers
        if "fire_at" in t and not t["event"].is_set()
    ]
    try:
        with open(_TIMERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.debug(f"[TimerStore] {len(data)} timer(s) guardados en disco")
    except Exception as e:
        logging.error(f"[TimerStore] Error guardando timers: {e}")


def load() -> List[dict]:
    """Load pending timers from disk. Returns list of {label, fire_at, remaining_seconds}."""
    if not os.path.exists(_TIMERS_FILE):
        return []
    try:
        with open(_TIMERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        now = time.time()
        pending = []
        for entry in data:
            fire_at = entry.get("fire_at", 0)
            remaining = fire_at - now
            pending.append(
                {
                    "label": entry["label"],
                    "fire_at": fire_at,
                    # Minimum 5s so the assistant finishes startup before firing
                    "remaining_seconds": max(5.0, remaining),
                    "already_expired": remaining <= 0,
                }
            )
        logging.info(f"[TimerStore] {len(pending)} timer(s) cargados desde disco")
        return pending
    except Exception as e:
        logging.error(f"[TimerStore] Error cargando timers: {e}")
        return []


def clear() -> None:
    """Delete the timers file (e.g. after all timers are cancelled)."""
    try:
        if os.path.exists(_TIMERS_FILE):
            os.remove(_TIMERS_FILE)
            logging.debug("[TimerStore] Archivo de timers eliminado")
    except Exception as e:
        logging.error(f"[TimerStore] Error borrando archivo: {e}")
