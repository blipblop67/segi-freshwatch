"""SQLite (WAL) persistence for readings and alerts. Safe to call from any thread."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts REAL NOT NULL, camera_id TEXT NOT NULL, zone_id TEXT NOT NULL, produce TEXT,
    score REAL, raw_score REAL, p_fresh REAL, color_score REAL, grade TEXT NOT NULL,
    brown REAL, dark REAL, mould REAL, foreground REAL
);
CREATE INDEX IF NOT EXISTS ix_readings_zone_ts ON readings(camera_id, zone_id, ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, camera_id TEXT NOT NULL,
    zone_id TEXT, kind TEXT NOT NULL, grade TEXT, message TEXT NOT NULL, details TEXT,
    snapshot TEXT, acknowledged_ts REAL, acknowledged_by TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_ts ON alerts(ts);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add_reading(self, row: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO readings VALUES (:ts,:camera_id,:zone_id,:produce,:score,:raw_score,:p_fresh,"
                ":color_score,:grade,:brown,:dark,:mould,:foreground)", row)

    def history(self, camera_id: str, zone_id: str, since: float, max_points: int = 500) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, score, grade, p_fresh, color_score FROM readings "
                "WHERE camera_id=? AND zone_id=? AND ts>=? ORDER BY ts", (camera_id, zone_id, since)).fetchall()
        step = max(1, len(rows) // max_points)
        return [dict(r) for r in rows[::step]]

    def add_alert(self, ts: float, camera_id: str, zone_id: Optional[str], kind: str, grade: Optional[str],
                  message: str, details: dict, snapshot: Optional[str]) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO alerts (ts,camera_id,zone_id,kind,grade,message,details,snapshot) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, camera_id, zone_id, kind, grade, message, json.dumps(details), snapshot))
            return int(cur.lastrowid)

    def alerts(self, active_only: bool = False, limit: int = 100) -> list[dict]:
        q = "SELECT * FROM alerts" + (" WHERE acknowledged_ts IS NULL" if active_only else "") + \
            " ORDER BY ts DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(q, (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d["details"] or "{}")
            out.append(d)
        return out

    def ack_alert(self, alert_id: int, who: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE alerts SET acknowledged_ts=?, acknowledged_by=? WHERE id=? AND acknowledged_ts IS NULL",
                (time.time(), who, alert_id))
            return cur.rowcount == 1

    def purge(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            n = self._db.execute("DELETE FROM readings WHERE ts < ?", (cutoff,)).rowcount
            n += self._db.execute("DELETE FROM alerts WHERE ts < ? AND acknowledged_ts IS NOT NULL",
                                  (cutoff,)).rowcount
        return n
