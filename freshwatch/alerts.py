"""Alert de-duplication, persistence and webhook delivery (non-blocking)."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AlertConfig, StorageConfig
from .metrics import ALERTS
from .storage import Store

log = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, cfg: AlertConfig, storage_cfg: StorageConfig, store: Store, site: dict) -> None:
        self.cfg = cfg
        self.store = store
        self.site = site
        self.snapshot_dir: Optional[Path] = storage_cfg.snapshot_dir
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last: dict[tuple[str, Optional[str], str], float] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._deliver_loop, name="alert-webhook", daemon=True)
        self._worker.start()

    def raise_alert(self, camera_id: str, zone_id: Optional[str], kind: str, message: str,
                    grade: Optional[str] = None, details: Optional[dict] = None,
                    frame: Optional[np.ndarray] = None, roi: Optional[tuple] = None) -> Optional[int]:
        if kind not in self.cfg.alert_on:
            return None
        key = (camera_id, zone_id, kind)
        now = time.time()
        with self._lock:
            if now - self._last.get(key, 0.0) < self.cfg.cooldown_s:
                return None
            self._last[key] = now

        snap = self._save_snapshot(frame, roi, camera_id, zone_id, kind, now) if frame is not None else None
        aid = self.store.add_alert(now, camera_id, zone_id, kind, grade, message, details or {}, snap)
        ALERTS.labels(kind=kind).inc()
        log.warning("ALERT #%d [%s] %s/%s: %s", aid, kind, camera_id, zone_id, message)
        payload = {"id": aid, "ts": now, "store": self.site, "camera_id": camera_id, "zone_id": zone_id,
                   "kind": kind, "grade": grade, "message": message, "details": details or {}}
        if self.cfg.webhook_url:
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                log.error("webhook queue full; dropping alert %d", aid)
        return aid

    def clear(self, camera_id: str, zone_id: Optional[str], kind: str) -> None:
        """Allow an immediate re-alert once a condition has recovered."""
        with self._lock:
            self._last.pop((camera_id, zone_id, kind), None)

    def _save_snapshot(self, frame, roi, camera_id, zone_id, kind, ts) -> Optional[str]:
        if not self.snapshot_dir:
            return None
        img = frame.copy()
        if roi:
            x, y, w, h = roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        name = f"{int(ts)}_{camera_id}_{zone_id or 'cam'}_{kind}.jpg"
        cv2.imwrite(str(self.snapshot_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return name

    def _deliver_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            body = json.dumps(payload).encode()
            for attempt in range(1, self.cfg.webhook_retries + 1):
                try:
                    req = urllib.request.Request(self.cfg.webhook_url, data=body, method="POST",
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.cfg.webhook_timeout_s) as resp:
                        if resp.status < 300:
                            break
                except Exception as e:  # noqa: BLE001 - network errors are expected
                    log.warning("webhook attempt %d/%d failed: %s", attempt, self.cfg.webhook_retries, e)
                    time.sleep(min(2 ** attempt, 30))

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2)
