"""Per-camera analysis worker: frame -> GPU (CUDA kernels + TensorRT) -> zone states -> alerts/storage.

GPU work for one analysis pass, all on the worker's own CUDA stream:
    pinned frame --H2D--> d_frame
    fc_color_stats(d_frame, zones)              -> d_stats  (HSV spoilage cues)
    fc_preprocess_rois(d_frame, zones)          -> d_input  (NCHW, normalised)
    TensorRT execute_async_v3(d_input)          -> logits
    D2H stats + logits, single stream sync
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from . import cuda_ops as co
from .alerts import AlertManager
from .camera import CameraSource, Frame
from .config import CameraConfig, Settings, ZoneConfig
from .metrics import ANALYSES, ANALYSIS_ERRORS, STAGE_LATENCY, ZONE_SCORE, ZONE_SEVERITY
from .scoring import SEVERITY, Grade, ZoneState, make_reading, update_state
from .storage import Store
from .trt_engine import FreshnessEngine, softmax

if TYPE_CHECKING:
    from .detector import Detection, FruitDetector

log = logging.getLogger(__name__)

GRADE_BGR = {
    Grade.FRESH: (80, 175, 76), Grade.AGING: (0, 193, 255), Grade.STALE: (0, 120, 245),
    Grade.SPOILED: (40, 40, 220), Grade.EMPTY: (170, 170, 170), Grade.UNKNOWN: (200, 200, 200),
}
# Same wording as the dashboard
GRADE_LABEL = {
    Grade.FRESH: "Fresh", Grade.AGING: "Sell first", Grade.STALE: "Mark down",
    Grade.SPOILED: "Remove", Grade.EMPTY: "Restock", Grade.UNKNOWN: "Calibrating",
}
ALERT_GRADES = {Grade.STALE: "stale", Grade.SPOILED: "spoiled", Grade.EMPTY: "empty"}
STATS_SIZE = ctypes.sizeof(co.ColorStats)
TRACK_IOU = 0.3   # a detection continues a track when its box overlaps the track's last box this much
CROP_PAD = 0.05   # grade a little context around each detected item


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


class CameraWorker:
    def __init__(self, cam: CameraConfig, settings: Settings, engine: Optional[FreshnessEngine],
                 store: Store, alerts: AlertManager, detector: "Optional[FruitDetector]" = None) -> None:
        self.cam = cam
        self.s = settings
        self.engine = engine
        self.store = store
        self.alerts = alerts
        self.source = CameraSource(cam)
        self.rois = [z.roi for z in cam.zones]
        # buffers need a size; a camera without zones (and without detection) never runs a GPU pass
        n = settings.detector.max_items if cam.detect else max(len(cam.zones), 1)

        self.stream = co.Stream()
        self.h_frame = co.PinnedArray((cam.height, cam.width, 3))
        self.d_frame = co.DeviceBuffer(self.h_frame.nbytes)
        self.d_stats = co.DeviceBuffer(STATS_SIZE * n)
        self.h_stats = co.PinnedArray((STATS_SIZE * n,))
        c = settings.color
        self.color_params = co.ColorParams(c.brown_hue[0], c.brown_hue[1], c.brown_sat_min, c.brown_val[0],
                                           c.brown_val[1], c.dark_val_max, c.mould_sat_max, c.mould_val_min,
                                           c.fg_sat_min, c.fg_val_min)
        self.ctx = None
        if engine is not None:
            size = engine.input_size
            self.d_input = co.DeviceBuffer(n * 3 * size * size * 4)
            self.ctx = engine.create_context()
        self.known_produce = set(engine.produce) if engine else set()
        # Detect mode: produce is found anywhere in the frame; each find is graded like a zone ("track").
        self.det_ctx = detector.create_context() if cam.detect and detector is not None else None
        self._tracks: dict[str, dict] = {}  # zone id -> {"produce", "box", "seen"}
        self._next_track = 1
        # A detected item is never "empty", and finds come and go, so grade after 2 agreeing readings.
        self.scoring = settings.scoring.model_copy(update={"min_foreground_ratio": 0.0, "hysteresis_readings": 2}) \
            if cam.detect else settings.scoring
        zones = [] if cam.detect else cam.zones  # saved zones are kept for when detect is turned off
        self.zone_cfgs = {z.id: z for z in zones}

        self.states = {z.id: ZoneState(cam.id, z.id, z.name, z.produce, z.roi) for z in zones}
        self._state_lock = threading.Lock()
        self._mismatch_count: dict[str, int] = {}
        self._last_persist: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"worker-{cam.id}", daemon=True)
        self.last_analysis_ts = 0.0
        self.last_latency_ms: dict[str, float] = {}
        self._camera_down_alerted = False
        self._preview: tuple[int, bytes] = (-1, b"")
        self._preview_lock = threading.Lock()

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.source.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():  # also safe for a worker that was never started
            self._thread.join(timeout=10)
        self.source.stop()
        if self.ctx:
            self.ctx.close()
            self.d_input.free()
        if self.det_ctx:
            self.det_ctx.close()
        for b in (self.d_frame, self.d_stats):
            b.free()
        for h in (self.h_frame, self.h_stats):
            h.free()
        self.stream.close()

    @property
    def healthy(self) -> bool:
        return self.source.connected and time.time() - self.last_analysis_ts < 3 * self.cam.analyze_interval_s + 5

    # ---------------------------------------------------------------- analysis
    def analyze(self, frame: Frame, upload: bool = True):
        """Run one GPU pass over self.rois. Returns (cues, p_fresh, produce_idx, produce_conf) per zone.

        upload=False: the frame is already on the GPU (the detector pass uploaded it)."""
        n = len(self.rois)
        t0 = time.perf_counter()
        if upload:
            np.copyto(self.h_frame.array, frame.image)
            co.h2d(self.d_frame, self.h_frame, self.stream)
        w, h = self.cam.width, self.cam.height
        co.color_stats(self.d_frame, w, h, self.rois, self.color_params, self.s.color.sample_step,
                       self.d_stats, self.stream)
        co.d2h(self.h_stats, self.d_stats, self.stream)

        p_fresh = prod_idx = prod_conf = None
        if self.ctx is not None:
            eng = self.engine
            co.preprocess_rois(self.d_frame, w, h, self.rois, self.d_input, eng.input_size, eng.mean, eng.std,
                               self.stream)
            t1 = time.perf_counter()
            pl, fl = [], []
            plane = 3 * eng.input_size ** 2 * 4
            for start in range(0, n, eng.max_batch):
                k = min(eng.max_batch, n - start)
                view = co.DeviceBuffer.view(self.d_input.address + start * plane)
                a, b = self.ctx.infer(view, k, self.stream, sync=start + k >= n)
                pl.append(a), fl.append(b)
            produce_p = softmax(np.concatenate(pl))
            p_fresh = softmax(np.concatenate(fl))[:, 0]
            prod_idx = produce_p.argmax(1)
            prod_conf = produce_p.max(1)
            STAGE_LATENCY.labels(camera=self.cam.id, stage="inference").observe(time.perf_counter() - t1)
        else:
            self.stream.sync()

        stats = (co.ColorStats * n).from_buffer_copy(self.h_stats.array.tobytes())
        cues = [co.ColorCues.from_stats(s) for s in stats]
        dt = time.perf_counter() - t0
        STAGE_LATENCY.labels(camera=self.cam.id, stage="gpu_total").observe(dt)
        self.last_latency_ms["gpu_total"] = dt * 1e3
        return cues, p_fresh, prod_idx, prod_conf

    def _process(self, frame: Frame) -> None:
        if self.cam.detect:
            zones, upload = self._detect(frame), False
        else:
            zones, upload = self.cam.zones, True
        if not zones:  # preview-only camera, or nothing detected
            return
        self.rois = [z.roi for z in zones]
        cues, p_fresh, prod_idx, prod_conf = self.analyze(frame, upload=upload)
        t0 = time.perf_counter()
        sc = self.scoring
        for i, z in enumerate(zones):
            pf = None if p_fresh is None else float(p_fresh[i])
            pred = None if prod_idx is None else self.engine.produce[int(prod_idx[i])]
            conf = None if prod_conf is None else float(prod_conf[i])
            with self._state_lock:
                st = self.states[z.id]
                if self.cam.detect:
                    z = self._relabel(st, z, pred, conf)
                r = make_reading(z, cues[i], st.baseline, pf, pred, conf, self.known_produce, sc, self.s.color,
                                 frame.ts)
                change = update_state(st, r, sc)
            labels = dict(camera=self.cam.id, zone=z.id, produce=z.produce)
            if st.score is not None:
                ZONE_SCORE.labels(**labels).set(st.score)
            ZONE_SEVERITY.labels(**labels).set(SEVERITY[st.grade])
            if change:
                self._on_grade_change(st, change, frame)
            if not self.cam.detect:  # in detect mode the detector already decided what the item is
                self._check_mismatch(z.id, r, frame)
            if change or frame.ts - self._last_persist.get(z.id, 0.0) >= self.s.storage.persist_interval_s:
                self._persist(st)
                self._last_persist[z.id] = frame.ts
        self.last_latency_ms["postprocess"] = (time.perf_counter() - t0) * 1e3

    # ---------------------------------------------------------------- detect mode
    def _detect(self, frame: Frame) -> list[ZoneConfig]:
        """Upload the frame, find produce, and update tracks. Returns the items seen in this frame."""
        if self.det_ctx is None:
            return []
        t0 = time.perf_counter()
        np.copyto(self.h_frame.array, frame.image)
        co.h2d(self.d_frame, self.h_frame, self.stream)
        dets = self.det_ctx.detect(self.d_frame, self.cam.width, self.cam.height, self.stream)
        dt = time.perf_counter() - t0
        STAGE_LATENCY.labels(camera=self.cam.id, stage="detector").observe(dt)
        self.last_latency_ms["detector"] = dt * 1e3
        return self._track(dets, frame.ts)

    def _pad(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = box
        px, py = int(w * CROP_PAD), int(h * CROP_PAD)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(self.cam.width, x + w + px), min(self.cam.height, y + h + py)
        return x0, y0, x1 - x0, y1 - y0

    def _track(self, dets: "list[Detection]", now: float) -> list[ZoneConfig]:
        """Greedy IoU matching of detections to tracks of the same produce; unseen tracks expire."""
        pairs = sorted(((_iou(t["box"], self._pad(d.box)), tid, j) for tid, t in self._tracks.items()
                        for j, d in enumerate(dets) if t["produce"] == d.produce), reverse=True)
        assign: dict[int, str] = {}
        for v, tid, j in pairs:
            if v < TRACK_IOU:
                break
            if tid not in assign.values() and j not in assign:
                assign[j] = tid
        seen = []
        with self._state_lock:
            for j, d in enumerate(dets):
                box = self._pad(d.box)
                tid = assign.get(j)
                if tid is None:
                    tid = f"D{self._next_track}"
                    self._next_track += 1
                    taken = {st.name for st in self.states.values()}
                    k = next(k for k in range(1, 100) if f"{d.produce.title()} {k}" not in taken)
                    self.states[tid] = ZoneState(self.cam.id, tid, f"{d.produce.title()} {k}", d.produce, box)
                    self._tracks[tid] = {"produce": d.produce}
                self._tracks[tid].update(box=box, seen=now, score=d.score)
                self.states[tid].roi = box
                seen.append(ZoneConfig(id=tid, name=self.states[tid].name, produce=d.produce, roi=box))
            gone = [tid for tid, t in self._tracks.items() if now - t["seen"] > self.s.detector.track_ttl_s]
            for tid in gone:
                del self._tracks[tid], self.states[tid]
                self._last_persist.pop(tid, None)
        for tid in gone:  # the item left the display: its open alerts are resolved
            for kind in ALERT_GRADES.values():
                self.alerts.clear(self.cam.id, tid, kind)
        return seen

    def _relabel(self, st: ZoneState, z: ZoneConfig, pred: Optional[str], conf: Optional[float]) -> ZoneConfig:
        """The detector finds where produce is; COCO has no strawberry, guava, jujube... and calls them apples.
        When the freshness model is sure which of its fruits an item is, show that name instead."""
        if pred is None or pred == st.produce or (conf or 0) < self.s.scoring.min_produce_confidence:
            return z.model_copy(update={"produce": st.produce, "name": st.name})
        taken = {s.name for s in self.states.values() if s is not st}
        k = next(k for k in range(1, 100) if f"{pred.title()} {k}" not in taken)
        st.produce, st.name = pred, f"{pred.title()} {k}"
        st.reset_baseline()  # colour baseline of the old label does not apply
        return z.model_copy(update={"produce": pred, "name": st.name})

    def _on_grade_change(self, st: ZoneState, change, frame: Frame) -> None:
        old, new = change
        log.info("%s/%s %s: %s -> %s (score=%s)", self.cam.id, st.zone_id, st.produce, old.value, new.value,
                 None if st.score is None else round(st.score, 3))
        if new in ALERT_GRADES:
            kind = ALERT_GRADES[new]
            msg = f"{st.name} ({st.produce}) is {new.value.upper()}: {st.action}"
            self.alerts.raise_alert(self.cam.id, st.zone_id, kind, msg, grade=new.value,
                                    details=st.to_dict(self.s.scoring), frame=frame.image, roi=st.roi)
        if SEVERITY[new] < SEVERITY[old]:
            for kind in ALERT_GRADES.values():
                self.alerts.clear(self.cam.id, st.zone_id, kind)

    def _check_mismatch(self, zone_id: str, r, frame: Frame) -> None:
        self._mismatch_count[zone_id] = self._mismatch_count.get(zone_id, 0) + 1 if r.mismatch and not r.empty else 0
        if self._mismatch_count[zone_id] == self.s.scoring.hysteresis_readings * 2:
            st = self.states[zone_id]
            self.alerts.raise_alert(
                self.cam.id, zone_id, "mismatch",
                f"{st.name}: expected {st.produce}, camera sees {r.produce_pred} ({r.produce_conf:.0%}). "
                "Check bin labelling / price tag.",
                details={"expected": st.produce, "seen": r.produce_pred, "conf": r.produce_conf},
                frame=frame.image, roi=st.roi)

    def _persist(self, st: ZoneState) -> None:
        r = st.last
        self.store.add_reading({
            "ts": r.ts, "camera_id": self.cam.id, "zone_id": st.zone_id, "produce": st.produce,
            "score": st.score, "raw_score": r.raw_score, "p_fresh": r.p_fresh, "color_score": r.color_score,
            "grade": st.grade.value, "brown": r.cues.brown_ratio, "dark": r.cues.dark_ratio,
            "mould": r.cues.mould_ratio, "foreground": r.cues.foreground_ratio})

    def _run(self) -> None:
        interval = self.cam.analyze_interval_s
        seq = 0
        while not self._stop.is_set():
            t_start = time.monotonic()
            frame = self.source.wait_newer(seq, timeout=interval)
            if frame is None or time.time() - frame.ts > max(10.0, 3 * interval):
                if not self._camera_down_alerted and time.time() - self.last_analysis_ts > 30:
                    self._camera_down_alerted = True
                    self.alerts.raise_alert(self.cam.id, None, "camera_down",
                                            f"Camera {self.cam.name or self.cam.id} not delivering frames: "
                                            f"{self.source.last_error or 'no frames'}")
                continue
            if self._camera_down_alerted:
                self._camera_down_alerted = False
                self.alerts.clear(self.cam.id, None, "camera_down")
            seq = frame.seq
            try:
                self._process(frame)
                self.last_analysis_ts = time.time()
                ANALYSES.labels(camera=self.cam.id).inc()
            except Exception:  # noqa: BLE001 - keep the worker alive, surface via metrics/logs
                ANALYSIS_ERRORS.labels(camera=self.cam.id).inc()
                log.exception("analysis failed on camera %s", self.cam.id)
                self._stop.wait(1.0)
            self._stop.wait(max(0.0, interval - (time.monotonic() - t_start)))

    def reset_baseline(self, zone_id: str) -> bool:
        """Call after restocking/re-arranging a crate so colour drift is measured from the new batch."""
        with self._state_lock:
            st = self.states.get(zone_id)
            if st is None:
                return False
            st.reset_baseline()
        log.info("%s/%s colour baseline reset", self.cam.id, zone_id)
        return True

    # ---------------------------------------------------------------- views
    def snapshot(self) -> list[dict]:
        with self._state_lock:
            out = [st.to_dict(self.s.scoring) for st in self.states.values()]
        for d in out:
            z = self.zone_cfgs.get(d["zone_id"])
            w = z.model_weight if z is not None and z.model_weight is not None else self.s.scoring.model_weight
            d["camera_name"] = self.cam.name or self.cam.id
            d["analysis"] = "ai+colour" if d["produce"] in self.known_produce and w > 0 else "colour"
            d["detected"] = self.cam.detect
            if self.cam.detect and d["zone_id"] in self._tracks:
                d["detect_score"] = round(self._tracks[d["zone_id"]].get("score", 0.0), 3)
        return out

    def preview_jpeg(self, annotate: bool = True) -> Optional[bytes]:
        frame = self.source.latest()
        if frame is None:
            return None
        with self._preview_lock:
            if self._preview[0] == frame.seq:
                return self._preview[1]
            img = frame.image.copy()
            if annotate:
                with self._state_lock:
                    states = list(self.states.values())
                for st in states:
                    self._draw_zone(img, st)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.s.api.jpeg_quality])
            data = buf.tobytes() if ok else b""
            self._preview = (frame.seq, data)
            return data

    @staticmethod
    def _draw_zone(img: np.ndarray, st: ZoneState) -> None:
        x, y, w, h = st.roi
        color = GRADE_BGR[st.grade]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        label = f"{st.name}  {GRADE_LABEL[st.grade]}"
        if st.score is not None:
            label += f" {st.score * 100:.0f}"
        scale = 0.55
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw + 10 > w:  # narrow zone: keep the tag inside its box (the box colour already shows the grade)
            label = st.name if st.score is None else f"{st.name} {st.score * 100:.0f}"
            scale = 0.45
            while scale > 0.3 and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] + 10 > w:
                scale -= 0.05
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (x, y), (x + min(tw + 10, w), y + th + 10), color, -1)
        cv2.putText(img, label, (x + 5, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
