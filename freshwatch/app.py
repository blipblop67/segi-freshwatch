"""Service wiring + FastAPI HTTP API and dashboard."""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import __version__, cuda_ops
from .alerts import AlertManager
from .camera import redact
from .camera_admin import CameraOverrides, build_rtsp, discover, grab_frame, parse_rtsp, probe_rtsp, with_changes
from .config import CameraConfig, Settings, ZoneConfig
from .pipeline import CameraWorker
from .scoring import SEVERITY
from .storage import Store
from .trt_engine import FreshnessEngine

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"


class Service:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.started = time.time()
        self.gpu = cuda_ops.device_info()
        self.gpu["device"] = _device_model() or self.gpu["name"]
        log.info("GPU: %s (sm_%s, %.1f GB)", self.gpu["name"], self.gpu["sm"].replace(".", ""),
                 self.gpu["total_mem"] / 1e9)
        self.engine: Optional[FreshnessEngine] = None
        self.engine_error: Optional[str] = None
        try:
            self.engine = FreshnessEngine(settings.model.engine, settings.model.labels, settings.model.max_batch)
        except Exception as e:  # noqa: BLE001
            if settings.model.required:
                raise
            self.engine_error = str(e)
            log.error("model unavailable, running DEGRADED on colour cues only: %s", e)
        # Object detector for cameras that find produce anywhere (detect: true). Optional: without it
        # those cameras show live video only, and the dashboard says why.
        self.detector = None
        self.detector_error: Optional[str] = None
        try:
            from .detector import FruitDetector
            self.detector = FruitDetector(settings.detector)
        except Exception as e:  # noqa: BLE001
            self.detector_error = str(e)
            if any(c.detect for c in settings.cameras):
                log.error("detector unavailable, detect cameras show live video only: %s", e)
        self.store = Store(settings.storage.db_path)
        site = settings.site.model_dump()
        self.alerts = AlertManager(settings.alerts, settings.storage, self.store, site)
        # Camera edits from the dashboard live next to the database and are applied over the YAML.
        self.overrides = CameraOverrides(settings.storage.db_path.parent / "cameras.json")
        self.yaml_camera_ids = {c.id for c in settings.cameras}
        settings.cameras = self.overrides.apply(settings.cameras)
        self.workers = {c.id: CameraWorker(c, settings, self.engine, self.store, self.alerts, self.detector)
                        for c in settings.cameras}
        self._cameras_lock = threading.Lock()
        self._stop = threading.Event()
        self._janitor = threading.Thread(target=self._housekeeping, name="janitor", daemon=True)

    def start(self) -> None:
        for w in self.workers.values():
            w.start()
        self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        for w in self.workers.values():
            w.stop()
        self.alerts.close()
        self.store.close()

    # Workers are swapped copy-on-write: readers iterate a dict that is never mutated in place.
    def put_camera(self, cam: CameraConfig, zones_changed: bool = False) -> None:
        """Add a camera, or restart one with new settings, without restarting the service."""
        with self._cameras_lock:
            new = CameraWorker(cam, self.s, self.engine, self.store, self.alerts, self.detector)
            old = self.workers.get(cam.id)
            if old is not None:
                old.stop()
                self.alerts.clear(cam.id, None, "camera_down")
            new.start()
            self.workers = {**self.workers, cam.id: new}
            self.overrides.save_camera(cam, added=cam.id not in self.yaml_camera_ids, zones_changed=zones_changed)
        log.info("camera %s %s (%s)", cam.id, "updated" if old else "added", redact(cam.source))

    def remove_camera(self, cam_id: str) -> bool:
        with self._cameras_lock:
            old = self.workers.get(cam_id)
            if old is None:
                return False
            self.workers = {k: v for k, v in self.workers.items() if k != cam_id}
            old.stop()
            self.alerts.clear(cam_id, None, "camera_down")
            self.overrides.remove_camera(cam_id, from_yaml=cam_id in self.yaml_camera_ids)
        log.info("camera %s removed", cam_id)
        return True

    def _housekeeping(self) -> None:
        while not self._stop.wait(3600):
            try:
                n = self.store.purge(self.s.storage.retention_days)
                if n:
                    log.info("retention purge removed %d rows", n)
            except Exception:  # noqa: BLE001
                log.exception("retention purge failed")


class CameraForm(BaseModel):
    """Camera connection as entered on the Settings page."""
    name: str = Field(min_length=1, max_length=60)
    note: str = Field("", max_length=60)
    kind: Literal["rtsp", "keep"] = "rtsp"  # "keep": leave a file/USB/CSI source as it is, only rename
    host: str = ""
    port: int = 554
    path: str = "/"
    username: str = Field("", max_length=64)
    password: Optional[str] = Field(None, max_length=128)  # None: keep the saved password
    codec: Literal["h264", "h265"] = "h264"
    detect: bool = False  # find produce anywhere in the picture instead of grading fixed zones


class CameraTest(CameraForm):
    camera_id: Optional[str] = None  # to reuse the saved password while testing an edit


def _camera_view(w: CameraWorker, yaml_ids: set[str]) -> dict:
    cam = w.cam
    rtsp = parse_rtsp(cam.source)
    kind = "rtsp" if rtsp else cam.source.split(":", 1)[0]
    label = {"file": "Video file", "usb": "USB camera", "csi": "CSI camera", "gst": "Custom pipeline"}.get(kind, "")
    out = {"id": cam.id, "name": cam.name, "note": cam.note, "kind": kind, "codec": cam.codec, "detect": cam.detect,
           "zones": len(cam.zones), "connected": w.source.connected, "error": w.source.last_error,
           "from_config": cam.id in yaml_ids,
           "source": redact(cam.source) if rtsp else f"{label}: {Path(cam.source.split(':', 1)[-1]).name}"}
    if rtsp:
        out |= {k: rtsp[k] for k in ("host", "port", "path", "username")} | {"has_password": bool(rtsp["password"])}
    return out


def create_app(settings: Settings, service: Optional[Service] = None) -> FastAPI:
    svc_holder: dict[str, Service] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = service or Service(settings)
        svc_holder["svc"] = svc
        svc.start()
        try:
            yield
        finally:
            svc.stop()

    app = FastAPI(title="SEGI FreshWatch", version=__version__, lifespan=lifespan)

    def svc() -> Service:
        return svc_holder["svc"]

    def require_key(x_api_key: Optional[str] = Header(default=None)) -> str:
        key = settings.api.api_key
        if key and x_api_key != key:
            raise HTTPException(401, "invalid or missing X-API-Key")
        return "api-key" if key else "anonymous"

    def worker(camera_id: str):
        w = svc().workers.get(camera_id)
        if w is None:
            raise HTTPException(404, f"unknown camera {camera_id}")
        return w

    @app.get("/", include_in_schema=False)
    def index():
        # no-cache: browsers revalidate, so a service upgrade never leaves a stale dashboard open
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "uptime_s": round(time.time() - svc().started, 1)}

    @app.get("/readyz")
    def readyz():
        s = svc()
        cams = {cid: w.healthy for cid, w in s.workers.items()}
        ready = all(cams.values())
        body = {"ready": ready, "degraded": s.engine is None, "model_error": s.engine_error, "cameras": cams}
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/status")
    def status():
        s = svc()
        zones = [z for w in s.workers.values() for z in w.snapshot()]
        summary: dict[str, int] = {}
        for z in zones:
            summary[z["grade"]] = summary.get(z["grade"], 0) + 1
        return {
            "site": settings.site.model_dump(),
            "version": __version__,
            "time": time.time(),
            "uptime_s": round(time.time() - s.started, 1),
            "gpu": s.gpu,
            "model": None if s.engine is None else {"produce": s.engine.produce, "input_size": s.engine.input_size,
                                                    "max_batch": s.engine.max_batch},
            "degraded": s.engine is None,
            "summary": summary,
            "worst_severity": max((SEVERITY_BY_NAME[z["grade"]] for z in zones), default=0),
            "cameras": [{"id": w.cam.id, "name": w.cam.name, "note": w.cam.note, "connected": w.source.connected,
                         "error": w.source.last_error, "healthy": w.healthy,
                         "last_analysis": w.last_analysis_ts, "latency_ms": w.last_latency_ms,
                         "width": w.cam.width, "height": w.cam.height, "detect": w.cam.detect,
                         "detector_error": s.detector_error if w.cam.detect and w.det_ctx is None else None}
                        for w in s.workers.values()],
            "zones": sorted(zones, key=lambda z: (-z["severity"], z["camera_id"], z["zone_id"])),
            "active_alerts": len(s.store.alerts(active_only=True, limit=1000)),
        }

    @app.get("/api/v1/zones/{camera_id}/{zone_id}/history")
    def history(camera_id: str, zone_id: str, hours: float = Query(24, gt=0, le=24 * 31),
                max_points: int = Query(500, ge=10, le=5000)):
        w = worker(camera_id)
        if zone_id not in w.states:
            raise HTTPException(404, f"unknown zone {zone_id}")
        since = time.time() - hours * 3600
        persisted = svc().store.history(camera_id, zone_id, since, max_points)
        live = [{"ts": t, "score": sc} for t, sc in list(w.states[zone_id].history) if t >= since]
        return {"camera_id": camera_id, "zone_id": zone_id, "persisted": persisted, "live": live[-max_points:]}

    @app.post("/api/v1/zones/{camera_id}/{zone_id}/baseline/reset")
    def reset_baseline(camera_id: str, zone_id: str, _: str = Depends(require_key)):
        if not worker(camera_id).reset_baseline(zone_id):
            raise HTTPException(404, f"unknown zone {zone_id}")
        return {"ok": True}

    @app.get("/api/v1/alerts")
    def alerts(active: bool = False, limit: int = Query(100, ge=1, le=1000)):
        return svc().store.alerts(active_only=active, limit=limit)

    @app.post("/api/v1/alerts/{alert_id}/ack")
    def ack(alert_id: int, request: Request, who: str = Depends(require_key)):
        by = request.headers.get("X-User", who)
        if not svc().store.ack_alert(alert_id, by):
            raise HTTPException(404, "alert not found or already acknowledged")
        return {"ok": True}

    @app.get("/api/v1/alerts/{alert_id}/snapshot.jpg")
    def alert_snapshot(alert_id: int):
        d = settings.storage.snapshot_dir
        rows = [a for a in svc().store.alerts(limit=1000) if a["id"] == alert_id]
        if not rows or not rows[0]["snapshot"] or d is None:
            raise HTTPException(404, "no snapshot")
        path = (d / rows[0]["snapshot"]).resolve()
        if path.parent != d.resolve() or not path.is_file():
            raise HTTPException(404, "no snapshot")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/cameras/{camera_id}/snapshot.jpg")
    def snapshot(camera_id: str, annotate: bool = True):
        data = worker(camera_id).preview_jpeg(annotate)
        if not data:
            raise HTTPException(503, "no frame yet")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/cameras/{camera_id}/stream.mjpg")
    async def stream(camera_id: str, request: Request):
        worker(camera_id)
        period = 1.0 / settings.api.preview_fps

        async def gen():
            while not await request.is_disconnected():
                w = svc().workers.get(camera_id)  # looked up each frame: follows a camera restarted from Settings
                if w is None:
                    break
                data = await asyncio.to_thread(w.preview_jpeg, True)
                if data:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                           str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                await asyncio.sleep(period)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # ---------------------------------------------------------------- camera settings (staff key)
    def resolve_source(form: CameraForm, current: Optional[CameraConfig]) -> str:
        if form.kind == "keep":
            if current is None or parse_rtsp(current.source) is not None:
                raise HTTPException(422, "only an existing non-RTSP camera can keep its source")
            return current.source
        password = form.password
        if password is None:  # not retyped: keep the saved one
            saved = parse_rtsp(current.source) if current else None
            password = saved["password"] if saved else ""
        try:
            return build_rtsp(form.host, form.port, form.path, form.username.strip(), password)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    def camera_config(form: CameraForm, current: Optional[CameraConfig], cam_id: str) -> CameraConfig:
        source = resolve_source(form, current)
        try:
            if current is None:
                return CameraConfig(id=cam_id, name=form.name.strip(), note=form.note, source=source,
                                    codec=form.codec, detect=form.detect, analyze_interval_s=2.0)
            return with_changes(current, name=form.name.strip(), note=form.note, source=source, codec=form.codec,
                                detect=form.detect)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/v1/settings/cameras")
    def list_cameras(_: str = Depends(require_key)):
        s = svc()
        return [_camera_view(w, s.yaml_camera_ids) for w in s.workers.values()]

    @app.post("/api/v1/settings/cameras", status_code=201)
    async def add_camera(form: CameraForm, _: str = Depends(require_key)):
        s = svc()
        slug = re.sub(r"[^a-z0-9]+", "-", form.name.lower()).strip("-")[:30] or "camera"
        cam_id, i = slug, 2
        while cam_id in s.workers:
            cam_id, i = f"{slug}-{i}", i + 1
        cam = camera_config(form, None, cam_id)
        await asyncio.to_thread(s.put_camera, cam)
        return _camera_view(s.workers[cam_id], s.yaml_camera_ids)

    @app.put("/api/v1/settings/cameras/{camera_id}")
    async def update_camera(camera_id: str, form: CameraForm, _: str = Depends(require_key)):
        s = svc()
        cam = camera_config(form, worker(camera_id).cam, camera_id)
        await asyncio.to_thread(s.put_camera, cam)
        return _camera_view(s.workers[camera_id], s.yaml_camera_ids)

    @app.put("/api/v1/settings/cameras/{camera_id}/zones")
    async def set_zones(camera_id: str, zones: list[ZoneConfig], _: str = Depends(require_key)):
        """Replace a camera's produce displays (ROIs in its width x height frame); grading restarts for it."""
        try:
            cam = CameraConfig.model_validate(worker(camera_id).cam.model_dump() | {
                "zones": [z.model_dump() for z in zones]})
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        await asyncio.to_thread(svc().put_camera, cam, True)
        return _camera_view(svc().workers[camera_id], svc().yaml_camera_ids) | {
            "zone_list": [z.model_dump(mode="json") for z in cam.zones]}

    @app.delete("/api/v1/settings/cameras/{camera_id}")
    async def delete_camera(camera_id: str, _: str = Depends(require_key)):
        if not await asyncio.to_thread(svc().remove_camera, camera_id):
            raise HTTPException(404, f"unknown camera {camera_id}")
        return {"ok": True}

    @app.post("/api/v1/settings/cameras/test")
    async def test_camera(form: CameraTest, _: str = Depends(require_key)):
        current = svc().workers[form.camera_id].cam if form.camera_id in svc().workers else None
        if form.kind == "keep":
            raise HTTPException(422, "only IP cameras can be tested")
        source = resolve_source(form, current)
        res = await asyncio.to_thread(probe_rtsp, source)
        if not res["ok"]:
            return res
        cam = CameraConfig(id="test", source=source, codec=res["codec"])
        jpg, msg = await asyncio.to_thread(grab_frame, cam)
        if jpg is None:
            return res | {"ok": False, "stage": "decode", "message": f"{res['message']} {msg}"}
        return res | {"message": f"{res['message']} Picture received.",
                      "snapshot": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}

    @app.post("/api/v1/settings/discover")
    async def discover_cameras(_: str = Depends(require_key)):
        return await asyncio.to_thread(discover)

    return app


SEVERITY_BY_NAME = {g.value: v for g, v in SEVERITY.items()}


def _device_model() -> Optional[str]:
    """Friendly board name, e.g. "Jetson Orin Nano Super", from the device tree."""
    try:
        raw = Path("/proc/device-tree/model").read_bytes().decode(errors="ignore").strip("\x00 \n")
    except OSError:
        return None
    for noise in ("NVIDIA ", " Engineering Reference", " Developer Kit"):
        raw = raw.replace(noise, "")
    return " ".join(raw.split()) or None
