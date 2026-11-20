"""Frame sources with hardware decode where available, auto-reconnect and latest-frame semantics.

Orin Nano has NVDEC (nvv4l2decoder) but no NVENC, so decode is hardware and the
only encode we do is JPEG for previews/evidence.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlsplit, urlunsplit

import cv2
import numpy as np

from .config import CameraConfig
from .metrics import CAMERA_UP, FRAMES_CAPTURED

log = logging.getLogger(__name__)


def _scale(w: int, h: int) -> str:
    return (f"nvvidconv ! video/x-raw,width={w},height={h},format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


def _gst_quote(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def split_credentials(url: str) -> tuple[str, str, str]:
    """rtsp://user:pass@host:port/path -> (url without credentials, user, password)."""
    u = urlsplit(url)
    if u.username is None:
        return url, "", ""
    netloc = u.hostname if ":" not in (u.hostname or "") else f"[{u.hostname}]"
    if u.port:
        netloc += f":{u.port}"
    return urlunsplit(u._replace(netloc=netloc)), unquote(u.username), unquote(u.password or "")


def redact(source: str) -> str:
    """Source string safe for logs and the dashboard (password removed)."""
    if not source.startswith("rtsp://"):
        return source
    bare, user, pw = split_credentials(source)
    if not user:
        return source
    return bare.replace("rtsp://", f"rtsp://{user}:***@" if pw else f"rtsp://{user}@", 1)


def build_pipeline(cfg: CameraConfig) -> tuple[str, int]:
    """Return (OpenCV source string, backend) for a camera config."""
    src, w, h, fps = cfg.source, cfg.width, cfg.height, int(round(cfg.fps))
    if src.startswith("gst:"):
        return src[4:], cv2.CAP_GSTREAMER
    if src.startswith("rtsp://"):
        # Credentials go in rtspsrc properties, quoted, so any password characters are safe in the pipeline.
        url, user, pw = split_credentials(src)
        auth = f" user-id={_gst_quote(user)} user-pw={_gst_quote(pw)}" if user else ""
        c = cfg.codec
        return (f"rtspsrc location={_gst_quote(url)}{auth} latency=200 protocols=tcp ! rtp{c}depay ! {c}parse ! "
                f"nvv4l2decoder ! {_scale(w, h)}"), cv2.CAP_GSTREAMER
    if src.startswith("csi:"):
        sid = int(src[4:] or 0)
        return (f"nvarguscamerasrc sensor-id={sid} ! video/x-raw(memory:NVMM),width={w},height={h},"
                f"framerate={fps}/1 ! {_scale(w, h)}"), cv2.CAP_GSTREAMER
    if src.startswith("usb:"):
        dev = src[4:]
        return (f"v4l2src device={dev} ! image/jpeg,width={w},height={h},framerate={fps}/1 ! "
                f"nvv4l2decoder mjpeg=1 ! {_scale(w, h)}"), cv2.CAP_GSTREAMER
    if src.startswith("file:"):
        return src[5:], cv2.CAP_FFMPEG
    raise ValueError(f"unsupported camera source: {src}")


AUTH_RETRY_S = 15 * 60


class LoginRefused(RuntimeError):
    """The camera rejected the username/password."""


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    seq: int


class CameraSource:
    """Background reader thread that always holds the newest frame."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.uri, self.backend = build_pipeline(cfg)
        self.is_file = cfg.source.startswith("file:")
        self._frame: Optional[Frame] = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{cfg.id}", daemon=True)
        self.connected = False
        self.last_error: Optional[str] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._frame

    def wait_newer(self, seq: int, timeout: float) -> Optional[Frame]:
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None and self._frame.seq > seq, timeout=timeout)
            return self._frame

    def _check_login(self) -> None:
        """One RTSP DESCRIBE before GStreamer connects (GStreamer retries a refused login several times).

        IP cameras lock the account after repeated failed logins, which also cuts off the vendor's own
        app, so a wrong password must cost one attempt per AUTH_RETRY_S, not one every few seconds.
        """
        from .camera_admin import probe_rtsp  # camera_admin imports this module

        res = probe_rtsp(self.cfg.source)
        if res["ok"]:
            if res["codec"] != self.cfg.codec:  # the camera knows best: follow its actual encoding
                log.warning("camera %s sends %s, not %s: using %s", self.cfg.id, res["codec"], self.cfg.codec,
                            res["codec"])
                self.uri, self.backend = build_pipeline(self.cfg.model_copy(update={"codec": res["codec"]}))
            return
        if res["stage"] == "auth":
            raise LoginRefused(f"{res['message'].rstrip('.')}. Fix it in Cameras settings "
                               f"(next try in {AUTH_RETRY_S // 60} min, to avoid locking the camera)")
        raise RuntimeError(res["message"].rstrip("."))

    def _open(self) -> cv2.VideoCapture:
        if self.cfg.source.startswith("rtsp://"):
            self._check_login()
        cap = cv2.VideoCapture(self.uri, self.backend)
        if not cap.isOpened():
            if self.cfg.source.startswith("rtsp://"):
                raise RuntimeError("cannot connect to the camera. Check its IP address and login in Cameras settings")
            raise RuntimeError(f"cannot open {redact(self.cfg.source)}")
        return cap

    def _run(self) -> None:
        seq, backoff = 0, 1.0
        while not self._stop.is_set():
            try:
                cap = self._open()
            except LoginRefused as e:
                self._set_down(str(e))
                self._stop.wait(AUTH_RETRY_S)  # saving new settings replaces this worker straight away
                continue
            except Exception as e:  # noqa: BLE001
                self._set_down(str(e))
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            period = 1.0 / self.cfg.fps if self.is_file else 0.0
            next_t = time.monotonic()
            log.info("camera %s connected (%s)", self.cfg.id, redact(self.cfg.source))
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    if self.is_file:  # loop demo footage
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, img = cap.read()
                    if not ok:
                        self._set_down("read failed / stream ended")
                        break
                if img.shape[1] != self.cfg.width or img.shape[0] != self.cfg.height:
                    img = cv2.resize(img, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
                seq += 1
                with self._cond:
                    self._frame = Frame(np.ascontiguousarray(img), time.time(), seq)
                    self._cond.notify_all()
                if not self.connected:
                    self.connected, self.last_error = True, None
                    CAMERA_UP.labels(camera=self.cfg.id).set(1)
                FRAMES_CAPTURED.labels(camera=self.cfg.id).inc()
                if period:
                    next_t += period
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                    else:
                        next_t = time.monotonic()
            cap.release()

    def _set_down(self, err: str) -> None:
        if self.connected or self.last_error != err:
            log.error("camera %s down: %s", self.cfg.id, err)
        self.connected, self.last_error = False, err
        CAMERA_UP.labels(camera=self.cfg.id).set(0)
