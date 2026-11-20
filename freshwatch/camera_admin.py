"""Camera settings edited from the dashboard: saved overrides, RTSP connection test and ONVIF discovery.

Changes made in Settings are stored next to the database (``cameras.json``, mode 0600 because it can
hold camera passwords) and applied on top of the YAML config at every start, so the YAML stays the
installation default and the dashboard owns only what staff changed.
"""
from __future__ import annotations

import base64
import errno
import hashlib
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote, urlsplit

import cv2

from .camera import build_pipeline, split_credentials
from .config import CameraConfig

log = logging.getLogger(__name__)

EDITABLE = ("name", "note", "source", "codec", "detect")


# ---------------------------------------------------------------- RTSP URLs
def parse_rtsp(source: str) -> Optional[dict]:
    """rtsp://user:pass@host:port/path?q -> fields for the settings form (None if not RTSP)."""
    if not source.startswith("rtsp://"):
        return None
    bare, user, pw = split_credentials(source)
    u = urlsplit(bare)
    path = u.path + (f"?{u.query}" if u.query else "")
    return {"host": u.hostname or "", "port": u.port or 554, "path": path or "/", "username": user, "password": pw}


def build_rtsp(host: str, port: int, path: str, username: str = "", password: str = "") -> str:
    host = host.strip()
    if not re.fullmatch(r"[A-Za-z0-9.\-]{1,253}", host):
        raise ValueError("IP address must look like 192.168.1.124 (or a host name)")
    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be 1-65535")
    path = "/" + path.strip().lstrip("/")
    if re.search(r"[\s\"]", path):
        raise ValueError("stream path cannot contain spaces or quotes")
    auth = ""
    if username:
        auth = quote(username, safe="") + (":" + quote(password, safe="") if password else "") + "@"
    return f"rtsp://{auth}{host}:{int(port)}{path}"


# ---------------------------------------------------------------- overrides file
class CameraOverrides:
    """Persisted dashboard edits: {"cameras": {id: {field: value, "added": bool, ...}}, "removed": [id]}."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data: dict = {"cameras": {}, "removed": []}
        if path.is_file():
            try:
                self.data = json.loads(path.read_text())
                self.data.setdefault("cameras", {})
                self.data.setdefault("removed", [])
            except (OSError, ValueError):
                log.exception("ignoring unreadable camera settings file %s", path)

    def apply(self, cameras: list[CameraConfig]) -> list[CameraConfig]:
        """YAML cameras + saved edits -> effective camera list."""
        out: list[CameraConfig] = []
        removed = set(self.data["removed"])
        saved = self.data["cameras"]
        for cam in cameras:
            if cam.id in removed:
                continue
            patch = {k: v for k, v in saved.get(cam.id, {}).items() if k in EDITABLE or k == "zones"}
            out.append(_validated(cam, patch) if patch else cam)
        known = {c.id for c in out}
        for cid, d in saved.items():
            if d.get("added") and cid not in known and cid not in removed:
                try:
                    out.append(CameraConfig.model_validate({k: v for k, v in d.items() if k != "added"} | {"id": cid}))
                except ValueError:
                    log.exception("ignoring invalid saved camera %s", cid)
        return out

    def save_camera(self, cam: CameraConfig, added: bool, zones_changed: bool = False) -> None:
        """Zones are stored once they were set from the dashboard (always for dashboard-added cameras)."""
        with self._lock:
            entry = {k: getattr(cam, k) for k in EDITABLE}
            if added:
                entry |= {"added": True, "width": cam.width, "height": cam.height, "fps": cam.fps,
                          "analyze_interval_s": cam.analyze_interval_s}
            if added or zones_changed or "zones" in self.data["cameras"].get(cam.id, {}):
                entry["zones"] = [z.model_dump(mode="json") for z in cam.zones]
            self.data["cameras"][cam.id] = entry
            if cam.id in self.data["removed"]:
                self.data["removed"].remove(cam.id)
            self._write()

    def remove_camera(self, cam_id: str, from_yaml: bool) -> None:
        with self._lock:
            self.data["cameras"].pop(cam_id, None)
            if from_yaml and cam_id not in self.data["removed"]:
                self.data["removed"].append(cam_id)
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


def _validated(cam: CameraConfig, patch: dict) -> CameraConfig:
    return CameraConfig.model_validate(cam.model_dump() | patch)


def with_changes(cam: CameraConfig, **patch) -> CameraConfig:
    return _validated(cam, {k: v for k, v in patch.items() if v is not None})


# ---------------------------------------------------------------- RTSP probe
class ProbeResult(dict):
    """{"ok", "stage": network|auth|path|rtsp|codec|decode, "message", "codec"?, "status"?}"""


def _rtsp_request(s: socket.socket, url: str, cseq: int, auth: Optional[str]) -> tuple[int, dict, str]:
    req = (f"DESCRIBE {url} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\nUser-Agent: FreshWatch\r\n"
           + (f"Authorization: {auth}\r\n" if auth else "") + "\r\n")
    s.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    lines = head.decode(errors="replace").split("\r\n")
    m = re.match(r"RTSP/1\.\d (\d{3})", lines[0])
    if not m:
        raise ConnectionError("the device on this port did not answer as an RTSP camera")
    headers: dict[str, str] = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(":")
        headers.setdefault(k.strip().lower(), v.strip())
    need = int(headers.get("content-length", "0") or 0)
    while len(body) < need:
        chunk = s.recv(4096)
        if not chunk:
            break
        body += chunk
    return int(m.group(1)), headers, body.decode(errors="replace")


def _auth_header(challenge: str, user: str, pw: str, url: str) -> str:
    if challenge.lower().startswith("basic"):
        return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
    p = dict(re.findall(r'(\w+)="?([^",]*)"?', challenge))

    def md5(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()  # noqa: S324 - RTSP Digest is defined on MD5

    ha1, ha2 = md5(f"{user}:{p.get('realm', '')}:{pw}"), md5(f"DESCRIBE:{url}")
    fields = f'username="{user}", realm="{p.get("realm", "")}", nonce="{p.get("nonce", "")}", uri="{url}"'
    if "auth" in p.get("qop", "").split(","):
        nc, cnonce = "00000001", secrets.token_hex(8)
        resp = md5(f"{ha1}:{p['nonce']}:{nc}:{cnonce}:auth:{ha2}")
        fields += f', qop=auth, nc={nc}, cnonce="{cnonce}"'
    else:
        resp = md5(f"{ha1}:{p.get('nonce', '')}:{ha2}")
    if "opaque" in p:
        fields += f', opaque="{p["opaque"]}"'
    return f'Digest {fields}, response="{resp}"'


def probe_rtsp(source: str, timeout: float = 4.0) -> ProbeResult:
    """Ask the camera for its stream description: reachability, credentials, path and codec in ~1 s."""
    f = parse_rtsp(source)
    if f is None:
        return ProbeResult(ok=False, stage="source", message="Not an RTSP address.")
    host, port = f["host"], f["port"]
    bare, user, pw = split_credentials(source)
    try:
        # The login must go on the connection that got the challenge: cameras such as Reolink tie the
        # Digest nonce to it and refuse the same login sent on a new connection.
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            status, headers, sdp = _rtsp_request(s, bare, 2, None)
            if status == 401 and user:
                auth = _auth_header(headers.get("www-authenticate", "Basic"), user, pw, bare)
                status, headers, sdp = _rtsp_request(s, bare, 3, auth)
    except ConnectionRefusedError:
        return ProbeResult(ok=False, stage="network",
                           message=f"{host} is on the network but refused port {port}. Check the RTSP port, and "
                                   "that RTSP is enabled in the camera's settings.")
    except OSError as e:
        if isinstance(e, TimeoutError) or e.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH):
            # timeout, or "no route to host": nothing on the network answers at this address
            return ProbeResult(ok=False, stage="network",
                               message=f"No answer from {host}:{port}. Check the IP address, that the camera is "
                                       "powered on, and that it is on the same network as this device.")
        return ProbeResult(ok=False, stage="network", message=f"Cannot reach {host}:{port}: {e.strerror or e}")
    if status == 401:
        return ProbeResult(ok=False, stage="auth", status=status,
                           message="Wrong username or password, or wrong stream path (some cameras, e.g. "
                                   "Reolink, answer an unknown path this way)." if user and pw else
                           "The camera needs a username and password. Enter them and test again.")
    if status in (403,):
        return ProbeResult(ok=False, stage="auth", status=status, message="The camera refused this user (403).")
    if status == 404:
        return ProbeResult(ok=False, stage="path", status=status,
                           message="The camera has no stream at this path. Check the stream path for your brand.")
    if status != 200:
        return ProbeResult(ok=False, stage="rtsp", status=status, message=f"The camera answered RTSP {status}.")
    video = sdp.split("m=video", 1)[-1] if "m=video" in sdp else sdp
    m = re.search(r"a=rtpmap:\d+\s+(H264|H265|HEVC)/", video, re.I)
    codec = None if not m else ("h264" if m.group(1).upper() == "H264" else "h265")
    if codec is None:
        return ProbeResult(ok=False, stage="codec", status=status,
                           message="The stream is not H.264 or H.265. Change the video encoding in the camera.")
    return ProbeResult(ok=True, stage="rtsp", status=status, codec=codec,
                       message=f"Camera answered, {'H.264' if codec == 'h264' else 'H.265'} stream.")


def grab_frame(cam: CameraConfig, timeout: float = 15.0) -> tuple[Optional[bytes], str]:
    """Decode one frame through the production pipeline (NVDEC) -> (JPEG or None, message)."""
    result: dict = {}

    def run():
        uri, backend = build_pipeline(cam)
        cap = cv2.VideoCapture(uri, backend)
        try:
            if not cap.isOpened():
                result["err"] = "The stream opened but could not be decoded."
                return
            ok, img = cap.read()
            if not ok:
                result["err"] = "The stream opened but sent no picture."
                return
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result["jpg"] = buf.tobytes() if ok else None
        finally:
            cap.release()

    t = threading.Thread(target=run, name=f"cam-test-{cam.id}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"No picture within {timeout:.0f} s."
    return result.get("jpg"), result.get("err", "Picture received.")


# ---------------------------------------------------------------- ONVIF discovery
_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{id}</w:MessageID>
<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>"""


def _scope(text: str, key: str) -> str:
    m = re.search(rf"onvif://www\.onvif\.org/{key}/(\S+?)[\s<]", text)
    return unquote(m.group(1)) if m else ""


def _arp_table() -> dict[str, str]:
    try:
        rows = Path("/proc/net/arp").read_text().splitlines()[1:]
    except OSError:
        return {}
    return {r.split()[0]: r.split()[3] for r in rows if len(r.split()) >= 4 and r.split()[3] != "00:00:00:00:00:00"}


def discover(timeout: float = 3.0) -> list[dict]:
    """WS-Discovery probe for ONVIF cameras on the local network -> [{ip, name, model, mac}]."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        s.settimeout(0.3)
        msg = _PROBE.format(id=uuid.uuid4()).encode()
        for _ in range(2):
            s.sendto(msg, ("239.255.255.250", 3702))
        found: dict[str, dict] = {}
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, (ip, _) = s.recvfrom(65535)
            except socket.timeout:
                continue
            text = data.decode(errors="replace")
            found[ip] = {"ip": ip, "name": _scope(text, "name"), "model": _scope(text, "hardware")}
    finally:
        s.close()
    arp = _arp_table()
    for d in found.values():
        d["mac"] = arp.get(d["ip"], "")
    return sorted(found.values(), key=lambda d: tuple(int(p) for p in d["ip"].split(".")))
