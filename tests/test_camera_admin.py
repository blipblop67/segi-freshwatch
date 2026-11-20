"""Camera settings: RTSP URL handling, saved overrides and the RTSP probe (against a fake camera)."""
import hashlib
import re
import secrets
import socket
import stat
import threading
import time

import pytest

from freshwatch.camera import CameraSource, build_pipeline, redact, split_credentials
from freshwatch.camera_admin import CameraOverrides, build_rtsp, parse_rtsp, probe_rtsp
from freshwatch.config import CameraConfig, ZoneConfig


def test_rtsp_url_round_trip_with_awkward_password():
    url = build_rtsp("192.168.1.124", 554, "h264Preview_01_main", "admin", 'p@ss:w/"rd!')
    assert "p@ss" not in url  # percent-encoded
    f = parse_rtsp(url)
    assert f == {"host": "192.168.1.124", "port": 554, "path": "/h264Preview_01_main",
                 "username": "admin", "password": 'p@ss:w/"rd!'}
    assert redact(url) == "rtsp://admin:***@192.168.1.124:554/h264Preview_01_main"
    assert split_credentials(url)[0] == "rtsp://192.168.1.124:554/h264Preview_01_main"
    assert parse_rtsp(build_rtsp("10.0.0.5", 554, "/cam/realmonitor?channel=1&subtype=0"))["path"] == \
        "/cam/realmonitor?channel=1&subtype=0"


@pytest.mark.parametrize("host,port,path", [("192.168.1.1; rm", 554, "/"), ("10.0.0.1", 70000, "/"),
                                            ("10.0.0.1", 554, "/a b")])
def test_rtsp_url_rejects_bad_input(host, port, path):
    with pytest.raises(ValueError):
        build_rtsp(host, port, path)


def test_pipeline_keeps_credentials_out_of_location_and_uses_codec():
    cam = CameraConfig(id="c", source=build_rtsp("10.0.0.2", 554, "/s", "admin", 'a"b'), codec="h265")
    uri, _ = build_pipeline(cam)
    assert 'location="rtsp://10.0.0.2:554/s"' in uri
    assert 'user-pw="a\\"b"' in uri
    assert "rtph265depay ! h265parse" in uri


def test_camera_without_zones_is_valid():
    assert CameraConfig(id="new", source="rtsp://10.0.0.2:554/s").zones == []


def test_overrides_edit_add_remove_and_reload(tmp_path):
    yaml_cams = [CameraConfig(id="a", name="A", source="file:/x.mp4",
                              zones=[ZoneConfig(id="Z", name="Z", produce="apple", roi=(0, 0, 50, 50))]),
                 CameraConfig(id="b", name="B", source="rtsp://10.0.0.9:554/s")]
    path = tmp_path / "cameras.json"
    ov = CameraOverrides(path)
    a = yaml_cams[0].model_copy(update={"name": "Fruit aisle", "source": "rtsp://u:pw@10.0.0.3:554/s"})
    ov.save_camera(a, added=False)
    ov.save_camera(CameraConfig(id="new", name="New", source="rtsp://10.0.0.4:554/s", codec="h265"), added=True)
    ov.remove_camera("b", from_yaml=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # may hold camera passwords

    cams = {c.id: c for c in CameraOverrides(path).apply(yaml_cams)}
    assert set(cams) == {"a", "new"}
    assert cams["a"].name == "Fruit aisle" and cams["a"].source.startswith("rtsp://u:pw@10.0.0.3")
    assert len(cams["a"].zones) == 1  # zones still come from the YAML
    assert cams["new"].codec == "h265" and cams["new"].zones == []


# ---------------------------------------------------------------- fake RTSP camera (Digest auth)
SDP = "v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H265/90000\r\n"


class FakeCamera:
    def __init__(self, user="admin", pw="secret", path="/live"):
        self.user, self.pw, self.path = user, pw, path
        self.logins = 0  # requests that carried credentials
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    def _session(self, conn):
        nonce = secrets.token_hex(8)  # like Reolink: valid only on the connection that issued it
        with conn:
            while req := conn.recv(4096).decode():
                url = req.split()[1]
                auth = re.search(r"Authorization: Digest (.*)\r\n", req)
                self.logins += bool(auth)
                p = dict(re.findall(r'(\w+)="?([^",]*)"?', auth.group(1))) if auth else {}
                if not auth or not self._valid(p, url, nonce):
                    conn.sendall(b'RTSP/1.0 401 Unauthorized\r\nCSeq: 2\r\n'
                                 b'WWW-Authenticate: Digest realm="cam", nonce="' + nonce.encode() + b'"\r\n\r\n')
                elif not url.endswith(self.path):  # unknown path after login
                    conn.sendall(b"RTSP/1.0 404 Not Found\r\nCSeq: 3\r\n\r\n")
                else:
                    conn.sendall(f"RTSP/1.0 200 OK\r\nCSeq: 3\r\nContent-Length: {len(SDP)}\r\n\r\n{SDP}".encode())

    def _valid(self, p, url, nonce):
        md5 = lambda s: hashlib.md5(s.encode()).hexdigest()  # noqa: E731, S324
        expect = md5(f"{md5(f'{self.user}:cam:{self.pw}')}:{nonce}:{md5(f'DESCRIBE:{url}')}")
        return p.get("username") == self.user and p.get("response") == expect

    def close(self):
        self.sock.close()


@pytest.fixture()
def fake_cam():
    cam = FakeCamera()
    yield cam
    cam.close()


def test_probe_detects_codec_with_digest_login(fake_cam):
    r = probe_rtsp(build_rtsp("127.0.0.1", fake_cam.port, "/live", "admin", "secret"))
    assert r["ok"] and r["codec"] == "h265"


@pytest.mark.parametrize("user,pw,path,stage", [("admin", "wrong", "/live", "auth"), ("", "", "/live", "auth"),
                                                ("admin", "secret", "/nope", "path")])
def test_probe_explains_failures(fake_cam, user, pw, path, stage):
    r = probe_rtsp(build_rtsp("127.0.0.1", fake_cam.port, path, user, pw))
    assert not r["ok"] and r["stage"] == stage


def test_probe_unreachable_port():
    s = socket.create_server(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = probe_rtsp(build_rtsp("127.0.0.1", port, "/live"))
    assert not r["ok"] and r["stage"] == "network"


def test_wrong_password_costs_one_login_not_a_lockout(fake_cam):
    src = CameraSource(CameraConfig(id="c", source=build_rtsp("127.0.0.1", fake_cam.port, "/live", "admin", "bad")))
    src.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not src.last_error:
            time.sleep(0.05)
        time.sleep(2)  # the old code retried within seconds
        assert "Wrong username or password" in src.last_error and not src.connected
        assert fake_cam.logins == 1
    finally:
        t0 = time.time()
        src.stop()
        assert time.time() - t0 < 2  # stopping (e.g. after saving a new password) is immediate
