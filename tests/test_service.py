"""End-to-end: real camera worker on the demo video, real TensorRT engine, HTTP API."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from freshwatch.app import create_app
from freshwatch.config import load_settings

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "config" / "demo.yaml"
pytestmark = pytest.mark.skipif(
    not (DEMO.is_file() and (ROOT / "models" / "freshness_fp16.engine").is_file()
         and (ROOT / "data" / "demo" / "shelf.mp4").is_file()),
    reason="needs `make engine demo-video`")


@pytest.fixture()
def client(tmp_path):
    s = load_settings(DEMO)
    s.storage.db_path = tmp_path / "t.db"
    s.storage.snapshot_dir = tmp_path / "snaps"
    s.api.api_key = "secret"
    with TestClient(create_app(s)) as c:
        deadline = time.time() + 30
        while time.time() < deadline and c.get("/readyz").status_code != 200:
            time.sleep(0.5)
        yield c


def test_end_to_end(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["ready"] and not ready["degraded"]

    deadline = time.time() + 20
    while time.time() < deadline:
        st = client.get("/api/v1/status").json()
        if all(z["grade"] != "unknown" for z in st["zones"]):
            break
        time.sleep(0.5)
    zones = {z["produce"]: z for z in st["zones"]}
    assert len(zones) == 8
    assert zones["grape"]["grade"] == "spoiled"          # rotten from frame 0
    assert zones["orange"]["grade"] == "fresh"
    assert all(z["last"]["produce_pred"] == p for p, z in zones.items())

    snap = client.get("/api/v1/cameras/aisle3-cam1/snapshot.jpg")
    assert snap.status_code == 200 and snap.content[:2] == b"\xff\xd8"

    alerts = client.get("/api/v1/alerts?active=true").json()
    grape = [a for a in alerts if a["zone_id"] == zones["grape"]["zone_id"]]
    assert grape and grape[0]["kind"] == "spoiled"
    aid = grape[0]["id"]
    assert client.get(f"/api/v1/alerts/{aid}/snapshot.jpg").status_code == 200
    assert client.post(f"/api/v1/alerts/{aid}/ack").status_code == 401
    assert client.post(f"/api/v1/alerts/{aid}/ack", headers={"X-API-Key": "secret"}).status_code == 200

    hist = client.get(f"/api/v1/zones/aisle3-cam1/{zones['apple']['zone_id']}/history?hours=1").json()
    assert hist["live"]
    assert b"freshwatch_zone_score" in client.get("/metrics").content
    assert client.get("/api/v1/cameras/nope/snapshot.jpg").status_code == 404


def test_camera_settings(client, tmp_path):
    key = {"X-API-Key": "secret"}
    assert client.get("/api/v1/settings/cameras").status_code == 401
    [demo] = client.get("/api/v1/settings/cameras", headers=key).json()
    assert demo["kind"] == "file"

    # rename the demo camera, keeping its video file and zones
    r = client.put(f"/api/v1/settings/cameras/{demo['id']}", headers=key, json={"name": "Aisle 3", "kind": "keep"})
    assert r.status_code == 200 and r.json()["name"] == "Aisle 3" and r.json()["zones"] == 8

    # add an IP camera (nothing listens there: it must just show as offline)
    body = {"name": "Veg aisle", "host": "127.0.0.1", "port": 9, "path": "/h264Preview_01_main",
            "username": "admin", "password": "s3cret!"}
    assert client.post("/api/v1/settings/cameras", headers=key, json=body | {"host": "1.2.3.4; x"}).status_code == 422
    new = client.post("/api/v1/settings/cameras", headers=key, json=body).json()
    assert new["id"] == "veg-aisle" and new["has_password"] and new["zones"] == 0
    status = client.get("/api/v1/status").json()
    assert {c["id"] for c in status["cameras"]} == {demo["id"], "veg-aisle"}
    assert "s3cret" not in client.get("/api/v1/settings/cameras", headers=key).text

    # change only the IP: the saved password is kept
    r = client.put("/api/v1/settings/cameras/veg-aisle", headers=key, json=body | {"port": 10, "password": None})
    assert r.json()["port"] == 10 and r.json()["has_password"]
    saved = (tmp_path / "cameras.json").read_text()
    assert "s3cret" in saved and "127.0.0.1:10" in saved

    test = client.post("/api/v1/settings/cameras/test", headers=key, json=body | {"camera_id": "veg-aisle"}).json()
    assert not test["ok"] and test["stage"] == "network"

    zones = [{"id": "B1", "name": "Banana tray", "produce": "banana", "roi": [100, 100, 300, 200]}]
    assert client.put("/api/v1/settings/cameras/veg-aisle/zones", headers=key,
                      json=[zones[0] | {"roi": [1200, 600, 300, 200]}]).status_code == 422  # outside the frame
    r = client.put("/api/v1/settings/cameras/veg-aisle/zones", headers=key, json=zones)
    assert r.status_code == 200 and r.json()["zones"] == 1
    client.put("/api/v1/settings/cameras/veg-aisle", headers=key, json=body | {"name": "Veg", "password": None})
    assert client.get("/api/v1/settings/cameras", headers=key).json()[-1]["zones"] == 1  # an edit keeps zones
    assert '"Banana tray"' in (tmp_path / "cameras.json").read_text()

    assert client.delete("/api/v1/settings/cameras/veg-aisle", headers=key).status_code == 200
    assert [c["id"] for c in client.get("/api/v1/status").json()["cameras"]] == [demo["id"]]
