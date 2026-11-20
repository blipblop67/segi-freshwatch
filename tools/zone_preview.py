"""Draw each camera's configured zones on a frame, for checking ROIs while setting up cameras.

    python tools/zone_preview.py --config config/real.yaml --out data/real/zones [--grid]

Works with every source type the service supports (file, RTSP, CSI, USB, gst).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freshwatch.camera import build_pipeline  # noqa: E402
from freshwatch.config import load_settings  # noqa: E402


def grab(cam, skip: int = 10):
    uri, backend = build_pipeline(cam)
    cap = cv2.VideoCapture(uri, backend)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {cam.source}")
    img, deadline = None, time.monotonic() + 15
    for _ in range(skip):  # let live sources settle exposure
        ok, frame = cap.read()
        if ok:
            img = frame
        elif time.monotonic() > deadline:
            break
    cap.release()
    if img is None:
        raise RuntimeError(f"no frame from {cam.source}")
    if img.shape[1] != cam.width or img.shape[0] != cam.height:
        img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    return img


def draw(img, cam, grid: bool):
    h, w = img.shape[:2]
    if grid:
        for x in range(0, w, 100):
            cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)
            cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for y in range(0, h, 100):
            cv2.line(img, (0, y), (w, y), (200, 200, 200), 1)
            cv2.putText(img, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for z in cam.zones:
        x, y, zw, zh = z.roi
        out = x + zw > w or y + zh > h
        color = (0, 0, 255) if out else (0, 220, 0)
        cv2.rectangle(img, (x, y), (x + zw, y + zh), color, 2)
        label = f"{z.id} {z.produce}" + (" OUTSIDE FRAME" if out else "")
        cv2.putText(img, label, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, label, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="zones")
    ap.add_argument("--grid", action="store_true", help="overlay a 100 px coordinate grid")
    a = ap.parse_args()
    s = load_settings(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for cam in s.cameras:
        try:
            p = out / f"{cam.id}.jpg"
            cv2.imwrite(str(p), draw(grab(cam), cam, a.grid))
            print(p)
        except Exception as e:  # noqa: BLE001
            print(f"{cam.id}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
