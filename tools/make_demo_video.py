"""Compose a demo "wet-market shelf" video + matching zone config from HELD-OUT test images.

Eight black crates (2 x 4), one produce type each, with a scripted timeline:
  banana, strawberry, jujube decay fresh -> rotten; grape is spoiled from the start;
  a shopper's hand passes over the apples (must NOT raise an alert);
  the guava crate is emptied then restocked.

    python tools/make_demo_video.py --data data/processed --out data/demo
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

W, H, FPS = 1280, 720, 15
COLS, ROWS = 4, 2
MARGIN, GAP, TOP = 24, 18, 64

# produce: (fresh->rotten start s, transition s) or None
SCRIPT = {
    "banana": (20, 12), "strawberry": (35, 10), "jujube": (50, 12), "grape": "rotten",
    "apple": None, "orange": None, "guava": None, "pomegranate": None,
}
HAND = ("apple", 40, 46)          # produce, start s, end s
EMPTY = ("guava", 62, 78)         # produce, empty from, restocked at


def load_images(root: Path, produce: str, fresh: str, size, rng) -> np.ndarray:
    with open(root / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["produce"] == produce
                and r["freshness"] == fresh]
    r = rng.choice(rows)
    img = cv2.imread(str(root / r["path"]))
    return fit(img, size)


def fit(img: np.ndarray, size) -> np.ndarray:
    w, h = size
    s = max(w / img.shape[1], h / img.shape[0])  # cover
    r = cv2.resize(img, (int(img.shape[1] * s + 1), int(img.shape[0] * s + 1)), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config-out", default="config/demo.yaml")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cw = (W - 2 * MARGIN - (COLS - 1) * GAP) // COLS
    ch = (H - TOP - MARGIN - (ROWS - 1) * GAP) // ROWS
    inset = 14
    crates, zones = [], []
    for i, produce in enumerate(SCRIPT):
        r, c = divmod(i, COLS)
        x, y = MARGIN + c * (cw + GAP), TOP + r * (ch + GAP)
        ix, iy, iw, ih = x + inset, y + inset, cw - 2 * inset, ch - 2 * inset
        fresh = load_images(root, produce, "fresh", (iw, ih), rng)
        rotten = load_images(root, produce, "rotten", (iw, ih), rng)
        crates.append((produce, (x, y, cw, ch), (ix, iy, iw, ih), fresh, rotten))
        zones.append({"id": f"{'AB'[r]}{c + 1}", "name": f"{produce.title()} crate", "produce": produce,
                      "roi": [ix, iy, iw, ih]})

    base = np.full((H, W, 3), (205, 212, 210), np.uint8)
    # Composited from held-out test images; the dashboard labels this camera as a simulation.
    cv2.putText(base, "CAM 03  AISLE 3", (W - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 50, 45), 2, cv2.LINE_AA)
    for _, (x, y, w, h), *_ in crates:
        cv2.rectangle(base, (x, y), (x + w, y + h), (28, 28, 30), -1)

    path = out / "shelf.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = a.seconds * FPS
    for f in range(n):
        t = f / FPS
        frame = base.copy()
        for produce, _, (ix, iy, iw, ih), fresh, rotten in crates:
            s = SCRIPT[produce]
            if s == "rotten":
                mix = 1.0
            elif s is None:
                mix = 0.0
            else:
                mix = float(np.clip((t - s[0]) / s[1], 0, 1))
            tile = cv2.addWeighted(fresh, 1 - mix, rotten, mix, 0) if 0 < mix < 1 else (rotten if mix else fresh)
            if produce == EMPTY[0] and EMPTY[1] <= t < EMPTY[2]:
                tile = np.full_like(tile, 34)
            frame[iy:iy + ih, ix:ix + iw] = tile
            if produce == HAND[0] and HAND[1] <= t < HAND[2]:
                p = (t - HAND[1]) / (HAND[2] - HAND[1])
                cx = int(ix - 60 + p * (iw + 120))
                cv2.ellipse(frame, (cx, iy + ih // 2), (70, 110), 20, 0, 360, (120, 160, 215), -1)
                cv2.rectangle(frame, (cx - 40, iy + ih // 2 + 60), (cx + 40, iy + ih + 40), (120, 160, 215), -1)
        # lighting flicker + sensor noise
        gain = 1.0 + 0.03 * np.sin(2 * np.pi * t / 7.0)
        noisy = frame.astype(np.float32) * gain + nrng.normal(0, 3.0, frame.shape)
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()

    # Demo-friendly settings: fast alert cooldown and persistence so a 90 s loop shows everything.
    cfg = {
        "site": {"store_id": "segi-demo-01", "name": "SEGI Cash & Carry - Wet Market (Demo)"},
        "model": {"engine": "models/freshness_fp16.engine", "labels": "models/labels.json"},
        "scoring": {"hysteresis_readings": 3},
        "alerts": {"cooldown_s": 60},
        "storage": {"db_path": "data/demo/freshwatch.db", "persist_interval_s": 5,
                    "snapshot_dir": "data/demo/snapshots"},
        "cameras": [{"id": "aisle3-cam1", "name": "Aisle 3 - Fruit", "source": f"file:{path}",
                     "width": W, "height": H, "fps": FPS, "analyze_interval_s": 1.0, "zones": zones}],
    }
    Path(a.config_out).write_text("# Generated by tools/make_demo_video.py\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {n} frames) and {a.config_out}")


if __name__ == "__main__":
    main()
