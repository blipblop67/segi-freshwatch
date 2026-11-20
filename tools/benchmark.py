"""Validate and benchmark the PRODUCTION GPU path (CUDA preprocess kernel + TensorRT FP16).

1. Accuracy: every held-out test image goes through fc_preprocess_rois + TensorRT
   exactly as a camera zone would (no PyTorch involved).
2. Latency: full GPU pass (H2D 720p frame, colour stats, preprocess, inference,
   D2H) for 1..16 zones.

    python tools/benchmark.py --engine models/freshness_fp16.engine --labels models/labels.json
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from freshwatch import cuda_ops as co  # noqa: E402
from freshwatch.trt_engine import FreshnessEngine, softmax  # noqa: E402


def accuracy(eng: FreshnessEngine, ctx, data: Path, stream: co.Stream) -> dict:
    with open(data / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    s = eng.input_size
    d_in = co.DeviceBuffer(3 * s * s * 4)
    ok_f = ok_p = 0
    conf = [[0, 0], [0, 0]]
    for r in rows:
        img = cv2.imread(str(data / r["path"]))
        h, w = img.shape[:2]
        host = co.PinnedArray((h, w, 3))
        host.array[:] = img
        dev = co.DeviceBuffer(host.nbytes)
        co.h2d(dev, host, stream)
        co.preprocess_rois(dev, w, h, [(0, 0, w, h)], d_in, s, eng.mean, eng.std, stream)
        pl, fl = ctx.infer(d_in, 1, stream)
        pf = int(softmax(fl)[0].argmax())
        t = eng.freshness.index(r["freshness"])
        conf[t][pf] += 1
        ok_f += pf == t
        ok_p += eng.produce[int(pl[0].argmax())] == r["produce"]
        host.free()
        dev.free()
    d_in.free()
    n = len(rows)
    return {"n": n, "freshness_acc": ok_f / n, "produce_acc": ok_p / n, "confusion_fresh_rotten": conf}


def latency(eng: FreshnessEngine, ctx, stream: co.Stream, iters: int) -> list[dict]:
    W, H = 1280, 720
    frame = co.PinnedArray((H, W, 3))
    frame.array[:] = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    d_frame = co.DeviceBuffer(frame.nbytes)
    s = eng.input_size
    out = []
    params = co.ColorParams(10, 40, 0.3, 0.12, 0.55, 0.16, 0.12, 0.75, 0.18, 0.14)
    for n in (1, 4, 8, 16):
        n = min(n, eng.max_batch)
        rois = [((i % 4) * 300 + 20, (i // 4) * 170 + 10, 280, 160) for i in range(n)]
        d_in = co.DeviceBuffer(n * 3 * s * s * 4)
        d_st = co.DeviceBuffer(ctypes.sizeof(co.ColorStats) * n)
        h_st = co.PinnedArray((ctypes.sizeof(co.ColorStats) * n,))
        times = {"total": [], "preprocess+stats": [], "trt": []}
        for i in range(iters + 10):
            t0 = time.perf_counter()
            co.h2d(d_frame, frame, stream)
            co.color_stats(d_frame, W, H, rois, params, 2, d_st, stream)
            co.d2h(h_st, d_st, stream)
            co.preprocess_rois(d_frame, W, H, rois, d_in, s, eng.mean, eng.std, stream)
            stream.sync()
            t1 = time.perf_counter()
            ctx.infer(d_in, n, stream)
            t2 = time.perf_counter()
            if i >= 10:
                times["total"].append(t2 - t0)
                times["preprocess+stats"].append(t1 - t0)
                times["trt"].append(t2 - t1)
        row = {"zones": n}
        for k, v in times.items():
            a = np.array(v) * 1e3
            row[f"{k}_ms_p50"] = round(float(np.percentile(a, 50)), 3)
            row[f"{k}_ms_p99"] = round(float(np.percentile(a, 99)), 3)
        row["zones_per_s"] = round(n / (row["total_ms_p50"] / 1e3), 1)
        out.append(row)
        d_in.free()
        d_st.free()
        h_st.free()
    frame.free()
    d_frame.free()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="models/freshness_fp16.engine")
    ap.add_argument("--labels", default="models/labels.json")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="models/benchmark.json")
    a = ap.parse_args()

    eng = FreshnessEngine(Path(a.engine), Path(a.labels), 16)
    ctx = eng.create_context()
    stream = co.Stream()
    res = {"device": co.device_info()}
    if (Path(a.data) / "manifest.csv").is_file():
        res["accuracy_production_path"] = accuracy(eng, ctx, Path(a.data), stream)
        print("accuracy (CUDA preprocess + TensorRT FP16):", json.dumps(res["accuracy_production_path"]))
    res["latency"] = latency(eng, ctx, stream, a.iters)
    for r in res["latency"]:
        print(r)
    Path(a.out).write_text(json.dumps(res, indent=2))
    ctx.close()
    stream.close()


if __name__ == "__main__":
    main()
