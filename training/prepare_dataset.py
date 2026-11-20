"""Convert the AgML fresh/rotten parquet shards into a resized JPEG tree.

Dataset: Project-AgML/fresh_rotten_fruit_classification (config "raw", CC-BY-4.0).
The "augmented" config is NOT used: it contains augmented copies of the same
photos, so random splits would leak test images into training.

Split is deterministic (content hash) so re-running never reshuffles:
    hash % 10 == 0 -> test, == 1 -> val, else train.

Output:  <out>/<split>/<produce>_<fresh|rotten>/<sha>.jpg  + <out>/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

LABELS = ["fresh", "rotten"]


def _process(args):
    data, label, produce, out_dir, short_side = args
    sha = hashlib.sha1(data).hexdigest()[:16]
    bucket = int(sha, 16) % 10
    split = "test" if bucket == 0 else "val" if bucket == 1 else "train"
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    s = short_side / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    cls = f"{produce}_{LABELS[label]}"
    dst = Path(out_dir) / split / cls / f"{sha}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=92)
    return split, cls, str(dst.relative_to(out_dir)), produce, LABELS[label]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--short-side", type=int, default=288)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    shards = sorted(Path(a.raw).glob("raw-*.parquet"))
    if not shards:
        raise SystemExit(f"no raw-*.parquet shards in {a.raw}")
    rows, seen = [], set()
    with ProcessPoolExecutor(a.workers) as pool:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["image", "label", "fruit_type"]).to_pylist()
                jobs = [(r["image"]["bytes"], r["label"], r["fruit_type"], a.out, a.short_side) for r in t]
                for res in pool.map(_process, jobs, chunksize=4):
                    if res[2] in seen:          # exact duplicate photo
                        continue
                    seen.add(res[2])
                    rows.append(res)
            print(f"{shard.name}: {len(rows)} images so far", flush=True)

    with open(Path(a.out) / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "path", "produce", "freshness"])
        w.writerows(sorted(rows))
    for split in ("train", "val", "test"):
        print(split, sum(r[0] == split for r in rows))


if __name__ == "__main__":
    main()
