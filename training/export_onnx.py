"""Export the trained checkpoint to ONNX with a dynamic batch axis.

    python training/export_onnx.py --ckpt models/freshness_best.pt --out models/freshness.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from train import FreshnessNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/freshness_best.pt")
    ap.add_argument("--out", default="models/freshness.onnx")
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    model = FreshnessNet(len(ck["produce"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, a.size, a.size)
    torch.onnx.export(
        model, dummy, a.out, opset_version=17, dynamo=False,
        input_names=["images"], output_names=["produce_logits", "freshness_logits"],
        dynamic_axes={"images": {0: "batch"}, "produce_logits": {0: "batch"}, "freshness_logits": {0: "batch"}},
    )
    m = onnx.load(a.out)
    onnx.checker.check_model(m)
    meta = m.metadata_props.add()
    meta.key, meta.value = "labels", json.dumps({"produce": ck["produce"]})
    onnx.save(m, a.out)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
