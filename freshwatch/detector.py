"""Produce detector: YOLOX (Apache-2.0, COCO) on TensorRT, fed by the CUDA letterbox kernel.

For cameras with ``detect: true``: instead of grading fixed zones, every analysis pass finds produce
anywhere in the frame, and each find is graded by the freshness model like a zone.

    d_frame --fc_preprocess_rois(letterbox, BGR, 0..255)--> d_input [1,3,640,640]
            --TensorRT FP16--> [8400, 4 box + objectness + 80 class scores] --D2H--> decode + NMS (CPU)

The ONNX export from the YOLOX release keeps raw grid offsets, so boxes are decoded here
(grid + stride per anchor point), as in YOLOX's own demo_postprocess.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import tensorrt as trt

from . import cuda_ops as co
from .config import DetectorConfig
from .trt_engine import _TrtLogger

log = logging.getLogger(__name__)

INPUT, OUTPUT = "images", "output"
STRIDES = (8, 16, 32)


@dataclass
class Detection:
    produce: str
    score: float
    box: tuple[int, int, int, int]  # x, y, w, h in frame pixels


def anchor_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor (x, y) grid cell and stride, in YOLOX's output order (stride 8, then 16, then 32)."""
    grids, strides = [], []
    for s in STRIDES:
        g = size // s
        ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        grids.append(np.stack([xs, ys], -1).reshape(-1, 2))
        strides.append(np.full((g * g, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


class FruitDetector:
    """One deserialised engine shared by all detect cameras; each worker gets its own context."""

    def __init__(self, cfg: DetectorConfig) -> None:
        if not cfg.engine.is_file():
            raise FileNotFoundError(f"detector engine not found: {cfg.engine} (run `make detector`)")
        self.cfg = cfg
        co.lib()
        self._logger = _TrtLogger()
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(cfg.engine.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialise {cfg.engine}; rebuild it for this JetPack/TensorRT")
        in_shape = tuple(self.engine.get_tensor_shape(INPUT))
        out_shape = tuple(self.engine.get_tensor_shape(OUTPUT))
        self.size = int(in_shape[2])
        self.n_anchors, self.n_out = int(out_shape[1]), int(out_shape[2])
        self.grid, self.stride = anchor_grid(self.size)
        if len(self.grid) != self.n_anchors:
            raise RuntimeError(f"unexpected detector output {out_shape} for input {in_shape}")
        self.class_ids = np.array(sorted(cfg.classes), dtype=np.int64)
        self.class_names = [cfg.classes[i] for i in self.class_ids]
        self._lock = threading.Lock()
        log.info("loaded detector %s (%dx%d): %s", cfg.engine.name, self.size, self.size, self.class_names)

    def create_context(self) -> "DetectorContext":
        with self._lock:
            return DetectorContext(self)

    def decode(self, out: np.ndarray, width: int, height: int) -> list[Detection]:
        """Raw [anchors, 85] output -> produce detections in frame pixels, after NMS."""
        cfg = self.cfg
        scores = out[:, 4:5] * out[:, 5 + self.class_ids]  # objectness x class probability (both sigmoided)
        best = scores.argmax(1)
        conf = scores[np.arange(len(scores)), best]
        keep = np.nonzero(conf >= cfg.min_score)[0]
        if keep.size == 0:
            return []
        p, g, st = out[keep], self.grid[keep], self.stride[keep]
        cxcy = (p[:, :2] + g) * st
        wh = np.exp(np.clip(p[:, 2:4], -10, 10)) * st
        # undo the centred letterbox done by the CUDA kernel
        s = max(width / self.size, height / self.size)
        off = np.array([(self.size - width / s) / 2, (self.size - height / s) / 2], np.float32)
        xy0 = (cxcy - wh / 2 - off) * s
        wh = wh * s
        x0 = np.clip(xy0[:, 0], 0, width - 1)
        y0 = np.clip(xy0[:, 1], 0, height - 1)
        x1 = np.clip(xy0[:, 0] + wh[:, 0], 0, width)
        y1 = np.clip(xy0[:, 1] + wh[:, 1], 0, height)
        boxes = np.stack([x0, y0, x1 - x0, y1 - y0], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf[keep].tolist(), best[keep].tolist(),
                                      cfg.min_score, cfg.nms_iou)
        dets = []
        for i in np.array(idx).reshape(-1):
            x, y, w, h = (int(round(v)) for v in boxes[i])
            if min(w, h) >= cfg.min_box_px:
                dets.append(Detection(self.class_names[best[keep][i]], float(conf[keep][i]), (x, y, w, h)))
        dets.sort(key=lambda d: -d.score)
        return dets[:cfg.max_items]


class DetectorContext:
    def __init__(self, det: FruitDetector) -> None:
        self.det = det
        self.ctx = det.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("could not create TensorRT execution context for the detector")
        self.d_in = co.DeviceBuffer(3 * det.size * det.size * 4)
        self.d_out = co.DeviceBuffer(det.n_anchors * det.n_out * 4)
        self.h_out = co.PinnedArray((det.n_anchors, det.n_out), np.float32)

    def detect(self, d_frame: co.DeviceBuffer, width: int, height: int, stream: co.Stream) -> list[Detection]:
        """Detect produce in a frame already on the GPU (same stream as its upload)."""
        s = self.det.size
        # YOLOX takes BGR pixel values 0..255: (x/255 - 0) / (1/255) = x
        co.preprocess_rois(d_frame, width, height, [(0, 0, width, height)], self.d_in, s, (0.0, 0.0, 0.0),
                           (1 / 255, 1 / 255, 1 / 255), stream, letterbox=True, keep_bgr=True)
        self.ctx.set_tensor_address(INPUT, self.d_in.address)
        self.ctx.set_tensor_address(OUTPUT, self.d_out.address)
        if not self.ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("detector execute_async_v3 failed")
        co.d2h(self.h_out, self.d_out, stream)
        stream.sync()
        return self.det.decode(self.h_out.array, width, height)

    def close(self) -> None:
        self.d_in.free()
        self.d_out.free()
        self.h_out.free()
        self.ctx = None
