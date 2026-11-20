"""TensorRT 10 runner for the two-head freshness classifier.

One deserialised engine is shared by all cameras; each camera worker gets its
own TrtContext (execution context + output buffers) and CUDA stream, so
inference on several cameras runs concurrently without locking.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import cuda_ops as co

log = logging.getLogger(__name__)

INPUT = "images"
OUTPUTS = ("produce_logits", "freshness_logits")


class _TrtLogger(trt.ILogger):
    def __init__(self) -> None:
        super().__init__()

    def log(self, severity, msg):  # noqa: D401 - TensorRT callback
        if severity <= trt.ILogger.ERROR:
            log.error("TensorRT: %s", msg)
        elif severity <= trt.ILogger.WARNING:
            log.warning("TensorRT: %s", msg)


class FreshnessEngine:
    def __init__(self, engine_path: Path, labels_path: Path, max_batch: int) -> None:
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path} (run `make engine`)")
        labels = json.loads(labels_path.read_text())
        self.produce: list[str] = labels["produce"]
        self.freshness: list[str] = labels["freshness"]
        self.input_size: int = int(labels["input_size"])
        self.mean = tuple(labels["mean"])
        self.std = tuple(labels["std"])

        co.lib()  # make sure the CUDA context exists on this process
        self._logger = _TrtLogger()
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialise {engine_path}; rebuild it for this JetPack/TensorRT")

        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        missing = {INPUT, *OUTPUTS} - names
        if missing:
            raise RuntimeError(f"engine is missing tensors {missing}; got {names}")
        max_shape = self.engine.get_tensor_profile_shape(INPUT, 0)[2]
        self.max_batch = min(max_batch, int(max_shape[0]))
        if self.engine.get_tensor_dtype(INPUT) != trt.float32:
            raise RuntimeError("engine input must be float32 (build with default I/O types)")
        self._ctx_lock = threading.Lock()
        log.info("loaded engine %s: produce=%s max_batch=%d size=%d", engine_path.name, self.produce,
                 self.max_batch, self.input_size)

    def create_context(self) -> "TrtContext":
        with self._ctx_lock:
            return TrtContext(self)


class TrtContext:
    def __init__(self, eng: FreshnessEngine) -> None:
        self.eng = eng
        self.ctx = eng.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("could not create TensorRT execution context (out of memory?)")
        b = eng.max_batch
        self.out_dims = {OUTPUTS[0]: len(eng.produce), OUTPUTS[1]: len(eng.freshness)}
        self.d_out = {k: co.DeviceBuffer(b * n * 4) for k, n in self.out_dims.items()}
        self.h_out = {k: co.PinnedArray((b, n), np.float32) for k, n in self.out_dims.items()}

    def infer(self, d_input: co.DeviceBuffer, n: int, stream: co.Stream, sync: bool = True):
        """Run on `n` preprocessed crops already in d_input; returns (produce_logits, freshness_logits)."""
        if not 0 < n <= self.eng.max_batch:
            raise ValueError(f"batch {n} outside 1..{self.eng.max_batch}")
        s = self.eng.input_size
        self.ctx.set_input_shape(INPUT, (n, 3, s, s))
        self.ctx.set_tensor_address(INPUT, d_input.address)
        for k in OUTPUTS:
            self.ctx.set_tensor_address(k, self.d_out[k].address)
        if not self.ctx.execute_async_v3(stream.ptr):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for k, dim in self.out_dims.items():
            co.d2h(self.h_out[k], self.d_out[k], stream, n * dim * 4)
        if sync:
            stream.sync()
        return self.h_out[OUTPUTS[0]].array[:n].copy(), self.h_out[OUTPUTS[1]].array[:n].copy()

    def close(self) -> None:
        for b in self.d_out.values():
            b.free()
        for h in self.h_out.values():
            h.free()
        self.ctx = None


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
