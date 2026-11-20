"""ctypes binding for libfreshcuda (cuda/src/freshcuda.cu).

Owns device/pinned buffers and CUDA streams so the rest of the service never
needs pycuda or cupy. Every call raises CudaError on failure.
"""
from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

MAX_ROIS = 32
HUE_BINS = 18


class CudaError(RuntimeError):
    pass


class Roi(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32), ("w", ctypes.c_int32), ("h", ctypes.c_int32)]


class ColorStats(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint32),
        ("foreground", ctypes.c_uint32),
        ("brown", ctypes.c_uint32),
        ("dark", ctypes.c_uint32),
        ("mould", ctypes.c_uint32),
        ("sum_sat", ctypes.c_float),
        ("sum_val", ctypes.c_float),
        ("hue_hist", ctypes.c_uint32 * HUE_BINS),
    ]


class ColorParams(ctypes.Structure):
    _fields_ = [
        ("brown_hue_lo", ctypes.c_float), ("brown_hue_hi", ctypes.c_float),
        ("brown_sat_min", ctypes.c_float),
        ("brown_val_lo", ctypes.c_float), ("brown_val_hi", ctypes.c_float),
        ("dark_val_max", ctypes.c_float),
        ("mould_sat_max", ctypes.c_float), ("mould_val_min", ctypes.c_float),
        ("fg_sat_min", ctypes.c_float), ("fg_val_min", ctypes.c_float),
    ]


@dataclass(frozen=True)
class ColorCues:
    """Per-ROI spoilage cues derived from ColorStats."""

    foreground_ratio: float
    brown_ratio: float
    dark_ratio: float
    mould_ratio: float
    mean_saturation: float
    mean_value: float
    hue_hist: tuple[float, ...]

    @classmethod
    def from_stats(cls, s: ColorStats) -> "ColorCues":
        total = max(s.total, 1)
        fg = max(s.foreground, 1)
        hist = np.frombuffer(bytes(s.hue_hist), dtype=np.uint32).astype(np.float64)
        hist = hist / max(hist.sum(), 1.0)
        return cls(
            foreground_ratio=s.foreground / total,
            brown_ratio=s.brown / fg,
            dark_ratio=s.dark / total,
            mould_ratio=s.mould / total,
            mean_saturation=s.sum_sat / fg,
            mean_value=s.sum_val / fg,
            hue_hist=tuple(float(x) for x in hist),
        )


def _find_library() -> Path:
    env = os.environ.get("FRESHWATCH_CUDA_LIB")
    candidates = [Path(env)] if env else []
    root = Path(__file__).resolve().parent.parent
    candidates += [root / "build" / "libfreshcuda.so", Path("/opt/freshwatch/lib/libfreshcuda.so"),
                   Path("/usr/local/lib/libfreshcuda.so")]
    for c in candidates:
        if c.is_file():
            return c
    raise CudaError("libfreshcuda.so not found; build it with `make cuda` or set FRESHWATCH_CUDA_LIB")


_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()

_vp = ctypes.c_void_p
_SIGNATURES = {
    "fc_last_error": ([], ctypes.c_char_p),
    "fc_init": ([ctypes.c_int], ctypes.c_int),
    "fc_device_info": ([ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
    "fc_malloc": ([ctypes.POINTER(_vp), ctypes.c_size_t], ctypes.c_int),
    "fc_free": ([_vp], ctypes.c_int),
    "fc_host_alloc": ([ctypes.POINTER(_vp), ctypes.c_size_t], ctypes.c_int),
    "fc_host_free": ([_vp], ctypes.c_int),
    "fc_memset_async": ([_vp, ctypes.c_int, ctypes.c_size_t, _vp], ctypes.c_int),
    "fc_memcpy_h2d_async": ([_vp, _vp, ctypes.c_size_t, _vp], ctypes.c_int),
    "fc_memcpy_d2h_async": ([_vp, _vp, ctypes.c_size_t, _vp], ctypes.c_int),
    "fc_stream_create": ([ctypes.POINTER(_vp)], ctypes.c_int),
    "fc_stream_destroy": ([_vp], ctypes.c_int),
    "fc_stream_sync": ([_vp], ctypes.c_int),
    "fc_preprocess_rois": ([_vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Roi), ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), _vp, _vp], ctypes.c_int),
    "fc_color_stats": ([_vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Roi), ctypes.c_int,
                        ctypes.c_int, ctypes.POINTER(ColorParams), _vp, _vp], ctypes.c_int),
}


def lib() -> ctypes.CDLL:
    global _lib
    with _lib_lock:
        if _lib is None:
            handle = ctypes.CDLL(str(_find_library()))
            for name, (args, res) in _SIGNATURES.items():
                fn = getattr(handle, name)
                fn.argtypes, fn.restype = args, res
            _lib = handle
            _check(handle.fc_init(0), "fc_init")
        return _lib


def _check(rc: int, what: str) -> None:
    if rc != 0:
        msg = _lib.fc_last_error().decode() if _lib else ""
        raise CudaError(f"{what} failed ({rc}): {msg}")


def device_info() -> dict:
    name = ctypes.create_string_buffer(256)
    major, minor, mem = ctypes.c_int(), ctypes.c_int(), ctypes.c_size_t()
    _check(lib().fc_device_info(name, 256, ctypes.byref(major), ctypes.byref(minor), ctypes.byref(mem)),
           "fc_device_info")
    return {"name": name.value.decode(), "sm": f"{major.value}.{minor.value}", "total_mem": mem.value}


class Stream:
    def __init__(self) -> None:
        h = _vp()
        _check(lib().fc_stream_create(ctypes.byref(h)), "fc_stream_create")
        self.handle = h

    @property
    def ptr(self) -> int:
        return self.handle.value or 0

    def sync(self) -> None:
        _check(lib().fc_stream_sync(self.handle), "fc_stream_sync")

    def close(self) -> None:
        if self.handle:
            lib().fc_stream_destroy(self.handle)
            self.handle = _vp()


class DeviceBuffer:
    """Raw device allocation that grows on demand."""

    def __init__(self, nbytes: int = 0) -> None:
        self.ptr = _vp()
        self.nbytes = 0
        if nbytes:
            self.ensure(nbytes)

    def ensure(self, nbytes: int) -> None:
        if nbytes <= self.nbytes:
            return
        self.free()
        _check(lib().fc_malloc(ctypes.byref(self.ptr), nbytes), "fc_malloc")
        self.nbytes = nbytes

    def free(self) -> None:
        if self.ptr:
            lib().fc_free(self.ptr)
            self.ptr = _vp()
            self.nbytes = 0

    @property
    def address(self) -> int:
        return self.ptr.value or 0

    @classmethod
    def view(cls, address: int) -> "DeviceBuffer":
        """Non-owning alias into another allocation; never free() it."""
        b = cls()
        b.ptr = _vp(address)
        return b


class PinnedArray:
    """numpy view over page-locked host memory for async DMA."""

    def __init__(self, shape: Sequence[int], dtype=np.uint8) -> None:
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = _vp()
        _check(lib().fc_host_alloc(ctypes.byref(self.ptr), self.nbytes), "fc_host_alloc")
        buf = (ctypes.c_uint8 * self.nbytes).from_address(self.ptr.value)
        self.array = np.frombuffer(buf, dtype=self.dtype).reshape(self.shape)

    def free(self) -> None:
        if self.ptr:
            self.array = None
            lib().fc_host_free(self.ptr)
            self.ptr = _vp()


def h2d(dst: DeviceBuffer, src: PinnedArray, stream: Stream, nbytes: int | None = None) -> None:
    n = src.nbytes if nbytes is None else nbytes
    _check(lib().fc_memcpy_h2d_async(dst.ptr, src.ptr, n, stream.handle), "h2d")


def d2h(dst: PinnedArray, src: DeviceBuffer, stream: Stream, nbytes: int | None = None) -> None:
    n = dst.nbytes if nbytes is None else nbytes
    _check(lib().fc_memcpy_d2h_async(dst.ptr, src.ptr, n, stream.handle), "d2h")


def _roi_array(rois: Sequence[tuple[int, int, int, int]]):
    if not 0 < len(rois) <= MAX_ROIS:
        raise ValueError(f"1..{MAX_ROIS} ROIs per call, got {len(rois)}")
    arr = (Roi * len(rois))()
    for i, (x, y, w, h) in enumerate(rois):
        arr[i] = Roi(int(x), int(y), int(w), int(h))
    return arr


def preprocess_rois(frame: DeviceBuffer, width: int, height: int, rois, out: DeviceBuffer, out_size: int,
                    mean, std, stream: Stream, letterbox: bool = False, keep_bgr: bool = False) -> None:
    arr = _roi_array(rois)
    m = (ctypes.c_float * 3)(*mean)
    s = (ctypes.c_float * 3)(*std)
    flags = (1 if letterbox else 0) | (2 if keep_bgr else 0)  # FC_PRE_LETTERBOX | FC_PRE_KEEP_BGR
    _check(lib().fc_preprocess_rois(frame.ptr, width, height, width * 3, arr, len(rois), out_size,
                                    flags, m, s, out.ptr, stream.handle), "fc_preprocess_rois")


def color_stats(frame: DeviceBuffer, width: int, height: int, rois, params: ColorParams, sample_step: int,
                out: DeviceBuffer, stream: Stream) -> None:
    arr = _roi_array(rois)
    _check(lib().fc_color_stats(frame.ptr, width, height, width * 3, arr, len(rois), sample_step,
                                ctypes.byref(params), out.ptr, stream.handle), "fc_color_stats")
