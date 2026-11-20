"""GPU kernels vs. CPU (OpenCV / numpy) reference implementations."""
import ctypes

import cv2
import numpy as np
import pytest

from freshwatch import cuda_ops as co

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def stream():
    s = co.Stream()
    yield s
    s.close()


def _upload(img, stream):
    h, w, _ = img.shape
    host = co.PinnedArray((h, w, 3))
    host.array[:] = img
    dev = co.DeviceBuffer(host.nbytes)
    co.h2d(dev, host, stream)
    stream.sync()
    host.free()
    return dev


def _cpu_preprocess(img, roi, size):
    x, y, w, h = roi
    crop = cv2.resize(img[y:y + h, x:x + w], (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def test_device_info():
    info = co.device_info()
    assert info["total_mem"] > 0
    assert info["sm"].count(".") == 1


def test_preprocess_matches_opencv(stream):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
    rois = [(0, 0, 640, 480), (100, 50, 200, 300), (333, 211, 97, 141)]
    size = 224
    dev = _upload(img, stream)
    out = co.DeviceBuffer(len(rois) * 3 * size * size * 4)
    co.preprocess_rois(dev, 640, 480, rois, out, size, MEAN, STD, stream)
    host = co.PinnedArray((len(rois), 3, size, size), np.float32)
    co.d2h(host, out, stream)
    stream.sync()
    for i, roi in enumerate(rois):
        ref = _cpu_preprocess(img, roi, size)
        # Downscale: OpenCV INTER_LINEAR area-averages slightly differently; allow ~2/255 px error.
        err = np.abs(host.array[i] - ref)
        assert err.mean() < 0.03, (roi, err.mean())
    host.free()
    out.free()
    dev.free()


def test_color_stats_matches_numpy(stream):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)
    rois = [(0, 0, 480, 360), (40, 30, 200, 150)]
    p = co.ColorParams(10, 40, 0.3, 0.15, 0.65, 0.18, 0.15, 0.7, 0.2, 0.15)
    dev = _upload(img, stream)
    out = co.DeviceBuffer(ctypes.sizeof(co.ColorStats) * len(rois))
    co.color_stats(dev, 480, 360, rois, p, 1, out, stream)
    host = co.PinnedArray((len(rois) * ctypes.sizeof(co.ColorStats),))
    co.d2h(host, out, stream)
    stream.sync()
    stats = (co.ColorStats * len(rois)).from_buffer_copy(host.array.tobytes())

    for roi, s in zip(rois, stats, strict=False):
        x, y, w, h = roi
        hsv = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV_FULL).astype(np.float64)
        bgr = img[y:y + h, x:x + w].astype(np.float64) / 255
        v = bgr.max(axis=2)
        mn = bgr.min(axis=2)
        sat = np.where(v > 1e-6, (v - mn) / np.maximum(v, 1e-6), 0)
        hue = hsv[..., 0] * 360 / 256
        fg = (sat >= 0.2) & (v >= 0.15)
        assert s.total == w * h
        assert abs(int(s.foreground) - int(fg.sum())) <= 0.002 * w * h
        assert abs(int(s.dark) - int((v <= 0.18).sum())) <= 0.002 * w * h
        brown = fg & (hue >= 10) & (hue <= 40) & (sat >= 0.3) & (v >= 0.15) & (v <= 0.65)
        assert abs(int(s.brown) - int(brown.sum())) <= 0.01 * w * h
        assert abs(s.sum_sat / s.foreground - sat[fg].mean()) < 1e-3
        assert sum(s.hue_hist) == s.foreground
    host.free()
    out.free()
    dev.free()


def test_rejects_out_of_frame_roi(stream):
    dev = co.DeviceBuffer(100 * 100 * 3)
    out = co.DeviceBuffer(3 * 32 * 32 * 4)
    with pytest.raises(co.CudaError, match="roi outside frame"):
        co.preprocess_rois(dev, 100, 100, [(90, 90, 20, 20)], out, 32, MEAN, STD, stream)
    dev.free()
    out.free()
