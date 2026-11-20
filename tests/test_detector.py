"""Detect mode: YOLOX output decoding (letterbox undo, class filter, NMS) and item tracking."""
import time

import numpy as np

from freshwatch.alerts import AlertManager
from freshwatch.config import CameraConfig, DetectorConfig, Settings
from freshwatch.detector import Detection, FruitDetector, anchor_grid
from freshwatch.pipeline import CameraWorker
from freshwatch.storage import Store

SIZE = 640


def make_detector(**cfg) -> FruitDetector:
    """A FruitDetector without a TensorRT engine: enough to test decode()."""
    det = FruitDetector.__new__(FruitDetector)
    det.cfg = DetectorConfig(**cfg)
    det.size = SIZE
    det.grid, det.stride = anchor_grid(SIZE)
    det.n_anchors, det.n_out = len(det.grid), 85
    det.class_ids = np.array(sorted(det.cfg.classes))
    det.class_names = [det.cfg.classes[i] for i in det.class_ids]
    return det


def raw_box(det, out, anchor, cx, cy, w, h, obj, cls_id, cls_p):
    """Write one anchor's raw output so that it decodes to centre (cx, cy), size (w, h) in network pixels."""
    g, s = det.grid[anchor], det.stride[anchor, 0]
    out[anchor, :4] = [cx / s - g[0], cy / s - g[1], np.log(w / s), np.log(h / s)]
    out[anchor, 4] = obj
    out[anchor, 5 + cls_id] = cls_p


def test_decode_maps_letterboxed_box_back_to_frame_pixels():
    det = make_detector()
    out = np.zeros((det.n_anchors, 85), np.float32)
    # 1280x720 frame -> scale 2, centred vertically: 140 px of padding above (640 - 360) / 2
    raw_box(det, out, 1000, cx=320, cy=320, w=100, h=60, obj=0.9, cls_id=46, cls_p=0.9)   # banana
    raw_box(det, out, 1001, cx=322, cy=321, w=98, h=60, obj=0.8, cls_id=46, cls_p=0.9)    # duplicate -> NMS
    raw_box(det, out, 2000, cx=100, cy=300, w=80, h=80, obj=0.9, cls_id=0, cls_p=0.99)    # person: ignored
    raw_box(det, out, 3000, cx=500, cy=400, w=60, h=60, obj=0.3, cls_id=47, cls_p=0.5)    # apple, too unsure
    dets = det.decode(out, 1280, 720)
    assert [d.produce for d in dets] == ["banana"]
    x, y, w, h = dets[0].box
    assert abs(x - 540) <= 1 and abs(y - 300) <= 1 and abs(w - 200) <= 1 and abs(h - 120) <= 1
    assert abs(dets[0].score - 0.81) < 1e-3


def test_decode_drops_tiny_finds():
    det = make_detector(min_box_px=24)
    out = np.zeros((det.n_anchors, 85), np.float32)
    raw_box(det, out, 500, cx=200, cy=320, w=8, h=8, obj=0.9, cls_id=49, cls_p=0.9)  # 16x16 px in the frame
    assert det.decode(out, 1280, 720) == []


def test_tracking_keeps_identity_and_expires(tmp_path):
    s = Settings(cameras=[])
    s.storage.db_path = tmp_path / "t.db"
    s.storage.snapshot_dir = None
    store = Store(s.storage.db_path)
    cam = CameraConfig(id="c", source="file:/dev/null", detect=True)
    w = CameraWorker(cam, s, None, store, AlertManager(s.alerts, s.storage, store, s.site.model_dump()))
    try:
        t0 = time.time()
        a = w._track([Detection("banana", 0.9, (100, 100, 200, 100)), Detection("apple", 0.8, (600, 300, 80, 80))], t0)
        assert sorted(z.name for z in a) == ["Apple 1", "Banana 1"]
        banana = next(z.id for z in a if z.produce == "banana")
        # the banana moved a little and a second banana appeared
        b = w._track([Detection("banana", 0.9, (110, 105, 200, 100)), Detection("banana", 0.7, (900, 400, 150, 90))],
                     t0 + 2)
        assert {z.id for z in b if z.name == "Banana 1"} == {banana}
        assert sorted(z.name for z in b) == ["Banana 1", "Banana 2"]
        assert len(w.states) == 3  # the apple is unseen but not expired yet
        w._track([], t0 + 2 + s.detector.track_ttl_s + 1)
        assert w.states == {}
    finally:
        w.stop()


def test_freshness_model_names_fruit_coco_lacks(tmp_path):
    s = Settings(cameras=[])
    s.storage.db_path = tmp_path / "t.db"
    store = Store(s.storage.db_path)
    w = CameraWorker(CameraConfig(id="c", source="file:/dev/null", detect=True), s, None, store,
                     AlertManager(s.alerts, s.storage, store, s.site.model_dump()))
    try:
        [z] = w._track([Detection("apple", 0.8, (100, 100, 200, 200))], time.time())
        st = w.states[z.id]
        assert w._relabel(st, z, "strawberry", 0.5).produce == "apple"       # not sure enough: keep
        z2 = w._relabel(st, z, "strawberry", 0.95)
        assert (z2.produce, z2.name, st.name) == ("strawberry", "Strawberry 1", "Strawberry 1")
        assert w._track([Detection("apple", 0.8, (105, 100, 200, 200))], time.time())[0].id == z.id  # same item
    finally:
        w.stop()
