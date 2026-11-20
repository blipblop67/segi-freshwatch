# SEGI FreshWatch

AI freshness monitoring for the SEGI Cash & Carry wet market, running fully on-device on an
**NVIDIA Jetson Orin Nano Super (JetPack 6.2, CUDA 12.6, TensorRT 10.3)**.

Fixed cameras look down at produce crates. Every crate is a *zone*. For each zone FreshWatch:

1. measures **spoilage colour cues** (browning, dark rot spots, mould) with a custom **CUDA kernel**,
2. classifies **produce type + fresh/rotten** with a **TensorRT FP16** MobileNetV3 model fed by a
   custom **CUDA preprocessing kernel** (crop + resize + normalise straight into the TensorRT input),
3. fuses both into a 0-100 freshness score, smooths it over time, and publishes a grade with an
   action for staff:

| Grade | Action |
|---|---|
| FRESH | no action |
| AGING | rotate stock, sell first |
| STALE | mark down / discount today |
| SPOILED | remove from display immediately |
| EMPTY | restock bin |

Alerts (with an evidence photo) go to the dashboard, SQLite, Prometheus and an optional webhook
(Slack/Teams/Telegram bridge, store system, …).

## Architecture

```
camera (RTSP / CSI / USB / file)
  └─ GStreamer + NVDEC (nvv4l2decoder) ── latest-frame thread, auto-reconnect
       └─ per-camera worker, own CUDA stream ─────────────────────────────────────┐
            pinned host frame ─H2D─► d_frame                                       │
            fc_color_stats      (CUDA, shared-mem + warp-shuffle reductions)       │  ~8 ms for 8
            fc_preprocess_rois  (CUDA, batched bilinear crop/resize/normalise)     │  zones @ 720p
            TensorRT FP16 execute_async_v3 (dynamic batch 1-16)                    │
            D2H logits + stats, one stream sync ───────────────────────────────────┘
       └─ scoring: model × colour-drift fusion → EMA → hysteresis → grade
       └─ alerts (cooldown, evidence JPEG, webhook w/ retry) · SQLite WAL · Prometheus
  FastAPI: dashboard · REST · MJPEG preview · /metrics · /healthz · /readyz
```

| Path | What |
|---|---|
| `cuda/` | `libfreshcuda.so`: CUDA kernels + memory/stream helpers with a C ABI (sm_87) |
| `freshwatch/` | the service: `cuda_ops` (ctypes binding), `trt_engine`, `camera`, `pipeline`, `scoring`, `alerts`, `storage`, `app` (API), `web/` (dashboard) |
| `training/` | dataset prep, training, ONNX export, TensorRT engine build |
| `tools/` | demo-video generator, production-path benchmark |
| `deploy/` | systemd unit + installer, Dockerfile, docker-compose |
| `tests/` | CUDA kernels vs. OpenCV/numpy, scoring logic, end-to-end API with TensorRT |

### Why colour *drift*, not absolute colour
A fresh orange or banana is "brown-ish" in HSV, so absolute thresholds raise false alarms. Each
crate learns its own colour baseline from its first confidently-fresh readings. It re-learns
automatically after every restock (EMPTY → stocked) or via `POST /api/v1/zones/{cam}/{zone}/baseline/reset`.
Only the *increase* in browning, dark spots or mould is penalised.

### Why hysteresis
A grade only changes after N consecutive agreeing readings (default 3). A shopper's hand over a
crate never triggers "spoiled", and the first grade is published only after the zone has warmed up.

## Results (measured on this Orin Nano Super, 15 W mode)

**Model:** MobileNetV3-Large, two heads (8 produce types; fresh/rotten). Trained on the device in about 6 minutes.

- **Data:** [Project-AgML/fresh_rotten_fruit_classification](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification), `raw` split (CC-BY-4.0).
- **Splits:** deterministic content-hash split of 2,543 train / 308 val / 342 test images.
- **Leakage avoided:** the `augmented` split was deliberately not used, because it would leak copies of test photos into training.

| Evaluation on 342 held-out test images | Freshness acc. | Rotten recall | Produce acc. |
|---|---|---|---|
| PyTorch (FP32 / AMP) | 100.0 % | 100.0 % | 100.0 % |
| **Production path: CUDA preprocess kernel + TensorRT FP16** | **99.4 %** | 98.8 % (2 misses) | 100.0 % |

| Zones per pass (1280×720 frame) | Full GPU pass p50 | of which TensorRT | zones/s |
|---|---|---|---|
| 1 | 3.7 ms | 3.3 ms | 272 |
| 8 | 8.1 ms | 7.6 ms | 986 |
| 16 | 14.6 ms | 14.0 ms | 1093 |

The CUDA colour + preprocess kernels take about 0.4–0.7 ms. Re-run with `make bench`; results are written to `models/benchmark.json`.

> **Important caveat:** these photos are clean, well-lit close-ups. Real crates under market
> lighting will be harder. Before go-live, collect and label a few hundred crops from SEGI's own
> cameras with `/api/v1/cameras/{id}/snapshot.jpg`, then fine-tune and re-measure. The training
> pipeline is built for exactly that. The current model covers 8 fruits only. Vegetables
> (sayur, kangkung, tomatoes, chillies, …) need SEGI data. Until then those zones can still run on
> colour drift + empty detection: unknown produce never triggers a mismatch alert.

## Quick start (this Jetson)

```bash
make setup        # venv + Jetson CUDA PyTorch (pypi.jetson-ai-lab.io jp6/cu126) + deps
make cuda         # build libfreshcuda.so for sm_87
make dataset      # download (~2.9 GB) + prepare
make model        # train → ONNX → TensorRT FP16 engine (engine must be built on the target device)
make demo-video   # 90 s shelf video from held-out test images + config/demo.yaml
make real-footage # 4 free Pexels market clips (7.4 MB)
make showcase     # simulation + real cameras on one dashboard: http://<jetson-ip>:8080
make test lint
```

`make help` lists every target. For a customer presentation, follow [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md).

The demo shows banana, strawberry and jujube decaying, grapes spoiled from the start, a shopper's hand
passing over the apples (no alert), and the guava crate being emptied and restocked. The video loops,
so states reset every 90 s.

## Real market footage test

```bash
make real-footage   # 4 free Pexels clips of real stalls with a fixed camera (7.4 MB, data/real/SOURCES.md)
make real-demo      # 4 cameras, 22 zones: http://<jetson-ip>:8080
make zones          # draw the configured zones on a frame from each camera (any config: CONFIG=...)
```

Results after 2 minutes of looped playback (`config/real.yaml`, before the v1.1 calibration below):

| | Result |
|---|---|
| Stability | No zone changed grade, even with stallholders walking in front of the crates. |
| Trained fruit, close-up (bananas) | Correct: fresh, and the right produce type. Spotted ripe bananas scored fresh (0.90–0.94), which is right: spots mean ripe, not spoiled. |
| Trained fruit, wide stall shot | Hanging bananas are correct. **Wrong:** red apples graded STALE (p_fresh about 0.5) and identified as jujube. Mandarins were identified as jujube. A 40 px banana crop was graded AGING. |
| Untrained produce (14 zones: vegetables, pineapple, jackfruit …) | Colour-drift only. All graded fresh, with no false alerts. |
| False alerts | 3: one STALE (the apples) and two produce mismatches. |

**v1.1 calibration** (`config/showcase.yaml`): the produce-mismatch threshold is now 0.8, the red-apple zone runs on colour analysis only, and the 40 px zone was dropped. Result: no false alerts on the real cameras. The dashboard shows each zone's mode ("AI + colour" or "Colour").

This is the gap the model's caveat warns about. Studio close-ups do not transfer to dim, wide market shots with many fruits per crop. Before go-live, fine-tune on crops from SEGI's own cameras. Until then, treat the
produce-mismatch alert as advisory (raise `scoring.min_produce_confidence`).

## Production deployment

```bash
sudo bash deploy/install.sh --showcase   # customer demo: installs AND starts it on every boot (= make install-showcase)
sudo bash deploy/install.sh          # /opt/freshwatch, user `freshwatch`, systemd unit (hardened)
sudoedit /etc/freshwatch/freshwatch.yaml   # cameras + zone ROIs
sudoedit /etc/freshwatch/freshwatch.env    # FRESHWATCH_API_KEY=…, FRESHWATCH_WEBHOOK_URL=…, CAM1_USER=…, CAM1_PASS=…
sudo systemctl start freshwatch && journalctl -u freshwatch -f
```

The container alternative is `deploy/Dockerfile` + `deploy/docker-compose.yml` (base `nvcr.io/nvidia/l4t-jetpack:r36.4.0`).

**Camera sources:**
- `rtsp://…` uses H.264 hardware decode.
- `csi:0` is a CSI camera.
- `usb:/dev/video0` is a USB camera (MJPEG).
- `gst:<pipeline>` takes a custom GStreamer pipeline.
- `file:` plays and loops a video file.

Orin Nano has no NVENC, so FreshWatch never encodes video. It produces only JPEG previews and evidence shots.

**Find produce anywhere (detect mode):** tick **Find produce automatically** for a camera in ⚙ Cameras (or set `detect: true`).
- **No zones needed:** the camera finds bananas, apples, oranges, broccoli and carrots anywhere in the picture, and grades and tracks each one.
- **Detector:** YOLOX-S, COCO classes, licensed Apache-2.0. Run `make detector` once per Jetson to download it and build its engine.
- **Zones vs detect mode:** fixed zones still suit crates and bins that are always in the same place. Detect mode suits loose produce on a table or counter, and quick demos.
- **Caveat:** COCO has no class for most wet-market produce (sayur, chillies, durian…). Those need fixed zones, or a detector fine-tuned on SEGI footage.

**Adding cameras and changing IP addresses:** use **⚙ Cameras** in the dashboard's top bar.
- **Add or edit:** enter the camera's IP address, pick its brand (this fills in the RTSP port, path and user), then **Test connection**. The test reports what is wrong, detects H.264/H.265, and shows a picture.
- **Save:** saving restarts only that camera. Changes are kept in `<db dir>/cameras.json` (mode 0600, it holds camera passwords) and override the YAML.
- **Scan network:** lists ONVIF cameras with their MAC addresses, for when DHCP gave a camera a new address.
- **New cameras:** a new camera starts as live view only. Add its zones in the YAML (below) to start grading.
- **Staff key:** when `api_key` is set, these pages need it.

**Setting zones:** grab `/api/v1/cameras/<id>/snapshot.jpg?annotate=false` and read crate pixel
rectangles off it, then set `roi: [x, y, w, h]`. Up to 32 zones per camera.

**Operations:**
- **Health:** `/healthz` reports liveness. `/readyz` reports readiness: 503 if a camera is down, with `degraded: true` when running without the model (`model.required: false`).
- **Metrics:** `/metrics` exposes per-stage latency histograms, camera up, per-zone score and severity, and alert counters.
- **Logs:** JSON logs (`log_json: true`) go to journald.
- **Retention:** SQLite in WAL mode. Readings are persisted every 60 s or on any grade change, and purged after 30 days.
- **Performance:** for the best throughput, set the MAXN SUPER power mode with `nvpmodel`. Check modes with `sudo nvpmodel -q --verbose`.

## API

| Method | Path | |
|---|---|---|
| GET | `/` | dashboard |
| GET | `/api/v1/status` | all zones, grades, scores, actions, camera health |
| GET | `/api/v1/zones/{cam}/{zone}/history?hours=24` | score history |
| POST | `/api/v1/zones/{cam}/{zone}/baseline/reset` | after restocking/rearranging (API key) |
| GET | `/api/v1/alerts?active=true` | alerts |
| POST | `/api/v1/alerts/{id}/ack` | acknowledge (API key, `X-API-Key` header) |
| GET | `/api/v1/alerts/{id}/snapshot.jpg` | evidence photo |
| GET | `/api/v1/cameras/{cam}/snapshot.jpg` · `/stream.mjpg` | annotated preview |
| GET | `/metrics` · `/healthz` · `/readyz` | ops |

Interactive OpenAPI docs are served at `/docs`.

## Licence of data/model
The model weights are trained on Project-AgML/fresh_rotten_fruit_classification (CC-BY-4.0). Attribution is required if you redistribute them.
