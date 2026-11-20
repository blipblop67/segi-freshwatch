# Changelog

## 1.2.0 (2026-11-19)

### Find produce anywhere (detect mode)
- **What it does:**
  - A camera with **Find produce automatically** (`detect: true`) finds bananas, apples, oranges, broccoli and carrots anywhere in the picture, with no zones to mark out.
  - Each item found is graded and tracked like a display, with score, grade, trend and alerts.
  - Empty space is never graded.
- **Detector:** YOLOX-S (Apache-2.0, COCO) on TensorRT FP16. `make detector` downloads it and builds the engine.
  - **Preprocessing:** the frame is letterboxed by the existing CUDA preprocess kernel, which has a new `FC_PRE_KEEP_BGR` flag.
  - **Postprocessing:** boxes are decoded and NMS runs on the CPU.
- **Grading:**
  - Banana, apple and orange are graded by the AI model plus colour; broccoli and carrot by colour only.
  - Items are tracked between frames by box overlap. An item gone for 6 s leaves the dashboard and its open alerts are cleared.

### Camera settings
- **⚙ Cameras page on the dashboard:** add IP cameras, change a camera's IP address, port, stream path or login, rename cameras, or remove them. Changes apply live, with no service restart.
- **Brand presets** fill in the RTSP port, path and default user for Reolink, Hikvision, Dahua, TP-Link Tapo/VIGI, Uniview and Axis.
- **Test connection:**
  - Says in plain words what is wrong: not reachable, port refused, wrong login, or wrong stream path.
  - Detects H.264 vs H.265 automatically and shows a live picture, decoded through the same NVDEC pipeline as production.
- **Scan network:** finds ONVIF cameras and shows their MAC address, so a camera that got a new IP can be recognised.
- **Storage:** changes are saved in `cameras.json` next to the database (mode 0600) and applied over the YAML at every start. Passwords are never sent back to the browser, and are removed from logs and status messages.
- **Live-view-only cameras:** a camera can now have no zones. It shows its live picture until its displays are set up.
- **H.265 streams:** RTSP cameras now support H.265 (`codec: h265`). Credentials are passed to GStreamer as quoted properties, so any password characters work.

### Deployment
- **`make install-showcase` (`deploy/install.sh --showcase`):** installs the customer demo as the `freshwatch` systemd service, so it starts on boot. It brings along the demo videos and the dashboard's camera settings.
- **Config path:** the service unit takes the config path from `FRESHWATCH_CONFIG`.
- **Runtime-only venv:** the installed venv gets runtime packages only.

### Fixes
- **Reolink:**
  - The preset path is now `/Preview_01_main`, which current firmware uses for both H.264 and H.265. `/h264Preview_01_main` only exists on older firmware.
  - On current firmware, an unknown path is answered with 401 instead of 404, so a wrong path looked like a wrong password.
- **Connection test login:** the test now sends the login on the same connection that received the challenge. Reolink ties the Digest nonce to that connection.
- **Wrong passwords no longer lock cameras out.**
  - Before connecting, FreshWatch checks the login once.
  - If the camera refuses it, the next try is 15 minutes later. Before, it retried every 30 s, and GStreamer retried several more times on each try.
  - Repeated failed logins made a Reolink camera lock its account, which also disconnected the vendor's phone app.
  - The dashboard now says "Wrong username or password" instead of a generic connection error.
  - A camera that actually sends H.265 is now switched over automatically.
- **Shutdown:** Ctrl+C / `systemctl stop` no longer hangs while a dashboard is open. Before, the never-ending MJPEG previews kept the process alive.
- **Offline cameras:** an offline camera shows "No picture from this camera" instead of the previous camera's last frame.

## 1.1.0 (2026-11-19)

### Dashboard
- **Redesign for store managers.**
  - KPIs are worded as actions: Sell first, Mark down, Remove now, Restock.
  - A prioritised **Staff actions** list, a camera viewer whose tabs show each camera's worst grade, and per-display cards with trend sparklines.
  - Technical detail is folded into "Analysis details".
- **Each display shows its analysis mode:** "AI + colour", or "Colour" for produce the model was not trained on.
- **System card:** board name, GPU, model, analysis speed and uptime.
- **Themes:** light and dark, with a toggle. Everything is inline, so the dashboard works offline on a store LAN.

### Detection
- **Untrained produce:** zones for produce the model was never trained on now use colour drift only. The model's score for them was meaningless.
- **Wrong-produce alerts:** the default `scoring.min_produce_confidence` was raised from 0.35 to 0.8. These alerts are advisory, and real footage showed them to be noisy at lower confidence.

### Demo and setup
- **`config/showcase.yaml` (`make showcase`):** one dashboard that combines the simulated decay scenario with four real market clips.
- **Zone setup:** `tools/zone_preview.py` (`make zones`) draws the configured zones on a frame from each camera.
- **Real footage:** `make real-footage` downloads the four real market clips (free Pexels footage, attributed in `data/real/SOURCES.md`).
- **Simulation video:** the debug text burned into it was replaced with a CCTV-style on-screen label.

### Packaging
- **Install and lint:** added `pyproject.toml` (installable package, `freshwatch` CLI, ruff config), plus `make help` and `make lint`.
- **Docs:** added `docs/DEMO_RUNBOOK.md` for customer presentations.

## 1.0.0 (2026-10-30)
- First release:
  - CUDA preprocessing and colour-statistics kernels.
  - TensorRT FP16 MobileNetV3 with produce and freshness outputs.
  - Scoring with a per-zone baseline and hysteresis.
  - FastAPI service with dashboard, alerts and webhooks, SQLite storage and Prometheus metrics.
  - systemd and Docker deployment files.
