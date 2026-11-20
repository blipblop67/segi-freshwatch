# FreshWatch customer showcase: runbook

Internal notes for the presenter. Plan 10–15 minutes plus questions.

## Start on boot (once per Jetson)

```bash
make install-showcase
```
- **What it does:**
  - Copies FreshWatch to `/opt/freshwatch` and installs the `freshwatch` systemd service, which starts on every boot and restarts by itself after a crash.
  - Moves the dashboard's camera settings to `/var/lib/freshwatch/showcase/cameras.json`, a one-time copy. After that, change cameras in the dashboard only.
- **Asks for:** the sudo password.
- **Commands:**
  - Status: `systemctl status freshwatch`
  - Logs: `journalctl -u freshwatch -f`
  - Stop or start: `sudo systemctl stop freshwatch` / `sudo systemctl start freshwatch`
- **After code changes:** re-run `make install-showcase`. Dashboard settings and alert history are kept.
- **Reset alerts:** with the service installed, use `sudo systemctl stop freshwatch && sudo rm -rf /var/lib/freshwatch/showcase/freshwatch.db* /var/lib/freshwatch/showcase/snapshots && sudo systemctl start freshwatch` instead of `make showcase-reset`.

## 30 minutes before

1. **Power and network.** Connect the Jetson to power and to the same LAN or Wi-Fi as the presentation laptop.
   - Find the address: `hostname -I`. The first address is the one to use, e.g. `192.168.1.101`.
2. **Performance mode.**
   - Check: `sudo nvpmodel -q` should print `MAXN_SUPER`.
   - If it doesn't: `sudo nvpmodel -m 2`. On this unit, mode 2 is MAXN_SUPER; check the number with `sudo nvpmodel -q --verbose`.
3. **Start the service.** If it is installed to start on boot (above), it is already running: skip to step 4. Otherwise, from the project folder: Clear old alerts first so the alert log only shows today's demo (camera settings are kept):
   ```bash
   make showcase-reset && make showcase
   ```
   - **Cameras:** every camera in **⚙ Cameras** should show a green dot.
   - **If a real IP camera is not reachable on site:** remove it there, or the dashboard shows a red "offline" banner and `/readyz` reports not ready.
   - Wait for `Uvicorn running on http://0.0.0.0:8080`.
   - Then check that `curl -s localhost:8080/readyz` reports `"ready":true`.
4. **Open the dashboard** on the laptop at `http://<jetson-ip>:8080`.
   - Use full screen (F11) at 100% zoom; 1440 px wide or more looks best.
   - The ◐ button switches between light and dark themes. Light is easier to see on projectors.
5. **Let it run for about 2 minutes** before people arrive, so every display has a grade and a sparkline.

## Suggested flow

| # | Show | Say |
|---|------|-----|
| 1 | Top bar and KPI row | "Every display in the market gets a freshness grade and one clear staff action: sell first, mark down, remove, or restock." |
| 2 | **Aisle 3 · Fruit display** (simulated 90 s decay) | Watch bananas, strawberries and jujubes go from Fresh to Mark down to Remove. The grapes are spoiled from the start. A shopper's hand over the apples triggers **no** alert. The guava crate is emptied, flagged "Restock", and then refilled. |
| 3 | **Staff actions** panel | "Staff get a prioritised to-do list: worst first, oldest first." Open the **Alert log** and an **Evidence photo**. Each alert keeps the camera frame as proof. |
| 4 | **Fruit stall · wide view** (real footage) | "Real Philippine roadside stall. Stallholders walk in front of the crates all the time, and no false alarms are raised." |
| 5 | **Vegetable wet market** / **Vegetable counter** | "Vegetables run on our CUDA colour analysis today. The badge on each card says **Colour**. Once we collect SEGI's own photos, the AI model takes over for these too." |
| 6 | **System** card | "All of this runs on one Jetson Orin Nano at the market. Custom CUDA kernels plus TensorRT analyse a frame in tens of milliseconds. No video leaves the store, and nothing depends on the internet." |

The simulation loops every 90 seconds. If you miss a moment, just wait for it to come around again.

## Questions to expect, with honest answers

- **How accurate is it?**
  - Studio photos the model never saw: 99.4% fresh/rotten accuracy (342 images).
  - Real market footage: the model is reliable on close-ups and bananas, but weaker on dim wide shots.
  - That is why the pilot starts by collecting SEGI's own footage and fine-tuning the model on it.
- **Which produce does it know?** The AI model knows 8 fruits: apple, banana, grape, guava, jujube, orange, pomegranate and strawberry. All other produce, including every vegetable, gets colour-change and empty-display detection now, and AI grading after fine-tuning on SEGI data.
- **What hardware is needed?** One Jetson Orin Nano per site. It handles about 5 cameras at one analysis per second each, as in this demo. It uses the store's existing IP cameras over RTSP, or USB/CSI cameras.
- **Where does the data go?** It stays on the device: a local database, the dashboard and metrics. A webhook can push alerts to Slack, Teams, Telegram or the store system.
- **What does a pilot look like?**
  1. Install at one aisle for 2–4 weeks.
  2. Record and label crops from SEGI's cameras.
  3. Fine-tune the model, then measure false-alert and miss rates against staff checks.

**Do not claim:** that it detects spoilage in vegetables with AI today, that it works on any camera angle without setup, or that the 99.4% figure applies to real stores.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Dashboard says "Service unreachable" | Restart in the terminal: Ctrl+C, then `make showcase`. |
| A red "Camera … offline" banner | The camera reconnects by itself within about 30 s. For an IP camera, open **⚙ Cameras › Edit › Test connection**: it says whether the IP, login or stream path is wrong. |
| A display shows "Calibrating" | Normal for the first few seconds after start or after a restock. |
| Choppy video on the laptop | Use wired LAN, or close other browser tabs. Analysis on the Jetson is not affected. |

## After the demo

Stop the service with Ctrl+C.
