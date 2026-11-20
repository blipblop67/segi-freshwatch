#!/usr/bin/env bash
# Install FreshWatch as a systemd service on a JetPack 6.x Jetson (starts on boot, restarts on failure).
# Run from the project root AFTER `make setup cuda model`:
#   sudo bash deploy/install.sh              production: edit /etc/freshwatch/freshwatch.yaml, then start
#   sudo bash deploy/install.sh --showcase   customer demo (config/showcase.yaml) with its videos and the
#                                            cameras added in the dashboard; starts right away
# Re-run it after updating the code. Saved dashboard settings and alert history are kept.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
MODE=production
[ "${1:-}" = "--showcase" ] && MODE=showcase
SRC=$(pwd)
DST=/opt/freshwatch
STATE=/var/lib/freshwatch
[ -f "$SRC/freshwatch/__init__.py" ] || { echo "run from the project root"; exit 1; }

id freshwatch &>/dev/null || useradd --system --home "$STATE" --shell /usr/sbin/nologin \
  --groups video,render freshwatch
install -d -o freshwatch -g freshwatch "$STATE" "$STATE/snapshots"
install -d /etc/freshwatch "$DST"

rsync -a --delete --exclude data --exclude logs --exclude .pytest_cache --exclude '*.pt' --exclude .venv \
  "$SRC/" "$DST/"
# venv paths are absolute: rebuild it in place, runtime packages only
python3 -m venv --system-site-packages --clear "$DST/.venv"
sed '/^# Training/,$d' "$DST/requirements.txt" > "$DST/requirements-runtime.txt"
"$DST/.venv/bin/pip" install -q -r "$DST/requirements-runtime.txt"

[ -f /etc/freshwatch/freshwatch.yaml ] || install -m 0644 config/freshwatch.yaml /etc/freshwatch/freshwatch.yaml
[ -f /etc/freshwatch/freshwatch.env ] || install -m 0600 /dev/null /etc/freshwatch/freshwatch.env
sed -i "s#^  engine: models/#  engine: $DST/models/#; s#^  labels: models/#  labels: $DST/models/#" \
  /etc/freshwatch/freshwatch.yaml

install -m 0644 deploy/freshwatch.service /etc/systemd/system/freshwatch.service
DROPIN=/etc/systemd/system/freshwatch.service.d
if [ "$MODE" = showcase ]; then
  # The demo config uses paths under data/: point $DST/data at the writable state directory.
  install -d -o freshwatch -g freshwatch "$STATE/demo" "$STATE/real" "$STATE/showcase"
  for d in demo real; do
    [ -d "data/$d" ] && rsync -a --chown=freshwatch:freshwatch "data/$d/" "$STATE/$d/"
  done
  # Cameras added or edited in the dashboard (may hold camera passwords): copy once, never overwrite.
  if [ -f data/showcase/cameras.json ] && [ ! -f "$STATE/showcase/cameras.json" ]; then
    install -m 0600 -o freshwatch -g freshwatch data/showcase/cameras.json "$STATE/showcase/cameras.json"
  fi
  [ -e "$DST/data" ] && [ ! -L "$DST/data" ] && rm -rf "$DST/data"
  ln -sfn "$STATE" "$DST/data"
  install -d "$DROPIN"
  printf '[Service]\nEnvironment=FRESHWATCH_CONFIG=%s/config/showcase.yaml\n' "$DST" > "$DROPIN/showcase.conf"
else
  rm -f "$DROPIN/showcase.conf"
fi

# A copy started by hand (`make showcase`) would hold port 8080.
if pgrep -f "python -m freshwatch --config config/" >/dev/null; then
  echo "Stopping FreshWatch started by hand (make showcase/run) so the service can use port 8080"
  pkill -TERM -f "python -m freshwatch --config config/" || true
  sleep 6
fi

systemctl daemon-reload
if [ "$MODE" = showcase ]; then
  systemctl enable freshwatch
  systemctl restart freshwatch
  echo "FreshWatch showcase installed: starts on every boot. Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
  echo "Status: systemctl status freshwatch    Logs: journalctl -u freshwatch -f"
else
  systemctl enable freshwatch
  echo "Edit /etc/freshwatch/freshwatch.yaml (cameras/zones) and /etc/freshwatch/freshwatch.env (secrets),"
  echo "then: sudo systemctl start freshwatch && journalctl -u freshwatch -f"
fi
