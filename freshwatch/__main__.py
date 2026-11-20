"""Entry point: python -m freshwatch --config config/freshwatch.yaml"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import uvicorn

from .app import create_app
from .config import load_settings


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(r), "level": r.levelname, "logger": r.name, "msg": r.getMessage()}
        if r.exc_info:
            d["exc"] = self.formatException(r.exc_info)
        return json.dumps(d)


def main() -> None:
    ap = argparse.ArgumentParser(prog="freshwatch")
    ap.add_argument("--config", default="config/freshwatch.yaml")
    ap.add_argument("--check", action="store_true", help="validate config and exit")
    a = ap.parse_args()

    settings = load_settings(a.config)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else
                         logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.basicConfig(level=settings.log_level.upper(), handlers=[handler], force=True)
    if a.check:
        print(f"config OK: {len(settings.cameras)} camera(s), "
              f"{sum(len(c.zones) for c in settings.cameras)} zone(s)")
        return

    app = create_app(settings)
    uvicorn.run(app, host=settings.api.host, port=settings.api.port, log_config=None, access_log=False,
                # open MJPEG previews never end by themselves; don't let them block Ctrl+C / systemctl stop
                timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
