"""Typed configuration loaded from YAML (+ FRESHWATCH_* env overrides)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SiteConfig(BaseModel):
    store_id: str = "segi-demo-01"
    name: str = "SEGI Cash & Carry - Wet Market"
    timezone: str = "Asia/Kuala_Lumpur"


class ModelConfig(BaseModel):
    engine: Path = Path("models/freshness_fp16.engine")
    labels: Path = Path("models/labels.json")
    max_batch: int = Field(16, ge=1, le=32)
    # If the engine cannot be loaded: True -> refuse to start; False -> run on
    # CUDA colour cues only and report "degraded" on /readyz.
    required: bool = True


class DetectorConfig(BaseModel):
    """Object detector for cameras with `detect: true` (YOLOX, COCO classes)."""

    engine: Path = Path("models/detector/yolox_s_fp16.engine")
    min_score: float = Field(0.35, gt=0, lt=1)
    nms_iou: float = Field(0.45, gt=0, lt=1)
    min_box_px: int = Field(24, ge=8)  # smaller finds are too few pixels to grade
    max_items: int = Field(16, ge=1, le=32)
    track_ttl_s: float = Field(6.0, gt=0)  # a find unseen this long leaves the dashboard
    # COCO class id -> produce name. Banana, apple and orange are graded by the AI model too;
    # broccoli and carrot on colour only (the freshness model has not been trained on them).
    classes: dict[int, str] = {46: "banana", 47: "apple", 49: "orange", 50: "broccoli", 51: "carrot"}


class ColorConfig(BaseModel):
    """HSV thresholds for the CUDA spoilage-cue kernel (hue in degrees, rest 0..1)."""

    brown_hue: tuple[float, float] = (10.0, 40.0)
    brown_sat_min: float = 0.30
    brown_val: tuple[float, float] = (0.12, 0.55)
    dark_val_max: float = 0.16
    mould_sat_max: float = 0.12
    mould_val_min: float = 0.75
    fg_sat_min: float = 0.18
    fg_val_min: float = 0.14
    sample_step: int = Field(2, ge=1, le=16)
    # Penalties applied to (1 - penalty * ratio) to form the colour score.
    k_brown: float = 1.2
    k_dark: float = 2.0
    k_mould: float = 1.5


class GradeThresholds(BaseModel):
    fresh: float = 0.75
    aging: float = 0.55
    stale: float = 0.35  # below -> spoiled

    @model_validator(mode="after")
    def _ordered(self):
        if not (1 >= self.fresh > self.aging > self.stale >= 0):
            raise ValueError("grade thresholds must satisfy 1 >= fresh > aging > stale >= 0")
        return self


class ScoringConfig(BaseModel):
    model_weight: float = Field(0.8, ge=0, le=1)
    ema_alpha: float = Field(0.35, gt=0, le=1)
    grades: GradeThresholds = GradeThresholds()
    min_produce_confidence: float = 0.8  # wrong-produce alerts are advisory: only very confident ones
    min_foreground_ratio: float = 0.12  # below -> zone reported EMPTY
    hysteresis_readings: int = Field(3, ge=1)
    trend_window_s: float = 3600.0


class AlertConfig(BaseModel):
    cooldown_s: float = 900.0
    webhook_url: Optional[str] = None
    webhook_timeout_s: float = 5.0
    webhook_retries: int = 3
    alert_on: list[str] = ["stale", "spoiled", "empty", "mismatch", "camera_down"]


class StorageConfig(BaseModel):
    db_path: Path = Path("data/freshwatch.db")
    persist_interval_s: float = 60.0
    retention_days: int = 30
    snapshot_dir: Optional[Path] = Path("data/snapshots")  # evidence JPEGs on alerts


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: Optional[str] = None  # required on mutating endpoints when set
    preview_fps: float = 5.0
    jpeg_quality: int = 80


class ZoneConfig(BaseModel):
    id: str
    name: str
    produce: str
    roi: tuple[int, int, int, int]  # x, y, w, h in source-frame pixels
    model_weight: Optional[float] = None  # per-zone override

    @field_validator("roi")
    @classmethod
    def _positive(cls, v):
        if v[0] < 0 or v[1] < 0 or v[2] <= 8 or v[3] <= 8:
            raise ValueError("roi must be x>=0, y>=0, w>8, h>8")
        return v


class CameraConfig(BaseModel):
    id: str
    name: str = ""
    # file:<path> | rtsp://... | csi:<sensor-id> | usb:/dev/videoN | gst:<pipeline ending in appsink>
    source: str
    codec: Literal["h264", "h265"] = "h264"  # RTSP stream encoding (Settings > Test connection detects it)
    detect: bool = False  # find produce anywhere with the object detector instead of grading fixed zones
    note: str = ""  # shown next to the camera name on the dashboard, e.g. "Simulation"
    width: int = 1280
    height: int = 720
    fps: float = 15.0
    analyze_interval_s: float = Field(2.0, ge=0.1)
    zones: list[ZoneConfig] = []  # a camera without zones is preview-only until displays are set up

    @model_validator(mode="after")
    def _zones_fit(self):
        ids = [z.id for z in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError(f"camera {self.id}: duplicate zone ids")
        if len(self.zones) > 32:
            raise ValueError(f"camera {self.id}: up to 32 zones supported")
        for z in self.zones:
            x, y, w, h = z.roi
            if x + w > self.width or y + h > self.height:
                raise ValueError(f"camera {self.id} zone {z.id}: roi exceeds {self.width}x{self.height}")
        return self


class Settings(BaseModel):
    site: SiteConfig = SiteConfig()
    model: ModelConfig = ModelConfig()
    detector: DetectorConfig = DetectorConfig()
    color: ColorConfig = ColorConfig()
    scoring: ScoringConfig = ScoringConfig()
    alerts: AlertConfig = AlertConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
    cameras: list[CameraConfig]
    log_level: str = "INFO"
    log_json: bool = False


def load_settings(path: str | Path) -> Settings:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base = path.parent.parent if path.parent.name == "config" else path.parent
    s = Settings.model_validate(data)
    # Relative paths in the YAML are resolved from the project root.
    for obj, attr in [(s.model, "engine"), (s.model, "labels"), (s.detector, "engine"), (s.storage, "db_path"),
                      (s.storage, "snapshot_dir")]:
        p = getattr(obj, attr)
        if p is not None and not p.is_absolute():
            setattr(obj, attr, (base / p).resolve())
    for cam in s.cameras:
        cam.source = os.path.expandvars(cam.source)
        if cam.source.startswith("file:") and not Path(cam.source[5:]).is_absolute():
            cam.source = "file:" + str((base / cam.source[5:]).resolve())
    if key := os.environ.get("FRESHWATCH_API_KEY"):
        s.api.api_key = key
    if url := os.environ.get("FRESHWATCH_WEBHOOK_URL"):
        s.alerts.webhook_url = url
    if lvl := os.environ.get("FRESHWATCH_LOG_LEVEL"):
        s.log_level = lvl
    return s
