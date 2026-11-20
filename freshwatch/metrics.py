"""Prometheus metrics (scraped at /metrics)."""
from prometheus_client import Counter, Gauge, Histogram

FRAMES_CAPTURED = Counter("freshwatch_frames_captured_total", "Frames read from the source", ["camera"])
ANALYSES = Counter("freshwatch_analyses_total", "Analysis passes executed", ["camera"])
ANALYSIS_ERRORS = Counter("freshwatch_analysis_errors_total", "Analysis passes that raised", ["camera"])
STAGE_LATENCY = Histogram(
    "freshwatch_stage_seconds", "Latency of pipeline stages", ["camera", "stage"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0))
CAMERA_UP = Gauge("freshwatch_camera_up", "1 if the camera is delivering frames", ["camera"])
ZONE_SCORE = Gauge("freshwatch_zone_score", "Smoothed freshness score 0..1", ["camera", "zone", "produce"])
ZONE_SEVERITY = Gauge("freshwatch_zone_severity", "0 fresh .. 3 spoiled", ["camera", "zone", "produce"])
ALERTS = Counter("freshwatch_alerts_total", "Alerts raised", ["kind"])
