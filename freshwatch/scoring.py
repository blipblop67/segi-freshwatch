"""Turn per-frame model outputs + colour cues into stable, actionable zone states.

Score fusion
    raw = w * P(fresh) + (1 - w) * colour_score
    colour_score = clamp(1 - k_brown*dBrown - k_dark*dDark - k_mould*dMould)
where dX is the *increase* of a CUDA colour cue over the zone's fresh baseline.
Absolute colour thresholds don't work across produce (a fresh orange is
"brown-ish" in HSV), so each crate learns its own baseline from its first
confidently-fresh readings and re-learns it after every restock (EMPTY -> stocked)
or on demand via the API. Until a baseline exists the model alone is used.

The raw score is smoothed with an EMA, mapped to a grade, and the published
grade only changes after `hysteresis_readings` consecutive readings agree, so a
shopper's hand over a bin does not trigger a "spoiled" alert.
"""
from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .config import ColorConfig, ScoringConfig, ZoneConfig
from .cuda_ops import ColorCues


class Grade(str, enum.Enum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    SPOILED = "spoiled"
    EMPTY = "empty"
    UNKNOWN = "unknown"


ACTIONS = {
    Grade.FRESH: "OK - no action",
    Grade.AGING: "Rotate stock: move to front, sell first",
    Grade.STALE: "Mark down / discount today",
    Grade.SPOILED: "Remove from display immediately",
    Grade.EMPTY: "Restock bin",
    Grade.UNKNOWN: "Warming up / check camera view",
}

SEVERITY = {Grade.FRESH: 0, Grade.AGING: 1, Grade.UNKNOWN: 1, Grade.EMPTY: 2, Grade.STALE: 2, Grade.SPOILED: 3}

BASELINE_SAMPLES = 5
BASELINE_MIN_P_FRESH = 0.8


@dataclass
class Baseline:
    brown: float
    dark: float
    mould: float
    ts: float


@dataclass
class Reading:
    """One analysed observation of a zone."""

    ts: float
    p_fresh: Optional[float]          # None when running without the model
    produce_pred: Optional[str]
    produce_conf: Optional[float]
    color_score: Optional[float]      # None until the zone has a baseline
    raw_score: Optional[float]
    cues: ColorCues
    empty: bool
    mismatch: bool


@dataclass
class ZoneState:
    camera_id: str
    zone_id: str
    name: str
    produce: str
    roi: tuple[int, int, int, int]
    score: Optional[float] = None
    grade: Grade = Grade.UNKNOWN
    candidate: Grade = Grade.UNKNOWN
    candidate_count: int = 0
    last: Optional[Reading] = None
    updated_ts: float = 0.0
    grade_since_ts: float = field(default_factory=time.time)
    baseline: Optional[Baseline] = None
    _baseline_acc: list = field(default_factory=list)
    history: deque = field(default_factory=lambda: deque(maxlen=2000))  # (ts, score)

    @property
    def action(self) -> str:
        return ACTIONS[self.grade]

    def reset_baseline(self) -> None:
        self.baseline = None
        self._baseline_acc.clear()

    def learn_baseline(self, r: Reading) -> None:
        if self.baseline is not None or r.empty:
            return
        if r.p_fresh is not None and r.p_fresh < BASELINE_MIN_P_FRESH:
            self._baseline_acc.clear()  # only learn from a run of confidently fresh readings
            return
        self._baseline_acc.append(r.cues)
        if len(self._baseline_acc) >= BASELINE_SAMPLES:
            c = self._baseline_acc
            self.baseline = Baseline(float(np.median([x.brown_ratio for x in c])),
                                     float(np.median([x.dark_ratio for x in c])),
                                     float(np.median([x.mould_ratio for x in c])), r.ts)
            self._baseline_acc.clear()

    def hours_to_markdown(self, stale_threshold: float, window_s: float) -> Optional[float]:
        """Linear trend of the smoothed score -> hours until it crosses `stale`."""
        if self.score is None or self.score <= stale_threshold or len(self.history) < 10:
            return None
        now = self.history[-1][0]
        pts = np.array([(t, s) for t, s in self.history if now - t <= window_s])
        if len(pts) < 10 or pts[-1, 0] - pts[0, 0] < 300:
            return None
        slope = np.polyfit(pts[:, 0] - pts[0, 0], pts[:, 1], 1)[0]  # score / s
        if slope >= -1e-7:
            return None
        return float((self.score - stale_threshold) / -slope / 3600.0)

    def to_dict(self, cfg: ScoringConfig) -> dict:
        r = self.last
        rnd = lambda v: None if v is None else round(v, 4)  # noqa: E731
        return {
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "name": self.name,
            "produce": self.produce,
            "roi": list(self.roi),
            "score": rnd(self.score),
            "grade": self.grade.value,
            "action": self.action,
            "severity": SEVERITY[self.grade],
            "grade_since": self.grade_since_ts,
            "updated": self.updated_ts,
            "hours_to_markdown": self.hours_to_markdown(cfg.grades.stale, cfg.trend_window_s),
            "baseline": None if self.baseline is None else {k: rnd(v) for k, v in asdict(self.baseline).items()},
            "last": None if r is None else {
                "p_fresh": rnd(r.p_fresh),
                "produce_pred": r.produce_pred,
                "produce_conf": rnd(r.produce_conf),
                "color_score": rnd(r.color_score),
                "raw_score": rnd(r.raw_score),
                "mismatch": r.mismatch,
                "cues": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(r.cues).items()
                         if k != "hue_hist"},
            },
        }


def color_score(cues: ColorCues, baseline: Optional[Baseline], c: ColorConfig) -> Optional[float]:
    if baseline is None:
        return None
    pen = (c.k_brown * max(0.0, cues.brown_ratio - baseline.brown)
           + c.k_dark * max(0.0, cues.dark_ratio - baseline.dark)
           + c.k_mould * max(0.0, cues.mould_ratio - baseline.mould))
    return float(min(max(1.0 - pen, 0.0), 1.0))


def grade_for(score: float, cfg: ScoringConfig) -> Grade:
    g = cfg.grades
    if score >= g.fresh:
        return Grade.FRESH
    if score >= g.aging:
        return Grade.AGING
    if score >= g.stale:
        return Grade.STALE
    return Grade.SPOILED


def make_reading(zone: ZoneConfig, cues: ColorCues, baseline: Optional[Baseline], p_fresh: Optional[float],
                 produce_pred: Optional[str], produce_conf: Optional[float], known_produce: set[str],
                 scfg: ScoringConfig, ccfg: ColorConfig, ts: float) -> Reading:
    cs = color_score(cues, baseline, ccfg)
    w = zone.model_weight if zone.model_weight is not None else scfg.model_weight
    if zone.produce not in known_produce or w <= 0:
        p_fresh = None  # the model was never trained on this produce: colour drift only
    if p_fresh is None:
        raw = cs
    elif cs is None:
        raw = p_fresh
    else:
        raw = w * p_fresh + (1 - w) * cs
    mismatch = (produce_pred is not None and zone.produce in known_produce and produce_pred != zone.produce
                and (produce_conf or 0) >= scfg.min_produce_confidence)
    return Reading(ts=ts, p_fresh=p_fresh, produce_pred=produce_pred, produce_conf=produce_conf,
                   color_score=cs, raw_score=raw, cues=cues,
                   empty=cues.foreground_ratio < scfg.min_foreground_ratio, mismatch=mismatch)


def update_state(st: ZoneState, r: Reading, cfg: ScoringConfig) -> Optional[tuple[Grade, Grade]]:
    """Apply a reading. Returns (old, new) when the published grade changes."""
    st.last = r
    st.updated_ts = r.ts
    if r.empty:
        target = Grade.EMPTY
    elif r.raw_score is None:          # degraded mode, baseline still being learned
        target = Grade.UNKNOWN
    else:
        st.score = r.raw_score if st.score is None or st.grade == Grade.EMPTY else \
            cfg.ema_alpha * r.raw_score + (1 - cfg.ema_alpha) * st.score
        st.history.append((r.ts, st.score))
        target = grade_for(st.score, cfg)

    if target == st.candidate:
        st.candidate_count += 1
    else:
        st.candidate, st.candidate_count = target, 1

    change = None
    if target != st.grade and st.candidate_count >= cfg.hysteresis_readings:
        old, st.grade = st.grade, target
        st.grade_since_ts = r.ts
        if old == Grade.EMPTY:
            st.reset_baseline()        # restocked: learn the new batch's baseline
        change = (old, target)
    st.learn_baseline(r)
    return change
