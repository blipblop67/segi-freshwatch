from freshwatch.config import ColorConfig, ScoringConfig, ZoneConfig
from freshwatch.cuda_ops import ColorCues
from freshwatch.scoring import BASELINE_SAMPLES, Grade, ZoneState, make_reading, update_state

ZONE = ZoneConfig(id="A1", name="Bananas", produce="banana", roi=(0, 0, 100, 100))
SC = ScoringConfig(hysteresis_readings=3)
CC = ColorConfig()


def cues(fg=0.6, brown=0.05, dark=0.01, mould=0.0):
    return ColorCues(fg, brown, dark, mould, 0.5, 0.6, tuple([1 / 18] * 18))


def feed(st, n, p_fresh, c=None, t0=0.0, pred="banana", conf=0.9):
    changes = []
    for i in range(n):
        r = make_reading(ZONE, c or cues(), st.baseline, p_fresh, pred, conf, {"banana", "apple"}, SC, CC, t0 + i)
        ch = update_state(st, r, SC)
        if ch:
            changes.append(ch)
    return changes


def new_state():
    return ZoneState("cam", "A1", "Bananas", "banana", (0, 0, 100, 100))


def test_first_grade_needs_hysteresis():
    st = new_state()
    assert feed(st, 2, 0.95) == []
    assert st.grade == Grade.UNKNOWN
    assert feed(st, 1, 0.95, t0=2) == [(Grade.UNKNOWN, Grade.FRESH)]


def test_brief_occlusion_does_not_flip_grade():
    st = new_state()
    feed(st, 10, 0.95)
    assert st.grade == Grade.FRESH
    feed(st, 2, 0.0, t0=10)            # hand over the bin for two readings
    feed(st, 3, 0.95, t0=12)
    assert st.grade == Grade.FRESH


def test_decay_reaches_spoiled():
    st = new_state()
    feed(st, 10, 0.95)
    feed(st, 15, 0.02, cues(brown=0.5, dark=0.2), t0=10)
    assert st.grade == Grade.SPOILED


def test_baseline_learned_then_drift_penalised():
    st = new_state()
    feed(st, BASELINE_SAMPLES, 0.95, cues(brown=0.30))   # naturally "brown-ish" produce
    assert st.baseline is not None and abs(st.baseline.brown - 0.30) < 1e-9
    r = make_reading(ZONE, cues(brown=0.30), st.baseline, 0.95, "banana", 0.9, {"banana"}, SC, CC, 99)
    assert r.color_score == 1.0                           # no drift -> no penalty
    r = make_reading(ZONE, cues(brown=0.60), st.baseline, 0.95, "banana", 0.9, {"banana"}, SC, CC, 99)
    assert r.color_score < 0.7


def test_no_baseline_from_rotten_stock():
    st = new_state()
    feed(st, 10, 0.1)
    assert st.baseline is None


def test_empty_then_restock_resets_baseline():
    st = new_state()
    feed(st, 10, 0.95)
    assert st.baseline is not None
    feed(st, 3, 0.95, cues(fg=0.02), t0=10)
    assert st.grade == Grade.EMPTY
    feed(st, 3, 0.95, t0=13)
    assert st.grade == Grade.FRESH
    assert st.baseline is None or st.baseline.ts >= 13


def test_mismatch_only_for_known_produce():
    r = make_reading(ZONE, cues(), None, 0.9, "apple", 0.95, {"banana", "apple"}, SC, CC, 0)
    assert r.mismatch
    kangkung = ZoneConfig(id="V1", name="Kangkung", produce="kangkung", roi=(0, 0, 100, 100))
    r = make_reading(kangkung, cues(), None, 0.9, "apple", 0.95, {"banana", "apple"}, SC, CC, 0)
    assert not r.mismatch


def test_degraded_mode_without_model():
    st = new_state()
    feed(st, 3, None)
    assert st.grade == Grade.UNKNOWN        # still learning baseline
    feed(st, 6, None, t0=3)
    assert st.grade == Grade.FRESH


def test_untrained_produce_ignores_model_and_learns_colour_baseline():
    cucumber = ZoneConfig(id="V2", name="Cucumber", produce="cucumber", roi=(0, 0, 100, 100))
    st = ZoneState("cam", "V2", "Cucumber", "cucumber", (0, 0, 100, 100))
    for i in range(BASELINE_SAMPLES + 3):  # model says "rotten" for an unseen class: must be ignored
        r = make_reading(cucumber, cues(), st.baseline, 0.05, "grape", 0.6, {"banana", "apple"}, SC, CC, i)
        assert r.p_fresh is None
        update_state(st, r, SC)
    assert st.baseline is not None and st.grade == Grade.FRESH
