"""Stage 5: snap note start times and durations to a musical grid.

When beat_times are supplied, positions are measured in BEATS rather than
seconds. This is what makes rubato playable: if the performer slows down, the
beats spread out in real time, but a note filling one beat is still notated as
a quarter note. Quantizing against a single average BPM instead turns every
tempo swing into wrong note values."""
from fractions import Fraction
import numpy as np
from config import QUANT_GRID

GRID_UNIT = min(QUANT_GRID)

# Only absorb gaps up to one grid unit. Larger gaps are real rests — absorbing
# them (the old 2x value) removed nearly all breathing from the score.
MAX_GAP_TO_CLOSE = GRID_UNIT

BEAT_EXTENSION = 8  # beats of linear extrapolation past the last tracked beat


def _snap(value: Fraction, unit: Fraction) -> Fraction:
    return round(value / unit) * unit


def build_beat_map(beat_times):
    """Build a monotonic (seconds -> quarterLength) mapping anchored so that
    t=0 maps to ql=0. Anchoring at 0 rather than at the first detected beat
    keeps every part in the score on the same timeline."""
    beats = np.asarray(beat_times, dtype=float)
    if beats.size < 2:
        return None

    intervals = np.diff(beats)
    median_interval = float(np.median(intervals))
    if not np.isfinite(median_interval) or median_interval <= 0:
        return None

    ql_at_beats = np.arange(beats.size, dtype=float) + beats[0] / median_interval

    # A synthetic point at t=0 is exactly what linear extrapolation at the
    # median tempo would give, so this stays consistent.
    post_times = beats[-1] + median_interval * np.arange(1, BEAT_EXTENSION + 1)
    post_ql = ql_at_beats[-1] + np.arange(1, BEAT_EXTENSION + 1)

    times = np.concatenate([[0.0], beats, post_times])
    qls = np.concatenate([[0.0], ql_at_beats, post_ql])

    # np.interp needs strictly increasing x
    keep = np.concatenate([[True], np.diff(times) > 0])
    return times[keep], qls[keep]


def _seconds_to_ql(t: float, beat_map) -> float:
    times, qls = beat_map
    return float(np.interp(t, times, qls))


def quantize_notes(notes: list[dict], bpm: float, beat_times=None) -> list[dict]:
    beat_map = build_beat_map(beat_times) if beat_times is not None else None
    seconds_per_quarter = 60.0 / bpm
    out = []

    for n in notes:
        if beat_map is not None:
            start_raw = _seconds_to_ql(n["start"], beat_map)
            end_raw = _seconds_to_ql(n["end"], beat_map)
        else:
            start_raw = n["start"] / seconds_per_quarter
            end_raw = n["end"] / seconds_per_quarter

        raw_start_ql = Fraction(start_raw).limit_denominator(96)
        raw_dur_ql = Fraction(max(end_raw - start_raw, 0.0)).limit_denominator(96)

        start_ql = _snap(raw_start_ql, GRID_UNIT)
        if start_ql < 0:
            start_ql = Fraction(0)
        snapped_dur = min(QUANT_GRID, key=lambda q: abs(q - raw_dur_ql))

        out.append({
            "pitch": n["pitch"],
            "start_ql": start_ql,
            "quarterLength": snapped_dur,
            "velocity": n["velocity"],
            "second_pitch": n.get("second_pitch"),
        })

    out.sort(key=lambda n: n["start_ql"])

    # Trim real overlaps
    for i in range(1, len(out)):
        prev = out[i - 1]
        prev_end = prev["start_ql"] + prev["quarterLength"]
        if out[i]["start_ql"] < prev_end:
            prev["quarterLength"] = max(Fraction(0), out[i]["start_ql"] - prev["start_ql"])

    out = [n for n in out if n["quarterLength"] > 0]

    # Close only rounding-sized gaps; anything larger stays a rest
    for i in range(1, len(out)):
        prev = out[i - 1]
        prev_end = prev["start_ql"] + prev["quarterLength"]
        gap = out[i]["start_ql"] - prev_end
        if 0 < gap <= MAX_GAP_TO_CLOSE:
            prev["quarterLength"] += gap

    mode = "beat-relative" if beat_map is not None else f"fixed {bpm:.1f} BPM"
    print(f"[quantize] quantized {len(out)} notes ({mode})")
    return out