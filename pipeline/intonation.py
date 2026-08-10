"""Resolve notes whose measured pitch landed near a semitone boundary.

Violinists play expressive intonation — leading tones pushed sharp, thirds
pulled — so a note can sit 55 cents above B-flat and round to B natural. When
the measurement is that ambiguous, the key is better evidence than the
rounding.

This is deliberately narrow. It only touches notes where the confidence-
weighted mean pitch sits inside the ambiguous band, and even then only chooses
between the two semitones it was already torn between. A clearly-measured note
is never moved, no matter how chromatic it is — which is what separates this
from blanket key-snapping, and from the Viterbi smoothing that wrecked the
chromatic passages."""
import numpy as np

# How far from a semitone centre counts as "the tracker couldn't decide".
# 0.35 => the band from 35 to 65 cents between two semitones.
AMBIGUOUS_MARGIN = 0.35

MIN_CONFIDENT_NOTES = 12  # below this there isn't enough evidence to call a key

# Krumhansl-Kessler key profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_DEGREES = [0, 2, 4, 5, 7, 9, 11]
# Natural minor plus the leading tone — the raised 7th appears constantly in
# minor-key writing, so treating it as out-of-key would be wrong.
MINOR_DEGREES = [0, 2, 3, 5, 7, 8, 10, 11]

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _is_ambiguous(note: dict) -> bool:
    raw = note.get("pitch_raw")
    if raw is None:
        return False
    return abs(raw - round(raw)) >= AMBIGUOUS_MARGIN


def estimate_key(pitches, weights=None):
    """Krumhansl-Schmuckler key finding over a pitch-class histogram.
    Returns (tonic_pitch_class, is_minor, correlation) or None."""
    if len(pitches) < MIN_CONFIDENT_NOTES:
        return None

    hist = np.zeros(12)
    for i, p in enumerate(pitches):
        w = 1.0 if weights is None else float(weights[i])
        hist[int(p) % 12] += w

    if hist.sum() <= 0:
        return None
    hist = hist / hist.sum()

    best = None
    for tonic in range(12):
        rotated = np.roll(hist, -tonic)
        for profile, is_minor in ((KK_MAJOR, False), (KK_MINOR, True)):
            corr = float(np.corrcoef(rotated, profile)[0, 1])
            if not np.isfinite(corr):
                continue
            if best is None or corr > best[2]:
                best = (tonic, is_minor, corr)
    return best


def _scale_pitch_classes(tonic: int, is_minor: bool) -> set:
    degrees = MINOR_DEGREES if is_minor else MAJOR_DEGREES
    return {(tonic + d) % 12 for d in degrees}


def resolve_ambiguous_pitches(notes: list[dict]) -> list[dict]:
    if not notes:
        return notes

    confident = [n for n in notes if not _is_ambiguous(n)]
    ambiguous = [n for n in notes if _is_ambiguous(n)]

    if not ambiguous:
        print("[intonation] no ambiguous pitches")
        return notes

    est = estimate_key(
        [n["pitch"] for n in confident],
        [max(n.get("velocity", 64), 1) for n in confident],
    )
    if est is None:
        print(f"[intonation] {len(ambiguous)} ambiguous note(s) but not enough "
              f"confident notes to estimate a key — leaving them alone")
        return notes

    tonic, is_minor, corr = est
    in_key = _scale_pitch_classes(tonic, is_minor)
    print(f"[intonation] key estimate: {PITCH_NAMES[tonic]} "
          f"{'minor' if is_minor else 'major'} (r={corr:.2f}) "
          f"from {len(confident)} confident notes")

    changed = 0
    for n in ambiguous:
        raw = n["pitch_raw"]
        low, high = int(np.floor(raw)), int(np.ceil(raw))
        if low == high:
            continue

        low_ok = (low % 12) in in_key
        high_ok = (high % 12) in in_key

        # Only act when the key breaks the tie cleanly. If both candidates are
        # in the key, or neither is, the key has nothing to say — keep the
        # measurement.
        if low_ok == high_ok:
            continue

        preferred = low if low_ok else high
        if preferred != n["pitch"]:
            n["pitch"] = preferred
            changed += 1

    print(f"[intonation] {len(ambiguous)} ambiguous note(s), {changed} re-resolved by key")
    return notes