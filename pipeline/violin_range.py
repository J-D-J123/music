"""Stage 6: transpose notes into the violin's playable range."""
import numpy as np
from config import VIOLIN_MIN_NOTE, VIOLIN_MAX_NOTE

MAX_OCTAVE_SHIFTS = 6  # guard against runaway loops on garbage input


def _too_high(pitch: int) -> bool:
    return pitch > VIOLIN_MAX_NOTE


def _too_low(pitch: int) -> bool:
    return pitch < VIOLIN_MIN_NOTE


def _out_of_range(pitch: int) -> bool:
    return _too_high(pitch) or _too_low(pitch)


def _run_still_out(notes, i, j, high: bool) -> bool:
    """True only while EVERY note in the run is still out on the same side —
    stops us shifting a run past the opposite boundary."""
    test = _too_high if high else _too_low
    return all(test(notes[k]["pitch"]) for k in range(i, j))


def fit_to_violin_range(notes: list[dict]) -> list[dict]:
    if not notes:
        return notes

    avg_pitch = np.mean([n["pitch"] for n in notes])
    center = (VIOLIN_MIN_NOTE + VIOLIN_MAX_NOTE) / 2
    octave_shift = round((center - avg_pitch) / 12) * 12

    for n in notes:
        n["pitch"] = int(n["pitch"] + octave_shift)
        if n.get("second_pitch") is not None:
            n["second_pitch"] = int(n["second_pitch"] + octave_shift)

    # Shift whole out-of-range runs together (preserves melodic shape), and keep
    # shifting until the run is actually in range — a single 12-semitone nudge
    # leaves anything more than an octave out still out.
    i = 0
    while i < len(notes):
        if not _out_of_range(notes[i]["pitch"]):
            i += 1
            continue

        high = _too_high(notes[i]["pitch"])
        direction = -12 if high else 12

        j = i
        while j < len(notes) and _out_of_range(notes[j]["pitch"]) \
                and _too_high(notes[j]["pitch"]) == high:
            j += 1

        guard = 0
        while _run_still_out(notes, i, j, high) and guard < MAX_OCTAVE_SHIFTS:
            for k in range(i, j):
                notes[k]["pitch"] += direction
                if notes[k].get("second_pitch") is not None:
                    notes[k]["second_pitch"] += direction
            guard += 1

        i = j

    # A double stop's second note can still sit outside the range even when the
    # primary is fine. Transposing it alone would change the interval, so drop
    # the second voice instead of silently mangling the chord.
    dropped = 0
    for n in notes:
        sp = n.get("second_pitch")
        if sp is not None and _out_of_range(sp):
            n["second_pitch"] = None
            dropped += 1

    msg = f"[violin_range] applied {octave_shift:+d} semitone base shift, run-clamped outliers"
    if dropped:
        msg += f", dropped {dropped} out-of-range double stop(s)"
    print(msg)
    return notes