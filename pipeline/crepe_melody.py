"""Hybrid melody extraction: onset detection defines note boundaries, CREPE
supplies the pitch inside each boundary.

Split into analyze() (expensive — runs CREPE) and notes_from_analysis()
(cheap — segments and filters) so a parameter sweep can reuse one CREPE pass."""
import numpy as np
import crepe
import librosa

from config import VIOLIN_MIN_NOTE, VIOLIN_MAX_NOTE

CONFIDENCE_THRESHOLD = 0.35   # voicing: is there a note here at all?
PITCH_CONFIDENCE_THRESHOLD = 0.55  # pitch: which frames are trustworthy enough to name it?
MIN_NOTE_SECONDS = 0.03
ONSET_DELTA = 0.02

# The first few ms after an onset are bow-attack noise, where pitch is genuinely
# unstable. Including it drags the estimate off — worst on short notes.
ATTACK_SKIP_SECONDS = 0.03

MERGE_SAME_PITCH = True
MERGE_GAP_SECONDS = 0.06

# A violin physically cannot sound below its open G string, so frames reporting
# a lower pitch are cello/piano bleed in the 'other' stem, not the melody.
RESTRICT_TO_VIOLIN_RANGE = True
RANGE_TOLERANCE_SEMITONES = 1.0

# Slide (portamento) detection. CREPE's confidence drops while pitch is sweeping,
# so those frames read as unvoiced and the slide becomes a gap — which then
# notates as a rest. These settings recover the sweep instead.
DETECT_SLIDES = True
SLIDE_MIN_SEMITONES = 2      # below this it's vibrato or tracking wobble
SLIDE_MAX_SECONDS = 0.40     # longer than this is a rest, not a slide
SLIDE_INSIDE_FRACTION = 0.70 # how much of the sweep must sit between the two pitches
SLIDE_MONOTONIC_FRACTION = 0.60  # how consistently it must travel one direction


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 69 + 12 * np.log2(hz / 440.0)


def _dominant_pitch(midi_vals: np.ndarray, confs: np.ndarray):
    """Return (rounded_pitch, weighted_mean).

    The rounded pitch comes from a confidence-weighted vote over semitone bins,
    which is robust to a few wild frames. The weighted mean is returned too
    because it says HOW near a semitone boundary the note actually sat —
    intonation.py uses that to spot genuinely ambiguous readings."""
    bins = np.round(midi_vals).astype(int)
    weights = {}
    for b, c in zip(bins, confs):
        weights[b] = weights.get(b, 0.0) + float(c)
    voted = int(max(weights.items(), key=lambda kv: kv[1])[0])

    total = float(np.sum(confs))
    mean = float(np.sum(midi_vals * confs) / total) if total > 0 else float(voted)
    return voted, mean


def _merge_same_pitch(notes: list[dict], max_gap: float) -> list[dict]:
    """Glue adjacent fragments back together ONLY when they share a pitch and
    are contiguous. Different-pitch neighbours are never touched, which keeps
    chromatic runs and alternating figures intact."""
    if not notes:
        return notes

    merged = [dict(notes[0])]
    for raw in notes[1:]:
        n = dict(raw)
        prev = merged[-1]
        if n["pitch"] == prev["pitch"] and (n["start"] - prev["end"]) <= max_gap:
            prev["end"] = n["end"]
            prev["velocity"] = max(prev["velocity"], n["velocity"])
        else:
            merged.append(n)
    return merged


def _detect_slides(notes: list[dict], analysis: dict) -> list[dict]:
    """Find gaps that are actually portamento and mark them.

    A slide looks like: a pitch trace that (a) stays between the two notes'
    pitches, and (b) travels consistently in one direction. Silence looks like
    neither. Marked notes get extended through the sweep so no rest appears,
    and carry slide_to_next so notate.py can draw a glissando."""
    time = analysis["time"]
    frequency = analysis["frequency"]
    midi_pitches = analysis["midi_pitches"]

    found = 0
    for i in range(len(notes) - 1):
        a, b = notes[i], notes[i + 1]
        gap_start, gap_end = a["end"], b["start"]

        if gap_end <= gap_start or (gap_end - gap_start) > SLIDE_MAX_SECONDS:
            continue

        interval = b["pitch"] - a["pitch"]
        if abs(interval) < SLIDE_MIN_SEMITONES:
            continue

        mask = (time >= gap_start) & (time < gap_end) & (frequency > 0)
        if not np.any(mask):
            continue

        vals = midi_pitches[mask]
        lo, hi = min(a["pitch"], b["pitch"]), max(a["pitch"], b["pitch"])
        inside = float(np.mean((vals >= lo - 1) & (vals <= hi + 1)))
        if inside < SLIDE_INSIDE_FRACTION:
            continue

        diffs = np.diff(vals)
        if diffs.size:
            direction = np.sign(interval)
            travelling = float(np.mean(np.sign(diffs) == direction))
            if travelling < SLIDE_MONOTONIC_FRACTION:
                continue

        a["slide_to_next"] = True
        a["end"] = gap_end  # the sweep is part of the sounding note
        found += 1

    if found:
        print(f"[crepe_melody] detected {found} slide(s)")
    return notes


def analyze(audio_path, step_size_ms: int = 10) -> dict:
    """Load the audio and run CREPE. This is the slow part — do it once."""
    print(f"[crepe_melody] loading {audio_path}")
    y, sr = librosa.load(str(audio_path), sr=16000)

    print("[crepe_melody] running CREPE pitch tracking (slow on CPU)...")
    time, frequency, confidence, _ = crepe.predict(
        y, sr, step_size=step_size_ms, viterbi=True, verbose=0
    )

    return {
        "y": y,
        "sr": sr,
        "time": time,
        "frequency": frequency,
        "confidence": confidence,
        "midi_pitches": _hz_to_midi(frequency),
        "step_seconds": step_size_ms / 1000.0,
    }


def notes_from_analysis(
    analysis: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    pitch_confidence_threshold: float = PITCH_CONFIDENCE_THRESHOLD,
    min_note_seconds: float = MIN_NOTE_SECONDS,
    onset_delta: float = ONSET_DELTA,
    attack_skip_seconds: float = ATTACK_SKIP_SECONDS,
    merge_same_pitch: bool = MERGE_SAME_PITCH,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    restrict_to_violin_range: bool = RESTRICT_TO_VIOLIN_RANGE,
    range_tolerance_semitones: float = RANGE_TOLERANCE_SEMITONES,
    detect_slides: bool = DETECT_SLIDES,
) -> list[dict]:
    """Segment a CREPE analysis into notes. Cheap — safe to call repeatedly
    with different parameters against the same analysis."""
    y, sr = analysis["y"], analysis["sr"]
    time = analysis["time"]
    frequency = analysis["frequency"]
    confidence = analysis["confidence"]
    midi_pitches = analysis["midi_pitches"]
    step_seconds = analysis["step_seconds"]

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, units="time", backtrack=True, delta=onset_delta
    )
    total_duration = len(y) / sr
    boundaries = sorted(set([0.0] + list(onsets) + [total_duration]))

    playable = np.ones_like(frequency, dtype=bool)
    if restrict_to_violin_range:
        with np.errstate(invalid="ignore"):
            playable = (
                (midi_pitches >= VIOLIN_MIN_NOTE - range_tolerance_semitones)
                & (midi_pitches <= VIOLIN_MAX_NOTE + range_tolerance_semitones)
            )
        rejected = int(np.sum((frequency > 0) & ~playable))
        print(f"[crepe_melody] discarded {rejected} frames outside violin range")

    notes = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        if seg_end - seg_start < min_note_seconds:
            continue

        voiced = (
            (time >= seg_start) & (time < seg_end)
            & (confidence >= confidence_threshold) & (frequency > 0)
            & playable
        )
        if not np.any(voiced):
            continue

        voiced_times = time[voiced]
        note_end = min(float(voiced_times[-1]) + step_seconds, seg_end)
        if note_end - seg_start < min_note_seconds:
            continue

        pitch_frames = voiced & (time >= seg_start + attack_skip_seconds)
        confident = pitch_frames & (confidence >= pitch_confidence_threshold)

        if np.any(confident):
            chosen = confident
        elif np.any(pitch_frames):
            chosen = pitch_frames
        else:
            chosen = voiced

        voted, mean = _dominant_pitch(midi_pitches[chosen], confidence[chosen])

        notes.append({
            "pitch": voted,
            "pitch_raw": mean,
            "start": seg_start,
            "end": note_end,
            "velocity": int(float(np.mean(confidence[voiced])) * 127),
            "slide_to_next": False,
        })

    raw_count = len(notes)
    if merge_same_pitch:
        notes = _merge_same_pitch(notes, merge_gap_seconds)
        print(f"[crepe_melody] {len(boundaries) - 1} onset segments -> "
              f"{raw_count} notes -> {len(notes)} after same-pitch merge")
    else:
        print(f"[crepe_melody] {len(boundaries) - 1} onset segments -> {raw_count} notes")

    if detect_slides:
        notes = _detect_slides(notes, analysis)

    return notes


def extract_melody_crepe(audio_path, step_size_ms: int = 10) -> list[dict]:
    return notes_from_analysis(analyze(audio_path, step_size_ms))