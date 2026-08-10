"""Stage 3: collapse polyphonic MIDI into a single monophonic melody line."""
from pathlib import Path
import pretty_midi


def _correct_octave_errors(melody: list[dict], window: int = 5) -> list[dict]:
    """If a note is roughly an octave (or two) away from its local neighborhood median,
    it's likely a harmonic/overtone misdetection — snap it back into the local register."""
    if len(melody) < 3:
        return melody

    pitches = [n["pitch"] for n in melody]
    for i in range(len(melody)):
        lo = max(0, i - window)
        hi = min(len(melody), i + window + 1)
        neighborhood = pitches[lo:i] + pitches[i + 1:hi]
        if not neighborhood:
            continue
        local_median = sorted(neighborhood)[len(neighborhood) // 2]
        diff = melody[i]["pitch"] - local_median
        for octaves in (2, 1):
            if abs(diff) >= 12 * octaves - 2 and abs(diff) <= 12 * octaves + 2:
                shift = -12 * octaves if diff > 0 else 12 * octaves
                melody[i]["pitch"] += shift
                pitches[i] = melody[i]["pitch"]
                break

    return melody


def extract_melody(midi_path) -> list[dict]:
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    all_notes = []
    for instrument in midi_data.instruments:
        for n in instrument.notes:
            all_notes.append({"pitch": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity})

    if not all_notes:
        print("[melody] no notes found")
        return []

    all_notes.sort(key=lambda n: n["start"])
    events = sorted(set(n["start"] for n in all_notes) | set(n["end"] for n in all_notes))

    melody = []
    last_pitch = None

    for i in range(len(events) - 1):
        t0, t1 = events[i], events[i + 1]
        if t1 - t0 <= 0:
            continue
        active = [n for n in all_notes if n["start"] <= t0 < n["end"]]
        if not active:
            continue

        def score(n):
            pitch_score = n["pitch"] * 2.0
            vel_score = (n["velocity"] / 127.0) * 4
            continuity_penalty = 0
            if last_pitch is not None:
                jump = abs(n["pitch"] - last_pitch)
                if jump > 7:
                    continuity_penalty = (jump - 7) * 1.5
            return pitch_score + vel_score - continuity_penalty

        best = max(active, key=score)
        last_pitch = best["pitch"]

        if melody and melody[-1]["pitch"] == best["pitch"] and melody[-1]["end"] == t0:
            melody[-1]["end"] = t1
        else:
            melody.append({"pitch": best["pitch"], "start": t0, "end": t1, "velocity": best["velocity"]})

    melody = _correct_octave_errors(melody)

    MIN_NOTE_SECONDS = 0.04
    melody = [n for n in melody if (n["end"] - n["start"]) >= MIN_NOTE_SECONDS]

    print(f"[melody] extracted {len(melody)} notes after octave correction and filtering")
    return melody