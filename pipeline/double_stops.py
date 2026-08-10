"""Detect double stops by cross-referencing the CREPE-based monophonic melody
line against Basic Pitch's polyphonic MIDI, which can see multiple simultaneous
notes that CREPE (a single-pitch tracker) architecturally cannot."""
import pretty_midi

# Fast double stops are short. Requiring the second voice to sustain through a
# fixed 60% of the note means a sixteenth note (~128ms at 117 BPM) needs a
# ~77ms detection — right at Basic Pitch's own minimum-note-length floor. Short
# notes therefore get a proportionally easier bar than long ones.
MIN_OVERLAP_FRACTION = 0.60        # applied to notes at or above LONG_NOTE_SECONDS
SHORT_OVERLAP_FRACTION = 0.35      # applied to notes at or below SHORT_NOTE_SECONDS
SHORT_NOTE_SECONDS = 0.15
LONG_NOTE_SECONDS = 0.40

MIN_INTERVAL = 3   # semitones — avoid flagging near-unison jitter as a double stop
MAX_INTERVAL = 19  # semitones — wider than this is not a realistic violin double stop


def _required_overlap(duration: float) -> float:
    """Linearly interpolate the overlap requirement between the short and long
    note thresholds, so fast passages aren't held to a bar their duration
    makes physically hard to clear."""
    if duration <= SHORT_NOTE_SECONDS:
        return SHORT_OVERLAP_FRACTION
    if duration >= LONG_NOTE_SECONDS:
        return MIN_OVERLAP_FRACTION
    span = LONG_NOTE_SECONDS - SHORT_NOTE_SECONDS
    t = (duration - SHORT_NOTE_SECONDS) / span
    return SHORT_OVERLAP_FRACTION + t * (MIN_OVERLAP_FRACTION - SHORT_OVERLAP_FRACTION)


def detect_double_stops(violin_notes: list[dict], polyphonic_midi_path) -> list[dict]:
    """violin_notes: raw CREPE notes with 'start'/'end' in seconds (call this
    BEFORE quantize_notes, while timing is still in real seconds).
    Adds a 'second_pitch' key to notes that appear to be double stops."""
    midi_data = pretty_midi.PrettyMIDI(str(polyphonic_midi_path))

    all_notes = []
    for instrument in midi_data.instruments:
        for n in instrument.notes:
            all_notes.append({"pitch": n.pitch, "start": n.start, "end": n.end})

    flagged = 0
    short_flagged = 0

    for vn in violin_notes:
        seg_start, seg_end = vn["start"], vn["end"]
        seg_dur = seg_end - seg_start
        if seg_dur <= 0:
            continue

        candidates = {}
        for n in all_notes:
            overlap = min(seg_end, n["end"]) - max(seg_start, n["start"])
            if overlap <= 0:
                continue
            interval = abs(n["pitch"] - vn["pitch"])
            if interval < MIN_INTERVAL or interval > MAX_INTERVAL:
                continue
            candidates[n["pitch"]] = candidates.get(n["pitch"], 0) + overlap

        if not candidates:
            continue

        best_pitch, best_overlap = max(candidates.items(), key=lambda kv: kv[1])
        if best_overlap / seg_dur >= _required_overlap(seg_dur):
            vn["second_pitch"] = best_pitch
            flagged += 1
            if seg_dur <= SHORT_NOTE_SECONDS:
                short_flagged += 1

    print(f"[double_stops] flagged {flagged} of {len(violin_notes)} notes "
          f"as likely double stops ({short_flagged} on short notes)")
    return violin_notes