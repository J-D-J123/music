"""Stage 7: build music21 parts and export MusicXML (combined + one file per part)."""
import copy
from fractions import Fraction
from music21 import (
    stream, note, chord, duration, tempo as m21tempo,
    instrument as m21inst, metadata, key, pitch as m21pitch,
)

from config import OUTPUT_DIR

# Fallback spellings when a pitch class isn't in the key's scale.
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


def _spelling_map(k) -> dict:
    """pitch class -> preferred note name for this key.

    Notes built from raw MIDI numbers get a default spelling with no knowledge
    of the key, which is how A-sharp ends up in a piece with a B-flat in the
    key signature. Scale tones are spelled by the scale; everything else
    follows the direction of the key signature."""
    mapping = {}
    try:
        for p in k.getPitches("C4", "B4"):
            mapping[p.pitchClass] = p.name
    except Exception:
        pass

    fallback = FLAT_NAMES if getattr(k, "sharps", 0) <= 0 else SHARP_NAMES
    for pc in range(12):
        mapping.setdefault(pc, fallback[pc])
    return mapping


def _spelled(midi_num: int, mapping: dict):
    """A Pitch respelled to the key's preference, via enharmonic swap so the
    octave stays correct across the B-sharp / C-flat boundary."""
    p = m21pitch.Pitch(midi=midi_num)
    target = mapping.get(p.pitchClass)
    if target and p.name != target:
        try:
            alt = p.getEnharmonic()
            if alt.name == target:
                return alt
        except Exception:
            pass
    return p


def _notes_to_part(notes: list[dict], instrument_name: str, part_name: str,
                   bpm: float = None) -> stream.Part:
    part = stream.Part()
    inst_class = getattr(m21inst, instrument_name, None) or m21inst.Piano
    part.insert(0, inst_class())
    part.partName = part_name
    if bpm is not None:
        part.append(m21tempo.MetronomeMark(number=round(bpm)))

    # Key first, so every note can be spelled against it as it's created.
    detected_key = None
    try:
        probe = stream.Stream()
        for n in notes:
            probe.append(note.Note(midi=max(0, min(127, n["pitch"]))))
        detected_key = probe.analyze("key")
    except Exception as e:
        print(f"[notate] {part_name}: key detection skipped ({e})")

    mapping = _spelling_map(detected_key) if detected_key is not None else {}

    cursor = Fraction(0)
    for n in sorted(notes, key=lambda x: x["start_ql"]):
        gap = n["start_ql"] - cursor
        if gap > 0:
            r = note.Rest()
            r.duration = duration.Duration(quarterLength=gap)
            part.append(r)

        pitch_num = max(0, min(127, n["pitch"]))
        if n.get("second_pitch") is not None:
            pitch2 = max(0, min(127, n["second_pitch"]))
            element = chord.Chord([_spelled(pitch_num, mapping),
                                   _spelled(pitch2, mapping)])
        else:
            element = note.Note(_spelled(pitch_num, mapping))

        element.duration = duration.Duration(quarterLength=n["quarterLength"])
        part.append(element)
        cursor = n["start_ql"] + n["quarterLength"]

    if detected_key is not None:
        part.insert(0, key.Key(detected_key.tonic.name, detected_key.mode))
        print(f"[notate] {part_name}: detected key "
              f"{detected_key.tonic.name} {detected_key.mode}")

    return part.makeNotation()


def _write_solo(part_obj: stream.Part, title: str, label: str):
    solo = stream.Score()
    solo.metadata = metadata.Metadata()
    solo.metadata.title = f"{title} - {label}"
    solo.metadata.composer = "violin-transcriber"

    # deepcopy: this same Part is already inside the combined score. A music21
    # element living in two streams at once gets confused offsets and contexts.
    solo.insert(0, copy.deepcopy(part_obj))

    path = OUTPUT_DIR / f"{title.lower().replace(' ', '_')}_{label.lower()}.musicxml"
    solo.write("musicxml", fp=str(path))
    print(f"[notate] wrote part: {path}")
    return path


def notate_full_and_parts(violin_notes: list[dict], bpm: float, title: str,
                          context: list[dict] | None = None) -> dict:
    """context: list of {'name', 'instrument', 'notes'} dicts.
    Returns dict of output paths: 'full' plus one per part label.
    Parts with no notes are skipped entirely — not written, not returned."""
    context = context or []
    outputs = {}

    violin_part = _notes_to_part(violin_notes, "Violin", "Violin", bpm=bpm)

    combined = stream.Score()
    combined.metadata = metadata.Metadata()
    combined.metadata.title = title
    combined.metadata.composer = "violin-transcriber"

    # insert(0, ...), NOT append(). Score.append() places each Part at the
    # score's current highestTime, so parts end up one after another in time
    # instead of stacked as simultaneous staves.
    combined.insert(0, violin_part)

    context_parts = []
    for c in context:
        if not c["notes"]:
            print(f"[notate] skipping '{c['name']}' — no notes detected")
            continue
        p = _notes_to_part(c["notes"], c["instrument"], c["name"])
        context_parts.append((c["name"], p))
        combined.insert(0, p)

    combined_path = OUTPUT_DIR / f"{title.lower().replace(' ', '_')}_full.musicxml"
    combined.write("musicxml", fp=str(combined_path))
    print(f"[notate] wrote combined score: {combined_path}")
    outputs["full"] = combined_path

    outputs["violin"] = _write_solo(violin_part, title, "Violin")
    for name, p in context_parts:
        outputs[name.lower()] = _write_solo(p, title, name)

    return outputs