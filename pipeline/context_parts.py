"""Transcribe accompaniment stems into note lists for later notation."""
from pipeline.melody import extract_melody
from pipeline.quantize import quantize_notes
from config import STEM_INSTRUMENTS


def build_context_notes(stem_name: str, midi_path, bpm: float, beat_times=None) -> dict:
    notes = extract_melody(midi_path)
    notes = quantize_notes(notes, bpm, beat_times=beat_times)
    inst_name = STEM_INSTRUMENTS.get(stem_name, "Piano")
    print(f"[context_parts] built '{stem_name}' notes: {len(notes)}")
    return {"name": stem_name.capitalize(), "instrument": inst_name, "notes": notes}