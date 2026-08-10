"""Parameter sweep: render several melody-extraction variants from a single
CREPE pass so they can be compared side by side.

Double-stop detection is deliberately skipped here — this tool is for tuning
the melody parameters in isolation. Run run.py for the real pipeline."""
import argparse
from pathlib import Path

from config import INPUT_DIR, STEMS_DIR, DEMUCS_MODEL
from pipeline.separate import separate_all
from pipeline.crepe_melody import analyze, notes_from_analysis
from pipeline.tempo import detect_beats
from pipeline.quantize import quantize_notes
from pipeline.violin_range import fit_to_violin_range
from pipeline.notate import notate_full_and_parts

# Rhythm and confidence settings held fixed — both are settled. What varies is
# whether frames outside the violin's physical range are allowed to vote.
BASE = dict(
    confidence_threshold=0.35,
    pitch_confidence_threshold=0.55,
    min_note_seconds=0.03,
    onset_delta=0.02,
    merge_same_pitch=True,
    merge_gap_seconds=0.06,
)

VARIANTS = {
    "norange": dict(**BASE, restrict_to_violin_range=False),
    "range": dict(**BASE, restrict_to_violin_range=True, range_tolerance_semitones=1.0),
    "range_hard": dict(**BASE, restrict_to_violin_range=True, range_tolerance_semitones=0.0),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("track", help="filename in data/input, e.g. HungarianDance.mp3")
    p.add_argument("--fixed-tempo", action="store_true",
                   help="quantize against average BPM instead of tracked beats")
    args = p.parse_args()

    input_path = INPUT_DIR / args.track
    if not input_path.exists():
        raise SystemExit(f"not found: {input_path}")

    title = input_path.stem
    stem_path = STEMS_DIR / DEMUCS_MODEL / title / "other.wav"

    if stem_path.exists():
        print(f"[sweep] reusing existing stem {stem_path}")
    else:
        print("[sweep] no stem found — running separation")
        separate_all(str(input_path))
        if not stem_path.exists():
            raise SystemExit(f"separation produced no 'other' stem at {stem_path}")

    bpm, beat_times = detect_beats(str(input_path))
    if args.fixed_tempo:
        beat_times = None

    # One CREPE pass, reused for every variant.
    analysis = analyze(stem_path)

    for name, params in VARIANTS.items():
        print(f"\n[sweep] === variant: {name} ===")
        notes = notes_from_analysis(analysis, **params)
        notes = quantize_notes(notes, bpm, beat_times=beat_times)
        notes = fit_to_violin_range(notes)
        notate_full_and_parts(notes, bpm, f"{title}_sweep_{name}")

    print("\n[sweep] done — compare the *_sweep_*_violin.musicxml files in data/output")


if __name__ == "__main__":
    main()