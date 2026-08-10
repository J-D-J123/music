"""Interactive entrypoint: pick a track from data/input and run the full pipeline."""
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from config import INPUT_DIR, STEMS_DIR, MIDI_DIR, OUTPUT_DIR, CONTEXT_SKIP_STEMS
from pipeline.separate import separate_all
from pipeline.parallel_transcribe import transcribe_many
from pipeline.crepe_melody import extract_melody_crepe
from pipeline.intonation import resolve_ambiguous_pitches
from pipeline.tempo import detect_beats
from pipeline.quantize import quantize_notes
from pipeline.violin_range import fit_to_violin_range
from pipeline.context_parts import build_context_notes
from pipeline.notate import notate_full_and_parts
from pipeline.midi_versioning import get_track_midi_dir
from pipeline.double_stops import detect_double_stops
from pipeline.transcribe import transcribe

BANNER = r"""
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ───────♪───────────────♫────────────────♪──────────
  ──────────────♫────────────────♪─────────────────♫──
  ───♪────────────────♪────────────────────♫──────────
  ──────────────────────────♫──────────────────────♪──
  ───────────♪────────────────────♪────────────────────
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           v i o l i n - t r a n s c r i b e r
              audio  ->  sheet music
"""

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MUSESCORE_CANDIDATES = ["musescore3", "musescore4", "musescore", "mscore"]


def pick_input_file():
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)

    if not files:
        print(f"  (no audio files found in {INPUT_DIR} — drop one in and try again)")
        raise SystemExit(1)

    print("  which track would you like to transcribe?\n")
    for i, f in enumerate(files, start=1):
        print(f"   {i}. {f.name}")
    print()

    while True:
        choice = input("  pick a number ✧ ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("  ✗ not a valid choice, try again")


def find_musescore():
    for cmd in MUSESCORE_CANDIDATES:
        if shutil.which(cmd):
            return cmd
    return None


def pick_and_open_output(outputs: dict):
    labels = list(outputs.keys())

    print("  which part would you like to open in MuseScore?\n")
    for i, label in enumerate(labels, start=1):
        print(f"   {i}. {label}")
    print(f"   {len(labels) + 1}. skip — don't open anything")
    print()

    while True:
        choice = input("  pick a number ✧ ").strip()
        if choice.isdigit():
            n = int(choice)
            if n == len(labels) + 1:
                return
            if 1 <= n <= len(labels):
                path = outputs[labels[n - 1]]
                mscore = find_musescore()
                if mscore:
                    print(f"\n  ✧ opening {path.name} in MuseScore ✧\n")
                    subprocess.Popen([mscore, str(path)])
                else:
                    print(f"\n  ✗ couldn't find MuseScore installed — file is at:\n    {path}\n")
                return
        print("  ✗ not a valid choice, try again")


def run_once():
    input_path = pick_input_file()
    title = input_path.stem
    print(f"\n  ✧ transcribing '{input_path.name}' ✧\n")

    track_midi_dir = get_track_midi_dir(title, input_audio_path=input_path)

    with ThreadPoolExecutor() as pool:
        beats_future = pool.submit(detect_beats, str(input_path))
        stems = separate_all(str(input_path))
        bpm, beat_times = beats_future.result()

    if "other" not in stems:
        raise RuntimeError("Demucs produced no 'other' stem — nothing to transcribe as violin")

    # Violin line: CREPE + onset detection on the isolated 'other' stem.
    violin_notes = extract_melody_crepe(stems["other"])

    # Re-round notes that landed near a semitone boundary, using the key as the
    # tiebreak. Runs before double-stop detection so intervals are measured
    # against corrected pitches.
    violin_notes = resolve_ambiguous_pitches(violin_notes)

    # Basic Pitch on the same stem, once, for the polyphonic reference CREPE
    # can't provide (CREPE is single-pitch and cannot see double stops).
    other_midi = transcribe(stems["other"], output_dir=track_midi_dir)
    violin_notes = detect_double_stops(violin_notes, other_midi)

    violin_notes = quantize_notes(violin_notes, bpm, beat_times=beat_times)
    violin_notes = fit_to_violin_range(violin_notes)

    # Context parts. 'other' is excluded because it was just transcribed above,
    # and 'drums' because pitch-transcribing an unpitched stem yields noise.
    context_stems = {k: v for k, v in stems.items() if k not in CONTEXT_SKIP_STEMS}
    midi_paths = transcribe_many(context_stems, output_dir=track_midi_dir)

    context = []
    for name in context_stems:
        if name not in midi_paths:
            print(f"  ✗ transcription failed for '{name}' — skipping that part")
            continue
        context.append(build_context_notes(name, midi_paths[name], bpm, beat_times=beat_times))

    outputs = notate_full_and_parts(violin_notes, bpm, title, context=context)

    print("\n  🎼 done! files written:\n")
    for label, path in outputs.items():
        print(f"   - {label}: {path}")
    print()

    pick_and_open_output(outputs)


def main():
    while True:
        print(BANNER)
        run_once()
        print("\n  ────────────────────────────────────────\n")


if __name__ == "__main__":
    main()