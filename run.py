"""Interactive entrypoint: pick a track from data/input and run the full pipeline."""

import itertools
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import (
    INPUT_DIR,
    STEMS_DIR,
    MIDI_DIR,
    OUTPUT_DIR,
    CONTEXT_SKIP_STEMS,
    ML_CORRECTION_CHECKPOINT,
)
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
from pipeline.ml.correct_notes import correct_pitches


BANNER = r"""

───────♪───────────────♫────────────────♪──────────
──────────────♫────────────────♪─────────────────♫──
───♪────────────────♪────────────────────♫──────────
──────────────────────────♫──────────────────────♪──
───────────♪────────────────────♪────────────────────

       v i o l i n - t r a n s c r i b e r
          audio  ->  sheet music

"""


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MUSESCORE_CANDIDATES = ["musescore3", "musescore4", "musescore", "mscore"]


# ---------------------------------------------------------------------------
# Music-note status animation
# ---------------------------------------------------------------------------

def music_status(stop_event):
    """Display a looping music-note status while Demucs is running."""
    notes = itertools.cycle(["♪", "♫", "♬", "♫"])

    while not stop_event.is_set():
        note = next(notes)

        sys.stdout.write(
            f"\r  {note}  separating stems  {note}  "
        )
        sys.stdout.flush()

        time.sleep(0.25)

    # Clear the old status line and show completion.
    sys.stdout.write(
        "\r  ♪  stems separated  ♫                  \n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------

def pick_input_file():
    files = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in AUDIO_EXTS
    )

    if not files:
        print(
            f"  (no audio files found in {INPUT_DIR} — "
            "drop one in and try again)"
        )
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


# ---------------------------------------------------------------------------
# MuseScore
# ---------------------------------------------------------------------------

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
                    print(
                        f"\n  ✧ opening {path.name} in MuseScore ✧\n"
                    )
                    subprocess.Popen([mscore, str(path)])
                else:
                    print(
                        "\n  ✗ couldn't find MuseScore installed — "
                        f"file is at:\n    {path}\n"
                    )

                return

        print("  ✗ not a valid choice, try again")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_once():
    input_path = pick_input_file()
    title = input_path.stem

    print(
        f"\n  ✧ transcribing '{input_path.name}' ✧\n"
    )

    track_midi_dir = get_track_midi_dir(
        title,
        input_audio_path=input_path,
    )

    # -----------------------------------------------------------------------
    # Beat detection + Demucs separation
    # -----------------------------------------------------------------------

    with ThreadPoolExecutor() as pool:
        beats_future = pool.submit(
            detect_beats,
            str(input_path),
        )

        # Start our music-note status animation.
        stop_status = threading.Event()

        status_thread = threading.Thread(
            target=music_status,
            args=(stop_status,),
            daemon=True,
        )

        status_thread.start()

        try:
            # Demucs runs while the music-note animation is displayed.
            stems = separate_all(str(input_path))

        finally:
            # Always stop the animation, even if Demucs throws an error.
            stop_status.set()
            status_thread.join()

        bpm, beat_times = beats_future.result()

    # -----------------------------------------------------------------------
    # Verify Demucs output
    # -----------------------------------------------------------------------

    if "other" not in stems:
        raise RuntimeError(
            "Demucs produced no 'other' stem — "
            "nothing to transcribe as violin"
        )

    # -----------------------------------------------------------------------
    # Violin melody extraction
    # -----------------------------------------------------------------------

    # Violin line: CREPE + onset detection on the isolated 'other' stem.
    violin_notes = extract_melody_crepe(
        stems["other"]
    )

    # Re-round notes that landed near a semitone boundary, using the key as
    # the tiebreak. Runs before double-stop detection so intervals are
    # measured against corrected pitches.
    violin_notes = resolve_ambiguous_pitches(
        violin_notes
    )

    # -----------------------------------------------------------------------
    # ML note correction
    # -----------------------------------------------------------------------

    # Model-based correction: flags/fixes notes the trained next-note model is
    # both surprised by and confident about an alternative for.
    #
    # Same spot as intonation.py (before double-stop detection, so intervals
    # get measured against corrected pitches).
    #
    # Skips cleanly if no checkpoint has been trained yet, so this pipeline
    # still runs without pipeline/ml/.

    USE_ML_CORRECTION = False

    # Disabled: 29/167 corrections on Hungarian Dance made output
    # noticeably worse.
    if USE_ML_CORRECTION and ML_CORRECTION_CHECKPOINT.exists():
        try:
            violin_notes = correct_pitches(
                violin_notes,
                ML_CORRECTION_CHECKPOINT,
            )

        except Exception as e:
            print(
                f"  ✗ ML note correction failed, "
                f"continuing without it ({e})"
            )

    else:
        print(
            f"  (no trained note-prediction checkpoint at "
            f"{ML_CORRECTION_CHECKPOINT} -- skipping ML correction)"
        )

    # -----------------------------------------------------------------------
    # Double stops
    # -----------------------------------------------------------------------

    # Basic Pitch on the same stem, once, for the polyphonic reference
    # CREPE can't provide (CREPE is single-pitch and cannot see double stops).
    other_midi = transcribe(
        stems["other"],
        output_dir=track_midi_dir,
    )

    violin_notes = detect_double_stops(
        violin_notes,
        other_midi,
    )

    # -----------------------------------------------------------------------
    # Quantization + violin range
    # -----------------------------------------------------------------------

    violin_notes = quantize_notes(
        violin_notes,
        bpm,
        beat_times=beat_times,
    )

    violin_notes = fit_to_violin_range(
        violin_notes
    )

    # -----------------------------------------------------------------------
    # Context parts
    # -----------------------------------------------------------------------

    # 'other' is excluded because it was just transcribed above,
    # and 'drums' because pitch-transcribing an unpitched stem yields noise.
    context_stems = {
        k: v
        for k, v in stems.items()
        if k not in CONTEXT_SKIP_STEMS
    }

    midi_paths = transcribe_many(
        context_stems,
        output_dir=track_midi_dir,
    )

    context = []

    for name in context_stems:
        if name not in midi_paths:
            print(
                f"  ✗ transcription failed for '{name}' "
                "— skipping that part"
            )
            continue

        context.append(
            build_context_notes(
                name,
                midi_paths[name],
                bpm,
                beat_times=beat_times,
            )
        )

    # -----------------------------------------------------------------------
    # Notation
    # -----------------------------------------------------------------------

    outputs = notate_full_and_parts(
        violin_notes,
        bpm,
        title,
        context=context,
    )

    # -----------------------------------------------------------------------
    # Finished
    # -----------------------------------------------------------------------

    print("\n  🎼 done! files written:\n")

    for label, path in outputs.items():
        print(f"   - {label}: {path}")

    print()

    pick_and_open_output(outputs)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    while True:
        print(BANNER)
        run_once()

        print(
            "\n  ────────────────────────────────────────\n"
        )


if __name__ == "__main__":
    main()