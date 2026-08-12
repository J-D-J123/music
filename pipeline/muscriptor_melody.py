"""Violin line extraction via MuScriptor (multi-instrument transcription
model) instead of CREPE + onset detection.

Replaces extract_melody_crepe() + intonation.py's resolve_ambiguous_pitches()
in the violin-line stage. Everything downstream — double-stop detection,
quantization, range-fitting, notation — is untouched, and Basic Pitch keeps
running on the 'other' stem exactly as before for the double-stop reference
track; this only changes where the primary violin note events come from.

intonation.py is skipped on this path deliberately, not by oversight:
MuScriptor emits already-decided discrete note events, not CREPE's raw
per-frame pitch estimate, so there's no 'pitch_raw' confidence-weighted mean
for intonation.py's key-based tiebreak to act on.

Conditioned on BOTH violin and piano (not violin alone): Demucs's 'other'
stem sometimes still carries piano bleed-through, since separation is never
perfect. With --instruments violin only, MuScriptor would be forced to
decode that bleed-through AS violin (the docs are explicit: "every
instrument not in the list is forbidden from being decoded at all") --
silently turning real piano content into wrong violin notes. Conditioning on
both lets MuScriptor identify each correctly; run.py then routes the piano
notes into the existing Piano context part instead of the violin line.
"""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pretty_midi

# "small"/"medium"/"large". medium tested at ~108s for a 66s clip on a
# no-CUDA Ryzen 5 3600 -- tractable, and gave a visibly cleaner result than
# CREPE on Hungarian Dance. Bump to "large" if quality matters more than
# time; drop to "small" if iteration speed matters more.
MUSCRIPTOR_MODEL = "medium"

HASH_FILENAME = ".muscriptor_source_hash"

# General MIDI program numbers (0-indexed) used to route each returned
# instrument track. Same GM_VIOLIN_PROGRAMS=violin-is-40 convention already
# used in pipeline/ml/data_prep.py, kept consistent here. Piano covers the
# whole GM "piano family" bank (0-7: acoustic/bright/electric grand,
# honky-tonk, electric pianos, harpsichord, clavi) since Demucs' piano stem
# bleed-through could plausibly land as any of these, not just program 0.
GM_VIOLIN_PROGRAMS = {40}
GM_PIANO_PROGRAMS = set(range(0, 8))


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _highest_voice(all_notes: list[dict]) -> list[dict]:
    """Collapse simultaneous notes to a single line by keeping whichever is
    sounding highest at each point in time. Same idea as melody.py's
    collapse, duplicated here rather than imported so this module has no
    dependency on melody.py's context-parts-specific usage. Applied to both
    the violin and piano streams below, matching how every other context
    part in this pipeline (vocals/bass/guitar/piano via melody.py) is
    already reduced to a monophonic line for notation."""
    if not all_notes:
        return []

    all_notes = sorted(all_notes, key=lambda n: n["start"])
    events = sorted(set(n["start"] for n in all_notes) | set(n["end"] for n in all_notes))

    line = []
    for i in range(len(events) - 1):
        t0, t1 = events[i], events[i + 1]
        if t1 - t0 <= 0:
            continue
        active = [n for n in all_notes if n["start"] <= t0 < n["end"]]
        if not active:
            continue
        top = max(active, key=lambda n: n["pitch"])

        if line and line[-1]["pitch"] == top["pitch"] and line[-1]["end"] == t0:
            line[-1]["end"] = t1
        else:
            line.append({"pitch": top["pitch"], "start": t0, "end": t1,
                         "velocity": top.get("velocity", 64)})

    return line


def _run_muscriptor(other_wav: Path, output_midi: Path, model: str) -> None:
    if shutil.which("muscriptor") is None:
        raise RuntimeError(
            "muscriptor CLI not found on PATH -- install it "
            "(pip install git+https://github.com/muscriptor/muscriptor.git) "
            "inside this venv."
        )

    cmd = [
        "muscriptor", "transcribe", str(other_wav),
        "--model", model,
        # violin + piano: see module docstring for why piano is included
        # even though this pipeline's actual goal is the violin line.
        # If this errors on the exact group name, run
        # `muscriptor list-instruments` to check the accepted spelling --
        # the CLI accepts unambiguous abbreviations, and "piano" alone may
        # be ambiguous with "electric_piano" depending on the taxonomy.
        "--instruments", "violin,acoustic_piano",
        "--detect-tempo", "false",  # tempo.py/quantize.py already handle timing
        "--format", "midi",
        "--output", str(output_midi),
    ]
    print(f"[muscriptor_melody] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_melody_muscriptor(other_wav_path, output_dir: Path,
                               model: str = MUSCRIPTOR_MODEL) -> tuple[list[dict], list[dict]]:
    """other_wav_path: the Demucs 'other' stem (same input CREPE used).
    output_dir: the per-track midi dir already threaded through run.py --
    used to cache the MuScriptor MIDI the same way separate.py/
    midi_versioning.py cache their own outputs, since re-running MuScriptor
    on every pipeline run while iterating on downstream stages would be
    needlessly slow.

    Returns (violin_notes, piano_notes) -- both raw-seconds note dicts
    ({'pitch','start','end','velocity'}), unquantized, matching the format
    extract_melody_crepe() already hands to the rest of run.py. piano_notes
    will usually be empty (most 'other'-stem content genuinely is violin);
    it's only non-empty when MuScriptor actually identified piano
    bleed-through, and run.py is responsible for merging it into the
    existing Piano context part."""
    other_wav_path = Path(other_wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_midi = output_dir / "other_muscriptor.mid"
    hash_file = output_dir / HASH_FILENAME

    current_hash = _hash_file(other_wav_path)
    previous_hash = hash_file.read_text().strip() if hash_file.exists() else None

    if output_midi.exists() and previous_hash == current_hash:
        print(f"[muscriptor_melody] reusing cached transcription: {output_midi}")
    else:
        _run_muscriptor(other_wav_path, output_midi, model=model)
        hash_file.write_text(current_hash)

    midi_data = pretty_midi.PrettyMIDI(str(output_midi))
    violin_raw, piano_raw = [], []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        if instrument.program in GM_VIOLIN_PROGRAMS:
            bucket = violin_raw
        elif instrument.program in GM_PIANO_PROGRAMS:
            bucket = piano_raw
        else:
            # Shouldn't happen given --instruments restricts decoding to
            # just these two, but the mapping from MT3_FULL_PLUS groups to
            # GM program numbers isn't something this module controls --
            # flag it loudly rather than silently dropping or misrouting.
            print(f"[muscriptor_melody] WARNING: unexpected instrument "
                  f"program {instrument.program} ({len(instrument.notes)} "
                  f"note(s)) -- not violin or piano, discarding")
            continue

        for n in instrument.notes:
            bucket.append({"pitch": n.pitch, "start": n.start, "end": n.end,
                           "velocity": n.velocity})

    if not violin_raw and not piano_raw:
        print("[muscriptor_melody] WARNING: MuScriptor returned no notes")
        return [], []

    violin_notes = _highest_voice(violin_raw)
    piano_notes = _highest_voice(piano_raw)

    msg = f"[muscriptor_melody] extracted {len(violin_notes)} violin note(s)"
    if piano_notes:
        msg += f", {len(piano_notes)} piano note(s) (bleed-through in 'other' stem)"
    print(msg)

    return violin_notes, piano_notes