"""Stage 1: parse the three source datasets into a single unified set of
monophonic pitch-token sequences, ready for the PyTorch Dataset in dataset.py.

Sources handled:
  - MAESTRO        (MIDI, solo piano)          -> highest-voice extraction
  - PDMX           (JSON "MusicRender" objects) -> highest-voice extraction over
                                                    GM-program-40 (Violin) tracks,
                                                    falling back to all tracks
  - Zenodo Violin  (MIDI, solo violin)         -> loaded directly (already monophonic)

PDMX note: despite shipping under a "MusicXML dataset" name, PDMX's actual files
on disk (data/**/*.json) are JSONified MusicRender objects, not .musicxml/.mid —
confirmed against a real downloaded file, schema below. This is NOT handled via
music21 for that reason; it's parsed directly as JSON.

Output: a single pickle at OUTPUT_PATH containing a list of dicts:
    {"source": "maestro" | "pdmx" | "violin", "file": <path str>, "tokens": [int, ...]}

Tokens are pitch-only (no duration/velocity yet) and use the vocabulary defined
below. Keeping duration out for v1 keeps the model + tokenizer simple; it can be
added as a second stream later without touching this file's extraction logic.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pretty_midi

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
PAD, BOS, EOS, REST = 0, 1, 2, 3
NUM_SPECIAL_TOKENS = 4
PITCH_OFFSET = NUM_SPECIAL_TOKENS  # MIDI pitch p -> token p + PITCH_OFFSET
VOCAB_SIZE = 128 + NUM_SPECIAL_TOKENS  # 132

MIDI_MIN, MIDI_MAX = 0, 127

# Loose sanity range for "physically plausible melodic note" — generous on
# purpose. Per-source tightening (e.g. violin range) happens in the
# source-specific extractors below, not here.
PLAUSIBLE_MIN, PLAUSIBLE_MAX = 21, 108  # A0 - C8, standard piano range

# Violin-specific range, used to filter the Zenodo set and to bias the PDMX
# violin-relevance heuristic. Kept local to this file rather than importing
# from the audio-transcriber's config.py, since these are separate projects
# that happen to share a domain.
VIOLIN_MIN_NOTE, VIOLIN_MAX_NOTE = 55, 96  # G3 - C7

MIN_SEQUENCE_LENGTH = 8   # pieces shorter than this aren't useful training signal
MAX_GAP_AS_REST = 1       # gaps of exactly one "slot" become a single REST token;
                          # longer or shorter gaps are just not represented (v1)

AUDIO_MIDI_EXTS = {".mid", ".midi"}

# General MIDI program numbers (0-indexed, per the GM spec) that count as
# "violin" for PDMX track filtering. Kept narrow — just Violin — rather than
# the wider string family (viola/cello/ensemble patches), since those aren't
# a good proxy for a solo violin melodic line.
GM_VIOLIN_PROGRAMS = {40}


@dataclass
class ExtractedPiece:
    source: str
    file: str
    tokens: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pitch_to_token(pitch: int) -> int:
    pitch = max(MIDI_MIN, min(MIDI_MAX, pitch))
    return pitch + PITCH_OFFSET


def _tokens_from_note_events(notes: list[dict]) -> list[int]:
    """notes: list of {'pitch', 'start', 'end'} sorted by start, already
    monophonic (no overlaps). Returns BOS/EOS-wrapped token list with a REST
    token inserted for small gaps between notes."""
    if not notes:
        return []

    tokens = [BOS]
    prev_end = notes[0]["start"]
    for n in notes:
        gap = n["start"] - prev_end
        if gap > 0:
            tokens.append(REST)
        tokens.append(_pitch_to_token(n["pitch"]))
        prev_end = n["end"]
    tokens.append(EOS)
    return tokens


def _highest_voice(all_notes: list[dict]) -> list[dict]:
    """Collapse polyphony to a single line by keeping, at every point in time,
    whichever note is currently sounding highest."""
    if not all_notes:
        return []

    all_notes = sorted(all_notes, key=lambda n: n["start"])
    events = sorted(set(n["start"] for n in all_notes) | set(n["end"] for n in all_notes))

    line = []
    active = []
    note_idx = 0
    n_notes = len(all_notes)

    for i in range(len(events) - 1):
        t0, t1 = events[i], events[i + 1]
        if t1 - t0 <= 0:
            continue
            
        # 1. Add newly active notes
        while note_idx < n_notes and all_notes[note_idx]["start"] <= t0:
            active.append(all_notes[note_idx])
            note_idx += 1
            
        # 2. Prune notes that have already ended
        active = [n for n in active if n["end"] > t0]
        
        if not active:
            continue
            
        # 3. Find the highest pitch among the much smaller active pool
        top = max(active, key=lambda n: n["pitch"])

        if line and line[-1]["pitch"] == top["pitch"] and line[-1]["end"] == t0:
            line[-1]["end"] = t1
        else:
            line.append({"pitch": top["pitch"], "start": t0, "end": t1})

    return line


def _filter_plausible(notes: list[dict], lo: int, hi: int) -> list[dict]:
    return [n for n in notes if lo <= n["pitch"] <= hi]


# ---------------------------------------------------------------------------
# MAESTRO (MIDI, solo piano -> highest-voice melody)
# ---------------------------------------------------------------------------

def extract_maestro_file(path: Path) -> list[int]:
    midi_data = pretty_midi.PrettyMIDI(str(path))
    all_notes = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            all_notes.append({"pitch": n.pitch, "start": n.start, "end": n.end})

    melody = _highest_voice(all_notes)
    melody = _filter_plausible(melody, PLAUSIBLE_MIN, PLAUSIBLE_MAX)
    return _tokens_from_note_events(melody)


def extract_maestro_dir(root: Path) -> list[ExtractedPiece]:
    pieces = []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_MIDI_EXTS)
    for f in files:
        try:
            tokens = extract_maestro_file(f)
        except Exception as e:
            print(f"[data_prep] MAESTRO: skipping {f.name} ({e})")
            continue
        if len(tokens) >= MIN_SEQUENCE_LENGTH:
            pieces.append(ExtractedPiece(source="maestro", file=str(f), tokens=tokens))
    print(f"[data_prep] MAESTRO: extracted {len(pieces)} of {len(files)} files")
    return pieces


# ---------------------------------------------------------------------------
# PDMX (JSON MusicRender objects -> highest-voice melody, GM-program filter)
#
# Confirmed schema (real downloaded file):
#   {"resolution": 480,                 # ticks per quarter note
#    "tracks": [
#        {"name": <str or null>, "program": <int, GM program #>, "is_drum": bool,
#         "notes": [{"time": <int ticks>, "duration": <int ticks>,
#                    "pitch": <int MIDI>, "velocity": <int>, ...}, ...]},
#        ...
#    ], ...}
# "name" is frequently null (as in the sample file), so instrument identity
# is read primarily from "program" (General MIDI number; 40 = Violin) and
# only from "name" as a secondary check when it's actually populated.
# ---------------------------------------------------------------------------

def _is_violin_track(track: dict) -> bool:
    if track.get("program") in GM_VIOLIN_PROGRAMS:
        return True
    name = (track.get("name") or "").lower()
    return "violin" in name


def extract_pdmx_json_file(path: Path) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    resolution = data.get("resolution")
    if not resolution:
        raise ValueError("missing/zero 'resolution' — can't convert ticks to quarterLength")

    tracks = data.get("tracks", [])
    all_notes = []
    relevant_notes = []

    for track in tracks:
        if track.get("is_drum"):
            continue
        relevant = _is_violin_track(track)
        for n in track.get("notes", []):
            start = n["time"] / resolution
            end = start + n["duration"] / resolution
            if end <= start:
                continue
            entry = {"pitch": n["pitch"], "start": start, "end": end}
            all_notes.append(entry)
            if relevant:
                relevant_notes.append(entry)

    # Prefer violin-program tracks when present — much better signal than
    # blending in the accompaniment. PDMX coverage of "is this actually a
    # violin piece" is inconsistent, so fall back to every non-drum track
    # (highest-voice collapse still applies) when nothing matched.
    source_notes = relevant_notes if relevant_notes else all_notes

    melody = _highest_voice(source_notes)
    melody = _filter_plausible(melody, PLAUSIBLE_MIN, PLAUSIBLE_MAX)
    return _tokens_from_note_events(melody)


def _load_pdmx_subset_paths(root: Path, subset_file: Path) -> list[Path]:
    """subset_file is one of PDMX's subset_paths/{all,deduplicated,rated,
    rated_deduplicated}.txt files — one relative path per line, e.g.
    './data/0/abc.json'. Returns absolute paths under `root`."""
    paths = []
    with open(subset_file, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            rel = rel[2:] if rel.startswith("./") else rel
            paths.append(root / rel)
    print(f"[data_prep] PDMX: {len(paths)} paths listed in {subset_file.name}")
    return paths


def extract_pdmx_dir(root: Path, subset_file: Path | None = None) -> list[ExtractedPiece]:
    """root: the PDMX directory containing data/, metadata/, subset_paths/,
    and PDMX.csv (i.e. what `tar -xzf PDMX.tar.gz` unpacks).
    subset_file: optional path to one of the subset_paths/*.txt files, to
    process only that curated subset instead of all ~254K files (e.g. the
    rated_deduplicated subset is ~13K much-higher-quality scores)."""
    if subset_file is not None:
        files = [p for p in _load_pdmx_subset_paths(root, subset_file) if p.exists()]
    else:
        files = sorted((root / "data").rglob("*.json"))

    pieces = []
    for f in files:
        try:
            tokens = extract_pdmx_json_file(f)
        except Exception as e:
            print(f"[data_prep] PDMX: skipping {f.name} ({e})")
            continue
        if len(tokens) >= MIN_SEQUENCE_LENGTH:
            pieces.append(ExtractedPiece(source="pdmx", file=str(f), tokens=tokens))
    print(f"[data_prep] PDMX: extracted {len(pieces)} of {len(files)} files")
    return pieces


# ---------------------------------------------------------------------------
# Zenodo Violin MIDI (already monophonic solo violin -> load directly)
# ---------------------------------------------------------------------------

def extract_violin_file(path: Path) -> list[int]:
    midi_data = pretty_midi.PrettyMIDI(str(path))
    all_notes = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            all_notes.append({"pitch": n.pitch, "start": n.start, "end": n.end})

    # Trusted monophonic, but real files occasionally have a stray overlap
    # (grace notes, encoding artifacts) — collapse defensively rather than
    # assuming it's clean.
    notes = _highest_voice(all_notes)
    notes = _filter_plausible(notes, VIOLIN_MIN_NOTE, VIOLIN_MAX_NOTE)
    return _tokens_from_note_events(notes)


def extract_violin_dir(root: Path) -> list[ExtractedPiece]:
    pieces = []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_MIDI_EXTS)
    for f in files:
        try:
            tokens = extract_violin_file(f)
        except Exception as e:
            print(f"[data_prep] Zenodo Violin: skipping {f.name} ({e})")
            continue
        if len(tokens) >= MIN_SEQUENCE_LENGTH:
            pieces.append(ExtractedPiece(source="violin", file=str(f), tokens=tokens))
    print(f"[data_prep] Zenodo Violin: extracted {len(pieces)} of {len(files)} files")
    return pieces


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_dataset(maestro_dir: Path | None, pdmx_dir: Path | None,
                   violin_dir: Path | None,
                   pdmx_subset: Path | None = None) -> list[ExtractedPiece]:
    pieces: list[ExtractedPiece] = []
    if maestro_dir is not None:
        pieces += extract_maestro_dir(maestro_dir)
    if pdmx_dir is not None:
        pieces += extract_pdmx_dir(pdmx_dir, subset_file=pdmx_subset)
    if violin_dir is not None:
        pieces += extract_violin_dir(violin_dir)

    print(f"[data_prep] total pieces: {len(pieces)}")
    return pieces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-dir", type=Path, default=None)
    parser.add_argument("--pdmx-dir", type=Path, default=None,
                         help="the PDMX dir containing data/, metadata/, subset_paths/, PDMX.csv")
    parser.add_argument("--pdmx-subset", type=Path, default=None,
                         help="optional path to a subset_paths/*.txt file "
                              "(e.g. subset_paths/rated_deduplicated.txt) to process only "
                              "that curated subset instead of all ~254K PDMX files")
    parser.add_argument("--violin-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/processed_sequences.pkl"))
    args = parser.parse_args()

    if not any([args.maestro_dir, args.pdmx_dir, args.violin_dir]):
        parser.error("pass at least one of --maestro-dir / --pdmx-dir / --violin-dir")

    pieces = build_dataset(args.maestro_dir, args.pdmx_dir, args.violin_dir,
                            pdmx_subset=args.pdmx_subset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump([p.__dict__ for p in pieces], f)
    print(f"[data_prep] wrote {len(pieces)} sequences to {args.output}")


if __name__ == "__main__":
    main()