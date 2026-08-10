"""Organizes MIDI output per-track, archiving previous runs only when the
source audio has actually changed. Compares a hash of the input file against
the hash stored from the last run for that track title — if they match,
files are overwritten in place instead of piling up oldV1, oldV2, oldV3..."""
import hashlib
import shutil
from pathlib import Path
from config import MIDI_DIR

HASH_FILENAME = ".source_hash"


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_track_midi_dir(title: str, input_audio_path=None) -> Path:
    track_dir = MIDI_DIR / title
    track_dir.mkdir(parents=True, exist_ok=True)

    hash_file = track_dir / HASH_FILENAME
    current_hash = _hash_file(Path(input_audio_path)) if input_audio_path else None

    previous_hash = hash_file.read_text().strip() if hash_file.exists() else None

    existing_files = [p for p in track_dir.iterdir() if p.is_file() and p.name != HASH_FILENAME]

    if existing_files:
        if current_hash is not None and current_hash == previous_hash:
            print(f"[midi_versioning] source unchanged for '{title}' — overwriting in place, no archive")
        else:
            n = 1
            while (track_dir / f"oldV{n}").exists():
                n += 1
            archive_dir = track_dir / f"oldV{n}"
            archive_dir.mkdir(parents=True)
            for f in existing_files:
                shutil.move(str(f), str(archive_dir / f.name))
            print(f"[midi_versioning] source changed — archived {len(existing_files)} previous MIDI file(s) to {archive_dir}")

    if current_hash is not None:
        hash_file.write_text(current_hash)

    return track_dir