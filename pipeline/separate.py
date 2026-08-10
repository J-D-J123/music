"""Stage 1: source separation via Demucs.

Stems are cached per-track under data/stems/<model>/<Title>/. A hash of the
source audio is stored alongside them, so a rerun on the same file reuses the
existing stems instead of paying for separation again. Change the audio and
the hash won't match, so it re-separates."""
import hashlib
import subprocess
import sys
from pathlib import Path

from config import STEMS_DIR, DEMUCS_MODEL

STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]
HASH_FILENAME = ".source_hash"


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_stems(stem_dir: Path) -> dict:
    return {
        name: stem_dir / f"{name}.wav"
        for name in STEM_NAMES
        if (stem_dir / f"{name}.wav").exists()
    }


def separate_all(input_path: str, force: bool = False) -> dict:
    """Return {stem_name: wav_path}, running Demucs only when needed."""
    input_path = Path(input_path)
    track_name = input_path.stem
    stem_dir = STEMS_DIR / DEMUCS_MODEL / track_name

    current_hash = _hash_file(input_path)
    hash_file = stem_dir / HASH_FILENAME

    if not force and stem_dir.exists():
        cached = _collect_stems(stem_dir)
        previous_hash = hash_file.read_text().strip() if hash_file.exists() else None
        if cached and previous_hash == current_hash:
            print(f"[separate] reusing cached stems for '{track_name}': {list(cached.keys())}")
            return cached
        if cached:
            print(f"[separate] source changed for '{track_name}' — re-separating")

    cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL,
           "-o", str(STEMS_DIR), str(input_path)]
    print(f"[separate] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    stems = _collect_stems(stem_dir)
    if not stems:
        raise FileNotFoundError(f"Demucs produced no stems in {stem_dir}")

    stem_dir.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(current_hash)

    print(f"[separate] got stems: {list(stems.keys())}")
    return stems


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--force", action="store_true", help="re-separate even if cached")
    args = p.parse_args()
    separate_all(args.input, force=args.force)