from pathlib import Path
from fractions import Fraction

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "data" / "input"
STEMS_DIR = ROOT / "data" / "stems"
MIDI_DIR = ROOT / "data" / "midi"
OUTPUT_DIR = ROOT / "data" / "output"
ML_CHECKPOINT_DIR = ROOT / "data" / "ml" / "checkpoints"
ML_CORRECTION_CHECKPOINT = ML_CHECKPOINT_DIR / "style_finetune_final.pt"

for d in (INPUT_DIR, STEMS_DIR, MIDI_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050

# Violin range (MIDI note numbers)
VIOLIN_MIN_NOTE = 55   # G3
VIOLIN_MAX_NOTE = 96   # C7 — realistic top for playable violin notation

DEMUCS_MODEL = "htdemucs_6s"  # 6-stem: vocals, drums, bass, guitar, piano, other

# Allowed note durations, in quarter-note units.
# IMPORTANT: quantize.py snaps note START times to min(QUANT_GRID). Every value
# here must therefore be a whole multiple of that minimum, or starts and
# durations sit on different lattices and music21 invents nested tuplets to
# reconcile them (this is what produced the 12/24/48 bracket mess).
# Fraction(1,3) is deliberately absent for that reason — re-adding it means
# also dropping GRID_UNIT to Fraction(1,12), which makes rests ugly.
QUANT_GRID = [
    Fraction(4, 1),
    Fraction(2, 1),
    Fraction(3, 2),
    Fraction(1, 1),
    Fraction(3, 4),
    Fraction(1, 2),
    Fraction(1, 4),
]

STEM_INSTRUMENTS = {
    "vocals": "Vocalist",
    "piano": "Piano",
    "guitar": "Guitar",
    "bass": "Bass",
    "other": "Violoncello",  # best guess label — cello/strings/etc. all land here
}

# 'other' carries the violin line (handled by CREPE separately); 'drums' is
# unpitched, so transcribing it produces a staff of noise.
CONTEXT_SKIP_STEMS = {"other", "drums"}