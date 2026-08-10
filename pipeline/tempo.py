"""Stage 4: tempo and beat detection."""
import numpy as np
import librosa
from config import SAMPLE_RATE

FALLBACK_BPM = 120.0


def detect_beats(audio_path):
    """Return (bpm, beat_times). beat_times matters more than bpm for rubato
    music — it's what lets quantization follow the performer instead of a
    metronome the performer never played to."""
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # librosa returns tempo as an ndarray; float() on a 1-element array is
    # deprecated in numpy >= 1.25, so pull the scalar out explicitly.
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        print(f"[tempo] detection failed — falling back to {FALLBACK_BPM:.0f} BPM")
        bpm = FALLBACK_BPM

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"[tempo] detected {bpm:.1f} BPM average, {len(beat_times)} beats tracked")
    return bpm, beat_times


def detect_tempo(audio_path) -> float:
    """Average BPM only. Kept for callers that don't need beat positions."""
    return detect_beats(audio_path)[0]