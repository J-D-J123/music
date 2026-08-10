"""Stage 2: audio -> polyphonic MIDI via Basic Pitch."""
from pathlib import Path

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

from config import MIDI_DIR


def transcribe(audio_path: str, output_dir=None) -> Path:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir) if output_dir else MIDI_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[transcribe] running Basic Pitch on {audio_path}")

    _, midi_data, _ = predict(
        str(audio_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.45,
        frame_threshold=0.28,
        # 40ms rather than 75ms: on the 'other' stem this output exists only to
        # supply the second voice for double-stop detection, and a fast double
        # stop's second note can be shorter than 75ms. The melody line comes
        # from CREPE, so a looser floor here can't add noise to it.
        minimum_note_length=40,
    )

    out_path = output_dir / f"{audio_path.stem}_raw.mid"
    midi_data.write(str(out_path))
    print(f"[transcribe] saved raw MIDI to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("audio_path")
    args = p.parse_args()
    transcribe(args.audio_path)