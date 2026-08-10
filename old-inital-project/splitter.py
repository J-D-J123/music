import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt

def run_demucs(input_audio_path, output_dir="demucs_output"):
    """Run Demucs htdemucs_ft model on input audio."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "demucs",
        "-n", "htdemucs_ft",
        "-o", output_dir,
        input_audio_path,
    ]
    subprocess.run(cmd, check=True)
    # Output path example: demucs_output/htdemucs_ft/yourfile/other.wav
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    separated_other_path = os.path.join(output_dir, "htdemucs_ft", base_name, "other.wav")
    return separated_other_path

def bandpass_filter(audio, sr, lowcut=200, highcut=3000, order=6):
    """Apply bandpass filter to isolate violin frequency range."""
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    filtered = sosfilt(sos, audio)
    return filtered

def save_audio(audio, sr, output_path):
    sf.write(output_path, audio, sr)
    print(f"Saved filtered audio to: {output_path}")

def main(input_audio_path):
    print("[*] Running Demucs separation...")
    other_stem_path = run_demucs(input_audio_path)

    print("[*] Loading 'other.wav' stem...")
    audio, sr = librosa.load(other_stem_path, sr=None, mono=True)

    print("[*] Applying violin bandpass filter (200 Hz - 3 kHz)...")
    filtered_audio = bandpass_filter(audio, sr)

    output_violin_path = "violin_only.wav"
    save_audio(filtered_audio, sr, output_violin_path)

    print("[*] Done. Use 'violin_only.wav' as input to Basic-Pitch.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python isolate_violin.py your_audio_file.wav")
        sys.exit(1)
    main(sys.argv[1])
