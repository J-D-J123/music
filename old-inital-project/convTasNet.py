import torch
import torchaudio
import sys
from tasnet import ConvTasNetStereo

def separate_violin(audio_path, output_path="violin_only.wav"):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

    print("[*] Loading pretrained violin separation model...")
    model = ConvTasNetStereo.from_pretrained("cadenzachallenge/ConvTasNet_Violin_NonCausal").cpu()
    model.eval()

    with torch.no_grad():
        est_source = model(waveform)  # Output shape: (batch, channels, samples)
    
    # est_source is a dict with keys usually 'violin' or so, but here it outputs tensor directly
    # If it outputs a tensor, just save it
    if isinstance(est_source, dict):
        violin_waveform = est_source['violin'].cpu()
    else:
        violin_waveform = est_source.cpu()

    torchaudio.save(output_path, violin_waveform, sample_rate=sr)
    print(f"[+] Violin isolated and saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convTasNet_violin.py your_song.wav")
        sys.exit(1)
    separate_violin(sys.argv[1])
