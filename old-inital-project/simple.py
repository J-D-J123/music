import os
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model
from basic_pitch.audio import load_audio
from music21 import converter, instrument

# --- Set your file paths ---
wav_path = "your_violin_audio.wav"
midi_output_path = "output.mid"
musicxml_output_path = "output.musicxml"

# --- Load model and audio ---
model = Model(ICASSP_2022_MODEL_PATH)
audio, sr = load_audio(wav_path)

# --- Run prediction and save MIDI ---
output_dict = predict(model, audio, sr)
output_dict["midi"].write(midi_output_path)

# --- Convert MIDI to MusicXML and set violin instrument ---
score = converter.parse(midi_output_path)
violin_part = instrument.fromString('Violin')
for part in score.parts:
    part.insert(0, violin_part)
score.write('musicxml', fp=musicxml_output_path)

print(f"🎻 MusicXML sheet music saved to {musicxml_output_path}")
