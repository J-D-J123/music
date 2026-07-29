import os
import numpy as np
import pretty_midi
from pydub import AudioSegment
import librosa
import torch
from transformers import pipeline
from collections import defaultdict
import soundfile as sf
import subprocess
from demucs.separate import main as demucs_main
from demucs.audio import AudioFile

# Alternative approaches for using Spotify Basic-Pitch from Hugging Face
try:
    # Option 1: Using transformers pipeline (if available)
    from transformers import AutoModel, AutoProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Transformers not available, falling back to basic-pitch package")

try:
    # Original basic-pitch package as fallback
    from basic_pitch.inference import predict_and_save, predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    print("basic-pitch package not available")

# ---- CONFIG ----
INPUT_FILE = "other.wav"
CLEAN_FILE = "clean_violin2.wav"
OUTPUT_DIR = "output"

# Violin range (MIDI note numbers)
VIOLIN_MIN_NOTE = 55  # G3
VIOLIN_MAX_NOTE = 103  # G7

class BasicPitchHuggingFace:
    """Wrapper for Spotify Basic-Pitch model from Hugging Face"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the Basic-Pitch model from Hugging Face"""
        try:
            # Method 1: Direct model loading (if supported)
            print("[+] Loading Basic-Pitch model from Hugging Face...")
            
            # Note: This is a conceptual implementation
            # The actual implementation depends on how Spotify has structured their HF model
            
            # Option A: If it's a standard transformers model
            if HF_AVAILABLE:
                try:
                    self.processor = AutoProcessor.from_pretrained("spotify/basic-pitch")
                    self.model = AutoModel.from_pretrained("spotify/basic-pitch")
                    self.model.to(self.device)
                    print(f"[+] Model loaded on {self.device}")
                    return True
                except Exception as e:
                    print(f"[!] Could not load as transformers model: {e}")
            
            # Option B: Using audio-specific pipeline (if available)
            try:
                self.pipeline = pipeline(
                    "automatic-speech-recognition",  # This might need to be adjusted
                    model="spotify/basic-pitch",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("[+] Loaded as pipeline")
                return True
            except Exception as e:
                print(f"[!] Could not load as pipeline: {e}")
                
            return False
            
        except Exception as e:
            print(f"[!] Error loading Hugging Face model: {e}")
            return False
    
    def predict_notes(self, audio_path):
        """Predict notes from audio using HF model"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Method 1: Direct model inference (conceptual)
            if self.model and self.processor:
                # Preprocess audio
                inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process outputs to get MIDI notes
                # Note: This part depends on the actual model output format
                notes = self._process_model_outputs(outputs, sr)
                return notes
            
            # Method 2: Using pipeline (if available)
            elif hasattr(self, 'pipeline'):
                # This is conceptual - actual implementation depends on model structure
                result = self.pipeline(audio)
                return self._process_pipeline_output(result)
            
            return None
            
        except Exception as e:
            print(f"[!] Error in HF prediction: {e}")
            return None
    
    def _process_model_outputs(self, outputs, sr):
        """Process raw model outputs to extract MIDI notes"""
        # This is a placeholder - actual implementation depends on model output format
        # Spotify Basic-Pitch typically outputs note onset/offset probabilities
        
        # Conceptual processing:
        # 1. Extract onset and offset probabilities
        # 2. Apply thresholding
        # 3. Convert to MIDI note events
        
        notes = []
        # Implementation would go here based on actual model outputs
        return notes
    
    def _process_pipeline_output(self, result):
        """Process pipeline output to extract notes"""
        # Placeholder for pipeline output processing
        notes = []
        return notes

def convert_to_mono(input_file, output_file):
    """Convert audio to mono WAV format"""
    print(f"[+] Converting {input_file} to mono WAV...")
    sound = AudioSegment.from_file(input_file)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(22050)
    sound.export(output_file, format="wav")
    print("[+] Mono conversion done.")

def separate_violin(input_file, output_dir):
    """Use Demucs to separate violin from mix"""
    print(f"[+] Separating violin from {input_file}...")
    try:
        # Create temp directory for separation
        sep_dir = os.path.join(output_dir, "separated")
        os.makedirs(sep_dir, exist_ok=True)
        
        # Run Demucs separation
        cmd = f"python -m demucs --two-stems=vocals -n htdemucs {input_file} -o {sep_dir}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Find the separated violin track
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        violin_path = os.path.join(sep_dir, "htdemucs", base_name, "vocals.wav")
        
        if os.path.exists(violin_path):
            return violin_path
        
        # Fallback to other stems
        for stem in ["other.wav", "vocals.wav"]:
            candidate = os.path.join(sep_dir, "htdemucs", base_name, stem)
            if os.path.exists(candidate):
                return candidate
                
        return input_file
    except Exception as e:
        print(f"[!] Separation failed: {e}")
        return input_file

def apply_bandpass_filter(input_file, output_file, lowcut=196, highcut=3150):
    """Apply bandpass filter to focus on violin frequencies"""
    print(f"[+] Applying bandpass filter ({lowcut}-{highcut}Hz)...")
    try:
        y, sr = librosa.load(input_file, sr=22050)
        
        # Design bandpass filter
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Butterworth filter
        sos = librosa.filters.bandpass(low, high)
        y_filtered = librosa.sosfilt(sos, y)
        
        sf.write(output_file, y_filtered, sr)
        return True
    except Exception as e:
        print(f"[!] Filtering failed: {e}")
        return False

def run_basic_pitch_hf(input_wav, output_dir):
    """Run Basic-Pitch using Hugging Face model"""
    print(f"[+] Running Basic-Pitch (HF) on {input_wav}...")
    
    # Initialize HF model
    bp_hf = BasicPitchHuggingFace()
    
    if bp_hf.load_model():
        notes = bp_hf.predict_notes(input_wav)
        
        if notes:
            # Convert to MIDI and save
            midi_path = os.path.join(output_dir, "basic_pitch_hf.mid")
            notes_to_midi(notes, midi_path)
            return midi_path
        else:
            print("[!] No notes predicted from HF model")
            return None
    else:
        print("[!] Failed to load HF model")
        return None

def run_basic_pitch_original(input_wav, output_dir):
    """Run Basic-Pitch using original package"""
    print(f"[+] Running Basic-Pitch (original) on {input_wav}...")
    
    if not BASIC_PITCH_AVAILABLE:
        print("[!] basic-pitch package not available")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Using predict_and_save
    predict_and_save(
        [input_wav],
        output_directory=output_dir,
        save_midi=True,
        save_model_outputs=False,
        save_notes=False,
        sonify_midi=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH
    )
    
    # Find the generated MIDI file
    return find_midi_file(output_dir, input_wav)

def run_basic_pitch_direct_inference(input_wav, output_dir):
    """Run Basic-Pitch using direct inference for more control"""
    print(f"[+] Running Basic-Pitch (direct inference) on {input_wav}...")
    
    if not BASIC_PITCH_AVAILABLE:
        print("[!] basic-pitch package not available")
        return None
    
    try:
        # Load audio
        audio, sr = librosa.load(input_wav, sr=22050, mono=True)
        
        # Direct prediction
        model_output, midi_data, note_events = predict(
            audio,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        
        # Save MIDI
        midi_path = os.path.join(output_dir, "basic_pitch_direct.mid")
        midi_data.write(midi_path)
        
        print(f"[+] Direct inference complete: {midi_path}")
        return midi_path
        
    except Exception as e:
        print(f"[!] Error in direct inference: {e}")
        return None

def notes_to_midi(notes, output_path):
    """Convert note events to MIDI file"""
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    for note_data in notes:
        note = pretty_midi.Note(
            velocity=note_data.get('velocity', 80),
            pitch=note_data['pitch'],
            start=note_data['start'],
            end=note_data['end']
        )
        instrument.notes.append(note)
    
    midi_data.instruments.append(instrument)
    midi_data.write(output_path)

def extract_melody_line(midi_data, time_resolution=0.05):
    """Improved melody extraction with voice leading"""
    all_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'duration': note.end - note.start,
                'velocity': note.velocity
            })
    
    if not all_notes:
        return []
    
    # Sort by start time then pitch (higher pitches first)
    all_notes.sort(key=lambda x: (x['start'], -x['pitch']))
    
    melody_notes = []
    current_time = 0
    last_pitch = None
    
    while current_time < max(n['end'] for n in all_notes):
        # Find all notes active at current time
        active_notes = [n for n in all_notes 
                       if n['start'] <= current_time < n['end']]
        
        if active_notes:
            # Prefer notes that are:
            # 1. Higher pitch (melody tends to be highest voice)
            # 2. Longer duration (more likely to be melody)
            # 3. Stronger velocity
            best_note = max(active_notes, 
                          key=lambda n: (n['pitch'], 
                                       n['duration'], 
                                       n['velocity']))
            
            # Only add if pitch changed
            if best_note['pitch'] != last_pitch:
                melody_notes.append({
                    'pitch': best_note['pitch'],
                    'start': current_time,
                    'velocity': best_note['velocity'],
                    'duration': best_note['end'] - current_time
                })
                last_pitch = best_note['pitch']
        
        current_time += time_resolution
    
    return melody_notes

def transpose_to_violin_range(notes):
    """Transpose notes to fit violin range"""
    print("[+] Transposing to violin range...")
    
    if not notes:
        return notes
    
    avg_pitch = np.mean([note['pitch'] for note in notes])
    violin_center = (VIOLIN_MIN_NOTE + VIOLIN_MAX_NOTE) / 2
    octave_shift = round((violin_center - avg_pitch) / 12) * 12
    
    for note in notes:
        new_pitch = note['pitch'] + octave_shift
        
        if new_pitch < VIOLIN_MIN_NOTE:
            new_pitch = VIOLIN_MIN_NOTE + (new_pitch - VIOLIN_MIN_NOTE) % 12
        elif new_pitch > VIOLIN_MAX_NOTE:
            new_pitch = VIOLIN_MAX_NOTE - (new_pitch - VIOLIN_MAX_NOTE) % 12
        
        note['pitch'] = int(new_pitch)
    
    print(f"[+] Applied octave shift of {octave_shift} semitones")
    return notes

def create_violin_midi(original_midi_path, violin_midi_path):
    """Create a violin melody MIDI file"""
    print(f"[+] Creating violin melody MIDI: {violin_midi_path}")
    
    try:
        midi_data = pretty_midi.PrettyMIDI(original_midi_path)
        melody_notes = extract_melody_line(midi_data)
        
        if not melody_notes:
            print("[!] No melody notes extracted")
            return False
        
        melody_notes = transpose_to_violin_range(melody_notes)
        
        violin_midi = pretty_midi.PrettyMIDI()
        violin_instrument = pretty_midi.Instrument(program=40, name='Violin')
        
        for note_data in melody_notes:
            note = pretty_midi.Note(
                velocity=min(100, max(60, note_data['velocity'])),
                pitch=note_data['pitch'],
                start=note_data['start'],
                end=note_data['start'] + note_data['duration']
            )
            violin_instrument.notes.append(note)
        
        violin_midi.instruments.append(violin_instrument)
        violin_midi.write(violin_midi_path)
        
        print(f"[+] Violin MIDI saved with {len(violin_instrument.notes)} notes")
        return True
        
    except Exception as e:
        print(f"[!] Error creating violin MIDI: {e}")
        return False

def find_midi_file(output_dir, original_filename):
    """Find the generated MIDI file"""
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    midi_filename = f"{base_name}_basic_pitch.mid"
    midi_path = os.path.join(output_dir, midi_filename)
    
    if os.path.exists(midi_path):
        return midi_path
    
    for file in os.listdir(output_dir):
        if file.endswith('.mid'):
            return os.path.join(output_dir, file)
    
    return None

def main():
    """Main function with enhanced violin isolation"""
    print("=== ENHANCED VIOLIN MELODY CONVERTER ===")
    print("Now with source separation and frequency filtering")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Convert to mono
    convert_to_mono(INPUT_FILE, CLEAN_FILE)
    
    # Step 2: Source separation
    separated_file = separate_violin(CLEAN_FILE, OUTPUT_DIR)
    
    # Step 3: Frequency filtering
    filtered_file = os.path.join(OUTPUT_DIR, "filtered_violin.wav")
    apply_bandpass_filter(separated_file, filtered_file)
    
    # Step 4: Try different Basic-Pitch methods
    midi_file = None
    
    # Option 1: Try Hugging Face version
    if HF_AVAILABLE:
        print("\n[+] Trying Hugging Face Basic-Pitch...")
        midi_file = run_basic_pitch_hf(filtered_file, OUTPUT_DIR)
        
        if midi_file:
            print("[+] Hugging Face method succeeded!")
        else:
            print("[!] Hugging Face method failed, trying alternatives...")
    
    # Option 2: Try direct inference
    if not midi_file and BASIC_PITCH_AVAILABLE:
        print("\n[+] Trying direct inference...")
        midi_file = run_basic_pitch_direct_inference(filtered_file, OUTPUT_DIR)
        
        if midi_file:
            print("[+] Direct inference succeeded!")
        else:
            print("[!] Direct inference failed, trying predict_and_save...")
    
    # Option 3: Try original method
    if not midi_file and BASIC_PITCH_AVAILABLE:
        print("\n[+] Trying original predict_and_save...")
        run_basic_pitch_original(filtered_file, OUTPUT_DIR)
        midi_file = find_midi_file(OUTPUT_DIR, filtered_file)
        
        if midi_file:
            print("[+] Original method succeeded!")
    
    # Process the MIDI file
    if midi_file:
        print(f"\n[+] Processing MIDI file: {midi_file}")
        
        violin_midi_path = os.path.join(OUTPUT_DIR, "violin_melody.mid")
        if create_violin_midi(midi_file, violin_midi_path):
            print(f"\n=== SUCCESS! ===")
            print(f"Violin MIDI saved: {os.path.abspath(violin_midi_path)}")
        else:
            print("[!] Failed to create violin MIDI")
    else:
        print("[!] All Basic-Pitch methods failed")
        print("Please check:")
        print("1. Audio file exists and is readable")
        print("2. Required packages are installed")
        print("3. Model files are accessible")

if __name__ == "__main__":
    main()