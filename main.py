import os
import time
import subprocess
import sys
import numpy as np
import signal  # Added for timeout handling

# Configuration - Optimized for speed
SAMPLE_RATE = 22050  # Reduced from 44100 for faster processing
FRAME_SIZE = 2048    # Reduced from 4096 for faster processing
HOP_LENGTH = 1024    # Increased from 512 for faster processing

class TimeoutException(Exception):
    """Custom exception for timeout handling"""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutException("Processing took too long")

def install_requirements():
    """Install compatible package versions"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        print("✅ Updated pip")
    except subprocess.CalledProcessError:
        print("⚠️ Could not update pip, continuing...")
    
    requirements = [
        'numpy>=1.24.0',
        'librosa>=0.10.0',
        'pydub>=0.25.1',
        'music21>=8.1.0',
        'soundfile>=0.12.1',
        'matplotlib>=3.7.0',
        'scipy>=1.10.0',  # Added for faster smoothing
    ]
    
    for package in requirements:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠️ FFmpeg not found!")
        print("FFmpeg is required for audio file conversion.")
        print("\n📥 To install FFmpeg:")
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Or use package manager: brew install ffmpeg / choco install ffmpeg")
        print("3. Add to PATH and restart terminal")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    if not check_ffmpeg():
        missing_deps.append("ffmpeg")
    
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
    
    try:
        import pydub
    except ImportError:
        missing_deps.append("pydub")
    
    try:
        import music21
    except ImportError:
        missing_deps.append("music21")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    return True

def get_instrument_choice():
    """Display instrument options and get user choice with specific ranges"""
    instruments = {
        '1': ('Piano', 'Piano', {'fmin': 27.5, 'fmax': 4186, 'default_octave': 4}),
        '2': ('Guitar', 'Guitar', {'fmin': 82, 'fmax': 880, 'default_octave': 3}),
        '3': ('Violin', 'Violin', {'fmin': 196, 'fmax': 2637, 'default_octave': 4}),
        '4': ('Flute', 'Flute', {'fmin': 261, 'fmax': 2093, 'default_octave': 4}),
        '5': ('Trumpet', 'Trumpet', {'fmin': 164, 'fmax': 988, 'default_octave': 3}),
        '6': ('Saxophone', 'Saxophone', {'fmin': 138, 'fmax': 830, 'default_octave': 3}),
        '7': ('Clarinet', 'Clarinet', {'fmin': 146, 'fmax': 932, 'default_octave': 3}),
        '8': ('Cello', 'Cello', {'fmin': 65, 'fmax': 987, 'default_octave': 2}),
        '9': ('Voice', 'Voice', {'fmin': 130, 'fmax': 1046, 'default_octave': 3}),
        '10': ('Bass', 'Bass', {'fmin': 41, 'fmax': 294, 'default_octave': 1}),
        '11': ('Drums', 'Percussion', {'fmin': 50, 'fmax': 5000, 'default_octave': 3}),
        '12': ('Other', 'Piano', {'fmin': 27.5, 'fmax': 4186, 'default_octave': 4})
    }
    
    print("\n🎼 Select target instrument for sheet music:")
    for key, (display_name, _, _) in instruments.items():
        print(f"{key}. {display_name}")
    
    while True:
        try:
            choice = input("\nChoose instrument (1-12): ").strip()
            if choice in instruments:
                return instruments[choice]
            print("❌ Invalid choice. Please enter a number between 1-12.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nReturning to main menu...")
            return None, None, None

def plot_pitch_track(y, sr, pitches, magnitudes, instrument_range):
    """Visualize the pitch detection for debugging"""
    try:
        import librosa.display
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot spectrogram
        plt.subplot(2, 1, 1)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=sr, y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Plot pitch track
        plt.subplot(2, 1, 2)
        times = librosa.times_like(pitches)
        plt.plot(times, pitches.T, 'o', markersize=2)
        plt.ylim([instrument_range['fmin']*0.9, instrument_range['fmax']*1.1])
        plt.title('Pitch Track')
        plt.tight_layout()
        
        # Save plot instead of showing if running in non-interactive mode
        if 'DISPLAY' not in os.environ:
            plt.savefig('pitch_track.png')
            print("📊 Pitch visualization saved to pitch_track.png")
        else:
            plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate pitch visualization: {str(e)}")

def audio_to_midi(input_path, output_dir, instrument_name, instrument_range):
    """Convert audio to MIDI with instrument-specific improvements - SPEED OPTIMIZED"""
    try:
        import librosa
        from music21 import stream, note, duration, tempo, key, meter
        
        print(f"🎵 Processing audio for {instrument_name} (range: {instrument_range['fmin']}-{instrument_range['fmax']}Hz)...")
        print("⚡ Speed-optimized mode enabled!")
        
        # Load audio with resampling for better pitch detection
        print("⚙️ Loading audio file...")
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
        
        # Trim to first 60 seconds for faster processing
        max_duration = 60  # seconds
        if len(y) > max_duration * sr:
            y = y[:max_duration * sr]
            print(f"🎧 Loaded {len(y)/sr:.2f} seconds of audio (trimmed to {max_duration}s for speed)")
        else:
            print(f"🎧 Loaded {len(y)/sr:.2f} seconds of audio")
        
        # Harmonic-percussive source separation (simplified for speed)
        print("🔍 Separating harmonic components...")
        y_harmonic = librosa.effects.harmonic(y, margin=4)  # Reduced margin for speed
        
        # Improved pitch tracking (optimized for speed)
        print("🎶 Detecting pitches (optimized for speed)...")
        pitches, magnitudes = librosa.piptrack(
            y=y_harmonic,
            sr=sr,
            fmin=instrument_range['fmin'],
            fmax=instrument_range['fmax'],
            hop_length=HOP_LENGTH,
            n_fft=FRAME_SIZE,
            threshold=0.1  # Added threshold to ignore weak pitches
        )
        print(f"✅ Detected {pitches.shape[1]} pitch frames (fast mode)")
        
        # Optional visualization (skip by default for speed)
        try:
            visualize = input("Generate pitch visualization? (y/N - default N for speed): ").lower()
            if visualize == 'y':
                plot_pitch_track(y_harmonic, sr, pitches, magnitudes, instrument_range)
        except (KeyboardInterrupt, EOFError):
            print("Skipping visualization for speed...")
        
        # Get the most prominent pitch for each frame (optimized)
        print("🎚️ Extracting prominent pitches (fast mode)...")
        pitch_track = []
        mag_track = []
        magnitude_threshold = np.percentile(magnitudes, 75)  # Use 75th percentile instead of median for speed
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            mag = magnitudes[index, t]
            
            if mag > magnitude_threshold:
                pitch_track.append(pitch)
                mag_track.append(mag)
            else:
                pitch_track.append(0)
                mag_track.append(0)
        
        # Smooth the pitch track (simplified)
        print("🔄 Smoothing pitch track (fast mode)...")
        # Use simpler smoothing for speed
        try:
            from scipy.ndimage import median_filter
            pitch_track = median_filter(pitch_track, size=3)
        except ImportError:
            # Fallback if scipy not available
            pitch_track = librosa.util.smooth(pitch_track, 3)  # Reduced from 5
        
        # Convert to MIDI notes (optimized)
        print("🎹 Converting to MIDI notes...")
        midi_notes = []
        previous_note = None
        note_threshold = 1.0  # Slightly higher threshold for cleaner results
        
        for pitch in pitch_track:
            if pitch > 0:
                midi_note = librosa.hz_to_midi(pitch)
                if previous_note is None or abs(midi_note - previous_note) > note_threshold:
                    rounded_note = int(round(midi_note))
                    min_midi = librosa.hz_to_midi(instrument_range['fmin'])
                    max_midi = librosa.hz_to_midi(instrument_range['fmax'])
                    if min_midi <= rounded_note <= max_midi:
                        midi_notes.append(rounded_note)
                        previous_note = rounded_note
                    else:
                        midi_notes.append(0)
                else:
                    midi_notes.append(previous_note)
            else:
                midi_notes.append(0)
                previous_note = None
        
        print(f"🎼 Generated {len([n for n in midi_notes if n > 0])} MIDI notes")
        
        # Create music21 stream (simplified)
        print("📝 Creating score (fast mode)...")
        score = stream.Stream()
        score.append(tempo.MetronomeMark(number=120))  # Slightly faster tempo
        score.append(key.Key('C', 'major'))
        score.append(meter.TimeSignature('4/4'))
        
        # Note segmentation (optimized)
        print("⏱️ Segmenting notes...")
        current_note = None
        note_start = 0
        min_note_duration = 0.2  # Slightly longer minimum for cleaner results
        time_pos = 0  # Initialize time_pos
        
        for i, (midi_note, magnitude) in enumerate(zip(midi_notes, mag_track)):
            time_pos = i * HOP_LENGTH / SAMPLE_RATE
            
            if midi_note != current_note:
                if current_note is not None and (time_pos - note_start) >= min_note_duration:
                    n = note.Note(midi=current_note)
                    # Simplified duration calculation
                    dur = duration.Duration(quarterLength=min(4.0, max(0.25, (time_pos - note_start) * 0.6)))
                    n.duration = dur
                    score.append(n)
                
                current_note = midi_note if midi_note > 0 else None
                note_start = time_pos
        
        # Handle the last note
        if current_note is not None:
            n = note.Note(midi=current_note)
            dur = duration.Duration(quarterLength=min(4.0, max(0.25, (time_pos - note_start) * 0.6)))
            n.duration = dur
            score.append(n)
        
        # Post-process MIDI (fast mode)
        print("✨ Post-processing MIDI (fast mode)...")
        post_process_midi_fast(score)
        
        # Save as MIDI
        midi_path = os.path.join(output_dir, f"{instrument_name.lower()}_melody.mid")
        score.write('midi', fp=midi_path)
        print(f"💾 Saved MIDI to {midi_path}")
        
        return midi_path
        
    except Exception as e:
        print(f"❌ Error converting audio to MIDI: {str(e)}")
        return None

def post_process_midi_fast(midi_stream):
    """Clean up the MIDI output for better results - FAST VERSION"""
    # Simplified post-processing for speed
    notes_to_remove = []
    
    # Remove very short notes (faster approach)
    for element in midi_stream.flatten().notes:
        if hasattr(element, 'duration') and element.duration.quarterLength < 0.15:
            notes_to_remove.append(element)
    
    for note in notes_to_remove:
        midi_stream.remove(note, recurse=True)
    
    # Simplified quantization
    midi_stream.quantize(quarterLengthDivisors=[4], inPlace=True)  # Only quarter notes for speed
    
    # Quick duration adjustment
    for n in midi_stream.flatten().notes:
        if n.duration.quarterLength > 4:
            n.duration.quarterLength = 4
        elif n.duration.quarterLength < 0.25:
            n.duration.quarterLength = 0.25

def midi_to_musicxml(midi_path, instrument_name, output_path):
    """Convert MIDI to MusicXML with specified instrument - OPTIMIZED"""
    try:
        from music21 import converter, instrument, stream
        from music21.metadata import Metadata
        
        print(f"🎼 Converting to sheet music for {instrument_name} (fast mode)...")
        
        # Load the MIDI file
        midi_stream = converter.parse(midi_path)
        
        # Create a new score
        score = stream.Score()
        
        # Add metadata (simplified)
        score.metadata = Metadata()
        score.metadata.title = f"Melody - {instrument_name}"
        score.metadata.composer = "Audio Converter"
        
        # Create instrument part
        part = stream.Part()
        
        # Set instrument - use a more robust approach
        try:
            inst_class = getattr(instrument, instrument_name, None)
            if inst_class is None:
                inst_class = instrument.Piano  # Default fallback
            part.insert(0, inst_class())
        except Exception:
            part.insert(0, instrument.Piano())  # Safe fallback
        
        # Extract notes from the MIDI stream (optimized)
        for element in midi_stream.flatten().notesAndRests:
            part.append(element)
        
        # Add the part to the score
        score.append(part)
        
        # Write to MusicXML
        print(f"💾 Saving as {output_path}...")
        score.write('musicxml', fp=output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to MusicXML: {str(e)}")
        return False

def extract_melody_to_sheet_music(input_path, output_dir="output"):
    """Main processing function with timeout protection - SPEED OPTIMIZED"""
    print("⚡ Processing melody extraction for sheet music (SPEED MODE)...")
    start_time = time.time()
    
    try:
        # Setup timeout (3 minutes for fast mode)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)  # Reduced from 300 to 180 seconds
        
        # Get instrument choice
        result = get_instrument_choice()
        if result[0] is None:  # Check if user cancelled
            return None
        instrument_display, instrument_name, instrument_range = result
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert audio to MIDI
        midi_file = audio_to_midi(input_path, output_dir, instrument_name, instrument_range)
        if not midi_file:
            raise Exception("Failed to convert audio to MIDI")
        
        # Convert MIDI to MusicXML
        output_mxl = os.path.join(output_dir, f"{instrument_name.lower()}_melody.mxl")
        success = midi_to_musicxml(midi_file, instrument_name, output_mxl)
        
        if success:
            elapsed_time = time.time() - start_time
            print(f"✅ Sheet music created in {elapsed_time:.1f} seconds (FAST MODE)")
            print(f"🎼 Instrument: {instrument_display}")
            print(f"🎵 Output files:")
            print(f"- MIDI: {midi_file}")
            print(f"- MusicXML: {output_mxl}")
            return output_mxl
        else:
            return None
            
    except TimeoutException:
        print("❌ Processing timed out after 3 minutes")
        print("Try with a shorter audio clip (30-60 seconds)")
        return None
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return None
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Disable the alarm

def validate_audio_file(file_path):
    """Validate the input audio file"""
    if not os.path.exists(file_path):
        return False, "File not found"
    
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in audio_extensions:
        return False, f"Unsupported file type: {file_ext}"
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    if file_size > 50:
        print(f"⚠️ Warning: Large file ({file_size:.1f}MB). Processing may take longer.")
    
    return True, "Valid audio file"

def main_menu():
    """Display main menu and get user choice"""
    print("\n🎵 AUDIO TO SHEET MUSIC CONVERTER (SPEED OPTIMIZED)")
    print("1. Extract melody to sheet music (optimized for speed)")
    print("2. Exit")
    
    while True:
        try:
            choice = input("Choose (1-2): ").strip()
            if choice in ("1", "2"):
                return choice
            print("❌ Invalid choice. Please enter 1 or 2.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            sys.exit(0)

def main():
    """Main application loop"""
    print("🎵 AUDIO TO SHEET MUSIC CONVERTER - SPEED OPTIMIZED")
    print("=" * 60)
    
    if not check_dependencies():
        print("⚙️ Installing requirements...")
        install_requirements()
        print("✅ Installation complete! Please restart the script.")
        return
    
    print("\n⚡ SPEED OPTIMIZATIONS ACTIVE:")
    print("- Reduced sample rate (22kHz) for faster processing")
    print("- Auto-trim to 60 seconds maximum")
    print("- Optimized pitch detection with thresholds")
    print("- Simplified post-processing")
    print("- 3-minute timeout (reduced from 5)")
    
    print("\n⚠️ For best results:")
    print("- Use WAV files (fastest loading)")
    print("- Keep clips under 60 seconds")
    print("- Isolate the melody if possible")
    print("- Skip visualization for maximum speed")
    
    while True:
        try:
            choice = main_menu()
            
            if choice == "2":
                print("👋 Goodbye!")
                break
                
            # Get file path
            file_path = None
            while True:
                try:
                    file_path = input("\nEnter audio file path (or 'back'): ").strip('"\'')
                    if file_path.lower() == 'back':
                        break
                    
                    valid, message = validate_audio_file(file_path)
                    if valid:
                        break
                    print(f"❌ {message}")
                except (KeyboardInterrupt, EOFError):
                    print("\n\nReturning to main menu...")
                    break
            
            if file_path is None or file_path.lower() == 'back':
                continue
                
            # Process the file with timeout protection
            result = extract_melody_to_sheet_music(file_path)
            
            if result:
                print(f"\n🎉 Conversion successful!")
                print(f"🎼 Open '{os.path.basename(result)}' in MuseScore or other music notation software")
                print("⚡ Processing completed in SPEED MODE")
            else:
                print("❌ Processing failed. Please try a different file.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()