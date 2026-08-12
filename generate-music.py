import subprocess
from pathlib import Path
import torch
from pipeline.ml.model import NoteTransformer
from pipeline.ml.data_prep import BOS

try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoteTransformer().to(device)
checkpoint_path = Path("data/ml/checkpoints/style_finetune_final.pt")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 2. Score Setup
score = music21.stream.Score()
target_key = music21.key.Key("a")

# Violin Part (The "Controlled" Melody)
violin_part = music21.stream.Part()
violin_part.partName = "Violin"
violin_part.insert(0, music21.instrument.Violin())

# Piano Grand Staff
piano_right = music21.stream.PartStaff(id="PianoRH")
piano_right.partName = "Piano"
piano_right.id = "PianoRH"

piano_left = music21.stream.PartStaff(id="PianoLH")

# 3. Generation (Only use AI for Violin; keep Piano purely functional)
prompt_ids = torch.tensor([BOS, 69, 72, 76, 74], dtype=torch.long, device=device)
with torch.no_grad():
    raw_tokens = model.generate(prompt_ids, max_new_tokens=100, temperature=0.7).tolist()

natural_minor = {9, 11, 0, 2, 4, 5, 7}
def snap(n): return min([n+o for o in range(-6,7) if (n+o)%12 in natural_minor], key=lambda x: abs(x-n))
violin_pitches = [max(60, min(84, snap(t))) for t in raw_tokens if isinstance(t, int)]

# 4. Fill Violin
v_idx = 0
for m in range(16):
    measure = music21.stream.Measure(number=m+1)
    if m == 0:
        measure.insert(0, music21.clef.TrebleClef())
        measure.insert(0, target_key)
        measure.insert(0, music21.meter.TimeSignature("4/4"))
    
    beats = 0
    while beats < 4:
        p = violin_pitches[v_idx % len(violin_pitches)]
        n = music21.note.Note(p)
        n.duration.quarterLength = 0.5
        measure.append(n)
        beats += 0.5
        v_idx += 1
    violin_part.append(measure)

# 5. Fill Piano (STRICT RHYTHMIC ANCHOR - NO AI MELODY)
bass_notes = [45, 40] # A2, E2
chords = [[57, 60, 64], [55, 59, 62]] # Am, Em

for m in range(16):
    rh = music21.stream.Measure(number=m+1)
    lh = music21.stream.Measure(number=m+1)
    
    if m == 0:
        rh.insert(0, music21.clef.TrebleClef())
        lh.insert(0, music21.clef.BassClef())
        rh.insert(0, target_key)
        lh.insert(0, target_key)
        rh.insert(0, music21.meter.TimeSignature("4/4"))
        lh.insert(0, music21.meter.TimeSignature("4/4"))
    
    for beat in range(4):
        b = music21.note.Note(bass_notes[beat % 2])
        b.duration.quarterLength = 1.0
        lh.append(b)
        
        c = music21.chord.Chord(chords[beat % 2])
        c.duration.quarterLength = 1.0
        rh.append(c)
        
    piano_right.append(rh)
    piano_left.append(lh)

# 6. Group Piano Staves & Assembly
piano_group = music21.layout.StaffGroup([piano_right, piano_left], symbol="brace", barTogether=True)
piano_group.name = "Piano"
piano_group.abbreviation = "Pno."

score.insert(0, violin_part)
score.insert(0, piano_right)
score.insert(0, piano_left)
score.insert(0, piano_group)

output_file = Path("violin_piano_refined.musicxml")
score.write("musicxml", fp=str(output_file))
print(f"Saved refined score to: {output_file.absolute()}")

# 7. Open in MuseScore
choice = input("Open this score in MuseScore? (y/n): ").strip().lower()
if choice == "y":
    musescore_commands = ["musescore", "musescore4", "musescore3", "mscore"]
    opened = False
    for cmd in musescore_commands:
        try:
            subprocess.Popen([cmd, str(output_file)])
            print(f"Opened score using '{cmd}'.")
            opened = True
            break
        except FileNotFoundError:
            continue
    if not opened:
        print("MuseScore was not found. Open the MusicXML file manually.")