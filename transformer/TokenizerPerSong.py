import os
import muspy
import tempfile
import numpy as np
from pathlib import Path
from miditok import REMI, TokenizerConfig

# https://github.com/annahung31/EMOPIA
# https://annahung31.github.io/EMOPIA
# https://muspy.readthedocs.io/en/stable/_modules/muspy/datasets/emopia.html
# https://miditok.readthedocs.io/en/v3.0.6/examples.html

# 1. Load EMOPIA via muspy
# In MusPy, that parameter controls whether the dataset is automatically downloaded and unpacked for you
emopia = muspy.EMOPIADataset("data/emopia/", download_and_extract=False)
emopia.convert()

# 2. Build REMI tokenizer
config = TokenizerConfig(
    num_velocities=32,
    use_chords=False,
    use_rests=False,
    use_tempos=True,
    use_time_signatures=False,
    use_programs=False,
    num_tempos=32,
    tempo_range=(40, 250),
    beat_res={(0, 4): 8, (4, 12): 4},
)
# Build tokenizer
tokenizer = REMI(config)

# 3. Convert each muspy Music object -> temp MIDI -> REMI tokens
output_dir = Path("data/emopia/remi_tokens_per_song")
output_dir.mkdir(parents=True, exist_ok=True)

# List of dicts with tokens + label
remi_dataset = []

for idx in range(len(emopia)):
    # MusPy Music object of type muspy.Music
    music = emopia[idx]
    # {'emo_class', 'YouTube_ID', 'seg_id'}
    annotation = music.annotations[0].annotation
    youtube_id = annotation["YouTube_ID"]
    seg_id = annotation["seg_id"]
    emo_class = annotation["emo_class"]

    # Write muspy Music object to a temporary MIDI file
    # It uses Python’s built-in module tempfile to create a file in the system’s temporary directory,
    # Ensuring that the file name ends with .mid
    # By default, NamedTemporaryFile deletes the file automatically when it’s closed
    # Setting delete=False means:
    # The file will NOT be deleted automatically
    # Default locations:
    # Linux / macOS: /tmp/
    # Windows: C:\Users\<your_username>\AppData\Local\Temp\
    # The with statement gives you a file object called tmp
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        # Every file object in Python has a .name attribute
        # For NamedTemporaryFile, .name is the full path of the temporary file on your system
        tmp_path = tmp.name

    try:
        # Uses MusPy to save a music object as a MIDI file
        muspy.write_midi(tmp_path, music)

        # Tokenize with REMI
        # When you call tokenizer(tmp_path), it reads the MIDI file and outputs a list of TokSequence objects
        # It returns a list of TokSequence objects, not a list of integers yet
        # Usually, for one MIDI track or song, this list has length 1, so tokens[0] is the sequence for that song
        tokens = tokenizer(tmp_path)
        # tokens[0] gives the first TokSequence object
        # .ids -> gives a list of integers, where each integer is a token representing a musical event
        token_ids = tokens[0].ids

        entry = {
            "youtube_id": youtube_id,
            "seg_id": seg_id,
            # Q1 / Q2 / Q3 / Q4
            "emo_class": emo_class,
            # list of integers (REMI) representing the whole song
            "token_ids": token_ids,
        }
        remi_dataset.append(entry)

        # Optionally save per-song .npy
        # out_path = output_dir / f"{youtube_id}_seg{seg_id}_Q{emo_class}.npy"
        # Save the REMI token IDs as a NumPy array
        # np.save(out_path, np.array(token_ids, dtype=np.int32))

        print(f"[{idx+1}/{len(emopia)}] {youtube_id} seg{seg_id} Q{emo_class} -> {len(token_ids)} tokens")

    finally:
        # os.unlink(path) is essentially the same as deleting a file. It removes the file specified by path from the filesystem
        os.unlink(tmp_path)  # clean up temp file

# 4. Save tokenizer for reuse
# A tokenizer (like REMI) is not just a function that converts MIDI → integers. It contains:
# Vocabulary: The set of all tokens it knows (like NOTE_ON_60, TEMPO_120, BAR, etc.)
# Mapping tables: Which token maps to which integer (token -> id and id -> token)
# Configuration: Settings like number of velocities, use of chords, rests, tempos, etc
# So even if two tokenizers are both REMI, if their configuration differs, the resulting token IDs for the same MIDI will not match
# Without saving, you would need to recreate the tokenizer with the exact same config, which is error-prone if the config is complex
tokenizer.save_pretrained("data/emopia/remi_tokens_per_song")

print(f"\nDone! {len(remi_dataset)} songs tokenized")
print(f"Vocab size: {len(tokenizer)}")
print(f"Token types: {tokenizer.vocab}")
