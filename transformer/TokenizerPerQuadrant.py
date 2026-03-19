from pathlib import Path
# miditok: A Python library to convert MIDI files into sequences of tokens, which can later be used for machine learning models
# REMI: A specific tokenization scheme that converts MIDI events into a structured sequence of tokens, including notes, velocities, time shifts, and optionally tempos, chords, etc
from miditok import REMI, TokenizerConfig

# https://github.com/annahung31/EMOPIA

# Configure the tokenizer
config = TokenizerConfig(
    # A note event = a single note being played (Note-On_60 for middle C)
    # A chord event = multiple notes played together that form a harmonic unit, like a C major triad (C-E-G)
    # Tokenizing chords means you create tokens that explicitly represent chords, instead of just sequences of single notes
    # Chord events are not tokenized
    use_chords=False,
    # Time-Shift tokens already handle gaps between notes, so you don’t need separate Rest tokens
    # Rest (silent periods) are ignored
    use_rests=False,
    # Tempo changes are tokenized (so the sequence can capture speed changes in the music)
    use_tempos=True,
    # A time signature defines the meter of the music: how many beats per bar and what note value counts as one beat
    # Examples:
    # 4/4 -> 4 beats per bar, quarter note = 1 beat
    # 3/4 -> 3 beats per bar, quarter note = 1 beat
    # 6/8 -> 6 beats per bar, eighth note = 1 beat
    # REMI can tokenize time signature changes as TimeSig_4/4, TimeSig_3/4, etc
    # This would add tokens to the vocabulary for each unique time signature in your dataset
    # Time signatures are ignored
    use_time_signatures=False,
    # MIDI supports multiple instruments via program change events
    # Examples: Piano, Guitar, Violin, etc.
    # Each instrument has a program number in MIDI
    # In REMI, use_programs=True would add tokens for each instrument (e.g., Program_0_Piano, Program_24_Guitar) into your sequences
    # EMOPIA is a single-instrument piano dataset
    # Every MIDI file uses the same instrument
    # So tokenizing instrument programs adds no useful information — it would just be the same token everywhere
    # Instrument programs (like piano, guitar) are ignored
    use_programs=False,
    
    # MIDI note velocities (how loud the note is) are discretized into 32 levels
    num_velocities=32,
    # Tempo values are quantized into 32 possible tokens
    num_tempos=32,
    # Only consider tempos between 40 and 250 BPM
    # This defines the minimum and maximum tempo (in BPM) that the tokenizer will consider when creating tempo tokens
    # So:
    # 40 BPM = slow (adagio / largo)
    # 250 BPM = very fast (prestissimo)
    # Any tempo in your MIDI files will be:
    # - Clipped to this range if needed
    # - Quantized into num_tempos bins (you set num_tempos=32)
    # With: 
    # num_tempos=32
    # tempo_range=(40, 250)
    # This means: The range 40 -> 250 BPM is divided into 32 discrete tempo tokens
    # So instead of having infinite tempo values like:
    # Tempo_121.3, Tempo_121.7, Tempo_122.1
    # You get something like:
    # Tempo_120, Tempo_124, Tempo_128 ...
    # This makes it learnable for a model
    # Anything outside this range is:
    # Rare
    # Often noise or MIDI artifacts
    tempo_range=(40, 250),
    # Defines temporal resolution depending on the bar:
    # Bars 0–4 use 8 tokens per beat
    # Bars 4–12 use 4 tokens per beat
    # This is useful for focusing more on the early structure of the song with finer resolution
    # This defines the temporal resolution, i.e.:
    # How finely time is divided within each beat
    # 8 -> each beat is split into 8 time steps (very precise)
    # 4 -> each beat is split into 4 time steps (less precise)
    # These refer to beat positions within a bar (measure).
    # So:
    # (0, 4): 8 -> For beats 0 to 4 (the first bar in 4/4) -> use high resolution (8)
    # (4, 12): 4 -> For later beats -> use lower resolution (4)
    # In practice, this means:
    # Early parts of the sequence -> more precise timing
    # Later parts -> coarser timing
    # Why would you do this?
    # 1. Trade-off: precision vs sequence length
    # Higher resolution = more tokens
    # If you used resolution 8 everywhere:
    # Very accurate timing
    # BUT sequences become longer and heavier for models
    # The start of a piece is often:
    # Structurally important
    # Rhythmically expressive
    # Emotionally meaningful (very relevant for EMOPIA)
    # So you:
    # Keep fine detail early
    # Reduce detail later to save tokens
    beat_res={(0, 4): 8, (4, 12): 4},
)
# Build tokenizer
tokenizer = REMI(config)

# Path to local EMOPIA MIDI folder
midi_dir = Path("data/emopia/EMOPIA_2.2/midis")

# Tokenize all MIDIs and save tokens
output_dir = Path("data/emopia/remi_tokens_per_song")
output_dir.mkdir(exist_ok=True)

# tokenize_dataset() -> takes each MIDI file and:
# - Reads it
# - Converts it into a sequence of tokens according to your REMI configuration
# - Saves the token sequence in output_dir
tokenizer.tokenize_dataset(
    # midi_dir.glob("**/*.mid") -> finds all MIDI files recursively in your folder
    files_paths=list(midi_dir.glob("**/*.mid")),
    out_dir=output_dir,
)

print(f"Vocab size: {len(tokenizer)}")  # Number of unique tokens in your tokenizer (the vocabulary size)
print(f"Token types: {tokenizer.vocab}")  # Shows all token types (like Note-On_60_Velocity_20, Tempo_120, Time-Shift_4)
