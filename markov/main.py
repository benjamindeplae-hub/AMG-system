import os
from audio_playback import AudioPlayback
from models.orchestrate_markov_model import OrchestrateMarkovModel
from utils.midi_utils import *

if __name__ == "__main__":
    MAESTRO_MIDI_DIR = "./data/maestro-v3.0.0-midi/maestro-v3.0.0"
    SOUNDFONT_PATH = "./soundfonts/FluidR3_GM_GS.sf2"

    # currently not used variable
    REAL_PIANO_PIECE = "./maestro-v3.0.0-midi/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"

    BASEPATH = "./markov/"
    MODEL_PATH = BASEPATH + "markov_model.pkl"
    OUTPUT_PATH = BASEPATH + "markov_output.mid"

    mm = OrchestrateMarkovModel(order=2)

    if os.path.exists(MODEL_PATH):  
        print("Found saved model! Loading...")
        mm.load_model(MODEL_PATH)
    else:
        print("No saved model found. Training from scratch...")
        mm.train(MAESTRO_MIDI_DIR)
        print("Saving model...")
        mm.save_model(MODEL_PATH)

    print("Generating melody...")
    left_seq = mm.generate_melody(length=150)

    print("Saving melody to MIDI...")
    melody_to_midi(left_seq, OUTPUT_PATH)

    print("Synthesizing audio for playback...")
    AudioPlayback.play_midi_fluidsynth(OUTPUT_PATH, SOUNDFONT_PATH)
