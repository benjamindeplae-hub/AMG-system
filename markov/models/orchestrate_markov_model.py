import os
import time
import random
import pickle
from tqdm import tqdm
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from models.key_markov_model import KeyChordMarkovModel
from utils.midi_utils import *

class OrchestrateMarkovModel:
    def __init__(self, order=1):
        self.order = order
        self.chord_model = KeyChordMarkovModel(order, mode='chord')
        self.rhythm_model = KeyChordMarkovModel(order, mode='rhythm')
        self.left_model = KeyChordMarkovModel(order, mode='melody')
        self.right_model = KeyChordMarkovModel(order, mode='melody')

    """
    Save/Load trained model to file
    """
    def __getstate__(self):
        return {
            'order': self.order,
            'chord_model': self.chord_model,
            'rhythm_model': self.rhythm_model,
            'left_model': self.left_model,
            'right_model': self.right_model,
        }

    def __setstate__(self, state):
        self.order = state['order']
        self.chord_model = state['chord_model']
        self.rhythm_model = state['rhythm_model']
        self.left_model = state['left_model']
        self.right_model = state['right_model']

    def save_model(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)

    """
    Split notes by left or right hands
    """
    @staticmethod
    def split_hands(notes):
        # group into chords
        groups = group_simultaneous_notes(notes)

        left_hand = []
        right_hand = []

        last_left_pitch = 48   # starting guess
        last_right_pitch = 72

        for chord in groups:

            # sort notes by pitch
            chord = sorted(chord, key=lambda n: n.pitch)

            # if single note -> choose closest hand
            if len(chord) == 1:
                note = chord[0]

                left_cost = abs(note.pitch - last_left_pitch)
                right_cost = abs(note.pitch - last_right_pitch)

                if left_cost < right_cost:
                    left_hand.append(note)
                    last_left_pitch = note.pitch
                else:
                    right_hand.append(note)
                    last_right_pitch = note.pitch

            else:
                # split chord into low/high halves
                mid = len(chord) // 2
                left_part = chord[:mid]
                right_part = chord[mid:]

                for note in left_part:
                    left_hand.append(note)
                    last_left_pitch = note.pitch

                for note in right_part:
                    right_hand.append(note)
                    last_right_pitch = note.pitch

        return left_hand, right_hand

    """
    Training
    """
    @staticmethod
    def _process_file(path):
        start_time = time.perf_counter()
        notes = extract_notes_from_midi(path)
        key = fast_detect_key(notes)
        left_notes, right_notes = OrchestrateMarkovModel.split_hands(notes)

        print(f"Execution time: {time.perf_counter() - start_time:.4f} seconds")
        return (
            key,
            extract_chord_sequence(extract_delta_chord_sequence(right_notes)),
            extract_rhythm_sequence(notes, path),
            extract_delta_chord_sequence(left_notes),
            extract_delta_chord_sequence(right_notes),
        )
        
    def train(self, midi_folder, max_files=None, workers=os.cpu_count() - 1):
        start_time = time.perf_counter()
        paths = list(islice(find_midis(midi_folder), max_files))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(OrchestrateMarkovModel._process_file, path): path for path in paths}
            
            for future in tqdm(as_completed(futures), total=len(paths)):
                result = future.result()
                if result is None:
                    continue
                key, chord_seq, rhythm_seq, left_seq, right_seq = result
                self.chord_model._train_on_sequence(chord_seq, key)
                self.rhythm_model._train_on_sequence(rhythm_seq, key)
                self.left_model._train_on_sequence(left_seq, key)
                self.right_model._train_on_sequence(right_seq, key)

        self.chord_model._normalize_transitions()
        self.rhythm_model._normalize_transitions()
        self.left_model._normalize_transitions()
        self.right_model._normalize_transitions()
        print(f"Execution time: {time.perf_counter() - start_time:.4f} seconds")

    """
    Generate melody
    """
    def generate_melody(self, length=100, key=None):
        if key is None:
            key = random.choice(['major', 'minor'])

        # 1. Generate a shared chord progression that drives both hands
        chord_progression = self.chord_model.generate(length, key)
        rhythm_progression = self.rhythm_model.generate(length, key)

        # 2. Generate each hand conditioned on (key + chord progression)
        left_seq = self._generate_hand_conditioned(self.left_model, chord_progression, rhythm_progression, key, length)
        right_seq = self._generate_hand_conditioned(self.right_model, chord_progression, rhythm_progression, key, length)

        # 3. Merge and sort by onset time
        combined = left_seq + right_seq
        combined.sort(key=lambda x: x[0])
        return combined

    """
    Generate Helper
    """
    @staticmethod
    def chord_notes(event):
        if isinstance(event, tuple) and len(event) == 2:
            _delta, chord = event
            if isinstance(chord, tuple):
                return chord
        if isinstance(event, tuple):
            return event
        return (event,)

    def _generate_hand_conditioned(self, hand_model, chord_progression, rhythm_progression, key, length):
        """
        Generate one hand's sequence, nudging the Markov state back toward
        the current target chord every 4 steps so both hands stay harmonically
        aligned with the shared chord_progression.
        """
        if not hand_model.transitions[key]:
            raise ValueError(f"No model trained for {key}")

        states = hand_model.states[key]
        model = hand_model.transitions[key]

        # Pick a start state that contains a note from the first chord
        # Seed with a state whose notes overlap the first chord if possible
        first_chord = set(chord_progression[0]) if chord_progression else set()
        matching = [s for s in states
                    if any(n in first_chord
                           for event in s
                           for n in OrchestrateMarkovModel.chord_notes(event))]
        state = random.choice(matching) if matching else random.choice(states)

        result = []
        for i, (target_chord, (delta, duration)) in enumerate(zip(chord_progression[:length], rhythm_progression[:length])):
            target_set = set(target_chord)

            # Sample next event from the hand model
            next_event = hand_model._sample_next(state, key)
            _, pitches = next_event
            timed_event = (delta, duration, pitches)
            result.append(timed_event)
            
            # Update state
            # Advance the Markov state normally
            if hand_model.order == 1:
                state = (next_event,)
            else:
                state = tuple(list(state[1:]) + [next_event])

            # Every N steps, steer toward the current chord by
            # preferring a state whose notes overlap with target_chord
            # Every 4 steps: if the state has drifted out of the model or away
            # from the target chord, steer it back
            if i % 4 == 0 and state not in model:
                candidates = [s for s in states
                              if any(n in target_set 
                                     for event in s
                                     for n in self.chord_notes(event))]
                if candidates:
                    state = random.choice(candidates)
        return result
