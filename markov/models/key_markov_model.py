import os
import bisect
import random
import pickle
from tqdm import tqdm
from itertools import islice
from collections import deque
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from utils.midi_utils import *

class KeyChordMarkovModel:
    def __init__(self, order=1, mode='melody'):
        """
        mode: 'melody' -> uses extract_delta_chord_sequence (original behaviour)
              'rhythm' -> uses extract_rhythm_sequence
              'chord'  -> uses extract_chord_sequence
        """
        self.order = order
        self.mode = mode
        self.transitions = defaultdict(lambda: defaultdict(Counter))
        self.counts = defaultdict(lambda: defaultdict(Counter))
        self.states = defaultdict(list)
        self.state_buffers = defaultdict(list)

    """
    Initialize shuffled state buffers
    """
    def _init_state_buffers(self):
        for key in self.states:
            self.state_buffers[key] = list(self.states[key])
            random.shuffle(self.state_buffers[key])
    
    def _get_random_state(self, key):
        buffer = self.state_buffers[key]
        if not buffer:
            buffer = list(self.states[key])
            random.shuffle(buffer)
            self.state_buffers[key] = buffer
        return buffer.pop()

    """
    Save/Load trained model to file
    """
    def __getstate__(self):
        return {
            'order': self.order,
            'mode':  self.mode,
            'transitions': {k: dict(v) for k, v in self.transitions.items()},
            'counts': {k: dict(v) for k, v in self.counts.items()},
            'states': dict(self.states),
            'state_buffers': dict(self.state_buffers),
        }

    def __setstate__(self, state):
        self.order = state['order']
        self.mode = state['mode']
        self.transitions = defaultdict(lambda: defaultdict(Counter), 
                                       {k: defaultdict(Counter, v) for k, v in state['transitions'].items()})
        self.counts = defaultdict(lambda: defaultdict(Counter),
                                  {k: defaultdict(Counter, v) for k, v in state['counts'].items()})
        self.states = defaultdict(list, state['states'])
        self.state_buffers = defaultdict(list, state['state_buffers'])

    def save_model(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)

    """
    Count state transitions
    """
    def _train_on_sequence(self, sequence, key):
        if len(sequence) < self.order + 1:
            return
        
        counts_key = self.counts[key]
        order = self.order

        # Pre-fill the sliding window with the first `order` elements
        window = deque(sequence[:order], maxlen=order)
        state = tuple(window)  # only allocate once to start

        for i in range(order, len(sequence)):
            next_event = sequence[i]
            counts_key[state][next_event] += 1

            # Slide window: O(1) vs O(order) for tuple slicing
            window.append(next_event)
            state = tuple(window)  # still need tuple for hashing, but deque.append is O(1)
            
    """
    Parallel normalize counts to probabilities
    """
    def _normalize_transitions(self):
        def normalize_key(item):
            key, state_counts = item
            transitions = {}
            for state, counter in state_counts.items():
                notes = list(counter.keys())
                counts = np.array(list(counter.values()), dtype=np.float32)
                cumulative = np.cumsum(counts / counts.sum()).tolist()
                transitions[state] = (notes, cumulative)
            return key, transitions

        with ThreadPoolExecutor() as executor:
            for key, transitions in executor.map(normalize_key, self.counts.items()):
                self.transitions[key] = transitions
                self.states[key] = list(transitions.keys())

    """
    Training
    """
    def _process_file(self, path):
        notes = extract_notes_from_midi(path)
        key = fast_detect_key(notes)
        match self.mode:
            case 'rhythm':
                sequence = extract_rhythm_sequence(notes, path)
            case 'chord':
                sequence = extract_chord_sequence(extract_delta_chord_sequence(notes))
            case _:
                sequence = extract_delta_chord_sequence(notes)
        return key, sequence
    
    def train(self, midi_folder, max_files=None, workers=os.cpu_count() - 1):
        paths = list(islice(find_midis(midi_folder), max_files))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(KeyChordMarkovModel._process_file, path): path for path in paths}
            for future in tqdm(as_completed(futures), total=len(paths)):
                result = future.result()
                if result is None:
                    continue
                key, sequence = result
                if len(sequence) >= self.order + 1:
                    self._train_on_sequence(sequence, key)

        self._normalize_transitions()

    """
    Generate melody
    """
    def _sample_next(self, state, key):
        if key is None:
            key = random.choice(['major', 'minor'])
        if not self.states[key]:
            raise ValueError(f"No model trained for {key}")
        model = self.transitions[key]
        if state not in model:
            state = self._get_random_state(key)
        notes, cumsum = model[state]
        idx = bisect.bisect_left(cumsum, random.random())
        return notes[min(idx, len(notes)-1)]

    def generate(self, length=100, key=None):
        if key is None:
            key = random.choice(['major', 'minor'])
        if not self.transitions[key]:
            raise ValueError(f"No model trained for {key}")
        state = random.choice(self.states[key])
        result = []
        for _ in range(length):
            next_event = self._sample_next(state, key)
            result.append(next_event)
            if self.order == 1:
                state = (next_event,)
            else:
                state = tuple(list(state[1:]) + [next_event])
        return result
