import os
import math
import pickle
from tqdm import tqdm
from itertools import islice
from collections import deque
from collections import defaultdict
from markov.emotions.emotion_point import EmotionPoint
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from utils.midi_utils import *

"""
Canonical valence/arousal centre of each EMOPIA quadrant
Convention follows Russell (1980) and Hung et al. (2021)
"""
QUADRANT_EMOTION = {
    "Q1": EmotionPoint( 0.7, 0.7),  # happy / excited
    "Q2": EmotionPoint(-0.7, 0.7),  # tense / angry
    "Q3": EmotionPoint(-0.7, -0.7),  # sad / melancholic
    "Q4": EmotionPoint( 0.7, -0.7),  # calm / peaceful
}

class EmotionMarkovModel:
    QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]

    def __init__(self, order=1):
        self.order = order
        self.transitions = {
            q: defaultdict(lambda: defaultdict(float))
            for q in self.QUADRANTS
        }
        self.counts = {
            q: defaultdict(lambda: defaultdict(float))
            for q in self.QUADRANTS
        }
        self.states = {
            q: defaultdict(list) 
            for q in self.QUADRANTS
        }

    @staticmethod
    def _parse_quadrant(midi_path):
        # Extract Q1/Q2/Q3/Q4 from an EMOPIA filename, or return None
        # os.path.basename strips the directory path, leaving just the filename
        # re.match tries to match at the start of the string
        # Q[1-4] matches the literal character Q followed by exactly one digit from 1–4
        # return m.group(1) if m else None
        # If a match was found, returns the captured group, e.g. "Q1", "Q3"
        # If no match (file doesn't follow the EMOPIA naming convention), returns None
        m = re.match(r"(Q[1-4])", os.path.basename(midi_path))
        return m.group(1) if m else None

    @staticmethod
    def _collect_midi_files(midi_folder, max_files):
        entries = []
        for path in list(islice(find_midis(midi_folder), max_files)):
            quadrant = EmotionMarkovModel._parse_quadrant(path)
            if quadrant is None:
                continue
            entries.append((path, quadrant))

        entries.sort(key=lambda x: os.path.basename(x[0]))
        return entries
    
    """
    Save/Load trained model to file
    """
    def __getstate__(self):
        return {
            'order': self.order,
            'transitions': {
                q: {
                    key: dict(inner)
                    for key, inner in key_dict.items()
                }
                for q, key_dict in self.transitions.items()
            },
            'counts': {
                q: {
                    key: {
                        state: dict(counter)       # fix: serialize inner counter too
                        for state, counter in key_dict.items()
                    }
                    for key, key_dict in self.counts[q].items()
                }
                for q in self.QUADRANTS
            },
            'states': {
                q: {
                    key: list(v)
                    for key, v in key_dict.items()
                }
                for q, key_dict in self.states.items()
            },
        }

    def __setstate__(self, state):
        self.order = state['order']
        self.transitions = {
            q: defaultdict(lambda: defaultdict(float), {
                key: defaultdict(float, inner)
                for key, inner in key_dict.items()
            })
            for q, key_dict in state['transitions'].items()
        }
        self.counts = {
            q: defaultdict(lambda: defaultdict(lambda: defaultdict(float)), {
                key: defaultdict(lambda: defaultdict(float), {  # fix: restore nested defaultdicts
                    s: defaultdict(float, counter)
                    for s, counter in key_dict.items()
                })
                for key, key_dict in state['counts'][q].items()
            })
            for q in state['counts']
        }
        self.states = {
            q: defaultdict(list, {
                key: list(v)
                for key, v in key_dict.items()
            })
            for q, key_dict in state['states'].items()
        }

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)

    """
    Count state transitions
    """
    def _train_on_sequence(self, sequence, key, quadrant):
        if len(sequence) < self.order + 1:
            return

        counts_qk = self.counts[quadrant][key]
        states_qk = self.states[quadrant][key]
        order = self.order

        window = deque(sequence[:order], maxlen=order)
        state = tuple(window)

        for i in range(order, len(sequence)):
            next_event = sequence[i]
            counts_qk[state][next_event] += 1

            if state not in states_qk:
                states_qk.append(state)

            window.append(next_event)
            state = tuple(window)

    """
    Parallel normalize counts to probabilities
    """
    def _normalize_transitions(self):
        def normalise_key(item):
            (q, key), state_counts = item
            transitions = {}
            for state, counter in state_counts.items():
                notes = list(counter.keys())
                counts = np.array(list(counter.values()), dtype=np.float32)
                total = counts.sum()
                if total > 0:
                    cumulative = np.cumsum(counts / total).tolist()
                    transitions[state] = (notes, cumulative)
            return q, key, transitions

        items = {
            (q, key): self.counts[q][key]
            for q in self.QUADRANTS
            for key in self.counts[q]
        }

        with ThreadPoolExecutor() as executor:
            for q, key, transitions in executor.map(normalise_key, items.items()):
                self.transitions[q][key] = transitions
                self.states[q][key] = list(transitions.keys())

    """
    Training
    """
    @staticmethod
    def _process_file(args):
        path, quadrant = args
        notes = extract_notes_from_midi(path)
        key = fast_detect_key(notes)
        sequence = extract_delta_chord_sequence(notes)
        return quadrant, key, sequence
    
    def train(self, midi_folder, max_files=None, workers=os.cpu_count() - 1):
        midi_files = EmotionMarkovModel._collect_midi_files(midi_folder, max_files)

        print(f"[EmotionMarkovModel] Training on {len(midi_files)} EMOPIA files…")
        quadrant_counts: dict = defaultdict(int)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(EmotionMarkovModel._process_file, midi_file): midi_file for midi_file in midi_files}
            for future in tqdm(as_completed(futures), total=len(midi_files)):
                result = future.result()
                if result is None:
                    continue
                quadrant, key, sequence = result
                if len(sequence) >= self.order + 1:
                    self._train_on_sequence(sequence, key, quadrant)
                    quadrant_counts[quadrant] += 1

        self._normalize_transitions()
        print(f"[EmotionMarkovModel] Done. Files per quadrant: {dict(quadrant_counts)}")

    """
    New helpers for distribution
    """
    def get_distribution(self, quadrant, key, state):
        """
        Return (notes, cumulative) for the given (quadrant, key, state)
        Falls back to uniform over all known events in this quadrant+key
        when the exact state was not seen during training
        """
        # Looks up the exact state in transitions
        key_transitions = self.transitions[quadrant][key]

        if state in key_transitions:
            return key_transitions[state]  # (notes, cumulative) tuple

        # Uniform fallback, collect all events seen in this quadrant+key
        all_events = set()
        for notes, _ in key_transitions.values():
            all_events.update(notes)

        if all_events:
            all_events = sorted(all_events)
            p = 1.0 / len(all_events)
            cumulative = [p * (i + 1) for i in range(len(all_events))]
            return all_events, cumulative

        return None

    def interpolated_distribution(self, emotion, key, state):
        weights = {}
        for q, pt in QUADRANT_EMOTION.items():
            # For each quadrant, it computes the Euclidean distance from the target emotion to that quadrant's canonical point
            d = math.hypot(emotion.valence - pt.valence, emotion.arousal - pt.arousal)
            # A nearby quadrant gets a large weight
            # A distant point gets a tiny weight
            # The 1e-6 epsilon prevents a division-by-zero if the emotion lands exactly on a canonical point.
            weights[q] = 1.0 / (d + 1e-6)
        total_w = sum(weights.values())

        # Different quadrant distributions might not share the same set of events
        # This union ensures every event that appears in any quadrant is included in the output, even if it has zero probability in some quadrants
        # Collect all events across all quadrants
        all_events: set = set()
        for q in self.QUADRANTS:
            result = self.get_distribution(q, key, state)
            if result is not None:
                notes, _cumulative = result
                all_events.update(notes)
        if not all_events:
            return {}

        # Build per-quadrant probability dicts for easy blending
        blended: dict = defaultdict(float)
        for q in self.QUADRANTS:
            # Normalize weights
            w = weights[q] / total_w
            result = self.get_distribution(q, key, state)
            if result is None:
                continue
            notes, cumulative = result
            # Recover individual probabilities from cumulative
            probs = [cumulative[0]] + [cumulative[i] - cumulative[i-1] for i in range(1, len(cumulative))]
            for event, prob in zip(notes, probs):
                blended[event] += w * prob

        return dict(blended)
