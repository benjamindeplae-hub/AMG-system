import random

class AffectiveBias:
    """
    Wraps an OrchestrateMarkovModel and biases its Markov sampling at each
    step according to an EmotionCurve
 
    The bias works by re-weighting the transition distribution before each
    sample draw
    For each candidate next-event, a compatibility score is
    computed from its musical features vs. the target (valence, arousal),
    the distribution is then multiplied element-wise by these scores and
    renormalised
    A blend parameter alpha controls how strongly the emotion
    pulls vs. the learned Markov probabilities
 
    Musical feature -> emotion mappings:
    The mappings in _feature_targets() are based on:
        Eerola, T., & Vuoskoski, J. K. (2011).
        A comparison of the discrete and dimensional models of emotion in music
        Psychology of Music, 39(1), 18-49. (Table 2)
 
        Thayer, R. E. (1989). The Biopsychology of Mood and Arousal
        Oxford University Press
    """
    def __init__(self, model, curve, alpha):
        # OrchestrateMarkovModel
        self.model = model
        # EmotionCurve
        self.curve = curve
        # float [0, 1] blend weight: 0 = pure Markov, 1 = pure emotion
        self.alpha = max(0.0, min(1.0, alpha))
    
    """
    feature target mapping (Eerola & Vuoskoski 2011)
    """
    @staticmethod
    def _feature_targets(emotion):
        """
        Convert a (valence, arousal) point to target musical feature ranges
 
        Returns a dict with keys:
            pitch_class_target: int [0,11] preferred pitch class (0=C)
            prefer_minor: float [0,1] 0 = major, 1 = minor
            prefer_dense: float [0,1] note density preference
            prefer_high_pitch: float [0,1] pitch register preference
 
        Mapping rationale (Eerola & Vuoskoski 2011, Table 2):
            High Arousal maps to higher note density, higher register
            Low Arousal maps to lower note density, lower register
            Low Valence maps to minor mode, dissonant intervals (diminished/minor 2nd)
            High Valence maps to major mode, consonant intervals
        """
        v = emotion.valence  # [-1, 1]
        a = emotion.arousal  # [-1, 1]
 
        # Normalize to [0, 1]
        vn = (v + 1) / 2
        an = (a + 1) / 2
 
        return {
            # Simply the inverse of valence
            # When valence is high (happy, vn close to 1), 1.0 - vn is close to 0 -> prefer major
            # When valence is low (sad, vn close to 0), 1.0 - vn is close to 1 -> prefer minor
            "prefer_minor": 1.0 - vn,
            # High arousal (an close to 1) -> dense notes
            # Low arousal (an close to 0) -> sparse notes
            "prefer_dense": an,
            # This one combines both dimensions, but arousal dominates
            # Arousal contributes 60% and Valence contributes 20%
            "prefer_high_pitch": an * 0.6 + vn * 0.2,
            # High valence -> low dissonance
            # Low valence -> high dissonance
            "prefer_dissonant": 1.0 - vn,
        }

    @staticmethod
    def _score_event(event, targets, key):
        """
        Return an affective compatibility score in [0, 1] for a candidate
        next-event given the feature targets (≈ emotion point)
 
        event is a (delta, pitches) tuple as stored in the Markov model
        """
        # neutral for malformed events
        if not isinstance(event, tuple) or len(event) < 2:
            return 0.5
 
        delta, pitches = event[0], event[1]
        if isinstance(pitches, (int, float)):
            pitch_list = [int(pitches)]
        elif isinstance(pitches, tuple):
            pitch_list = list(pitches)
        else:
            return 0.5
 
        score = 0.0
        weight_total = 0.0
        """
        1. Mode compatibility
        """
        # which of the 12 semitones in an octave the note belongs to, regardless of which octave it's in
        # So C4 and C5 both have pitch class 0 if C is the root
        # 60 % 12 = 0 means C4 is C, and 72 % 12 = 0 means C5 is also C, just an octave higher
        # Semitone 0    1    2    3    4    5    6    7    8    9    10    11
        # Note     C    C#   D    D#   E    F    F#   G    G#   A    A#    B
        # EU       Do   Do#  Re   Re#  Mi   Fa   Fa#  Sol  Sol# La   La#   Si
        # Major key intervals: 0, 2, 4, 5, 7, 9, 11 = C D E  F G A  B  = Do Re Mi  Fa Sol La   Si 
        # Minor key intervals: 0, 2, 3, 5, 7, 8, 10 = C D D# F G G# A# = Do Re Re# Fa Sol Sol# La#
        MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}
        MINOR_PCS = {0, 2, 3, 5, 7, 8, 10}
 
        if pitch_list:
            # pitch class
            pcs = {p % 12 for p in pitch_list}
            prefer_minor = targets["prefer_minor"]
            if key == "minor":
                mode_match = len(pcs & MINOR_PCS) / len(pcs)
                mode_score = mode_match * prefer_minor       + (1 - mode_match) * (1 - prefer_minor)
            else:
                mode_match = len(pcs & MAJOR_PCS) / len(pcs)
                mode_score = mode_match * (1 - prefer_minor) + (1 - mode_match) * prefer_minor
            score += mode_score * 2.0
            weight_total += 2.0
 
        # ── 2. Register (pitch height) ─────────────────────────────────────────
        # MIDI pitch ~60 = middle C; normalise 36–84 range → [0,1]
        if pitch_list:
            mean_pitch = sum(pitch_list) / len(pitch_list)
            pitch_norm = max(0.0, min(1.0, (mean_pitch - 36) / 48))
            register_score = 1.0 - abs(pitch_norm - targets["prefer_high_pitch"])
            score += register_score * 1.5
            weight_total += 1.5
 
        # ── 3. Density (inverse of delta) ─────────────────────────────────────
        # Small delta → dense / high arousal feel
        # Assume delta is in ticks; normalise 0–480 range
        if isinstance(delta, (int, float)) and delta >= 0:
            delta_norm = max(0.0, min(1.0, delta / 480))
            density_score = 1.0 - abs((1 - delta_norm) - targets["prefer_dense"])
            score += density_score * 1.0
            weight_total += 1.0
 
        if weight_total == 0:
            return 0.5
 
        return score / weight_total
 




    # ── sampler factory ───────────────────────────────────────────────────────
 
    def _make_biased_sampler(self, submodel, key: str):
        """
        Return a replacement _sample_next function that blends the Markov
        distribution with affective compatibility scores.
 
        The step counter is closed over a mutable list so the nested function
        can increment it across calls without a class attribute.
        """
        step_counter = [0]
        alpha = self.alpha
        curve = self.curve
 
        def biased_sample_next(state, k):
            step = step_counter[0]
            step_counter[0] += 1
 
            emotion  = curve.get(step)
            targets  = AffectiveBias._feature_targets(emotion)
 
            raw_dist = dict(submodel.transitions[k].get(state, {}))
            if not raw_dist:
                states = submodel.states[k]
                if not states:
                    raise ValueError(f"No states for key={k}")
                return random.choice(states)[-1]
 
            # Compute affective score for each candidate event
            scores = {
                event: AffectiveBias._score_event(event, targets, k)
                for event in raw_dist
            }
 
            # Blend: new_weight = (1-α)*markov_prob + α*affective_score
            blended = {}
            for event, prob in raw_dist.items():
                blended[event] = (1.0 - alpha) * prob + alpha * scores[event]
 
            # Renormalise
            total = sum(blended.values()) or 1.0
            blended = {e: w / total for e, w in blended.items()}
 
            # Sample
            r = random.random()
            cumulative = 0.0
            for event, prob in blended.items():
                cumulative += prob
                if r <= cumulative:
                    return event
            return list(blended.keys())[-1]
 
        return biased_sample_next

    """
    Generate melody
    """
    def generate(self, length: int = 100, key: str = None) -> list:
        """
        Generate a song with emotion bias applied at each step.
        Returns the same list-of-(delta, duration, pitches) tuples as
        OrchestrateMarkovModel.generate_melody().
        """
        if key is None:
            key = random.choice(["major", "minor"])
 
        # Update curve's total_steps to match requested length
        self.curve.total_steps = length
 
        # Patch _sample_next on both melody models to inject affective bias
        originals = {}
        for name in ("left_model", "right_model"):
            sub = getattr(self.model, name)
            originals[name] = sub._sample_next
            sub._sample_next = self._make_biased_sampler(sub, key)
 
        try:
            song = self.model.generate_melody(length=length, key=key)
        finally:
            for name, orig in originals.items():
                getattr(self.model, name)._sample_next = orig
 
        return song