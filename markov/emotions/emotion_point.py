
class EmotionPoint:
    _map = {
        "angry": (-0.7, 0.8),
        "tense": (-0.6, 0.7),
        "fearful": (-0.7, 0.6),
        "sad": (-0.8, -0.6),
        "melancholic": (-0.5, -0.4),
        "calm": (0.4, -0.6),
        "peaceful": (0.6, -0.7),
        "tender": (0.5, -0.3),
        "happy": (0.8, 0.6),
        "excited": (0.7, 0.9),
        "joyful": (0.9, 0.5),
    }

    def __init__(self, valence, arousal):
        self.valence = max(-1.0, min(1.0, float(valence)))
        self.arousal = max(-1.0, min(1.0, float(arousal)))

    def __repr__(self):
        return f"EmotionPoint(v={self.valence:+.2f}, a={self.arousal:+.2f})"

    @classmethod
    def from_label(cls, label):
        key = label.lower().strip()
        if key not in cls._map:
            raise ValueError(f"Unknown label {label}")
        v, a = cls._map[key]
        return cls(v, a)