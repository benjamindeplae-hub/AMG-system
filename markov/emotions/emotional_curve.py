from markov.emotions.emotion_point import EmotionPoint

class EmotionCurve:
    """
    Maps generation steps -> (valence, arousal) using linear interpolation between user-defined anchor points
 
    Accepts four region-definition styles which can be freely mixed:
    1. bar-based       add_bar_region(bar_start, bar_end, valence, arousal)
    2. segment-based   add_segment(name, valence, arousal)
    3. percent-based   add_percent_region(pct_start, pct_end, valence, arousal)
    4. label-based     add_label_region(label, pct_start, pct_end)
    """
    SEGMENT_ORDER = ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro"]

    def __init__(self, total_steps=100, steps_per_bar=4):
        self.total_steps = total_steps
        self.steps_per_bar = steps_per_bar
        # Internal list of (step_index, EmotionPoint) anchors, unsorted
        self._anchors: list[tuple[int, EmotionPoint]] = []
        # Named segments: list of (name, valence, arousal) in insertion order
        self._segments: list[tuple[str, float, float]] = []

    """
    Bar-based
    """
    def add_bar_region(self, bar_start, bar_end, valence, arousal):
        step_start = (bar_start - 1) * self.steps_per_bar
        step_end = bar_end * self.steps_per_bar - 1
        pt = EmotionPoint(valence, arousal)
        self._anchors.append((step_start, pt))
        self._anchors.append((step_end, pt))

    """
    Segment-based
    """
    def add_segment(self, name, valence, arousal):
        self._segments.append((name.lower(), valence, arousal))

    """
    Percent-based
    Set the emotion for a percentage range [pct_start, pct_end] ⊆ [0, 1]
    """
    def add_percent_region(self, pct_start, pct_end, valence, arousal):
        step_start = int(pct_start * (self.total_steps - 1))  # -1 because zero-based index
        step_end = int(pct_end * (self.total_steps - 1))  # -1 because zero-based index
        pt = EmotionPoint(valence, arousal)
        self._anchors.append((step_start, pt))
        self._anchors.append((step_end, pt))
        return self

    """
    Label-based
    """
    def add_label_region(self, label, pct_start, pct_end):
        pt = EmotionPoint.from_label(label)
        return self.add_percent_region(pct_start, pct_end, pt.valence, pt.arousal)
    
    """
    Query 
    """
    def get(self, step):
        anchors = self._all_anchors()
        if not anchors:
            return EmotionPoint(0.0, 0.0)

        anchors.sort(key=lambda x: x[0])

        # anchors = list[tuple[int, EmotionPoint]]
        # anchors[0] = tuple[int, EmotionPoint]
        # anchors[0][1] = EmotionPoint
        if step <= anchors[0][0]:
            return anchors[0][1]
        if step >= anchors[-1][0]:
            return anchors[-1][1]

        for i in range(len(anchors) - 1):
            # anchors = list[tuple[int, EmotionPoint]]
            # anchors[i] = tuple[int, EmotionPoint]
            # s0 = int, p0 = EmotionPoint
            s0, p0 = anchors[i]
            s1, p1 = anchors[i + 1]
            if s0 <= step <= s1:
                # Imagine s0 = 20 and s1 = 60, and step = 30
                # step - s0 = 30 - 20 = 10 -> how many steps past the start of the segment
                # s1 - s0 = 60 - 20 = 40 -> the total length of the segment
                # t = 10 / 40 = 0.25 -> 25% of the way through
                # So t answers: "within just this segment, how far along am I?"
                t = (step - s0) / (s1 - s0) if s1 > s0 else 0.0
                # Imagine p0.valence = 0.2 and p1.valence = 0.8, and t = 0.25:
                # p1.valence - p0.valence = 0.6 -> the total emotional distance to travel
                # t * 0.6 = 0.15 -> how far you've travelled given your position
                # p0.valence + 0.15 = 0.35 -> your current valence
                return EmotionPoint(
                    valence = p0.valence + t * (p1.valence - p0.valence),
                    arousal = p0.arousal + t * (p1.arousal - p0.arousal),
                )
        return anchors[-1][1]

    def timeline(self):
        return [self.get(i) for i in range(self.total_steps)]

    def _all_anchors(self):
        # Merge explicit anchors with any segments
        combined = list(self._anchors)
        # Explicit takes priority: always let _anchors override segment-derived ones at the same step
        explicit_steps = {step for step, _ in self._anchors}

        if self._segments:
            n = len(self._segments)
            for idx, (name, v, a) in enumerate(self._segments):
                pct_start = idx / n
                pct_end = (idx + 1) / n
                step_start = int(pct_start * (self.total_steps - 1))
                step_end = int(pct_end * (self.total_steps - 1))
                pt = EmotionPoint(v, a)
                if step_start not in explicit_steps:
                    combined.append((step_start, pt))
                if step_end not in explicit_steps:
                    combined.append((step_end, pt))
        return combined