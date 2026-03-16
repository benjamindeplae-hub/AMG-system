import os
import re
import numpy as np
import pretty_midi
from itertools import islice

"""
Reason pitch % 12:
which of the 12 semitones in an octave the note belongs to, regardless of which octave it's in
So C4 and C5 both have pitch class 0 if C is the root
60 % 12 = 0 means C4 is C, and 72 % 12 = 0 means C5 is also C, just an octave higher
Semitone 0    1    2    3    4    5    6    7    8    9    10    11
Note     C    C#   D    D#   E    F    F#   G    G#   A    A#    B
EU       Do   Do#  Re   Re#  Mi   Fa   Fa#  Sol  Sol# La   La#   Si
"""

def find_midis(midi_folder):
    for root, _dirs, filenames in os.walk(midi_folder):
        for f in filenames:
            if f.endswith((".midi", ".mid")):
                yield os.path.join(root, f)

def extract_notes_from_midi(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes.append(note)
    return notes

def fast_detect_key(notes):
    pitches = np.array([note.pitch for note in notes]) % 12
    pc_counts = np.bincount(pitches, minlength=12)
    # Simple major/minor decision based on I-IV-V vs vi-ii-iii
    if pc_counts[0] + pc_counts[4] + pc_counts[7] > pc_counts[9] + pc_counts[2] + pc_counts[5]:
        return 'major'
    else:
        return 'minor'

def melody_to_midi(sequence, filepath):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    current_time = 0.0
    for delta, duration, chord in sequence:
        current_time += delta
        for pitch_class in chord:
            pitch = pitch_class + 60  # center around C4
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            piano.notes.append(note)
    midi.instruments.append(piano)
    midi.write(filepath)

def group_simultaneous_notes(notes, threshold=0.03):
    groups = []
    current_group = [notes[0]]
    for note in notes[1:]:
        if abs(note.start - current_group[0].start) < threshold:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)
    return groups

"""
extract_delta_chord_sequence
"""
def extract_delta_chord_sequence(notes, time_quant=0.25, time_tol=0.03):
    groups = group_simultaneous_notes(notes, time_tol)

    chords = []
    for group in groups:
        start_time = group[0].start
        pitches = tuple(sorted(set(n.pitch % 12 for n in group[:5])))
        chords.append((start_time, pitches))

    delta_events = []
    last_onset = None
    for start_time, chord in chords:
        onset_q = max(0, round(start_time / time_quant) * time_quant)
        delta = 0 if last_onset is None else onset_q - last_onset
        delta_events.append((delta, chord))
        last_onset = onset_q

    return delta_events

"""
extract_chord_sequence
"""
def extract_chord_sequence(delta_sequence):
    return [chord for (_delta, chord) in delta_sequence]

"""
extract_rhythm_sequence
"""
def extract_tempo(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    tempos, _ = midi.get_tempo_changes()
    valid = tempos[tempos > 0]
    return float(np.median(valid)) if len(valid) > 0 else 120.0

def beats_to_rhythmic_token(delta_beats, dur_beats):
    """
    Represent timing as (numerator, denominator) fractions of a beat.
    e.g. a dotted quarter = (3, 2), eighth = (1, 2), quarter = (1, 1)
    """
    GRID = [0.125, 0.25, 0.333, 0.375, 0.5, 0.667, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    def snap(val):
        return min(GRID, key=lambda g: abs(g - val))
    
    return (snap(delta_beats), snap(dur_beats))

def extract_rhythm_sequence(notes, midi_path, time_tol=0.02):
    if not notes:
        return []

    tempo = extract_tempo(midi_path)
    beat_duration = 60.0 / tempo  # seconds per beat

    notes = sorted(notes, key=lambda n: n.start)
    groups = group_simultaneous_notes(notes, time_tol)

    rhythm_tokens = []
    last_onset_beats = None

    for group in groups:
        onset_beats = group[0].start / beat_duration
        dur_beats   = max(n.end - n.start for n in group) / beat_duration

        delta_beats = 0.0 if last_onset_beats is None else onset_beats - last_onset_beats
        last_onset_beats = onset_beats

        token = beats_to_rhythmic_token(delta_beats, dur_beats)
        rhythm_tokens.append(token)

    return rhythm_tokens
