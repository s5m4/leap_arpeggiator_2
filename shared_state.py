"""
Shared state between all modules.
Thread-safe via a simple lock.
"""

import threading
from dataclasses import dataclass, field
import numpy as np


# Chord quality grid (3x3x3): [layer][row][col]
# Right hand X/Y/Z selects (col/layer/row).
CHORD_GRID = [
    [
        ["maj", "min", "min"],
        ["maj", "7", "min"],
        ["dim", "sus4", "min"],
    ],
    [
        ["7", "min", "sus4"],
        ["maj", "maj", "7"],
        ["dim", "min", "maj"],
    ],
    [
        ["sus4", "7", "min"],
        ["min", "maj", "dim"],
        ["maj", "sus4", "7"],
    ],
]

# Colors for each zone (R, G, B) in [layer][row][col]
CHORD_COLORS = [
    [
        [(210, 70, 70), (210, 120, 70), (210, 170, 70)],
        [(70, 160, 110), (70, 145, 210), (100, 70, 210)],
        [(170, 70, 210), (210, 70, 170), (150, 150, 150)],
    ],
    [
        [(255, 80, 80), (255, 140, 80), (255, 200, 80)],
        [(80, 200, 120), (80, 180, 255), (120, 80, 255)],
        [(200, 80, 255), (255, 80, 200), (180, 180, 180)],
    ],
    [
        [(255, 120, 120), (255, 170, 120), (255, 220, 120)],
        [(120, 230, 160), (120, 210, 255), (160, 120, 255)],
        [(230, 120, 255), (255, 120, 230), (220, 220, 220)],
    ],
]

# MIDI note intervals keyed by chord quality
CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "dim": [0, 3, 6],
    "sus4": [0, 5, 7],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


QUALITY_SUFFIX = {
    "maj": "maj",
    "min": "m",
    "7": "7",
    "dim": "dim",
    "sus4": "sus4",
}


def midi_to_name(midi_note: int) -> str:
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def note_name_class(midi_note: int) -> str:
    return NOTE_NAMES[midi_note % 12]


def chord_name_from_root(root_midi_note: int, quality: str) -> str:
    root = note_name_class(root_midi_note)
    suffix = QUALITY_SUFFIX.get(quality, quality)
    return f"{root}{suffix}"


@dataclass
class HandData:
    """Raw hand tracking data from Leap Motion (or mock)."""
    palm_position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # mm
    palm_normal: np.ndarray = field(default_factory=lambda: np.array([0, -1, 0], dtype=float))
    palm_direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1], dtype=float))
    finger_positions: list = field(default_factory=lambda: [np.zeros(3) for _ in range(5)])
    visible: bool = False
    hand_type: str = "unknown"  # "left" or "right"

    # Derived angles (computed in gesture interpreter)
    roll: float = 0.0   # radians
    pitch: float = 0.0  # radians


@dataclass
class MusicalState:
    """The musical parameters derived from gesture interpretation."""
    # Chord selection (right hand)
    chord_zone_layer: int = 1
    chord_zone_row: int = 1
    chord_zone_col: int = 1
    chord_quality: str = "maj"
    chord_type: str = "Cmaj"
    chord_intervals: list = field(default_factory=lambda: [0, 4, 7])

    # Base note & arpeggio params (left hand)
    base_note: int = 60  # MIDI note (C4)
    note_min: int = 36   # C2
    note_max: int = 60   # C4
    step_count: int = 4
    note_spread: float = 1.0  # multiplier on intervals

    # Arpeggiator output
    current_step: int = 0
    current_note: int = 60
    is_playing: bool = True
    bpm: float = 120.0

    # Raw hand data for visualization
    right_hand: HandData = field(default_factory=HandData)
    left_hand: HandData = field(default_factory=HandData)


class SharedState:
    """Thread-safe wrapper around MusicalState."""

    def __init__(self):
        self._state = MusicalState()
        self._lock = threading.Lock()

    def read(self) -> MusicalState:
        """Return a snapshot (shallow copy ok for our dataclass)."""
        with self._lock:
            import copy
            return copy.copy(self._state)

    def update(self, **kwargs):
        """Update one or more fields."""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def update_right_hand(self, hand: HandData):
        with self._lock:
            self._state.right_hand = hand

    def update_left_hand(self, hand: HandData):
        with self._lock:
            self._state.left_hand = hand
