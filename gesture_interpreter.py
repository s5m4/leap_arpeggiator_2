"""
Gesture Interpreter

Reads raw HandData from SharedState, applies smoothing and mapping,
writes musical parameters back to SharedState.

Key features:
  - Exponential moving average smoothing
  - Hysteresis for chord zone switching (prevents flickering)
  - Configurable mapping ranges
"""

import threading
import time
import math
import numpy as np

from shared_state import (
    SharedState, CHORD_GRID, CHORD_INTERVALS, chord_name_from_root,
)


class GestureInterpreter:
    """
    Maps hand tracking data to musical parameters.

    Right hand (palm XYZ position) → chord zone in 3×3×3 grid
    Left hand Y → base note
    Left hand roll → step count
    Left hand pitch → note spread
    """

    def __init__(self, state: SharedState, update_rate: float = 60.0):
        self.state = state
        self.update_rate = update_rate
        self._running = False
        self._thread = None

        # Smoothing factors (0 = no smoothing, 1 = frozen)
        self.smooth_position = 0.7
        self.smooth_angle = 0.6

        # Smoothed values
        self._smooth_right_x = 0.0
        self._smooth_right_y = 250.0
        self._smooth_right_z = 0.0
        self._smooth_left_y = 250.0
        self._smooth_left_roll = 0.0
        self._smooth_left_pitch = 0.0

        # Hysteresis: current zone with threshold
        self._current_zone = (1, 1, 1)  # (layer, row, col)
        self._hysteresis = 0.15  # fraction of zone width needed to switch

        # Mapping ranges (Leap Motion coordinate space, in mm)
        self.right_x_range = (-150, 150)   # left to right
        self.right_y_range = (150, 350)    # low to high
        self.right_z_range = (-150, 150)    # near to far
        self.left_y_range = (100, 400)      # low to high (above sensor)
        self.note_range = (36, 60)          # fallback note range (C2 to C4)

        # Left hand roll → step count mapping
        self.roll_range = (-0.8, 0.8)       # radians
        self.step_range = (1, 8)

        # Left hand pitch → note spread
        self.pitch_range = (-0.5, 0.5)
        self.spread_range = (0.5, 2.0)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _ema(self, old_val: float, new_val: float, alpha: float) -> float:
        """Exponential moving average."""
        return alpha * old_val + (1 - alpha) * new_val

    def _map_range(self, value: float, in_min: float, in_max: float,
                   out_min: float, out_max: float) -> float:
        """Linear map with clamping."""
        t = (value - in_min) / (in_max - in_min)
        t = max(0.0, min(1.0, t))
        return out_min + t * (out_max - out_min)

    def _get_zone_with_hysteresis(self, norm_x: float, norm_y: float, norm_z: float) -> tuple:
        """
        Map normalized (0-1) position to 3×3×3 grid zone.
        Uses hysteresis to prevent flickering at boundaries.
        """
        zone_size = 1.0 / 3.0
        threshold = self._hysteresis * zone_size

        def zone_axis(value, current):
            # Only switch if we've moved past the boundary + threshold
            if value < current * zone_size - threshold:
                return max(0, current - 1)
            elif value > (current + 1) * zone_size + threshold:
                return min(2, current + 1)
            return current

        new_layer = zone_axis(norm_y, self._current_zone[0])
        new_row = zone_axis(norm_z, self._current_zone[1])
        new_col = zone_axis(norm_x, self._current_zone[2])
        return (new_layer, new_row, new_col)

    def _loop(self):
        while self._running:
            snapshot = self.state.read()
            rh = snapshot.right_hand
            lh = snapshot.left_hand

            # ---- Right hand → chord zone ----
            if rh.visible:
                self._smooth_right_x = self._ema(
                    self._smooth_right_x, rh.palm_position[0], self.smooth_position
                )
                self._smooth_right_y = self._ema(
                    self._smooth_right_y, rh.palm_position[1], self.smooth_position
                )
                self._smooth_right_z = self._ema(
                    self._smooth_right_z, rh.palm_position[2], self.smooth_position
                )

                # Normalize to 0-1
                norm_x = self._map_range(
                    self._smooth_right_x,
                    self.right_x_range[0], self.right_x_range[1],
                    0.0, 1.0
                )
                norm_y = self._map_range(
                    self._smooth_right_y,
                    self.right_y_range[0], self.right_y_range[1],
                    0.0, 1.0
                )
                norm_z = self._map_range(
                    self._smooth_right_z,
                    self.right_z_range[0], self.right_z_range[1],
                    0.0, 1.0
                )

                self._current_zone = self._get_zone_with_hysteresis(norm_x, norm_y, norm_z)

            # ---- Left hand → base note, steps, spread ----
            if lh.visible:
                self._smooth_left_y = self._ema(
                    self._smooth_left_y, lh.palm_position[1], self.smooth_position
                )
                self._smooth_left_roll = self._ema(
                    self._smooth_left_roll, lh.roll, self.smooth_angle
                )
                self._smooth_left_pitch = self._ema(
                    self._smooth_left_pitch, lh.pitch, self.smooth_angle
                )

                # Height → base note (quantized to semitones)
                note_min = min(snapshot.note_min, snapshot.note_max)
                note_max = max(snapshot.note_min, snapshot.note_max)
                base_note_float = self._map_range(
                    self._smooth_left_y,
                    self.left_y_range[0], self.left_y_range[1],
                    note_min, note_max,
                )
                base_note = int(round(base_note_float))

                # Roll → step count
                step_float = self._map_range(
                    self._smooth_left_roll,
                    self.roll_range[0], self.roll_range[1],
                    self.step_range[0], self.step_range[1],
                )
                step_count = int(round(step_float))

                # Pitch → note spread
                spread = self._map_range(
                    self._smooth_left_pitch,
                    self.pitch_range[0], self.pitch_range[1],
                    self.spread_range[0], self.spread_range[1],
                )

                self.state.update(
                    base_note=base_note,
                    step_count=step_count,
                    note_spread=round(spread, 2),
                )

            # Keep chord naming dynamic to current left-hand root note.
            current_base_note = base_note if lh.visible else snapshot.base_note
            layer, row, col = self._current_zone
            chord_quality = CHORD_GRID[layer][row][col]
            chord_name = chord_name_from_root(current_base_note, chord_quality)
            self.state.update(
                chord_zone_layer=layer,
                chord_zone_row=row,
                chord_zone_col=col,
                chord_quality=chord_quality,
                chord_type=chord_name,
                chord_intervals=CHORD_INTERVALS.get(chord_quality, [0, 4, 7]),
            )

            time.sleep(1.0 / self.update_rate)
