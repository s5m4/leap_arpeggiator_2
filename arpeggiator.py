"""
Arpeggiator Engine

Generates note events based on musical parameters from SharedState.
Uses precise timing with drift compensation.

In this version: prints notes to console.
For real MIDI output, swap in python-rtmidi (see comments below).
"""

import threading
import time

from shared_state import SharedState, midi_to_name


class ArpeggiatorEngine:
    """
    Reads chord, base_note, step_count, and spread from SharedState.
    Cycles through arpeggio notes at the specified BPM.

    Currently outputs to console. To add real MIDI:
      pip install python-rtmidi
      Then uncomment the rtmidi sections below.
    """

    def __init__(self, state: SharedState):
        self.state = state
        self._running = False
        self._thread = None
        self._midi_out = None

    def start(self):
        # ---- Optional: real MIDI output ----
        # try:
        #     import rtmidi
        #     self._midi_out = rtmidi.MidiOut()
        #     # Open a virtual port (macOS/Linux) or first available port (Windows)
        #     if self._midi_out.get_ports():
        #         self._midi_out.open_port(0)
        #         print(f"  MIDI output: {self._midi_out.get_ports()[0]}")
        #     else:
        #         self._midi_out.open_virtual_port("LeapArpeggiator")
        #         print("  MIDI output: virtual port 'LeapArpeggiator'")
        # except ImportError:
        #     print("  No python-rtmidi found. Running without MIDI output.")
        #     self._midi_out = None

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        # if self._midi_out:
        #     self._midi_out.close_port()

    def _build_arpeggio(self, base_note: int, intervals: list,
                         step_count: int, spread: float) -> list:
        """
        Build a list of MIDI notes for the arpeggio.

        Args:
            base_note: root MIDI note
            intervals: chord intervals [0, 4, 7] etc.
            step_count: how many steps to produce
            spread: multiplier on intervals (1.0 = normal, 2.0 = double spacing)
        """
        notes = []
        octave = 0
        idx = 0
        for step in range(step_count):
            interval = intervals[idx % len(intervals)]
            note = base_note + int(interval * spread) + (octave * 12)
            note = max(0, min(127, note))  # clamp to MIDI range
            notes.append(note)
            idx += 1
            if idx >= len(intervals):
                idx = 0
                octave += 1
        return notes

    def _send_note_on(self, note: int, velocity: int = 100):
        """Send MIDI note on."""
        if self._midi_out:
            self._midi_out.send_message([0x90, note, velocity])

    def _send_note_off(self, note: int):
        """Send MIDI note off."""
        if self._midi_out:
            self._midi_out.send_message([0x80, note, 0])

    def _loop(self):
        """Main arpeggiator loop with drift-compensated timing."""
        step = 0
        last_note = -1

        while self._running:
            snap = self.state.read()
            bpm = snap.bpm
            step_duration = 60.0 / bpm / 2  # 8th notes

            # Build current arpeggio
            notes = self._build_arpeggio(
                snap.base_note,
                snap.chord_intervals,
                snap.step_count,
                snap.note_spread,
            )

            if not notes:
                time.sleep(0.01)
                continue

            # Get current note
            current_note = notes[step % len(notes)]

            # Note off for previous
            if last_note >= 0 and last_note != current_note:
                self._send_note_off(last_note)

            # Note on
            self._send_note_on(current_note)
            last_note = current_note

            # Update state for visualization
            self.state.update(
                current_step=step % len(notes),
                current_note=current_note,
            )

            step = (step + 1) % max(1, snap.step_count)

            # Precise sleep with drift compensation
            next_time = time.perf_counter() + step_duration
            while time.perf_counter() < next_time:
                remaining = next_time - time.perf_counter()
                if remaining > 0.001:
                    time.sleep(remaining * 0.8)

        # Clean up: note off
        if last_note >= 0:
            self._send_note_off(last_note)
