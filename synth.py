"""
Simple sine-wave synthesizer using sounddevice.

Provides a non-blocking audio stream that plays a single sine tone.
Call note_on(midi_note) to change pitch, note_off() to silence.
"""

import math
import threading
import numpy as np
import sounddevice as sd


def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


class SineSynth:
    """Mono sine oscillator with simple ADSR-ish envelope (attack + release)."""

    def __init__(self, sample_rate: int = 44100, volume: float = 0.3):
        self.sr = sample_rate
        self.volume = volume

        self._freq = 0.0
        self._phase = 0.0
        self._target_amp = 0.0    # 0 or 1
        self._current_amp = 0.0   # smoothed
        self._attack = 0.005      # seconds
        self._release = 0.04      # seconds
        self._lock = threading.Lock()
        self._stream = None

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=256,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def note_on(self, midi_note: int):
        with self._lock:
            self._freq = midi_to_freq(midi_note)
            self._target_amp = 1.0

    def note_off(self):
        with self._lock:
            self._target_amp = 0.0

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            freq = self._freq
            target = self._target_amp

        samples = np.empty(frames, dtype=np.float32)
        phase = self._phase
        amp = self._current_amp
        sr = self.sr
        vol = self.volume

        # Per-sample to get smooth envelope
        phase_inc = 2.0 * math.pi * freq / sr if freq > 0 else 0.0

        # Envelope smoothing rate (per sample)
        if target > amp:
            rate = 1.0 / max(1, self._attack * sr)
        else:
            rate = 1.0 / max(1, self._release * sr)

        for i in range(frames):
            amp += (target - amp) * rate
            samples[i] = math.sin(phase) * amp * vol
            phase += phase_inc

        # Keep phase in [0, 2pi) to avoid float drift
        phase %= 2.0 * math.pi

        self._phase = phase
        self._current_amp = amp
        outdata[:, 0] = samples
