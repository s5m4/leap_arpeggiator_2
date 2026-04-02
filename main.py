#!/usr/bin/env python3
"""
Leap Motion Arpeggiator — Main Entry Point

Wires together all modules:
  1. Leap Motion data source (mock or real)
  2. Gesture interpreter (smoothing + mapping)
  3. Arpeggiator engine (note generation)
  4. ModernGL 3D visualizer (native desktop window)

Usage:
  python main.py              # uses mock data (no Leap hardware needed)
  python main.py --real-leap  # uses real Ultraleap Gemini device
"""

import sys
import signal

from shared_state import SharedState
from leap_source import MockLeapSource, LeapMotionSource
from gesture_interpreter import GestureInterpreter
from arpeggiator import ArpeggiatorEngine
from visualizer import ArpeggiatorVisualizer


def main():
    use_real_leap = "--real-leap" in sys.argv

    print("=" * 60)
    print("  Leap Motion Arpeggiator")
    print("=" * 60)
    print()

    # 1. Shared state
    state = SharedState()

    # 2. Data source
    if use_real_leap:
        print("  > Using REAL Leap Motion device")
        source = LeapMotionSource(state)
    else:
        print("  > Using MOCK hand data (sine wave simulation)")
        print("    Tip: add --real-leap to use your Ultraleap device")
        source = MockLeapSource(state, update_rate=60)

    # 3. Gesture interpreter
    interpreter = GestureInterpreter(state, update_rate=60)

    # 4. Arpeggiator engine
    arpeggiator = ArpeggiatorEngine(state)

    # ---- Start everything ----
    print()
    source.start()
    print("  + Data source started")

    interpreter.start()
    print("  + Gesture interpreter started")

    arpeggiator.start()
    print("  + Arpeggiator engine started")

    print("  + Starting ModernGL visualizer...")

    # Handle Ctrl+C gracefully
    def shutdown(sig, frame):
        print("\n\n  Shutting down...")
        arpeggiator.stop()
        interpreter.stop()
        source.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Visualizer runs the main loop (blocking)
    try:
        ArpeggiatorVisualizer.run(state)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


if __name__ == "__main__":
    main()
