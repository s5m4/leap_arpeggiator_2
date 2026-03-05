"""
Leap Motion data source.

Provides:
  - MockLeapSource: simulates two hands with sine-wave movement (no hardware needed)
  - LeapMotionSource: real Ultraleap Gemini bindings (requires SDK + device)

Both run in their own thread and push HandData into SharedState.
"""

import threading
import time
import math
import sys
from pathlib import Path
import numpy as np

from shared_state import HandData, SharedState


def _import_leap_with_fallback():
    """
    Import `leap`, falling back to local vendored binding paths in this repo.
    """
    try:
        import leap
        return leap
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent
    candidate_dirs = [
        repo_root / "leapc-python-bindings" / "leapc-python-api" / "src",
        repo_root / "leapc-python-bindings" / "leapc-cffi" / "src",
        repo_root / "leapc-python-api" / "src",
        repo_root / "leapc-cffi" / "src",
    ]

    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    try:
        import leap
        return leap
    except ImportError as exc:
        searched = "\n".join(f"  - {path}" for path in candidate_dirs)
        raise ImportError(
            "Could not import 'leap'. Make sure you have:\n"
            "  1. Ultraleap Gemini Hand Tracking Software installed\n"
            "  2. leapc-python-bindings installed, OR use the bundled copy in this repo\n"
            "     (pip install -e leapc-python-bindings/leapc-python-api)\n"
            "  3. If needed, also install cffi layer:\n"
            "     (pip install -e leapc-python-bindings/leapc-cffi)\n"
            "Searched local binding paths:\n"
            f"{searched}\n"
            "See: https://github.com/ultraleap/leapc-python-bindings"
        ) from exc


class MockLeapSource:
    """
    Generates fake hand data using sine waves.
    Right hand sweeps across the chord cube.
    Left hand moves up/down and tilts.
    """

    def __init__(self, state: SharedState, update_rate: float = 60.0):
        self.state = state
        self.update_rate = update_rate
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        t0 = time.time()
        while self._running:
            t = time.time() - t0

            # ---- Right hand: sweeps across XZ plane ----
            # Leap coordinate space: X=left/right, Y=up, Z=toward/away
            # We map to a cube from roughly -150..150 on X, 150..350 on Y, -150..150 on Z
            rx = 100 * math.sin(t * 0.3)        # slow left-right sweep
            ry = 250 + 50 * math.sin(t * 0.5)   # gentle vertical bob
            rz = 80 * math.sin(t * 0.2)          # slow forward-back

            right_hand = HandData(
                palm_position=np.array([rx, ry, rz]),
                palm_normal=np.array([0, -1, 0], dtype=float),
                palm_direction=np.array([0, 0, -1], dtype=float),
                finger_positions=[
                    np.array([rx - 40, ry + 30, rz - 20]),  # thumb
                    np.array([rx - 20, ry + 50, rz - 40]),  # index
                    np.array([rx, ry + 55, rz - 45]),        # middle
                    np.array([rx + 20, ry + 50, rz - 40]),  # ring
                    np.array([rx + 35, ry + 40, rz - 30]),  # pinky
                ],
                visible=True,
                hand_type="right",
            )

            # ---- Left hand: height and roll/pitch ----
            lx = -120 + 30 * math.sin(t * 0.15)
            ly = 200 + 100 * math.sin(t * 0.25)  # bigger vertical range
            lz = 20 * math.sin(t * 0.1)

            roll = 0.5 * math.sin(t * 0.4)    # ±0.5 rad
            pitch = 0.3 * math.sin(t * 0.35)  # ±0.3 rad

            left_hand = HandData(
                palm_position=np.array([lx, ly, lz]),
                palm_normal=np.array([math.sin(roll), -math.cos(roll), 0], dtype=float),
                palm_direction=np.array([0, math.sin(pitch), -math.cos(pitch)], dtype=float),
                finger_positions=[
                    np.array([lx + 40, ly + 30, lz - 20]),
                    np.array([lx + 20, ly + 50, lz - 40]),
                    np.array([lx, ly + 55, lz - 45]),
                    np.array([lx - 20, ly + 50, lz - 40]),
                    np.array([lx - 35, ly + 40, lz - 30]),
                ],
                visible=True,
                hand_type="left",
                roll=roll,
                pitch=pitch,
            )

            self.state.update_right_hand(right_hand)
            self.state.update_left_hand(left_hand)

            time.sleep(1.0 / self.update_rate)


class LeapMotionSource:
    """
    Real Ultraleap Gemini LeapC Python bindings.

    Requires:
      - Ultraleap Gemini Hand Tracking Software installed
      - leapc-python-bindings set up (pip install -e leapc-python-api)

    To use: replace MockLeapSource with LeapMotionSource in main.py
    """

    def __init__(self, state: SharedState):
        self.state = state
        self._running = False
        self._thread = None

    def start(self):
        _import_leap_with_fallback()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        leap = _import_leap_with_fallback()

        class Listener(leap.Listener):
            def __init__(self, state):
                self.state = state

            def on_tracking_event(self, event):
                for hand in event.hands:
                    palm = hand.palm

                    # Extract finger tip positions
                    finger_positions = []
                    for digit in hand.digits:
                        tip = digit.distal.next_joint
                        finger_positions.append(
                            np.array([tip.x, tip.y, tip.z])
                        )

                    # Compute roll and pitch from palm normal/direction
                    normal = np.array([palm.normal.x, palm.normal.y, palm.normal.z])
                    direction = np.array([palm.direction.x, palm.direction.y, palm.direction.z])
                    roll_val = math.atan2(normal[0], -normal[1])
                    pitch_val = math.atan2(direction[1], -direction[2])

                    hand_data = HandData(
                        palm_position=np.array([palm.position.x, palm.position.y, palm.position.z]),
                        palm_normal=normal,
                        palm_direction=direction,
                        finger_positions=finger_positions,
                        visible=True,
                        hand_type="left" if hand.type == leap.HandType.Left else "right",
                        roll=roll_val,
                        pitch=pitch_val,
                    )

                    if hand_data.hand_type == "left":
                        self.state.update_left_hand(hand_data)
                    else:
                        self.state.update_right_hand(hand_data)

        connection = leap.Connection()
        listener = Listener(self.state)
        connection.add_listener(listener)

        with connection.open():
            while self._running:
                time.sleep(0.01)
