# 🎹 Leap Motion Arpeggiator

A hand-gesture-controlled arpeggiator using Leap Motion and a 3D web-based visualizer.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Leap Motion     │────▸│  Gesture Interpreter  │────▸│  Arpeggiator       │
│  (or Mock)       │     │  • smoothing (EMA)    │     │  Engine            │
│                  │     │  • hysteresis zones   │     │  • drift-comp timer│
│  leap_source.py  │     │  • range mapping      │     │  • MIDI output     │
└─────────────────┘     └──────────────────────┘     └────────────────────┘
        │                         │                           │
        └─────────────┬───────────┘                           │
                      ▼                                       │
              ┌──────────────┐                                │
              │ SharedState   │◂───────────────────────────────┘
              │ (thread-safe) │
              └──────┬───────┘
                     │
                     ▼
          ┌───────────────────┐
          │  Viser Visualizer  │
          │  • 3×3 chord cube  │
          │  • hand spheres    │
          │  • GUI sidebar     │
          │  • note trail      │
          │                    │
          │  http://localhost   │
          │  :8080              │
          └───────────────────┘
```

## Gesture Mapping

| Hand   | Gesture               | Musical Parameter          |
|--------|-----------------------|----------------------------|
| Right  | Palm XZ position      | Chord zone (3×3 grid)      |
| Left   | Palm height (Y)       | Base note (C2–C6)          |
| Left   | Wrist roll            | Arpeggio step count (1–8)  |
| Left   | Hand pitch            | Note interval spread       |

## Quick Start

### Without Leap Motion (mock data):
```bash
pip install viser numpy
python main.py
# Open http://localhost:8080
```

### With real Ultraleap device:
```bash
# 1. Install Ultraleap Gemini Hand Tracking Software
# 2. Set up leapc-python-bindings:
git clone https://github.com/ultraleap/leapc-python-bindings.git
cd leapc-python-bindings
pip install -r requirements.txt
pip install -e leapc-python-api

# 3. Run:
python main.py --real-leap
```

### With MIDI output:
```bash
pip install python-rtmidi
# Then uncomment the rtmidi sections in arpeggiator.py
```

## Files

| File                     | Purpose                                    |
|--------------------------|--------------------------------------------|
| `main.py`                | Entry point — wires everything together     |
| `shared_state.py`        | Thread-safe shared state + music constants  |
| `leap_source.py`         | Mock + real Leap Motion data sources        |
| `gesture_interpreter.py` | Smoothing, hysteresis, parameter mapping    |
| `arpeggiator.py`         | Note generation engine with precise timing  |
| `visualizer.py`          | Viser 3D scene + GUI                        |

## Next Steps

- [ ] Swap mock source for real Leap Motion
- [ ] Enable MIDI output (uncomment in arpeggiator.py)
- [ ] Add more chord voicings to the 3×3 grid
- [ ] Add arpeggio pattern modes (up, down, up-down, random)
- [ ] Use finger spread distance for velocity/dynamics
- [ ] Add a piano-roll style note history display
- [ ] Expand to 3×3×3 cube (use Y axis for chord families)
