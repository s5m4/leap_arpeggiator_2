  # CLAUDE.md

  This file provides guidance to Claude Code (claude.ai/code) when working with code in this
  repository.

  ## Project Overview

  A hand-gesture-controlled arpeggiator using Leap Motion hand tracking with a 3D web-based visualizer 
  (Viser). Maps hand position and orientation to musical parameters, generating MIDI arpeggios in      
  real-time. Python 3.12, uses a venv at `./venv/`.

  ## Running

  ```bash
  source venv/Scripts/activate   # Windows
  python main.py                 # Mock mode (no hardware)
  python main.py --real-leap     # Real Ultraleap Gemini device
  python main.py --port=9000     # Custom port
  # Opens at http://localhost:8080

  MIDI output requires pip install python-rtmidi and uncommenting rtmidi sections in arpeggiator.py.   

  Architecture

  Four threads communicate through a thread-safe SharedState (lock-protected):

  1. Data Source (leap_source.py) — MockLeapSource (sine-wave simulation) or LeapMotionSource (real    
  device via LeapC bindings in leapc-python-bindings/). Pushes HandData into SharedState at ~60 Hz.    
  2. Gesture Interpreter (gesture_interpreter.py) — Reads raw hand data, applies EMA smoothing
  (position α=0.7, angles α=0.6) and hysteresis (15% zone width) to prevent flickering. Maps right hand   XYZ → 3×3×3 chord grid zone, left hand height → base note, left roll → step count, left pitch →     
  interval spread.
  3. Arpeggiator Engine (arpeggiator.py) — Reads chord intervals and musical params from SharedState.  
  Generates note sequences with drift-compensated timing. Updates current_note/current_step in
  SharedState.
  4. Visualizer (visualizer.py, main thread) — Viser 3D web server rendering chord cube, hand spheres, 
  note trail, and GUI sidebar with BPM/range controls. GUI updates throttled to 100ms.

  Data flow: LeapSource → SharedState → GestureInterpreter → SharedState → ArpeggiatorEngine +
  Visualizer

  Key Files

  ┌────────────────────────┬────────────────────────────────────────────────────────────────────────┐  
  │          File          │                                Purpose                                 │  
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤  
  │ main.py                │ Entry point, wires modules, handles Ctrl+C shutdown                    │  
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤  
  │ shared_state.py        │ SharedState, HandData, MusicalState dataclasses, CHORD_GRID,           │  
  │                        │ CHORD_INTERVALS, CHORD_COLORS, NOTE_NAMES                              │  
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤  
  │ leap_source.py         │ Mock and real Leap Motion data sources                                 │  
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤  
  │ gesture_interpreter.py │ EMA smoothing, hysteresis, gesture-to-parameter mapping                │  
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤  
  │ arpeggiator.py         │ Note generation engine with precise timing                             │  
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤  
  │ visualizer.py          │ Viser 3D scene, GUI sidebar, note trail                                │  
  └────────────────────────┴────────────────────────────────────────────────────────────────────────┘  

  Threading Rules

  - Never modify SharedState fields directly — use state.update(**kwargs), state.update_right_hand(),  
  state.update_left_hand()
  - state.read() returns a shallow snapshot safe to use without holding the lock
  - All worker threads are daemon threads; they exit when the main (visualizer) thread exits

  Dependencies

  Core: viser, numpy, scipy, trimesh. Bundled: leapc-python-bindings/ (Ultraleap CFFI bindings with    
  LeapC.dll). Optional: python-rtmidi.