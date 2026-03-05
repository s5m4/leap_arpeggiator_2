"""
Viser 3D Visualization for Leap Motion Arpeggiator

Renders:
    - A semi-transparent 3×3×3 chord selection cube (right hand zone)
  - Spheres for palm positions of both hands
  - Small spheres for fingertips
  - Labels showing chord names in each zone
  - Active zone highlighting
  - GUI sidebar with real-time musical parameters
  - A "note trail" showing recently played notes
"""

import time
import math
import numpy as np

import viser

from shared_state import (
    SharedState, CHORD_GRID, CHORD_COLORS,
    midi_to_name, chord_name_from_root,
)


class ArpeggiatorVisualizer:
    """
    Creates and maintains the viser 3D scene.
    Reads from SharedState and updates scene objects each frame.
    """

    # Cube dimensions in scene units (we scale Leap mm → scene units)
    CUBE_SIZE = 3.0        # total cube edge length
    ZONE_SIZE = 1.0        # each zone is 1×1×1 (3×3×3 = 3)
    SCALE = 3.0 / 300.0   # 300mm Leap range → 3 scene units
    Y_OFFSET = -2.5        # shift Leap Y=250 to scene center

    def __init__(self, state: SharedState, port: int = 8080):
        self.state = state
        self.port = port
        self.server = None

        # Scene handles (set during build)
        self._zone_handles = {}     # (layer, row, col) → mesh handle
        self._label_handles = {}    # (layer, row, col) → label handle
        self._right_palm = None
        self._left_palm = None
        self._right_fingers = []
        self._left_fingers = []
        self._note_trail = []
        self._hud_base_note = None

        # GUI handles
        self._gui_chord = None
        self._gui_base_note = None
        self._gui_steps = None
        self._gui_spread = None
        self._gui_bpm = None
        self._gui_note_min = None
        self._gui_note_max = None
        self._gui_fps = None
        self._gui_perf_mode = None
        self._gui_show_fingers = None
        self._gui_show_trail = None
        self._gui_current_note = None

        # Active zone tracking for highlight
        self._active_zone = (-1, -1, -1)
        self._last_base_opacity = None
        self._last_label_base_note = None
        self._last_label_layer = None
        self._last_gui_update_time = 0.0
        self._gui_update_period = 0.1  # seconds
        self._last_controls = None
        self._last_show_fingers = None
        self._last_show_trail = None
        self._last_perf_mode = None

    def start(self):
        """Build the viser server, scene, and GUI. Then loop forever updating."""
        self.server = viser.ViserServer(port=self.port)
        self.server.scene.set_up_direction("+y")

        self._build_scene()
        self._build_gui()

        print(f"\n  ✦ Viser running at: http://localhost:{self.port}")
        print(f"    Open this URL in your browser to see the 3D visualization.\n")

        self._update_loop()

    def _leap_to_scene(self, pos: np.ndarray) -> tuple:
        """Convert Leap Motion mm coordinates to viser scene coordinates."""
        x = pos[0] * self.SCALE
        y = (pos[1] - 250) * self.SCALE  # center around Leap Y=250
        z = pos[2] * self.SCALE
        return (x, y, z)

    def _build_scene(self):
        """Create the static and dynamic scene objects."""
        s = self.server

        # ---- Grid floor ----
        s.scene.add_grid(
            "/grid",
            width=8.0,
            height=8.0,
            position=(0.0, -2.0, 0.0),
        )

        # ---- Chord cube zones (3×3×3) ----
        half = self.CUBE_SIZE / 2.0
        zone = self.ZONE_SIZE

        for layer in range(3):
            for row in range(3):
                for col in range(3):
                    # Position: center of each zone cell
                    cx = -half + zone * 0.5 + col * zone
                    cy = -half + zone * 0.5 + layer * zone
                    cz = -half + zone * 0.5 + row * zone

                    color_rgb = CHORD_COLORS[layer][row][col]
                    name = f"/cube/zone_{layer}_{row}_{col}"

                    handle = s.scene.add_box(
                        name=name,
                        dimensions=(zone * 0.92, zone * 0.92, zone * 0.92),
                        color=color_rgb,
                        position=(cx, cy, cz),
                        opacity=0.12,
                    )
                    self._zone_handles[(layer, row, col)] = handle

                    # Label for each cube (dynamic text updated in loop)
                    quality = CHORD_GRID[layer][row][col]
                    chord_name = chord_name_from_root(60, quality)
                    label_name = f"/cube/label_{layer}_{row}_{col}"
                    label_handle = s.scene.add_label(
                        label_name,
                        text=chord_name,
                        position=(cx, cy, cz),
                    )
                    self._label_handles[(layer, row, col)] = label_handle

        self._hud_base_note = s.scene.add_label(
            "/hud/base_note",
            text="Base: C4",
            position=(0.0, 2.2, 0.0),
        )

        # ---- Cube wireframe outline ----
        # Draw edges of the overall cube
        corners = []
        for x in [-half, half]:
            for y in [-0.2, 0.2]:
                for z in [-half, half]:
                    corners.append([x, y, z])

        # ---- Hand palm spheres ----
        self._right_palm = s.scene.add_icosphere(
            "/hands/right_palm",
            radius=0.15,
            color=(255, 100, 60),
            position=(0.5, 0, 0),
        )

        self._left_palm = s.scene.add_icosphere(
            "/hands/left_palm",
            radius=0.15,
            color=(60, 140, 255),
            position=(-0.5, 0, 0),
        )

        # ---- Fingertip spheres ----
        for i in range(5):
            rh = s.scene.add_icosphere(
                f"/hands/right_finger_{i}",
                radius=0.06,
                color=(255, 160, 120),
                position=(0.5, 0, 0),
            )
            self._right_fingers.append(rh)

            lh = s.scene.add_icosphere(
                f"/hands/left_finger_{i}",
                radius=0.06,
                color=(120, 170, 255),
                position=(-0.5, 0, 0),
            )
            self._left_fingers.append(lh)

        # ---- Note trail (small spheres showing recent arpeggio notes) ----
        for i in range(10):
            nh = s.scene.add_icosphere(
                f"/notes/trail_{i}",
                radius=0.04,
                color=(255, 255, 100),
                position=(0, -5, 0),  # hidden initially
                opacity=0.0,
            )
            self._note_trail.append(nh)

    def _build_gui(self):
        """Create the sidebar GUI with musical parameter readouts."""
        s = self.server

        s.gui.add_markdown("## 🎵 Leap Arpeggiator")

        with s.gui.add_folder("Musical State"):
            self._gui_chord = s.gui.add_text(
                "Chord",
                initial_value="G7",
                disabled=True,
            )
            self._gui_base_note = s.gui.add_text(
                "Base Note",
                initial_value="C4",
                disabled=True,
            )
            self._gui_steps = s.gui.add_number(
                "Steps",
                initial_value=4,
                disabled=True,
            )
            self._gui_spread = s.gui.add_number(
                "Spread",
                initial_value=1.0,
                disabled=True,
                step=0.01,
            )
            self._gui_current_note = s.gui.add_text(
                "Playing",
                initial_value="—",
                disabled=True,
            )

        with s.gui.add_folder("Settings"):
            self._gui_bpm = s.gui.add_slider(
                "BPM",
                min=40,
                max=240,
                step=1,
                initial_value=120,
            )
            self._gui_fps = s.gui.add_slider(
                "Visual FPS",
                min=10,
                max=30,
                step=1,
                initial_value=18,
            )
            self._gui_note_min = s.gui.add_slider(
                "Note Min",
                min=24,
                max=84,
                step=1,
                initial_value=36,
            )
            self._gui_note_max = s.gui.add_slider(
                "Note Max",
                min=24,
                max=84,
                step=1,
                initial_value=60,
            )
            self._gui_perf_mode = s.gui.add_checkbox(
                "Performance Mode",
                initial_value=True,
            )
            self._gui_show_fingers = s.gui.add_checkbox(
                "Show Fingertips",
                initial_value=False,
            )
            self._gui_show_trail = s.gui.add_checkbox(
                "Show Note Trail",
                initial_value=False,
            )
            self._gui_opacity = s.gui.add_slider(
                "Zone Opacity",
                min=0.05,
                max=0.5,
                step=0.01,
                initial_value=0.08,
            )

        with s.gui.add_folder("Hand Tracking"):
            self._gui_right_pos = s.gui.add_text(
                "Right Palm",
                initial_value="—",
                disabled=True,
            )
            self._gui_left_pos = s.gui.add_text(
                "Left Palm",
                initial_value="—",
                disabled=True,
            )
            self._gui_left_roll = s.gui.add_number(
                "Left Roll°",
                initial_value=0.0,
                disabled=True,
                step=0.1,
            )
            self._gui_left_pitch = s.gui.add_number(
                "Left Pitch°",
                initial_value=0.0,
                disabled=True,
                step=0.1,
            )

        s.gui.add_markdown(
            "---\n"
            "**Controls:**\n"
            "- Right hand XYZ → chord zone\n"
            "- Left hand height → base note\n"
            "- Left hand roll → step count\n"
            "- Left hand pitch → note spread"
        )

    def _update_loop(self):
        """Main render loop — reads state and updates scene + GUI."""
        note_trail_idx = 0
        last_note = -1
        last_step = -1

        while True:
            snap = self.state.read()
            now = time.perf_counter()
            gui_due = (now - self._last_gui_update_time) >= self._gui_update_period

            # ---- Update BPM from GUI slider ----
            note_min = int(self._gui_note_min.value)
            note_max = int(self._gui_note_max.value)

            if note_min >= note_max:
                if note_min < 84:
                    note_max = note_min + 1
                    self._gui_note_max.value = note_max
                else:
                    note_min = 83
                    note_max = 84
                    self._gui_note_min.value = note_min
                    self._gui_note_max.value = note_max

            controls = (self._gui_bpm.value, note_min, note_max)
            if controls != self._last_controls:
                self.state.update(
                    bpm=self._gui_bpm.value,
                    note_min=note_min,
                    note_max=note_max,
                )
                self._last_controls = controls

            perf_mode = bool(self._gui_perf_mode.value)
            show_fingers = bool(self._gui_show_fingers.value)
            show_trail = bool(self._gui_show_trail.value)
            perf_mode_changed = self._last_perf_mode is None or self._last_perf_mode != perf_mode

            if self._last_show_fingers is None or self._last_show_fingers != show_fingers:
                finger_opacity = 1.0 if show_fingers else 0.0
                for fh in self._right_fingers:
                    fh.opacity = finger_opacity
                for fh in self._left_fingers:
                    fh.opacity = finger_opacity
                self._last_show_fingers = show_fingers

            if self._last_show_trail is None or self._last_show_trail != show_trail:
                trail_opacity = 0.6 if show_trail else 0.0
                for nh in self._note_trail:
                    nh.opacity = trail_opacity if show_trail else 0.0
                self._last_show_trail = show_trail

            # ---- Update zone highlighting ----
            new_zone = (snap.chord_zone_layer, snap.chord_zone_row, snap.chord_zone_col)
            base_opacity = self._gui_opacity.value
            zone_changed = new_zone != self._active_zone
            opacity_changed = self._last_base_opacity is None or abs(base_opacity - self._last_base_opacity) > 1e-6

            if zone_changed:
                # Dim old zone
                if self._active_zone in self._zone_handles:
                    old_h = self._zone_handles[self._active_zone]
                    old_layer, old_row, old_col = self._active_zone
                    old_h.opacity = base_opacity
                    old_h.color = CHORD_COLORS[old_layer][old_row][old_col]

                # Highlight new zone
                if new_zone in self._zone_handles:
                    new_h = self._zone_handles[new_zone]
                    new_h.opacity = min(0.7, base_opacity * 3.0)
                    # Brighten the color
                    r, g, b = CHORD_COLORS[new_zone[0]][new_zone[1]][new_zone[2]]
                    new_h.color = (
                        min(255, r + 80),
                        min(255, g + 80),
                        min(255, b + 80),
                    )

                self._active_zone = new_zone

            if zone_changed or opacity_changed or perf_mode_changed:
                # Also update non-active zone opacities if active zone/layer or slider changed
                for zone_key, handle in self._zone_handles.items():
                    if zone_key == self._active_zone:
                        continue
                    if perf_mode:
                        if zone_key[0] == new_zone[0]:
                            handle.opacity = base_opacity * 0.55
                        else:
                            handle.opacity = 0.0
                    elif zone_key[0] == new_zone[0]:
                        handle.opacity = min(0.25, base_opacity * 1.35)
                    else:
                        handle.opacity = base_opacity * 0.55
                self._last_base_opacity = base_opacity

            # ---- Dynamic chord labels from current root note ----
            labels_need_update = (
                self._last_label_base_note != snap.base_note
                or self._last_label_layer != new_zone[0]
                or zone_changed
                or perf_mode_changed
            )

            if labels_need_update:
                for (layer, row, col), label_handle in self._label_handles.items():
                    quality = CHORD_GRID[layer][row][col]
                    chord_label = chord_name_from_root(snap.base_note, quality)
                    if (layer, row, col) == new_zone:
                        label_handle.text = chord_label
                    elif (not perf_mode) and layer == new_zone[0]:
                        label_handle.text = chord_label
                    else:
                        label_handle.text = ""
                self._last_label_base_note = snap.base_note
                self._last_label_layer = new_zone[0]
                self._last_perf_mode = perf_mode

            # ---- Update hand positions ----
            rh = snap.right_hand
            lh = snap.left_hand

            if rh.visible:
                rpos = self._leap_to_scene(rh.palm_position)
                self._right_palm.position = rpos
                if show_fingers:
                    for i, fh in enumerate(self._right_fingers):
                        if i < len(rh.finger_positions):
                            fh.position = self._leap_to_scene(rh.finger_positions[i])

            if lh.visible:
                lpos = self._leap_to_scene(lh.palm_position)
                self._left_palm.position = lpos
                if show_fingers:
                    for i, fh in enumerate(self._left_fingers):
                        if i < len(lh.finger_positions):
                            fh.position = self._leap_to_scene(lh.finger_positions[i])

            # ---- Update GUI readouts ----
            if gui_due:
                self._gui_chord.value = snap.chord_type
                self._gui_base_note.value = midi_to_name(snap.base_note)
                self._gui_steps.value = snap.step_count
                self._gui_spread.value = snap.note_spread

                if rh.visible:
                    self._gui_right_pos.value = (
                        f"({rh.palm_position[0]:.0f}, {rh.palm_position[1]:.0f}, {rh.palm_position[2]:.0f})"
                    )
                if lh.visible:
                    self._gui_left_pos.value = (
                        f"({lh.palm_position[0]:.0f}, {lh.palm_position[1]:.0f}, {lh.palm_position[2]:.0f})"
                    )
                    self._gui_left_roll.value = round(math.degrees(lh.roll), 1)
                    self._gui_left_pitch.value = round(math.degrees(lh.pitch), 1)

                if self._hud_base_note is not None:
                    self._hud_base_note.text = f"Base: {midi_to_name(snap.base_note)}"
                self._last_gui_update_time = now

            # ---- Note trail: show a breadcrumb for each arpeggiator step ----
            if snap.current_step != last_step:
                last_step = snap.current_step
                # Place a note indicator near the left hand
                if show_trail and lh.visible:
                    trail_pos = self._leap_to_scene(lh.palm_position)
                    # Offset each note slightly on X based on step
                    offset_x = (snap.current_step - snap.step_count / 2) * 0.12
                    idx = note_trail_idx % len(self._note_trail)
                    self._note_trail[idx].position = (
                        trail_pos[0] + offset_x,
                        trail_pos[1] - 0.3,
                        trail_pos[2],
                    )
                    self._note_trail[idx].opacity = 0.9
                    self._note_trail[idx].color = CHORD_COLORS[snap.chord_zone_layer][snap.chord_zone_row][snap.chord_zone_col]
                    note_trail_idx += 1

                if gui_due:
                    self._gui_current_note.value = midi_to_name(snap.current_note)

            # Fade old trail notes
            if show_trail:
                for i, nh in enumerate(self._note_trail):
                    age = (note_trail_idx - i) % len(self._note_trail)
                    if age > 0:
                        fade = max(0.0, 0.9 - age * 0.09)
                        nh.opacity = fade

            fps = max(10, int(self._gui_fps.value))
            time.sleep(1.0 / fps)
