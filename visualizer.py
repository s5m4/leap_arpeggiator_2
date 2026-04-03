"""
ModernGL + ImGui visualization for Leap Motion Arpeggiator.

Renders:
    - A semi-transparent 3x3x3 chord selection cube (right hand zone)
    - Spheres for palm positions of both hands
    - Small spheres for fingertips
    - Chord name overlays projected from 3D zone centers
    - Active zone highlighting
    - ImGui sidebar with real-time musical parameters
    - A "note trail" showing recently played notes
"""

import sys
import math
import numpy as np
import glm
import moderngl
import moderngl_window
import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

from shaders import MESH_VERTEX, MESH_FRAGMENT, LINE_VERTEX, LINE_FRAGMENT
from geometry import make_cube, make_sphere, make_grid, make_wire_cube
from shared_state import (
    SharedState, CHORD_GRID, CHORD_COLORS,
    midi_to_name, chord_name_from_root,
)


class ArpeggiatorVisualizer(moderngl_window.WindowConfig):
    gl_version = (3, 3)
    title = "Leap Arpeggiator"
    window_size = (1400, 900)
    resizable = True
    vsync = True

    _shared_state = None

    @classmethod
    def run(cls, state: SharedState, port: int = 8080):
        cls._shared_state = state
        # Strip custom args before moderngl_window parses sys.argv
        # (its parse_args falls back to sys.argv[1:] when args is falsy)
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            moderngl_window.run_window_config(cls)
        finally:
            sys.argv = original_argv

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state = self.__class__._shared_state

        # ImGui setup
        imgui.create_context()
        self.imgui_renderer = PygletProgrammablePipelineRenderer(self.wnd._window)

        # Compile shader programs
        self.mesh_prog = self.ctx.program(
            vertex_shader=MESH_VERTEX,
            fragment_shader=MESH_FRAGMENT,
        )
        self.line_prog = self.ctx.program(
            vertex_shader=LINE_VERTEX,
            fragment_shader=LINE_FRAGMENT,
        )

        # Build geometry VAOs
        self._build_vaos()

        # Camera state (orbit camera)
        self.cam_distance = 8.0
        self.cam_yaw = math.radians(-30)
        self.cam_pitch = math.radians(25)
        self.cam_target = glm.vec3(0.0, 0.0, 0.0)

        # GUI state
        self.gui_bpm = 120
        self.gui_note_min = 36
        self.gui_note_max = 60
        self.gui_perf_mode = True
        self.gui_show_fingers = False
        self.gui_show_trail = False
        self.gui_zone_opacity = 0.08
        self.gui_synth_enabled = False

        # Note trail tracking
        self._trail_positions = []  # list of (glm.vec3, color_tuple, age_counter)
        self._last_step = -1
        self._trail_max = 10

        # OpenGL state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def _build_vaos(self):
        # Cube (mesh)
        verts, norms, indices = make_cube()
        buf = np.hstack([verts, norms]).astype("f4")
        vbo = self.ctx.buffer(buf.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        self.cube_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=ibo,
        )

        # Sphere (mesh)
        verts, norms, indices = make_sphere(rings=12, sectors=24)
        buf = np.hstack([verts, norms]).astype("f4")
        vbo = self.ctx.buffer(buf.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        self.sphere_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=ibo,
        )

        # Grid (lines)
        grid_verts = make_grid(size=8.0, divisions=16)
        vbo = self.ctx.buffer(grid_verts.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.line_prog,
            [(vbo, "3f", "in_position")],
        )
        self.grid_vertex_count = len(grid_verts)

        # Wire cube (lines)
        wire_verts = make_wire_cube()
        vbo = self.ctx.buffer(wire_verts.tobytes())
        self.wire_cube_vao = self.ctx.vertex_array(
            self.line_prog,
            [(vbo, "3f", "in_position")],
        )
        self.wire_cube_vertex_count = len(wire_verts)

    def _eye_position(self):
        yaw = self.cam_yaw
        pitch = self.cam_pitch
        eye = self.cam_target + self.cam_distance * glm.vec3(
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.cos(yaw),
        )
        return eye

    def _view_matrix(self):
        eye = self._eye_position()
        return glm.lookAt(eye, self.cam_target, glm.vec3(0, 1, 0))

    def _proj_matrix(self):
        w, h = self.window_size
        aspect = w / max(h, 1)
        return glm.perspective(glm.radians(45.0), aspect, 0.1, 100.0)

    def mouse_drag_event(self, x, y, dx, dy):
        self.cam_yaw += dx * 0.003
        self.cam_pitch = max(
            math.radians(-89),
            min(math.radians(89), self.cam_pitch + dy * 0.003),
        )

    def mouse_scroll_event(self, x_offset, y_offset):
        self.cam_distance = max(2.0, min(30.0, self.cam_distance - y_offset * 0.5))

    def _leap_to_scene(self, pos):
        x = pos[0] * 3.0 / 300.0
        y = (pos[1] - 250) * 3.0 / 300.0
        z = pos[2] * 3.0 / 300.0
        return glm.vec3(x, y, z)

    @staticmethod
    def _glm_to_bytes(m):
        """Convert glm.mat4 to column-major bytes for OpenGL uniforms.
        PyGLM's bytes() gives row-major order; OpenGL expects column-major."""
        return bytes(glm.transpose(m))

    def _draw_mesh(self, vao, model, color, opacity, view, proj, eye):
        self.mesh_prog["u_model"].write(self._glm_to_bytes(model))
        self.mesh_prog["u_view"].write(self._glm_to_bytes(view))
        self.mesh_prog["u_proj"].write(self._glm_to_bytes(proj))
        self.mesh_prog["u_color"].value = tuple(color)
        self.mesh_prog["u_opacity"].value = opacity
        self.mesh_prog["u_light_dir"].value = (0.5, 1.0, 0.3)
        self.mesh_prog["u_view_pos"].value = tuple(eye)
        vao.render(moderngl.TRIANGLES)

    def _draw_lines(self, vao, count, color, opacity, view, proj):
        self.line_prog["u_view"].write(self._glm_to_bytes(view))
        self.line_prog["u_proj"].write(self._glm_to_bytes(proj))
        self.line_prog["u_color"].value = tuple(color)
        self.line_prog["u_opacity"].value = opacity
        vao.render(moderngl.LINES, vertices=count)

    def _world_to_screen(self, world_pos, view, proj):
        """Project a 3D world position to 2D screen coordinates.

        Returns (sx, sy, is_visible).
        """
        pos4 = glm.vec4(world_pos, 1.0)
        clip = proj * view * pos4
        if clip.w <= 0.0:
            return 0, 0, False
        ndc = glm.vec3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
        if ndc.z < -1.0 or ndc.z > 1.0:
            return 0, 0, False
        w, h = self.window_size
        sx = (ndc.x * 0.5 + 0.5) * w
        sy = (1.0 - (ndc.y * 0.5 + 0.5)) * h
        return sx, sy, True

    def render(self, time_val, frametime):
        self.ctx.clear(0.08, 0.08, 0.12, 1.0)

        snap = self.state.read()

        view = self._view_matrix()
        proj = self._proj_matrix()
        eye = self._eye_position()

        # --- Pass 1: opaque / near-opaque geometry (depth write ON) ---

        # Grid floor at Y=-2.0
        grid_view = glm.translate(view, glm.vec3(0.0, -2.0, 0.0))
        self.line_prog["u_view"].write(self._glm_to_bytes(grid_view))
        self.line_prog["u_proj"].write(self._glm_to_bytes(proj))
        self.line_prog["u_color"].value = (0.3, 0.3, 0.35)
        self.line_prog["u_opacity"].value = 0.4
        self.grid_vao.render(moderngl.LINES, vertices=self.grid_vertex_count)

        # Hands (opacity 0.9 — render before transparent cubes)
        self._render_hands(snap, view, proj, eye)

        # --- Pass 2: transparent geometry (depth write OFF) ---
        self.ctx.depth_mask = False

        # Chord cubes (opacity 0.08–0.7)
        self._render_chord_cube(snap, view, proj, eye)

        # Note trail
        if self.gui_show_trail:
            self._render_note_trail(snap, view, proj, eye)

        self.ctx.depth_mask = True

        # Sync GUI controls to shared state
        self._sync_gui_to_state(snap)

        # Render ImGui
        self._render_imgui(snap, view, proj)

    def _render_chord_cube(self, snap, view, proj, eye):
        active_layer = snap.chord_zone_layer
        active_row = snap.chord_zone_row
        active_col = snap.chord_zone_col
        base_opacity = self.gui_zone_opacity
        perf_mode = self.gui_perf_mode

        for layer in range(3):
            for row in range(3):
                for col in range(3):
                    cx = -1.5 + 0.5 + col * 1.0
                    cy = -1.5 + 0.5 + layer * 1.0
                    cz = -1.5 + 0.5 + row * 1.0

                    r, g, b = CHORD_COLORS[layer][row][col]
                    color = glm.vec3(r / 255.0, g / 255.0, b / 255.0)

                    is_active = (layer == active_layer and row == active_row and col == active_col)
                    same_layer = (layer == active_layer)

                    if is_active:
                        opacity = min(0.7, base_opacity * 3.0)
                        color = glm.vec3(
                            min(1.0, color.x + 0.3),
                            min(1.0, color.y + 0.3),
                            min(1.0, color.z + 0.3),
                        )
                    elif perf_mode:
                        if same_layer:
                            opacity = base_opacity * 0.55
                        else:
                            opacity = 0.0
                    else:
                        if same_layer:
                            opacity = min(0.25, base_opacity * 1.35)
                        else:
                            opacity = base_opacity * 0.55

                    if opacity < 0.005:
                        continue

                    model = glm.translate(glm.mat4(1.0), glm.vec3(cx, cy, cz))
                    model = glm.scale(model, glm.vec3(0.92))

                    self._draw_mesh(self.cube_vao, model, color, opacity, view, proj, eye)

        # Draw outer wireframe cube scaled to 3.0
        # Line shader has no u_model, so bake scale into view matrix
        wire_view = view * glm.scale(glm.mat4(1.0), glm.vec3(3.0))
        self._draw_lines(
            self.wire_cube_vao, self.wire_cube_vertex_count,
            (0.5, 0.5, 0.6), 0.3, wire_view, proj,
        )

    def _render_hands(self, snap, view, proj, eye):
        rh = snap.right_hand
        lh = snap.left_hand

        if rh.visible:
            rpos = self._leap_to_scene(rh.palm_position)
            model = glm.translate(glm.mat4(1.0), rpos)
            model = glm.scale(model, glm.vec3(0.15))
            self._draw_mesh(
                self.sphere_vao, model,
                glm.vec3(1.0, 0.4, 0.24), 0.9,
                view, proj, eye,
            )

            if self.gui_show_fingers:
                for i, fpos in enumerate(rh.finger_positions):
                    fp = self._leap_to_scene(fpos)
                    fmodel = glm.translate(glm.mat4(1.0), fp)
                    fmodel = glm.scale(fmodel, glm.vec3(0.06))
                    self._draw_mesh(
                        self.sphere_vao, fmodel,
                        glm.vec3(1.0, 0.63, 0.47), 0.85,
                        view, proj, eye,
                    )

        if lh.visible:
            lpos = self._leap_to_scene(lh.palm_position)
            model = glm.translate(glm.mat4(1.0), lpos)
            model = glm.scale(model, glm.vec3(0.15))
            self._draw_mesh(
                self.sphere_vao, model,
                glm.vec3(0.24, 0.55, 1.0), 0.9,
                view, proj, eye,
            )

            if self.gui_show_fingers:
                for i, fpos in enumerate(lh.finger_positions):
                    fp = self._leap_to_scene(fpos)
                    fmodel = glm.translate(glm.mat4(1.0), fp)
                    fmodel = glm.scale(fmodel, glm.vec3(0.06))
                    self._draw_mesh(
                        self.sphere_vao, fmodel,
                        glm.vec3(0.47, 0.67, 1.0), 0.85,
                        view, proj, eye,
                    )

    def _render_note_trail(self, snap, view, proj, eye):
        lh = snap.left_hand

        # Detect step change and add new trail position
        if snap.current_step != self._last_step:
            self._last_step = snap.current_step
            if lh.visible:
                base_pos = self._leap_to_scene(lh.palm_position)
                offset_x = (snap.current_step - snap.step_count / 2.0) * 0.12
                trail_pos = glm.vec3(base_pos.x + offset_x, base_pos.y - 0.3, base_pos.z)
                color_rgb = CHORD_COLORS[snap.chord_zone_layer][snap.chord_zone_row][snap.chord_zone_col]
                color = glm.vec3(color_rgb[0] / 255.0, color_rgb[1] / 255.0, color_rgb[2] / 255.0)
                self._trail_positions.append((trail_pos, color))
                if len(self._trail_positions) > self._trail_max:
                    self._trail_positions.pop(0)

        # Render trail spheres with fading
        total = len(self._trail_positions)
        for i, (pos, color) in enumerate(self._trail_positions):
            age = total - 1 - i  # 0 = newest
            fade = max(0.0, 0.9 - age * 0.09)
            if fade < 0.01:
                continue
            model = glm.translate(glm.mat4(1.0), pos)
            model = glm.scale(model, glm.vec3(0.04))
            self._draw_mesh(self.sphere_vao, model, color, fade, view, proj, eye)

    def _sync_gui_to_state(self, snap):
        note_min = self.gui_note_min
        note_max = self.gui_note_max

        if note_min >= note_max:
            if note_min < 84:
                note_max = note_min + 1
                self.gui_note_max = note_max
            else:
                note_min = 83
                note_max = 84
                self.gui_note_min = note_min
                self.gui_note_max = note_max

        self.state.update(
            bpm=float(self.gui_bpm),
            note_min=note_min,
            note_max=note_max,
            synth_enabled=self.gui_synth_enabled,
        )

    def _render_imgui(self, snap, view, proj):
        imgui.new_frame()

        # --- Floating chord labels overlay ---
        draw_list = imgui.get_background_draw_list()
        active_layer = snap.chord_zone_layer
        active_row = snap.chord_zone_row
        active_col = snap.chord_zone_col
        perf_mode = self.gui_perf_mode

        for layer in range(3):
            for row in range(3):
                for col in range(3):
                    is_active = (layer == active_layer and row == active_row and col == active_col)
                    same_layer = (layer == active_layer)

                    # In perf mode: show only same-layer labels
                    # In non-perf mode: show active + same-layer labels
                    if is_active:
                        pass  # always show active
                    elif same_layer:
                        pass  # show same-layer labels
                    else:
                        continue  # hide other layers

                    cx = -1.5 + 0.5 + col * 1.0
                    cy = -1.5 + 0.5 + layer * 1.0
                    cz = -1.5 + 0.5 + row * 1.0

                    sx, sy, visible = self._world_to_screen(
                        glm.vec3(cx, cy, cz), view, proj,
                    )
                    if not visible:
                        continue

                    quality = CHORD_GRID[layer][row][col]
                    label = chord_name_from_root(snap.base_note, quality)

                    if is_active:
                        col_u32 = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
                    else:
                        col_u32 = imgui.get_color_u32_rgba(0.8, 0.8, 0.8, 0.6)

                    draw_list.add_text(sx - 10, sy - 6, col_u32, label)

        # HUD label above cube: "Base: C4"
        base_label = f"Base: {midi_to_name(snap.base_note)}"
        hud_sx, hud_sy, hud_vis = self._world_to_screen(
            glm.vec3(0.0, 2.2, 0.0), view, proj,
        )
        if hud_vis:
            hud_col = imgui.get_color_u32_rgba(1.0, 1.0, 0.7, 0.9)
            draw_list.add_text(hud_sx - 20, hud_sy - 8, hud_col, base_label)

        # --- Main ImGui sidebar panel ---
        imgui.set_next_window_position(10, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(300, 700, imgui.FIRST_USE_EVER)
        imgui.begin("Leap Arpeggiator")

        # Musical State section
        expanded, _ = imgui.collapsing_header(
            "Musical State", flags=imgui.TREE_NODE_DEFAULT_OPEN,
        )
        if expanded:
            imgui.text(f"Chord: {snap.chord_type}")
            imgui.text(f"Base Note: {midi_to_name(snap.base_note)}")
            imgui.text(f"Steps: {snap.step_count}")
            imgui.text(f"Spread: {snap.note_spread:.2f}")
            imgui.text(f"Playing: {midi_to_name(snap.current_note)}")
            imgui.separator()

        # Settings section
        expanded, _ = imgui.collapsing_header(
            "Settings", flags=imgui.TREE_NODE_DEFAULT_OPEN,
        )
        if expanded:
            changed, val = imgui.slider_int("BPM", self.gui_bpm, 40, 240)
            if changed:
                self.gui_bpm = val

            changed, val = imgui.slider_int("Note Min", self.gui_note_min, 24, 84)
            if changed:
                self.gui_note_min = val

            changed, val = imgui.slider_int("Note Max", self.gui_note_max, 24, 84)
            if changed:
                self.gui_note_max = val

            changed, val = imgui.slider_float(
                "Zone Opacity", self.gui_zone_opacity, 0.05, 0.5,
            )
            if changed:
                self.gui_zone_opacity = val

            changed, val = imgui.checkbox("Performance Mode", self.gui_perf_mode)
            if changed:
                self.gui_perf_mode = val

            changed, val = imgui.checkbox("Show Fingertips", self.gui_show_fingers)
            if changed:
                self.gui_show_fingers = val

            changed, val = imgui.checkbox("Show Note Trail", self.gui_show_trail)
            if changed:
                self.gui_show_trail = val

            changed, val = imgui.checkbox("Sine Synth", self.gui_synth_enabled)
            if changed:
                self.gui_synth_enabled = val

            imgui.separator()

        # Hand Tracking section
        expanded, _ = imgui.collapsing_header(
            "Hand Tracking", flags=imgui.TREE_NODE_DEFAULT_OPEN,
        )
        if expanded:
            rh = snap.right_hand
            lh = snap.left_hand
            if rh.visible:
                p = rh.palm_position
                imgui.text(f"Right Palm: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})")
            else:
                imgui.text("Right Palm: --")

            if lh.visible:
                p = lh.palm_position
                imgui.text(f"Left Palm: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})")
                imgui.text(f"Left Roll: {math.degrees(lh.roll):.1f} deg")
                imgui.text(f"Left Pitch: {math.degrees(lh.pitch):.1f} deg")
            else:
                imgui.text("Left Palm: --")

            imgui.separator()

        # Controls help
        imgui.text_wrapped(
            "Controls:\n"
            "- Right hand XYZ -> chord zone\n"
            "- Left hand height -> base note\n"
            "- Left hand roll -> step count\n"
            "- Left hand pitch -> note spread\n"
            "- Mouse drag -> orbit camera\n"
            "- Mouse scroll -> zoom"
        )

        imgui.end()

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def resize(self, width, height):
        self.window_size = (width, height)
        self.ctx.viewport = (0, 0, width, height)
