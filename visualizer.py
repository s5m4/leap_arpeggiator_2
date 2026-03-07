"""
ModernGL + glfw + imgui Visualization for Leap Motion Arpeggiator

Renders:
    - A semi-transparent 3x3x3 chord selection cube (right hand zone)
    - Spheres for palm positions of both hands
    - Small spheres for fingertips
    - Labels showing chord names in each zone
    - Active zone highlighting
    - imgui sidebar with real-time musical parameters
    - A "note trail" showing recently played notes
"""

import time
import math
import numpy as np

import glfw
import moderngl
import imgui
from imgui.integrations.glfw import GlfwRenderer

from shared_state import (
    SharedState, CHORD_GRID, CHORD_COLORS,
    midi_to_name, chord_name_from_root,
)

# ---------------------------------------------------------------------------
# Shader sources
# ---------------------------------------------------------------------------

VERT_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_normal;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    gl_Position = u_proj * u_view * world;
}
"""

FRAG_SHADER = """
#version 330 core
in vec3 v_normal;

uniform vec4 u_color;
uniform vec3 u_light_dir;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(-u_light_dir);
    float diff = max(dot(n, l), 0.0);
    float ambient = 0.35;
    float light = ambient + diff * 0.65;
    frag_color = vec4(u_color.rgb * light, u_color.a);
}
"""

GRID_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 u_view;
uniform mat4 u_proj;
void main() {
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
}
"""

GRID_FRAG = """
#version 330 core
uniform vec4 u_color;
out vec4 frag_color;
void main() {
    frag_color = u_color;
}
"""


# ---------------------------------------------------------------------------
# Mesh generators
# ---------------------------------------------------------------------------

def _make_cube_mesh():
    """Unit cube centered at origin, 36 verts with normals."""
    # 6 faces, 2 triangles each
    V = np.array([
        # front (+Z)
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
        [-0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        # back (-Z)
        [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        # right (+X)
        [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5,  0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
        # left (-X)
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        # top (+Y)
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        # bottom (-Y)
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
    ], dtype='f4')

    N = np.array([
        *([[0,0,1]]*6),
        *([[0,0,-1]]*6),
        *([[1,0,0]]*6),
        *([[-1,0,0]]*6),
        *([[0,1,0]]*6),
        *([[0,-1,0]]*6),
    ], dtype='f4')

    return np.hstack([V, N])


def _make_sphere_mesh(stacks=12, slices=16):
    """UV sphere centered at origin, radius 1.0."""
    verts = []
    for i in range(stacks):
        phi0 = math.pi * i / stacks
        phi1 = math.pi * (i + 1) / stacks
        for j in range(slices):
            theta0 = 2.0 * math.pi * j / slices
            theta1 = 2.0 * math.pi * (j + 1) / slices

            # 4 corners of the quad
            p00 = _sphere_point(phi0, theta0)
            p10 = _sphere_point(phi1, theta0)
            p11 = _sphere_point(phi1, theta1)
            p01 = _sphere_point(phi0, theta1)

            # 2 triangles
            for p in [p00, p10, p11, p00, p11, p01]:
                verts.append(list(p) + list(p))  # position = normal for unit sphere

    return np.array(verts, dtype='f4')


def _sphere_point(phi, theta):
    sp = math.sin(phi)
    return (sp * math.cos(theta), math.cos(phi), sp * math.sin(theta))


def _make_grid_lines(size=8.0, divisions=8):
    """Floor grid at Y=-2.0 using GL_LINES."""
    lines = []
    half = size / 2.0
    step = size / divisions
    y = -2.0
    for i in range(divisions + 1):
        t = -half + i * step
        lines.append([t, y, -half])
        lines.append([t, y,  half])
        lines.append([-half, y, t])
        lines.append([ half, y, t])
    return np.array(lines, dtype='f4')


# ---------------------------------------------------------------------------
# Matrix math (pure numpy, no pyrr/glm)
# ---------------------------------------------------------------------------

def _look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype='f4')
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[3, :3] = 0
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype='f4')
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _model_matrix(tx=0, ty=0, tz=0, sx=1, sy=1, sz=1):
    m = np.eye(4, dtype='f4')
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def _project_to_screen(pos_3d, view, proj, width, height):
    """Project a 3D world position to 2D screen coordinates."""
    p = np.array([*pos_3d, 1.0], dtype='f4')
    clip = proj @ view @ p
    if abs(clip[3]) < 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    if ndc[2] < -1 or ndc[2] > 1:
        return None
    sx = (ndc[0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return (sx, sy)


# ---------------------------------------------------------------------------
# Visualizer class
# ---------------------------------------------------------------------------

class ArpeggiatorVisualizer:
    """
    Creates and maintains a native OpenGL 3D scene with imgui GUI.
    Reads from SharedState and updates scene objects each frame.
    """

    CUBE_SIZE = 3.0
    ZONE_SIZE = 1.0
    SCALE = 3.0 / 300.0
    Y_OFFSET = -2.5

    def __init__(self, state: SharedState, port: int = 8080):
        self.state = state
        self.port = port  # accepted but ignored (no web server)

        # Camera (spherical orbit)
        self._cam_yaw = -0.4
        self._cam_pitch = 0.35
        self._cam_dist = 9.0
        self._cam_target = np.array([0.0, 0.0, 0.0], dtype='f4')

        # Mouse state for orbit
        self._mouse_right_down = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # GL objects (set in _init_gl)
        self._ctx = None
        self._prog = None
        self._grid_prog = None
        self._cube_vao = None
        self._sphere_vao = None
        self._grid_vao = None
        self._grid_vert_count = 0
        self._sphere_vert_count = 0

        # imgui
        self._imgui_impl = None

        # GUI state
        self._bpm = 120
        self._fps = 18
        self._note_min = 36
        self._note_max = 60
        self._perf_mode = True
        self._show_fingers = False
        self._show_trail = False
        self._zone_opacity = 0.08

        # Active zone tracking
        self._active_zone = (-1, -1, -1)
        self._last_base_opacity = None
        self._last_label_base_note = None
        self._last_label_layer = None
        self._last_gui_update_time = 0.0
        self._gui_update_period = 0.1
        self._last_controls = None
        self._last_perf_mode = None
        self._last_show_fingers = None
        self._last_show_trail = None

        # Cached GUI readout strings (throttled updates)
        self._gui_chord_str = "Cmaj"
        self._gui_base_note_str = "C4"
        self._gui_steps_val = 4
        self._gui_spread_val = 1.0
        self._gui_playing_str = "—"
        self._gui_right_pos_str = "—"
        self._gui_left_pos_str = "—"
        self._gui_left_roll_val = 0.0
        self._gui_left_pitch_val = 0.0

        # Note trail
        self._note_trail_positions = []  # list of (x, y, z, r, g, b, opacity)
        self._note_trail_idx = 0
        self._last_step = -1

        # Zone data cache for rendering
        self._zone_render_data = {}  # (l,r,c) -> (cx, cy, cz, r, g, b, opacity)
        self._zone_labels = {}       # (l,r,c) -> str
        self._hud_base_text = "Base: C4"

        # Pre-compute zone centers
        half = self.CUBE_SIZE / 2.0
        zone = self.ZONE_SIZE
        for layer in range(3):
            for row in range(3):
                for col in range(3):
                    cx = -half + zone * 0.5 + col * zone
                    cy = -half + zone * 0.5 + layer * zone
                    cz = -half + zone * 0.5 + row * zone
                    r, g, b = CHORD_COLORS[layer][row][col]
                    self._zone_render_data[(layer, row, col)] = (cx, cy, cz, r, g, b, 0.12)
                    quality = CHORD_GRID[layer][row][col]
                    self._zone_labels[(layer, row, col)] = chord_name_from_root(60, quality)

        # Initialize note trail slots
        for _ in range(10):
            self._note_trail_positions.append((0, -5, 0, 255, 255, 100, 0.0))

    def start(self):
        """Initialize window, GL, imgui, and run the main loop."""
        self._init_window()
        self._init_gl()
        self._init_imgui()

        print(f"\n  * Visualizer window opened (native OpenGL)")
        print(f"    Right-drag to orbit, scroll to zoom.\n")

        try:
            self._update_loop()
        finally:
            self._cleanup()

    def _leap_to_scene(self, pos):
        x = pos[0] * self.SCALE
        y = (pos[1] - 250) * self.SCALE
        z = pos[2] * self.SCALE
        return (x, y, z)

    # ---- Window / GL / imgui init ----

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)

        self._window = glfw.create_window(1280, 800, "Leap Arpeggiator", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(0)  # we do our own FPS limiting

        # Mouse callbacks
        glfw.set_mouse_button_callback(self._window, self._mouse_button_cb)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_cb)
        glfw.set_scroll_callback(self._window, self._scroll_cb)

    def _init_gl(self):
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)

        # Main 3D shader
        self._prog = self._ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        # Grid shader
        self._grid_prog = self._ctx.program(
            vertex_shader=GRID_VERT,
            fragment_shader=GRID_FRAG,
        )

        # Cube VAO
        cube_data = _make_cube_mesh()
        cube_buf = self._ctx.buffer(cube_data.tobytes())
        self._cube_vao = self._ctx.vertex_array(
            self._prog,
            [(cube_buf, '3f 3f', 'in_position', 'in_normal')],
        )

        # Sphere VAO
        sphere_data = _make_sphere_mesh()
        self._sphere_vert_count = len(sphere_data)
        sphere_buf = self._ctx.buffer(sphere_data.tobytes())
        self._sphere_vao = self._ctx.vertex_array(
            self._prog,
            [(sphere_buf, '3f 3f', 'in_position', 'in_normal')],
        )

        # Grid VAO
        grid_data = _make_grid_lines()
        self._grid_vert_count = len(grid_data)
        grid_buf = self._ctx.buffer(grid_data.tobytes())
        self._grid_vao = self._ctx.vertex_array(
            self._grid_prog,
            [(grid_buf, '3f', 'in_position')],
        )

    def _init_imgui(self):
        imgui.create_context()
        self._imgui_impl = GlfwRenderer(self._window, attach_callbacks=False)

    # ---- Camera mouse callbacks ----

    def _mouse_button_cb(self, window, button, action, mods):
        # Let imgui handle first
        io = imgui.get_io()
        if io.want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right_down = (action == glfw.PRESS)
            self._last_mouse_x, self._last_mouse_y = glfw.get_cursor_pos(window)

    def _cursor_pos_cb(self, window, xpos, ypos):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return
        if self._mouse_right_down:
            dx = xpos - self._last_mouse_x
            dy = ypos - self._last_mouse_y
            self._cam_yaw -= dx * 0.005
            self._cam_pitch += dy * 0.005
            self._cam_pitch = max(-math.pi / 2 + 0.05, min(math.pi / 2 - 0.05, self._cam_pitch))
            self._last_mouse_x = xpos
            self._last_mouse_y = ypos

    def _scroll_cb(self, window, xoffset, yoffset):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return
        self._cam_dist -= yoffset * 0.5
        self._cam_dist = max(3.0, min(25.0, self._cam_dist))

    # ---- Camera matrices ----

    def _get_camera_matrices(self, width, height):
        eye = self._cam_target + np.array([
            self._cam_dist * math.cos(self._cam_pitch) * math.sin(self._cam_yaw),
            self._cam_dist * math.sin(self._cam_pitch),
            self._cam_dist * math.cos(self._cam_pitch) * math.cos(self._cam_yaw),
        ], dtype='f4')
        view = _look_at(eye, self._cam_target, np.array([0, 1, 0], dtype='f4'))
        aspect = width / max(height, 1)
        proj = _perspective(45.0, aspect, 0.1, 100.0)
        return view, proj, eye

    # ---- Rendering ----

    def _draw_sphere(self, view, proj, eye, x, y, z, radius, r, g, b, alpha=1.0):
        model = _model_matrix(x, y, z, radius, radius, radius)
        self._prog['u_model'].write(model.T.tobytes())
        self._prog['u_view'].write(view.T.tobytes())
        self._prog['u_proj'].write(proj.T.tobytes())
        self._prog['u_color'].value = (r / 255.0, g / 255.0, b / 255.0, alpha)
        self._prog['u_light_dir'].value = (0.3, -1.0, 0.5)
        self._sphere_vao.render()

    def _draw_cube_zone(self, view, proj, eye, cx, cy, cz, r, g, b, opacity):
        scale = self.ZONE_SIZE * 0.92
        model = _model_matrix(cx, cy, cz, scale, scale, scale)
        self._prog['u_model'].write(model.T.tobytes())
        self._prog['u_view'].write(view.T.tobytes())
        self._prog['u_proj'].write(proj.T.tobytes())
        self._prog['u_color'].value = (r / 255.0, g / 255.0, b / 255.0, opacity)
        self._prog['u_light_dir'].value = (0.3, -1.0, 0.5)
        self._cube_vao.render()

    def _render_scene(self, snap, view, proj, eye):
        # 1. Grid floor (opaque)
        self._grid_prog['u_view'].write(view.T.tobytes())
        self._grid_prog['u_proj'].write(proj.T.tobytes())
        self._grid_prog['u_color'].value = (0.35, 0.35, 0.35, 1.0)
        self._grid_vao.render(moderngl.LINES)

        # 2. Hand palm spheres (opaque)
        rh = snap.right_hand
        lh = snap.left_hand
        if rh.visible:
            rpos = self._leap_to_scene(rh.palm_position)
            self._draw_sphere(view, proj, eye, *rpos, 0.15, 255, 100, 60)
        if lh.visible:
            lpos = self._leap_to_scene(lh.palm_position)
            self._draw_sphere(view, proj, eye, *lpos, 0.15, 60, 140, 255)

        # 3. Fingertip spheres
        if self._show_fingers:
            if rh.visible:
                for i in range(min(5, len(rh.finger_positions))):
                    fp = self._leap_to_scene(rh.finger_positions[i])
                    self._draw_sphere(view, proj, eye, *fp, 0.06, 255, 160, 120)
            if lh.visible:
                for i in range(min(5, len(lh.finger_positions))):
                    fp = self._leap_to_scene(lh.finger_positions[i])
                    self._draw_sphere(view, proj, eye, *fp, 0.06, 120, 170, 255)

        # 4. Note trail spheres
        if self._show_trail:
            for (tx, ty, tz, tr, tg, tb, top) in self._note_trail_positions:
                if top > 0.01:
                    self._draw_sphere(view, proj, eye, tx, ty, tz, 0.04, tr, tg, tb, top)

        # 5. Chord cubes (transparent, sorted back-to-front)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.depth_mask = False

        # Sort zones by distance to camera (farthest first)
        zone_list = []
        for key, (cx, cy, cz, r, g, b, opacity) in self._zone_render_data.items():
            if opacity < 0.001:
                continue
            dist_sq = (cx - eye[0])**2 + (cy - eye[1])**2 + (cz - eye[2])**2
            zone_list.append((dist_sq, cx, cy, cz, r, g, b, opacity))
        zone_list.sort(key=lambda x: -x[0])

        for (_, cx, cy, cz, r, g, b, opacity) in zone_list:
            self._draw_cube_zone(view, proj, eye, cx, cy, cz, r, g, b, opacity)

        self._ctx.depth_mask = True
        self._ctx.disable(moderngl.BLEND)

    def _render_labels_overlay(self, snap, view, proj, width, height):
        """Draw zone labels and HUD text as imgui overlay."""
        draw_list = imgui.get_foreground_draw_list()
        new_zone = (snap.chord_zone_layer, snap.chord_zone_row, snap.chord_zone_col)

        for key, label in self._zone_labels.items():
            if not label:
                continue
            cx, cy, cz, _, _, _, _ = self._zone_render_data[key]
            screen = _project_to_screen((cx, cy, cz), view, proj, width, height)
            if screen is None:
                continue
            sx, sy = screen
            if 0 <= sx < width and 0 <= sy < height:
                if key == new_zone:
                    color = imgui.get_color_u32_rgba(1, 1, 1, 1)
                else:
                    color = imgui.get_color_u32_rgba(0.8, 0.8, 0.8, 0.7)
                text_size = imgui.calc_text_size(label)
                draw_list.add_text(sx - text_size.x / 2, sy - text_size.y / 2, color, label)

        # HUD base note label
        screen = _project_to_screen((0, 2.2, 0), view, proj, width, height)
        if screen:
            sx, sy = screen
            color = imgui.get_color_u32_rgba(1, 1, 0.6, 1)
            text_size = imgui.calc_text_size(self._hud_base_text)
            draw_list.add_text(sx - text_size.x / 2, sy - text_size.y / 2, color, self._hud_base_text)

    def _render_gui(self, snap, width, height):
        """Render imgui sidebar on the right side of the window."""
        sidebar_width = 280
        imgui.set_next_window_position(width - sidebar_width, 0)
        imgui.set_next_window_size(sidebar_width, height)

        imgui.begin(
            "Leap Arpeggiator",
            flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE,
        )

        # -- Musical State --
        if imgui.collapsing_header("Musical State", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text(f"Chord:     {self._gui_chord_str}")
            imgui.text(f"Base Note: {self._gui_base_note_str}")
            imgui.text(f"Steps:     {self._gui_steps_val}")
            imgui.text(f"Spread:    {self._gui_spread_val:.2f}")
            imgui.text(f"Playing:   {self._gui_playing_str}")

        imgui.spacing()

        # -- Settings --
        if imgui.collapsing_header("Settings", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            changed, self._bpm = imgui.slider_int("BPM", self._bpm, 40, 240)
            _, self._fps = imgui.slider_int("Visual FPS", self._fps, 10, 30)

            _, self._note_min = imgui.slider_int("Note Min", self._note_min, 24, 84)
            _, self._note_max = imgui.slider_int("Note Max", self._note_max, 24, 84)
            if self._note_min >= self._note_max:
                if self._note_min < 84:
                    self._note_max = self._note_min + 1
                else:
                    self._note_min = 83
                    self._note_max = 84

            _, self._perf_mode = imgui.checkbox("Performance Mode", self._perf_mode)
            _, self._show_fingers = imgui.checkbox("Show Fingertips", self._show_fingers)
            _, self._show_trail = imgui.checkbox("Show Note Trail", self._show_trail)
            _, self._zone_opacity = imgui.slider_float("Zone Opacity", self._zone_opacity, 0.05, 0.5)

        imgui.spacing()

        # -- Hand Tracking --
        if imgui.collapsing_header("Hand Tracking", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text(f"Right Palm:  {self._gui_right_pos_str}")
            imgui.text(f"Left Palm:   {self._gui_left_pos_str}")
            imgui.text(f"Left Roll:   {self._gui_left_roll_val:.1f} deg")
            imgui.text(f"Left Pitch:  {self._gui_left_pitch_val:.1f} deg")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # -- Control Legend --
        imgui.text_wrapped(
            "Controls:\n"
            "- Right hand XYZ -> chord zone\n"
            "- Left hand height -> base note\n"
            "- Left hand roll -> step count\n"
            "- Left hand pitch -> note spread\n"
            "\n"
            "Camera:\n"
            "- Right-drag to orbit\n"
            "- Scroll to zoom"
        )

        imgui.end()

    # ---- Main loop ----

    def _update_loop(self):
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            self._imgui_impl.process_inputs()
            imgui.new_frame()

            snap = self.state.read()
            now = time.perf_counter()
            gui_due = (now - self._last_gui_update_time) >= self._gui_update_period

            # Push GUI control changes to SharedState
            controls = (self._bpm, self._note_min, self._note_max)
            if controls != self._last_controls:
                self.state.update(bpm=self._bpm, note_min=self._note_min, note_max=self._note_max)
                self._last_controls = controls

            perf_mode = self._perf_mode
            perf_mode_changed = self._last_perf_mode is None or self._last_perf_mode != perf_mode

            # Update zone highlighting
            new_zone = (snap.chord_zone_layer, snap.chord_zone_row, snap.chord_zone_col)
            base_opacity = self._zone_opacity
            zone_changed = new_zone != self._active_zone
            opacity_changed = self._last_base_opacity is None or abs(base_opacity - self._last_base_opacity) > 1e-6

            if zone_changed or opacity_changed or perf_mode_changed:
                for key in self._zone_render_data:
                    cx, cy, cz, _, _, _, _ = self._zone_render_data[key]
                    r, g, b = CHORD_COLORS[key[0]][key[1]][key[2]]

                    if key == new_zone:
                        # Active zone: brighter & more opaque
                        r = min(255, r + 80)
                        g = min(255, g + 80)
                        b = min(255, b + 80)
                        opacity = min(0.7, base_opacity * 3.0)
                    elif perf_mode:
                        if key[0] == new_zone[0]:
                            opacity = base_opacity * 0.55
                        else:
                            opacity = 0.0
                    elif key[0] == new_zone[0]:
                        opacity = min(0.25, base_opacity * 1.35)
                    else:
                        opacity = base_opacity * 0.55

                    self._zone_render_data[key] = (cx, cy, cz, r, g, b, opacity)

                self._active_zone = new_zone
                self._last_base_opacity = base_opacity

            # Update labels
            labels_need_update = (
                self._last_label_base_note != snap.base_note
                or self._last_label_layer != new_zone[0]
                or zone_changed
                or perf_mode_changed
            )
            if labels_need_update:
                for key in self._zone_labels:
                    layer, row, col = key
                    quality = CHORD_GRID[layer][row][col]
                    chord_label = chord_name_from_root(snap.base_note, quality)
                    if key == new_zone:
                        self._zone_labels[key] = chord_label
                    elif (not perf_mode) and layer == new_zone[0]:
                        self._zone_labels[key] = chord_label
                    else:
                        self._zone_labels[key] = ""
                self._last_label_base_note = snap.base_note
                self._last_label_layer = new_zone[0]
                self._last_perf_mode = perf_mode

            # Update note trail
            rh = snap.right_hand
            lh = snap.left_hand
            if snap.current_step != self._last_step:
                self._last_step = snap.current_step
                if self._show_trail and lh.visible:
                    trail_pos = self._leap_to_scene(lh.palm_position)
                    offset_x = (snap.current_step - snap.step_count / 2) * 0.12
                    idx = self._note_trail_idx % len(self._note_trail_positions)
                    tr, tg, tb = CHORD_COLORS[snap.chord_zone_layer][snap.chord_zone_row][snap.chord_zone_col]
                    self._note_trail_positions[idx] = (
                        trail_pos[0] + offset_x, trail_pos[1] - 0.3, trail_pos[2],
                        tr, tg, tb, 0.9
                    )
                    self._note_trail_idx += 1

            # Fade trail
            if self._show_trail:
                for i in range(len(self._note_trail_positions)):
                    age = (self._note_trail_idx - i) % len(self._note_trail_positions)
                    if age > 0:
                        fade = max(0.0, 0.9 - age * 0.09)
                        tx, ty, tz, tr, tg, tb, _ = self._note_trail_positions[i]
                        self._note_trail_positions[i] = (tx, ty, tz, tr, tg, tb, fade)

            # Throttled GUI readout updates
            if gui_due:
                self._gui_chord_str = snap.chord_type
                self._gui_base_note_str = midi_to_name(snap.base_note)
                self._gui_steps_val = snap.step_count
                self._gui_spread_val = snap.note_spread
                self._gui_playing_str = midi_to_name(snap.current_note)
                self._hud_base_text = f"Base: {midi_to_name(snap.base_note)}"

                if rh.visible:
                    self._gui_right_pos_str = (
                        f"({rh.palm_position[0]:.0f}, {rh.palm_position[1]:.0f}, {rh.palm_position[2]:.0f})"
                    )
                if lh.visible:
                    self._gui_left_pos_str = (
                        f"({lh.palm_position[0]:.0f}, {lh.palm_position[1]:.0f}, {lh.palm_position[2]:.0f})"
                    )
                    self._gui_left_roll_val = round(math.degrees(lh.roll), 1)
                    self._gui_left_pitch_val = round(math.degrees(lh.pitch), 1)

                self._last_gui_update_time = now

            # ---- Render ----
            width, height = glfw.get_framebuffer_size(self._window)
            self._ctx.viewport = (0, 0, width, height)
            self._ctx.clear(0.08, 0.08, 0.12, 1.0)

            view, proj, eye = self._get_camera_matrices(width, height)

            self._render_scene(snap, view, proj, eye)

            # Window-size for label projection (use window size, not framebuffer on HiDPI)
            win_w, win_h = glfw.get_window_size(self._window)
            self._render_labels_overlay(snap, view, proj, win_w, win_h)
            self._render_gui(snap, win_w, win_h)

            imgui.render()
            self._imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self._window)

            fps = max(10, self._fps)
            time.sleep(1.0 / fps)

    def _cleanup(self):
        if self._imgui_impl:
            self._imgui_impl.shutdown()
        glfw.terminate()
