# ModernGL Visualizer Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Viser-based web visualizer with a native ModernGL + ImGui desktop window that renders the same 3D scene (chord cube, hand spheres, note trail, grid) with an equivalent GUI sidebar.

**Architecture:** The new visualizer uses `moderngl-window` for windowing/context, `moderngl` for GPU-accelerated 3D rendering with custom GLSL shaders, and `imgui` (via `moderngl-window`'s built-in integration) for the GUI panel. Geometry (cubes, spheres, grid lines) is generated as vertex buffers. A perspective camera with orbit controls replaces Viser's built-in camera. Only `visualizer.py` and `main.py` change — all other modules (shared_state, leap_source, gesture_interpreter, arpeggiator) remain untouched.

**Tech Stack:** Python 3.8+, moderngl, moderngl-window[pyglet], imgui[pyglet], numpy (already present)

---

## Analysis: What Changes and What Doesn't

### Unchanged Modules
- `shared_state.py` — Thread-safe state, data classes, chord grid, constants. **No changes.**
- `leap_source.py` — Mock & real Leap sources. **No changes.**
- `gesture_interpreter.py` — EMA smoothing, hysteresis, mapping. **No changes.**
- `arpeggiator.py` — Note generation, timing. **No changes.**

### Modules to Rewrite
- `visualizer.py` (512 lines) — **Complete rewrite.** Viser API replaced with ModernGL rendering pipeline.
- `main.py` (95 lines) — **Minor edits.** Update import and visualizer instantiation. The visualizer still runs on the main thread (OpenGL requirement).

### New Files
- `shaders.py` — GLSL vertex/fragment shader source strings for the 3D scene.
- `geometry.py` — Mesh generation functions (cube VAO, sphere VAO, grid VAO, line cube wireframe).

### Feature Parity Matrix

| Feature | Viser (current) | ModernGL (target) |
|---------|-----------------|-------------------|
| 3x3x3 chord cube boxes | `add_box()` | Instanced cube VAO with per-instance color/opacity/position |
| Chord zone highlighting | Modify `.opacity`/`.color` | Update instance buffer data |
| Chord labels in 3D | `add_label()` | ImGui overlay text projected from 3D coords (or bitmap font) |
| Hand palm spheres (2) | `add_icosphere()` | Sphere VAO rendered at hand positions |
| Fingertip spheres (10) | `add_icosphere()` | Same sphere VAO, smaller scale, conditional render |
| Note trail (10 spheres) | `add_icosphere()` | Same sphere VAO, per-trail color/opacity |
| Grid floor | `add_grid()` | Line-based grid VAO |
| HUD base note label | `add_label()` | ImGui overlay text |
| GUI sidebar | Viser GUI (sliders, text, checkboxes) | ImGui panel (same controls) |
| Camera orbit | Built-in Viser | `moderngl-window` OrbitCamera or custom |
| Access method | Browser http://localhost:8080 | Native desktop window |

---

## File Structure

```
leap_arpeggiator_2/
├── main.py              [MODIFY] Update visualizer import + instantiation
├── visualizer.py        [REWRITE] ModernGL window class with render loop
├── shaders.py           [CREATE] GLSL shader source strings
├── geometry.py          [CREATE] Mesh generation (cube, sphere, grid VAOs)
├── shared_state.py      [UNCHANGED]
├── leap_source.py       [UNCHANGED]
├── gesture_interpreter.py [UNCHANGED]
├── arpeggiator.py       [UNCHANGED]
└── requirements.txt     [CREATE] Pin dependencies
```

---

## Task 1: Create `requirements.txt` and Verify Install

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
numpy
moderngl
moderngl-window[pyglet]
imgui[pyglet]
PyGLM
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully. Verify with:

```bash
python -c "import moderngl; import moderngl_window; import imgui; import glm; print('All imports OK')"
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add ModernGL dependencies for native visualizer"
```

---

## Task 2: Create `shaders.py` — GLSL Shader Sources

**Files:**
- Create: `shaders.py`

- [ ] **Step 1: Write the shader module**

```python
"""
GLSL shader sources for the ModernGL arpeggiator visualizer.

Two programs:
  1. mesh_shader — Phong-lit shader for cubes and spheres.
     Accepts per-instance model matrix, color, and opacity via uniforms.
  2. line_shader — Unlit flat-color shader for grid lines and wireframes.
"""

# ──────────────────────────────────────────────────────────
# Mesh shader: Phong lighting for solid geometry
# ──────────────────────────────────────────────────────────

MESH_VERTEX = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_frag_pos = world_pos.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    gl_Position = u_proj * u_view * world_pos;
}
"""

MESH_FRAGMENT = """
#version 330 core

in vec3 v_normal;
in vec3 v_frag_pos;

uniform vec3 u_color;
uniform float u_opacity;
uniform vec3 u_light_dir;
uniform vec3 u_view_pos;

out vec4 frag_color;

void main() {
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(u_light_dir);

    // Ambient
    float ambient = 0.25;

    // Diffuse
    float diff = max(dot(norm, light_dir), 0.0);

    // Specular (Blinn-Phong)
    vec3 view_dir = normalize(u_view_pos - v_frag_pos);
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway), 0.0), 32.0) * 0.3;

    vec3 result = u_color * (ambient + diff * 0.65 + spec);
    frag_color = vec4(result, u_opacity);
}
"""

# ──────────────────────────────────────────────────────────
# Line shader: flat color for grid and wireframes
# ──────────────────────────────────────────────────────────

LINE_VERTEX = """
#version 330 core

in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_proj;

void main() {
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
}
"""

LINE_FRAGMENT = """
#version 330 core

uniform vec3 u_color;
uniform float u_opacity;

out vec4 frag_color;

void main() {
    frag_color = vec4(u_color, u_opacity);
}
"""
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import shaders; print('Shaders module OK')"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add shaders.py
git commit -m "feat: add GLSL shader sources for mesh and line rendering"
```

---

## Task 3: Create `geometry.py` — Mesh Generation

**Files:**
- Create: `geometry.py`

- [ ] **Step 1: Write the geometry module**

```python
"""
Mesh generation for the ModernGL arpeggiator visualizer.

Provides functions that return (vertices, normals, indices) numpy arrays
ready to be uploaded to ModernGL vertex buffers.
"""

import math
import numpy as np


def make_cube():
    """
    Unit cube centered at origin (side length 1.0).
    Returns (vertices, normals, indices) as numpy arrays.
    vertices: (24, 3) float32 — 6 faces x 4 verts
    normals:  (24, 3) float32
    indices:  (36,) uint32     — 6 faces x 2 tris x 3
    """
    # Each face has its own 4 vertices so normals are per-face (flat shading).
    h = 0.5
    # (position, normal) per face
    faces = [
        # +X
        ([(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)], (1, 0, 0)),
        # -X
        ([(-h, -h, h), (-h, h, h), (-h, h, -h), (-h, -h, -h)], (-1, 0, 0)),
        # +Y
        ([(-h, h, -h), (-h, h, h), (h, h, h), (h, h, -h)], (0, 1, 0)),
        # -Y
        ([(-h, -h, h), (-h, -h, -h), (h, -h, -h), (h, -h, h)], (0, -1, 0)),
        # +Z
        ([(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)], (0, 0, 1)),
        # -Z
        ([(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)], (0, 0, -1)),
    ]

    verts = []
    norms = []
    idxs = []
    for i, (quad, normal) in enumerate(faces):
        base = i * 4
        for p in quad:
            verts.append(p)
            norms.append(normal)
        idxs.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return (
        np.array(verts, dtype="f4"),
        np.array(norms, dtype="f4"),
        np.array(idxs, dtype="i4"),
    )


def make_sphere(rings=12, sectors=24):
    """
    UV sphere centered at origin with radius 1.0.
    Returns (vertices, normals, indices) as numpy arrays.
    """
    verts = []
    norms = []
    idxs = []

    for r in range(rings + 1):
        phi = math.pi * r / rings  # 0 to pi
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors  # 0 to 2pi
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.append((x, y, z))
            norms.append((x, y, z))  # unit sphere: normal = position

    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + (sectors + 1)
            idxs.extend([a, b, a + 1])
            idxs.extend([a + 1, b, b + 1])

    return (
        np.array(verts, dtype="f4"),
        np.array(norms, dtype="f4"),
        np.array(idxs, dtype="i4"),
    )


def make_grid(size=8.0, divisions=16):
    """
    Flat grid of lines on the XZ plane at Y=0.
    Returns vertices as (N, 3) float32 array. Render with GL_LINES.
    """
    half = size / 2.0
    step = size / divisions
    lines = []

    for i in range(divisions + 1):
        t = -half + i * step
        # Line along Z
        lines.append((t, 0.0, -half))
        lines.append((t, 0.0, half))
        # Line along X
        lines.append((-half, 0.0, t))
        lines.append((half, 0.0, t))

    return np.array(lines, dtype="f4")


def make_wire_cube():
    """
    Wireframe cube (12 edges) centered at origin, side length 1.0.
    Returns vertices as (24, 3) float32 array. Render with GL_LINES.
    """
    h = 0.5
    corners = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h),  (h, -h, h),  (h, h, h),  (-h, h, h),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # front face
        (4,5),(5,6),(6,7),(7,4),  # back face
        (0,4),(1,5),(2,6),(3,7),  # connecting edges
    ]
    lines = []
    for a, b in edges:
        lines.append(corners[a])
        lines.append(corners[b])

    return np.array(lines, dtype="f4")
```

- [ ] **Step 2: Verify geometry generation**

Run:
```bash
python -c "
from geometry import make_cube, make_sphere, make_grid, make_wire_cube
v, n, i = make_cube()
print(f'Cube: {v.shape[0]} verts, {i.shape[0]} indices')
v, n, i = make_sphere()
print(f'Sphere: {v.shape[0]} verts, {i.shape[0]} indices')
g = make_grid()
print(f'Grid: {g.shape[0]} line verts')
w = make_wire_cube()
print(f'Wire cube: {w.shape[0]} line verts')
"
```
Expected:
```
Cube: 24 verts, 36 indices
Sphere: 325 verts, 1728 indices
Grid: 68 line verts
Wire cube: 24 line verts
```

- [ ] **Step 3: Commit**

```bash
git add geometry.py
git commit -m "feat: add mesh generation for cube, sphere, grid, wireframe"
```

---

## Task 4: Create the ModernGL Visualizer — Window, Camera, and Shader Setup

**Files:**
- Rewrite: `visualizer.py`

This task creates the skeleton: window class, shader compilation, VAO creation, camera, and an empty render loop that shows the grid floor. Tasks 5-8 will add the scene objects incrementally.

- [ ] **Step 1: Write the visualizer skeleton**

```python
"""
ModernGL 3D Visualization for Leap Motion Arpeggiator

Native desktop window replacing the previous Viser-based web visualizer.
Uses moderngl for rendering and imgui for the GUI sidebar.
"""

import time
import math
import numpy as np
import glm

import moderngl
import moderngl_window
from moderngl_window import geometry as mglw_geo
from moderngl_window.context.base import KeyModifiers

import imgui
from imgui.integrations.pyglet import create_renderer as create_imgui_renderer

from shaders import MESH_VERTEX, MESH_FRAGMENT, LINE_VERTEX, LINE_FRAGMENT
from geometry import make_cube, make_sphere, make_grid, make_wire_cube
from shared_state import (
    SharedState, CHORD_GRID, CHORD_COLORS,
    midi_to_name, chord_name_from_root,
)


class ArpeggiatorVisualizer(moderngl_window.WindowConfig):
    """
    ModernGL window that renders the arpeggiator 3D scene and ImGui controls.
    Reads from SharedState each frame — same contract as the old Viser visualizer.
    """

    gl_version = (3, 3)
    title = "Leap Arpeggiator"
    window_size = (1400, 900)
    resizable = True
    vsync = True
    resource_dir = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── State reference (injected before run) ──
        self.state: SharedState = self.__class__._shared_state

        # ── ImGui setup ──
        imgui.create_context()
        self.imgui_renderer = create_imgui_renderer(self.wnd._window)

        # ── Compile shader programs ──
        self.mesh_prog = self.ctx.program(
            vertex_shader=MESH_VERTEX,
            fragment_shader=MESH_FRAGMENT,
        )
        self.line_prog = self.ctx.program(
            vertex_shader=LINE_VERTEX,
            fragment_shader=LINE_FRAGMENT,
        )

        # ── Create VAOs ──
        self._build_vaos()

        # ── Camera state ──
        self.cam_distance = 8.0
        self.cam_yaw = -30.0      # degrees
        self.cam_pitch = 25.0     # degrees
        self.cam_target = glm.vec3(0.0, 0.0, 0.0)
        self._mouse_dragging = False
        self._last_mouse = (0, 0)

        # ── Coordinate conversion constants (same as old visualizer) ──
        self.SCALE = 3.0 / 300.0
        self.CUBE_SIZE = 3.0
        self.ZONE_SIZE = 1.0

        # ── GUI state defaults ──
        self.gui_bpm = 120
        self.gui_note_min = 36
        self.gui_note_max = 60
        self.gui_fps = 18
        self.gui_perf_mode = True
        self.gui_show_fingers = False
        self.gui_show_trail = False
        self.gui_zone_opacity = 0.08

        # ── Tracking for dirty detection ──
        self._active_zone = (-1, -1, -1)
        self._last_gui_update_time = 0.0
        self._note_trail_positions = []  # list of (x, y, z, r, g, b, opacity)
        self._note_trail_idx = 0
        self._last_step = -1

        # ── OpenGL state ──
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        )

    def _build_vaos(self):
        """Create all vertex array objects from geometry."""
        # Cube
        cv, cn, ci = make_cube()
        cube_buf = self.ctx.buffer(np.hstack([cv, cn]).tobytes())
        cube_ibo = self.ctx.buffer(ci.tobytes())
        self.cube_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(cube_buf, "3f 3f", "in_position", "in_normal")],
            index_buffer=cube_ibo,
        )

        # Sphere
        sv, sn, si = make_sphere(rings=12, sectors=24)
        sphere_buf = self.ctx.buffer(np.hstack([sv, sn]).tobytes())
        sphere_ibo = self.ctx.buffer(si.tobytes())
        self.sphere_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(sphere_buf, "3f 3f", "in_position", "in_normal")],
            index_buffer=sphere_ibo,
        )

        # Grid (lines)
        gv = make_grid(size=8.0, divisions=16)
        grid_buf = self.ctx.buffer(gv.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.line_prog,
            [(grid_buf, "3f", "in_position")],
        )
        self.grid_vertex_count = gv.shape[0]

        # Wire cube (lines)
        wv = make_wire_cube()
        wire_buf = self.ctx.buffer(wv.tobytes())
        self.wire_cube_vao = self.ctx.vertex_array(
            self.line_prog,
            [(wire_buf, "3f", "in_position")],
        )
        self.wire_cube_vertex_count = wv.shape[0]

    # ─────────────────────────────────────────────
    # Camera helpers
    # ─────────────────────────────────────────────

    def _view_matrix(self) -> glm.mat4:
        yaw_rad = math.radians(self.cam_yaw)
        pitch_rad = math.radians(self.cam_pitch)
        eye = glm.vec3(
            self.cam_distance * math.cos(pitch_rad) * math.sin(yaw_rad),
            self.cam_distance * math.sin(pitch_rad),
            self.cam_distance * math.cos(pitch_rad) * math.cos(yaw_rad),
        )
        eye += self.cam_target
        return glm.lookAt(eye, self.cam_target, glm.vec3(0, 1, 0))

    def _proj_matrix(self) -> glm.mat4:
        aspect = self.window_size[0] / max(1, self.window_size[1])
        return glm.perspective(glm.radians(45.0), aspect, 0.1, 100.0)

    def _eye_position(self) -> glm.vec3:
        yaw_rad = math.radians(self.cam_yaw)
        pitch_rad = math.radians(self.cam_pitch)
        eye = glm.vec3(
            self.cam_distance * math.cos(pitch_rad) * math.sin(yaw_rad),
            self.cam_distance * math.sin(pitch_rad),
            self.cam_distance * math.cos(pitch_rad) * math.cos(yaw_rad),
        )
        return eye + self.cam_target

    # ─────────────────────────────────────────────
    # Input handling (camera orbit + zoom)
    # ─────────────────────────────────────────────

    def mouse_drag_event(self, x, y, dx, dy):
        self.cam_yaw += dx * 0.3
        self.cam_pitch = max(-89.0, min(89.0, self.cam_pitch + dy * 0.3))

    def mouse_scroll_event(self, x_offset, y_offset):
        self.cam_distance = max(2.0, min(30.0, self.cam_distance - y_offset * 0.5))

    # ─────────────────────────────────────────────
    # Coordinate conversion (Leap mm → scene units)
    # ─────────────────────────────────────────────

    def _leap_to_scene(self, pos) -> glm.vec3:
        x = pos[0] * self.SCALE
        y = (pos[1] - 250) * self.SCALE
        z = pos[2] * self.SCALE
        return glm.vec3(x, y, z)

    # ─────────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────────

    def _draw_mesh(self, vao, model: glm.mat4, color: tuple, opacity: float,
                   view: glm.mat4, proj: glm.mat4, eye: glm.vec3):
        """Render a mesh VAO with the Phong shader."""
        self.mesh_prog["u_model"].write(bytes(model))
        self.mesh_prog["u_view"].write(bytes(view))
        self.mesh_prog["u_proj"].write(bytes(proj))
        self.mesh_prog["u_color"].value = color
        self.mesh_prog["u_opacity"].value = opacity
        self.mesh_prog["u_light_dir"].value = (0.5, 1.0, 0.3)
        self.mesh_prog["u_view_pos"].value = (eye.x, eye.y, eye.z)
        vao.render(moderngl.TRIANGLES)

    def _draw_lines(self, vao, count: int, color: tuple, opacity: float,
                    view: glm.mat4, proj: glm.mat4):
        """Render a line VAO with the flat shader."""
        self.line_prog["u_view"].write(bytes(view))
        self.line_prog["u_proj"].write(bytes(proj))
        self.line_prog["u_color"].value = color
        self.line_prog["u_opacity"].value = opacity
        vao.render(moderngl.LINES, vertices=count)

    # ─────────────────────────────────────────────
    # Main render (called by moderngl_window each frame)
    # ─────────────────────────────────────────────

    def render(self, time_val: float, frametime: float):
        self.ctx.clear(0.08, 0.08, 0.12, 1.0)

        snap = self.state.read()
        view = self._view_matrix()
        proj = self._proj_matrix()
        eye = self._eye_position()

        # ── Grid floor at Y = -2.0 ──
        self.line_prog["u_view"].write(bytes(
            glm.translate(view, glm.vec3(0.0, -2.0, 0.0))
        ))
        self.line_prog["u_proj"].write(bytes(proj))
        self.line_prog["u_color"].value = (0.25, 0.25, 0.30)
        self.line_prog["u_opacity"].value = 0.5
        self.grid_vao.render(moderngl.LINES, vertices=self.grid_vertex_count)

        # ── Chord cube zones (3x3x3) ──
        self._render_chord_cube(snap, view, proj, eye)

        # ── Hand spheres ──
        self._render_hands(snap, view, proj, eye)

        # ── Note trail ──
        if self.gui_show_trail:
            self._render_note_trail(snap, view, proj, eye)

        # ── Push GUI controls to shared state ──
        self._sync_gui_to_state(snap)

        # ── ImGui overlay ──
        self._render_imgui(snap)

    def _render_chord_cube(self, snap, view, proj, eye):
        """Draw 27 semi-transparent chord zone cubes + active zone highlight."""
        half = self.CUBE_SIZE / 2.0
        zone = self.ZONE_SIZE
        active = (snap.chord_zone_layer, snap.chord_zone_row, snap.chord_zone_col)
        base_opacity = self.gui_zone_opacity
        perf_mode = self.gui_perf_mode

        for layer in range(3):
            for row in range(3):
                for col in range(3):
                    key = (layer, row, col)
                    cx = -half + zone * 0.5 + col * zone
                    cy = -half + zone * 0.5 + layer * zone
                    cz = -half + zone * 0.5 + row * zone

                    r, g, b = CHORD_COLORS[layer][row][col]
                    color = (r / 255.0, g / 255.0, b / 255.0)

                    if key == active:
                        opacity = min(0.7, base_opacity * 3.0)
                        # Brighten
                        color = (
                            min(1.0, color[0] + 0.3),
                            min(1.0, color[1] + 0.3),
                            min(1.0, color[2] + 0.3),
                        )
                    elif perf_mode:
                        if layer == active[0]:
                            opacity = base_opacity * 0.55
                        else:
                            opacity = 0.0
                    elif layer == active[0]:
                        opacity = min(0.25, base_opacity * 1.35)
                    else:
                        opacity = base_opacity * 0.55

                    if opacity < 0.005:
                        continue  # skip invisible cubes

                    scale = zone * 0.92
                    model = glm.translate(glm.mat4(1.0), glm.vec3(cx, cy, cz))
                    model = glm.scale(model, glm.vec3(scale, scale, scale))

                    self._draw_mesh(self.cube_vao, model, color, opacity, view, proj, eye)

        # Outer wireframe cube (shows the full 3x3x3 boundary)
        wire_model = glm.scale(glm.mat4(1.0), glm.vec3(self.CUBE_SIZE))
        self._draw_lines(
            self.wire_cube_vao, self.wire_cube_vertex_count,
            (0.4, 0.4, 0.5), 0.3, view, proj,
        )

    def _render_hands(self, snap, view, proj, eye):
        """Draw palm and fingertip spheres for both hands."""
        rh = snap.right_hand
        lh = snap.left_hand

        if rh.visible:
            pos = self._leap_to_scene(rh.palm_position)
            model = glm.translate(glm.mat4(1.0), pos)
            model = glm.scale(model, glm.vec3(0.15))
            self._draw_mesh(self.sphere_vao, model, (1.0, 0.4, 0.24), 1.0, view, proj, eye)

            if self.gui_show_fingers:
                for i, fp in enumerate(rh.finger_positions):
                    fpos = self._leap_to_scene(fp)
                    fm = glm.translate(glm.mat4(1.0), fpos)
                    fm = glm.scale(fm, glm.vec3(0.06))
                    self._draw_mesh(self.sphere_vao, fm, (1.0, 0.63, 0.47), 1.0, view, proj, eye)

        if lh.visible:
            pos = self._leap_to_scene(lh.palm_position)
            model = glm.translate(glm.mat4(1.0), pos)
            model = glm.scale(model, glm.vec3(0.15))
            self._draw_mesh(self.sphere_vao, model, (0.24, 0.55, 1.0), 1.0, view, proj, eye)

            if self.gui_show_fingers:
                for i, fp in enumerate(lh.finger_positions):
                    fpos = self._leap_to_scene(fp)
                    fm = glm.translate(glm.mat4(1.0), fpos)
                    fm = glm.scale(fm, glm.vec3(0.06))
                    self._draw_mesh(self.sphere_vao, fm, (0.47, 0.67, 1.0), 1.0, view, proj, eye)

    def _render_note_trail(self, snap, view, proj, eye):
        """Draw small spheres showing recently played notes."""
        lh = snap.left_hand
        if not lh.visible:
            return

        if snap.current_step != self._last_step:
            self._last_step = snap.current_step
            trail_pos = self._leap_to_scene(lh.palm_position)
            offset_x = (snap.current_step - snap.step_count / 2) * 0.12

            r, g, b = CHORD_COLORS[snap.chord_zone_layer][snap.chord_zone_row][snap.chord_zone_col]
            self._note_trail_positions.append((
                trail_pos.x + offset_x,
                trail_pos.y - 0.3,
                trail_pos.z,
                r / 255.0, g / 255.0, b / 255.0,
            ))
            # Keep last 10
            if len(self._note_trail_positions) > 10:
                self._note_trail_positions.pop(0)

        for i, (tx, ty, tz, tr, tg, tb) in enumerate(self._note_trail_positions):
            age = len(self._note_trail_positions) - i
            fade = max(0.0, 0.9 - age * 0.09)
            if fade < 0.01:
                continue
            model = glm.translate(glm.mat4(1.0), glm.vec3(tx, ty, tz))
            model = glm.scale(model, glm.vec3(0.04))
            self._draw_mesh(self.sphere_vao, model, (tr, tg, tb), fade, view, proj, eye)

    def _sync_gui_to_state(self, snap):
        """Push GUI slider values to SharedState when they change."""
        note_min = self.gui_note_min
        note_max = self.gui_note_max
        if note_min >= note_max:
            if note_min < 84:
                note_max = note_min + 1
            else:
                note_min = 83
                note_max = 84
            self.gui_note_min = note_min
            self.gui_note_max = note_max

        self.state.update(
            bpm=float(self.gui_bpm),
            note_min=note_min,
            note_max=note_max,
        )

    def _render_imgui(self, snap):
        """Draw the ImGui sidebar panel."""
        imgui.new_frame()

        imgui.set_next_window_position(10, 10, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(300, 700, condition=imgui.FIRST_USE_EVER)

        imgui.begin("Leap Arpeggiator", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        # ── Musical State ──
        if imgui.collapsing_header("Musical State", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text(f"Chord: {snap.chord_type}")
            imgui.text(f"Base Note: {midi_to_name(snap.base_note)}")
            imgui.text(f"Steps: {snap.step_count}")
            imgui.text(f"Spread: {snap.note_spread:.2f}")
            imgui.text(f"Playing: {midi_to_name(snap.current_note)}")

        # ── Settings ──
        if imgui.collapsing_header("Settings", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            changed, self.gui_bpm = imgui.slider_int("BPM", self.gui_bpm, 40, 240)
            changed, self.gui_note_min = imgui.slider_int("Note Min", self.gui_note_min, 24, 84)
            changed, self.gui_note_max = imgui.slider_int("Note Max", self.gui_note_max, 24, 84)
            changed, self.gui_zone_opacity = imgui.slider_float(
                "Zone Opacity", self.gui_zone_opacity, 0.05, 0.5
            )
            _, self.gui_perf_mode = imgui.checkbox("Performance Mode", self.gui_perf_mode)
            _, self.gui_show_fingers = imgui.checkbox("Show Fingertips", self.gui_show_fingers)
            _, self.gui_show_trail = imgui.checkbox("Show Note Trail", self.gui_show_trail)

        # ── Hand Tracking ──
        if imgui.collapsing_header("Hand Tracking")[0]:
            rh = snap.right_hand
            lh = snap.left_hand
            if rh.visible:
                imgui.text(f"Right Palm: ({rh.palm_position[0]:.0f}, {rh.palm_position[1]:.0f}, {rh.palm_position[2]:.0f})")
            else:
                imgui.text("Right Palm: --")
            if lh.visible:
                imgui.text(f"Left Palm: ({lh.palm_position[0]:.0f}, {lh.palm_position[1]:.0f}, {lh.palm_position[2]:.0f})")
                imgui.text(f"Left Roll: {math.degrees(lh.roll):.1f} deg")
                imgui.text(f"Left Pitch: {math.degrees(lh.pitch):.1f} deg")
            else:
                imgui.text("Left Palm: --")

        imgui.separator()
        imgui.text_colored("Controls:", 0.7, 0.7, 0.7)
        imgui.bullet_text("Right hand XYZ -> chord zone")
        imgui.bullet_text("Left hand height -> base note")
        imgui.bullet_text("Left hand roll -> step count")
        imgui.bullet_text("Left hand pitch -> note spread")
        imgui.bullet_text("Mouse drag -> orbit camera")
        imgui.bullet_text("Scroll -> zoom")

        imgui.end()

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    # ─────────────────────────────────────────────
    # Static launcher (called from main.py)
    # ─────────────────────────────────────────────

    @classmethod
    def run(cls, state: SharedState, port: int = 8080):
        """
        Entry point matching the old API.
        port is accepted for backward-compat but ignored (native window).
        """
        cls._shared_state = state
        moderngl_window.run_window_config(cls, args=[])
```

- [ ] **Step 2: Verify it starts (with mock state)**

Run:
```bash
python -c "
from shared_state import SharedState
from visualizer import ArpeggiatorVisualizer
state = SharedState()
# Just test that import + class construction path is valid
print('Visualizer class loaded OK')
"
```
Expected: `Visualizer class loaded OK`

- [ ] **Step 3: Commit**

```bash
git add visualizer.py
git commit -m "feat: rewrite visualizer with ModernGL, ImGui, and orbit camera"
```

---

## Task 5: Update `main.py` to Use the New Visualizer

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update main.py**

Replace the full contents of `main.py` with:

```python
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
    print("  Leap Motion Arpeggiator (ModernGL)")
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

    # ---- Start background threads ----
    print()
    source.start()
    print("  + Data source started")

    interpreter.start()
    print("  + Gesture interpreter started")

    arpeggiator.start()
    print("  + Arpeggiator engine started")

    print("  + Starting ModernGL visualizer...")
    print()

    # Handle Ctrl+C gracefully
    def shutdown(sig, frame):
        print("\n\n  Shutting down...")
        arpeggiator.stop()
        interpreter.stop()
        source.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # 5. Visualizer runs on main thread (OpenGL requires it)
    try:
        ArpeggiatorVisualizer.run(state)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: update main.py to launch ModernGL visualizer"
```

---

## Task 6: Integration Test — Full Application Smoke Test

**Files:**
- No new files. Run the full application.

- [ ] **Step 1: Run the application with mock data**

Run:
```bash
python main.py
```

Expected behavior:
- Console prints startup messages
- A native desktop window opens (1400x900)
- Dark background with a grid floor
- 3x3x3 chord cube with semi-transparent colored boxes
- One highlighted (brighter) box = active chord zone
- Orange sphere (right palm) and blue sphere (left palm) moving smoothly
- ImGui panel in top-left with Musical State, Settings, Hand Tracking sections
- BPM slider, note range sliders, checkboxes all functional
- Mouse drag orbits the camera, scroll zooms
- No crashes, no OpenGL errors in console

- [ ] **Step 2: Test GUI interactions**

In the running window:
- Drag the BPM slider — confirm it changes (reflected in state)
- Toggle "Show Fingertips" — 10 additional smaller spheres appear near hands
- Toggle "Show Note Trail" — small colored breadcrumbs appear below left hand
- Toggle "Performance Mode" — only the active layer of cubes is visible
- Adjust "Zone Opacity" — cube transparency changes
- Orbit camera by dragging mouse — scene rotates smoothly
- Scroll to zoom in/out

- [ ] **Step 3: Fix any issues found, then commit**

```bash
git add -A
git commit -m "fix: address integration issues from smoke testing"
```

(Only if fixes were needed. Skip if everything works.)

---

## Task 7: Polish — Chord Labels as ImGui Overlay Text

**Files:**
- Modify: `visualizer.py`

The Viser version showed chord name labels floating in 3D space inside each cube zone. In ModernGL, the cleanest approach is to project 3D zone centers to 2D screen coordinates and draw them as ImGui overlay text.

- [ ] **Step 1: Add a 3D-to-screen projection helper**

Add this method to `ArpeggiatorVisualizer` after `_eye_position`:

```python
def _world_to_screen(self, world_pos: glm.vec3, view: glm.mat4, proj: glm.mat4) -> tuple:
    """Project a 3D world position to 2D screen pixel coordinates.
    Returns (x, y, is_visible). is_visible is False if behind camera."""
    clip = proj * view * glm.vec4(world_pos, 1.0)
    if clip.w <= 0:
        return (0, 0, False)
    ndc = glm.vec3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
    if abs(ndc.x) > 1.0 or abs(ndc.y) > 1.0 or ndc.z > 1.0:
        return (0, 0, False)
    sx = (ndc.x * 0.5 + 0.5) * self.window_size[0]
    sy = (1.0 - (ndc.y * 0.5 + 0.5)) * self.window_size[1]
    return (sx, sy, True)
```

- [ ] **Step 2: Add chord label rendering in `_render_imgui`**

Add this block inside `_render_imgui`, after `imgui.new_frame()` and before the main window `imgui.begin(...)`:

```python
# ── Floating chord labels (projected from 3D) ──
view = self._view_matrix()
proj = self._proj_matrix()
half = self.CUBE_SIZE / 2.0
zone_sz = self.ZONE_SIZE
active = (snap.chord_zone_layer, snap.chord_zone_row, snap.chord_zone_col)

draw_list = imgui.get_background_draw_list()
for layer in range(3):
    for row in range(3):
        for col in range(3):
            key = (layer, row, col)
            # Determine visibility
            if self.gui_perf_mode:
                if key != active and layer != active[0]:
                    continue
            elif key != active and layer != active[0]:
                continue

            cx = -half + zone_sz * 0.5 + col * zone_sz
            cy = -half + zone_sz * 0.5 + layer * zone_sz
            cz = -half + zone_sz * 0.5 + row * zone_sz
            sx, sy, visible = self._world_to_screen(
                glm.vec3(cx, cy, cz), view, proj
            )
            if not visible:
                continue

            quality = CHORD_GRID[layer][row][col]
            label = chord_name_from_root(snap.base_note, quality)
            if key == active:
                draw_list.add_text(sx - 15, sy - 8, imgui.get_color_u32_rgba(1, 1, 1, 1), label)
            else:
                draw_list.add_text(sx - 15, sy - 8, imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 0.6), label)

# ── HUD base note label (top of cube) ──
sx, sy, visible = self._world_to_screen(glm.vec3(0, 2.2, 0), view, proj)
if visible:
    base_text = f"Base: {midi_to_name(snap.base_note)}"
    draw_list.add_text(sx - 30, sy - 8, imgui.get_color_u32_rgba(1, 1, 0.4, 1), base_text)
```

- [ ] **Step 3: Verify labels appear correctly**

Run: `python main.py`
Expected:
- Chord names (e.g., "Cmaj", "Gm") float over each visible cube zone
- Active zone label is white and bright
- Non-active zone labels are dimmer
- "Base: C4" label floats above the cube
- Labels track correctly when orbiting the camera

- [ ] **Step 4: Commit**

```bash
git add visualizer.py
git commit -m "feat: add 3D-projected chord labels and base note HUD via ImGui overlay"
```

---

## Task 8: Handle Window Resize and Clean Shutdown

**Files:**
- Modify: `visualizer.py`

- [ ] **Step 1: Add resize handling**

Add this method to `ArpeggiatorVisualizer`:

```python
def resize(self, width: int, height: int):
    self.window_size = (width, height)
    self.ctx.viewport = (0, 0, width, height)
```

- [ ] **Step 2: Verify resize works**

Run: `python main.py`
Resize the window by dragging edges. The viewport and projection should adapt correctly — no stretching, no black bars.

- [ ] **Step 3: Commit**

```bash
git add visualizer.py
git commit -m "fix: handle window resize for correct aspect ratio and viewport"
```

---

## Task 9: Final Verification and Cleanup

- [ ] **Step 1: Run the full application and verify all features**

Run: `python main.py`

Checklist:
- [ ] Window opens with correct title "Leap Arpeggiator"
- [ ] Grid floor visible
- [ ] 3x3x3 chord cube renders with colors from CHORD_COLORS
- [ ] Active zone highlighted (brighter, higher opacity)
- [ ] Performance mode hides non-active layers
- [ ] Right hand (orange) and left hand (blue) spheres move
- [ ] Fingertips toggle on/off
- [ ] Note trail toggle on/off
- [ ] Chord labels projected correctly in 3D
- [ ] Base note HUD label above cube
- [ ] ImGui panel shows correct musical state values
- [ ] BPM slider changes arpeggiator speed
- [ ] Note min/max sliders work with validation (min < max)
- [ ] Zone opacity slider changes cube transparency
- [ ] Camera orbit (mouse drag) works smoothly
- [ ] Camera zoom (scroll) works
- [ ] Window resize works without distortion
- [ ] Ctrl+C shuts down cleanly
- [ ] No OpenGL errors or Python exceptions in console

- [ ] **Step 2: Remove viser from imports if still referenced anywhere**

Run: `grep -r "import viser" *.py`
Expected: No matches. If found, remove the import.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup after ModernGL rewrite"
```

---

## Appendix: Key Differences from Viser Version

| Aspect | Viser | ModernGL |
|--------|-------|----------|
| Window | Browser tab (WebSocket) | Native desktop (pyglet) |
| Camera | Built-in orbit | Custom orbit via mouse events |
| GUI | Viser GUI (sliders/text/folders) | ImGui panels |
| 3D Text | `add_label()` | Screen-space ImGui overlay |
| Meshes | `add_box()`, `add_icosphere()` | Custom VAOs with GLSL shaders |
| Transparency | Per-object `.opacity` | Alpha blending + per-draw uniform |
| Grid | `add_grid()` | Line VAO on XZ plane |
| Performance | ~18 FPS browser | VSync native (~60 FPS) |
| Dependencies | `viser` | `moderngl`, `moderngl-window`, `imgui`, `PyGLM` |
