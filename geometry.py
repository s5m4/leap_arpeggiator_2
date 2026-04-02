"""Mesh generation utilities returning numpy arrays."""

import numpy as np
import math


def make_cube():
    """Unit cube centered at origin with flat-shading normals.

    Returns:
        vertices: (24, 3) float32
        normals:  (24, 3) float32
        indices:  (36,)   int32
    """
    # Half-extent
    h = 0.5
    # Each face: 4 vertices sharing the face normal
    # Faces: +X, -X, +Y, -Y, +Z, -Z
    face_data = [
        # +X face
        ([h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h], [1, 0, 0]),
        # -X face
        ([-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h], [-1, 0, 0]),
        # +Y face
        ([-h, h, -h], [-h, h, h], [h, h, h], [h, h, -h], [0, 1, 0]),
        # -Y face
        ([-h, -h, h], [-h, -h, -h], [h, -h, -h], [h, -h, h], [0, -1, 0]),
        # +Z face
        ([-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h], [0, 0, 1]),
        # -Z face
        ([h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h], [0, 0, -1]),
    ]

    vertices = np.empty((24, 3), dtype="f4")
    normals = np.empty((24, 3), dtype="f4")
    indices = np.empty(36, dtype="i4")

    for i, (v0, v1, v2, v3, n) in enumerate(face_data):
        base = i * 4
        vertices[base] = v0
        vertices[base + 1] = v1
        vertices[base + 2] = v2
        vertices[base + 3] = v3
        normals[base:base + 4] = n
        ti = i * 6
        indices[ti:ti + 6] = [base, base + 1, base + 2, base, base + 2, base + 3]

    return vertices, normals, indices


def make_sphere(rings=12, sectors=24):
    """UV sphere, radius 1.0, centered at origin.

    Returns:
        vertices: ((rings+1)*(sectors+1), 3) float32
        normals:  ((rings+1)*(sectors+1), 3) float32
        indices:  (rings*sectors*6,)         int32
    """
    num_verts = (rings + 1) * (sectors + 1)
    vertices = np.empty((num_verts, 3), dtype="f4")

    idx = 0
    for r in range(rings + 1):
        phi = math.pi * r / rings  # 0 to pi
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors  # 0 to 2pi
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            vertices[idx] = (x, y, z)
            idx += 1

    # For a unit sphere, normals == positions
    normals = vertices.copy()

    # Indices
    indices = np.empty(rings * sectors * 6, dtype="i4")
    idx = 0
    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + (sectors + 1)
            indices[idx] = a
            indices[idx + 1] = b
            indices[idx + 2] = a + 1
            indices[idx + 3] = a + 1
            indices[idx + 4] = b
            indices[idx + 5] = b + 1
            idx += 6

    return vertices, normals, indices


def make_grid(size=8.0, divisions=16):
    """Flat grid of lines on XZ plane at Y=0.

    Returns:
        vertices: (N, 3) float32 for GL_LINES
    """
    half = size / 2.0
    step = size / divisions
    lines = []

    for i in range(divisions + 1):
        x = -half + i * step
        # Line along Z axis
        lines.append([x, 0.0, -half])
        lines.append([x, 0.0, half])
        # Line along X axis
        z = -half + i * step
        lines.append([-half, 0.0, z])
        lines.append([half, 0.0, z])

    return np.array(lines, dtype="f4")


def make_wire_cube():
    """Wireframe unit cube (12 edges) centered at origin.

    Returns:
        vertices: (24, 3) float32 for GL_LINES
    """
    h = 0.5
    # 8 corners of the cube
    corners = [
        [-h, -h, -h],  # 0
        [h, -h, -h],   # 1
        [h, h, -h],    # 2
        [-h, h, -h],   # 3
        [-h, -h, h],   # 4
        [h, -h, h],    # 5
        [h, h, h],     # 6
        [-h, h, h],    # 7
    ]
    # 12 edges as pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]
    verts = []
    for a, b in edges:
        verts.append(corners[a])
        verts.append(corners[b])

    return np.array(verts, dtype="f4")
