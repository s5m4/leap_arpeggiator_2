"""GLSL shader source strings for ModernGL rendering."""

# ---------------------------------------------------------------------------
# Mesh shader – Blinn-Phong lighting for cubes and spheres
# ---------------------------------------------------------------------------

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
    float diffuse = 0.65 * diff;

    // Specular (Blinn-Phong)
    vec3 view_dir = normalize(u_view_pos - v_frag_pos);
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway), 0.0), 32.0);
    float specular = 0.3 * spec;

    vec3 result = (ambient + diffuse + specular) * u_color;
    frag_color = vec4(result, u_opacity);
}
"""

# ---------------------------------------------------------------------------
# Line shader – flat color for grid and wireframes
# ---------------------------------------------------------------------------

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
