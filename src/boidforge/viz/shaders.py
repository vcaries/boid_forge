"""GLSL 3.30 shader sources for the boid renderer.

All programs target the OpenGL 3.3 core profile (the lowest common denominator
across the desktop GPUs this replays on). They are plain string constants — no
logic — kept in one place so the renderer module stays focused on GL resource
management and draw orchestration.

Pipeline:

* :data:`POINT_VERTEX` / :data:`POINT_FRAGMENT` draw each boid as a soft,
  velocity-coloured glow sprite (additive, HDR).
* :data:`FULLSCREEN_VERTEX` feeds every post-processing pass.
* :data:`SOLID_FRAGMENT` fades the trail-accumulation buffer.
* :data:`BRIGHT_FRAGMENT` / :data:`BLUR_FRAGMENT` build the bloom.
* :data:`COMPOSITE_FRAGMENT` tonemaps the HDR scene to the display.
"""

from __future__ import annotations

#: Vertex shader for instanced boid point sprites.
POINT_VERTEX: str = """
#version 330
in vec2 in_pos;
in vec2 in_vel;
in float in_aux;   // precomputed normalized density in [0,1] (density mode)

uniform mat4 u_mvp;
uniform float u_point_size;       // base sprite diameter in pixels
uniform float u_min_size;         // floor so distant boids stay visible
uniform float u_speed_lo;         // speed mapped to the cold end of the ramp
uniform float u_speed_inv_range;  // 1 / (speed_hi - speed_lo)
uniform int u_color_mode;         // 0 speed, 1 heading, 2 uniform, 3 density
uniform float u_uniform_t;        // LUT coordinate when color mode == uniform

out float v_t;
out float v_speed;

const float TWO_PI = 6.28318530718;

void main() {
    float speed = length(in_vel);
    // Map speed across the *observed* [lo, hi] range so the colour ramp spans
    // the actual distribution — slow boids land cold, fast boids hot — instead
    // of all clustering at one end when speeds are tightly clamped.
    float norm = clamp((speed - u_speed_lo) * u_speed_inv_range, 0.0, 1.0);

    if (u_color_mode == 1) {
        float ang = atan(in_vel.y, in_vel.x) / TWO_PI + 0.5;
        v_t = ang;
    } else if (u_color_mode == 2) {
        v_t = u_uniform_t;
    } else if (u_color_mode == 3) {
        v_t = clamp(in_aux, 0.0, 1.0);
    } else {
        v_t = norm;
    }
    v_speed = norm;

    gl_Position = u_mvp * vec4(in_pos, 0.0, 1.0);
    // Faster boids render slightly larger for a sense of energy.
    float size = u_point_size * (0.65 + 0.7 * norm);
    gl_PointSize = clamp(size, u_min_size, 64.0);
}
"""

#: Fragment shader for boid point sprites: soft core plus a wide halo.
POINT_FRAGMENT: str = """
#version 330
in float v_t;
in float v_speed;

uniform sampler2D u_lut;   // 256x1 RGB colormap
uniform float u_glow;      // halo strength
uniform float u_intensity; // overall emission multiplier

out vec4 frag;

void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(pc, pc);
    if (r2 > 1.0) discard;

    float r = sqrt(r2);
    float core = smoothstep(1.0, 0.0, r);      // tight disc
    float halo = exp(-r2 * 3.5);               // soft falloff
    float a = core * core * 0.75 + halo * u_glow;

    vec3 col = texture(u_lut, vec2(v_t, 0.5)).rgb;
    // Brighter, whiter hot core for the fastest boids.
    col = mix(col, vec3(1.0), v_speed * 0.25 * core);

    vec3 emission = col * u_intensity * a;
    frag = vec4(emission, a);
}
"""

#: Pass-through vertex shader for fullscreen post-processing triangles.
FULLSCREEN_VERTEX: str = """
#version 330
in vec2 in_pos;   // clip-space position of a fullscreen triangle
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

#: Solid-colour fragment used to fade the trail buffer toward black.
SOLID_FRAGMENT: str = """
#version 330
uniform vec4 u_color;
out vec4 frag;
void main() {
    frag = u_color;
}
"""

#: Bright-pass: keep only luminance above a threshold to seed the bloom.
BRIGHT_FRAGMENT: str = """
#version 330
in vec2 v_uv;
uniform sampler2D u_scene;
uniform float u_threshold;
out vec4 frag;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float keep = max(lum - u_threshold, 0.0) / max(lum, 1e-4);
    frag = vec4(c * keep, 1.0);
}
"""

#: Separable 9-tap Gaussian blur (run once per axis via u_dir).
BLUR_FRAGMENT: str = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2 u_dir;   // texel step along the blur axis
out vec4 frag;

const float W0 = 0.227027;
const float W1 = 0.194595;
const float W2 = 0.121622;
const float W3 = 0.054054;
const float W4 = 0.016216;

void main() {
    vec3 sum = texture(u_tex, v_uv).rgb * W0;
    sum += texture(u_tex, v_uv + u_dir * 1.0).rgb * W1;
    sum += texture(u_tex, v_uv - u_dir * 1.0).rgb * W1;
    sum += texture(u_tex, v_uv + u_dir * 2.0).rgb * W2;
    sum += texture(u_tex, v_uv - u_dir * 2.0).rgb * W2;
    sum += texture(u_tex, v_uv + u_dir * 3.0).rgb * W3;
    sum += texture(u_tex, v_uv - u_dir * 3.0).rgb * W3;
    sum += texture(u_tex, v_uv + u_dir * 4.0).rgb * W4;
    sum += texture(u_tex, v_uv - u_dir * 4.0).rgb * W4;
    frag = vec4(sum, 1.0);
}
"""

#: Final composite: tonemap HDR scene, add bloom, vignette, gamma-correct.
COMPOSITE_FRAGMENT: str = """
#version 330
in vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_exposure;
uniform float u_bloom_strength;
uniform float u_vignette;      // 0 = off, 1 = strong
uniform vec3 u_background;
out vec4 frag;

// ACES filmic tonemap approximation (Narkowicz).
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb;
    hdr += texture(u_bloom, v_uv).rgb * u_bloom_strength;
    hdr *= u_exposure;

    vec3 mapped = aces(hdr);
    // Composite over the background so faded trails settle on the base colour.
    mapped = u_background + mapped;

    // Radial vignette.
    vec2 d = v_uv - 0.5;
    float vig = 1.0 - u_vignette * dot(d, d) * 2.2;
    mapped *= clamp(vig, 0.0, 1.0);

    mapped = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
    frag = vec4(mapped, 1.0);
}
"""
