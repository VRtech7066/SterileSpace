// GLSL ES 3.00 WebGL2 shaders. The update step runs with transform
// feedback exactly like the desktop version. UPDATE_VS MUST stay numerically identical to src/logic/simRef.js.

// Cursor repulsion, applied to the *drawn* position only (clip space), so the physics step stays identical to
// src/logic/simRef.js. Particles near the cursor are pushed radially away with a smoothstep falloff; because the
// offset is a pure function of the (eased) cursor position, they glide away and flow back as it moves off.
const MOUSE_GLSL = `
uniform vec2 u_mouse;       // cursor in NDC, -1..1 (y up)
uniform vec2 u_mouse_cfg;   // x: framebuffer aspect (w / h), y: repulsion strength 0..1 (eased on the CPU)
const float MOUSE_RADIUS = 0.35;   // in NDC units of screen height
const float MOUSE_PUSH = 0.2;      // maximum displacement
vec4 mouse_repel(vec4 clip) {
  vec2 ndc = clip.xy / clip.w;
  vec2 d = ndc - u_mouse;
  d.x *= u_mouse_cfg.x;                                         // circular falloff on a non-square screen
  float dist = length(d);
  float f = 1.0 - smoothstep(0.0, MOUSE_RADIUS, dist);
  vec2 dir = d / max(dist, 1e-4);
  dir.x /= u_mouse_cfg.x;
  ndc += dir * (f * MOUSE_PUSH * u_mouse_cfg.y);
  return vec4(ndc * clip.w, clip.zw);
}

// 3D Simplex & Curl Noise for fluid organic nebula
vec4 permute(vec4 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

vec3 curlNoise(vec3 p) {
  const float e = 0.035;
  const vec3 o1 = vec3(13.1, 7.4, 3.8);
  const vec3 o2 = vec3(31.7, 47.1, 19.3);
  const vec3 o3 = vec3(91.3, 11.2, 53.7);

  float dz_dy = snoise(p + vec3(0.0, e, 0.0) + o3) - snoise(p - vec3(0.0, e, 0.0) + o3);
  float dy_dz = snoise(p + vec3(0.0, 0.0, e) + o2) - snoise(p - vec3(0.0, 0.0, e) + o2);
  float dx_dz = snoise(p + vec3(0.0, 0.0, e) + o1) - snoise(p - vec3(0.0, 0.0, e) + o1);
  float dz_dx = snoise(p + vec3(e, 0.0, 0.0) + o3) - snoise(p - vec3(e, 0.0, 0.0) + o3);
  float dy_dx = snoise(p + vec3(e, 0.0, 0.0) + o2) - snoise(p - vec3(e, 0.0, 0.0) + o2);
  float dx_dy = snoise(p + vec3(0.0, e, 0.0) + o1) - snoise(p - vec3(0.0, e, 0.0) + o1);

  return vec3(dz_dy - dy_dz, dx_dz - dz_dx, dy_dx - dx_dy) / (2.0 * e);
}

// Fast analytical divergence-free fluid flow (zero frame drops on GPU)
vec3 harmonic_flow(vec3 p, float t) {
  vec3 w1 = vec3(
    sin(p.y * 3.6 + t * 0.85) + cos(p.z * 3.0 - t * 0.7),
    sin(p.z * 3.2 + t * 0.75) + cos(p.x * 2.8 + t * 0.65),
    sin(p.x * 3.0 - t * 0.60) + cos(p.y * 3.4 + t * 0.8)
  );
  vec3 w2 = vec3(
    cos(p.y * 7.2 - t * 1.25) * 0.40,
    sin(p.z * 6.6 + t * 1.10) * 0.40,
    cos(p.x * 6.8 - t * 1.15) * 0.40
  );
  return w1 * 0.55 + w2;
}

void calc_nebula(vec3 inP, float time, float nebulaMix, vec3 baseCol, out vec3 outP, out vec3 outCol, out float highlight, out float spatialMask) {
  if (nebulaMix < 0.001) {
    outP = inP;
    outCol = baseCol;
    highlight = 0.0;
    spatialMask = 1.0;
    return;
  }

  // 1. Spatial Scale & Natural Free Flow (Volumetric Z Depth):
  vec3 baseShape = inP * vec3(2.6, 2.4, 3.2);
  baseShape.y -= 0.22; // Shift entire nebula down so it pools nicely behind the card

  // Fluid displacement: heavily push particle noise into the Z-axis for macro 3D depth
  vec3 flow = harmonic_flow(baseShape * 0.45, time);
  vec3 disp = flow * vec3(0.35, 0.35, 0.95);
  disp.y += sin(baseShape.x * 1.5 + time * 0.8) * 0.09;
  disp.z += cos(baseShape.x * 2.2 + baseShape.y * 1.8 + time * 0.75) * 0.55;

  // Unclamped particles: move freely off-screen along the noise field
  outP = baseShape + disp;
  outP = mix(inP, outP, nebulaMix);

  // Vertical envelope keeps nebula centered/pooled, while horizontal flow softly fades far off-screen
  float yRel = (outP.y + 0.22) / 1.15;
  float yMask = clamp(1.0 - smoothstep(0.55, 1.20, yRel * yRel), 0.0, 1.0);
  float xFade = clamp(1.0 - smoothstep(2.6, 3.5, abs(outP.x)), 0.0, 1.0);
  spatialMask = yMask * xFade;

  // 2. Strict Primary Gradient: Cyan -> Purple -> Gold
  const vec3 CYAN_BASE    = vec3(0.0, 0.8, 1.0);
  const vec3 CYAN_HIGH    = vec3(0.4, 0.9, 1.0);

  const vec3 PURPLE_BASE  = vec3(0.4, 0.0, 0.8);
  const vec3 PURPLE_HIGH  = vec3(0.7, 0.2, 0.9);

  const vec3 GOLD_BASE    = vec3(1.0, 0.7, 0.2);
  const vec3 GOLD_HIGH    = vec3(0.8, 0.5, 0.1);

  const vec3 SILVER_WHITE = vec3(0.95, 0.95, 1.0);

  // Flow-based harmonic coordinates for interwoven blending
  float waveCol = sin(outP.x * 1.1 + (outP.y + 0.22) * 1.4 + time * 0.6) * 0.5 + 0.5;
  float waveCross = cos(outP.x * 1.2 - outP.z * 1.1 - time * 0.5) * 0.5 + 0.5;
  float tCol = clamp(mix(waveCol, waveCross, 0.5) * 0.75 + (outP.x * 0.16 + 0.5) * 0.25, 0.0, 1.0);

  // Local flow curvature highlights
  float hlCyan   = clamp(sin((outP.y + 0.22) * 3.0 + time * 0.9) * 0.5 + 0.5, 0.0, 1.0);
  float hlPurple = clamp(cos(outP.x * 2.8 - time * 0.8) * 0.5 + 0.5, 0.0, 1.0);
  float hlGold   = clamp(sin(outP.z * 3.2 + time * 0.7) * 0.5 + 0.5, 0.0, 1.0);

  vec3 colCyan   = mix(CYAN_BASE, CYAN_HIGH, hlCyan);
  vec3 colPurple = mix(PURPLE_BASE, PURPLE_HIGH, hlPurple);
  vec3 colGold   = mix(GOLD_BASE, GOLD_HIGH, hlGold);

  // Primary flowing gradient strictly mapped between Cyan, Purple, and Gold
  vec3 finalCol;
  if (tCol < 0.46) {
    float k = smoothstep(0.08, 0.44, tCol);
    finalCol = mix(colCyan, colPurple, k);
  } else {
    float k = smoothstep(0.48, 0.88, tCol);
    finalCol = mix(colPurple, colGold, k);
  }

  // Cross-woven threads of contrasting Gold & Cyan
  float thread = sin(outP.x * 2.4 - (outP.y + 0.22) * 2.6 + outP.z * 1.8 + time * 0.95);
  if (thread > 0.48) {
    finalCol = mix(finalCol, colGold, smoothstep(0.48, 0.85, thread) * 0.65);
  } else if (thread < -0.48) {
    finalCol = mix(finalCol, colCyan, smoothstep(0.48, 0.85, -thread) * 0.65);
  }

  // 3. Silver as a Sharp Highlight (Strict Color Threshold):
  // High-frequency harmonic interference generates sharp, thin piercing strands.
  // Apply Silver/White only to the absolute highest 2-5% peaks of the color noise field.
  float sWave1 = sin(outP.x * 4.2 + (outP.y + 0.22) * 5.0 + outP.z * 2.8 + time * 1.15);
  float sWave2 = cos(outP.x * 3.1 - (outP.y + 0.22) * 3.8 - outP.z * 4.4 - time * 0.90);
  float colorNoise = (sWave1 * 0.5 + 0.5) * (sWave2 * 0.5 + 0.5);

  float silverStrand = smoothstep(0.82, 0.96, colorNoise);
  finalCol = mix(finalCol, SILVER_WHITE * 1.45, silverStrand * 0.95);

  // Base brightness boost to lift colors out of dark space
  finalCol *= 1.25;

  outCol = mix(baseCol, finalCol, nebulaMix);
  highlight = silverStrand;
}
`;

export const UPDATE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_vel;
layout(location=2) in vec3 in_target;   // this particle's point on the current model (normalized units)
layout(location=3) in vec4 in_seed;     // xyz: random unit direction, w: random 0..1
layout(location=4) in vec4 in_offset;   // xyz: explode vector, w: explode stage 0..1 (machines only)
layout(location=5) in float in_group;   // part id (label index) for picking, -1 = none

uniform float u_dt;
uniform float u_time;
uniform int   u_mode;      // 0 idle, 1 sphere, 2 form, 3 dissolve
uniform float u_form_t;    // seconds since FORM began
uniform float u_kick;      // 1.0 on the single frame of a burst, else 0.0
uniform mat3  u_rot;       // model rotation (tilt * slow spin)
uniform float u_sphere_r;
uniform float u_explode;   // 0 assembled .. 1 fully exploded (driven by hand openness)
uniform float u_scale;     // zoom-out that keeps the exploded view on screen
uniform vec3  u_excenter;  // centre of the fully exploded cloud
uniform float u_sel;       // selected part id (-100 = none)
uniform vec3  u_pull;      // world-space pull applied to the selected part (pinch = pull it out toward you)

out vec3 out_pos;
out vec3 out_vel;

const float SWIRL = 0.45;
const float ORBIT = 1.6;
const float RAD_DAMP = 5.0;
const float K_RAD = 14.0;
const float DAMP = 2.2;
const float KICK = 2.2;
const float K_FORM = 80.0;
const float STAGGER = 0.6;
const float OUT_ACC = 2.5;

// Gradient of phi = sum(sin(k.p + w t)); the flow cross(n, grad phi) is tangent to the sphere and
// divergence-free on it, so particles swirl in vortices without clumping.
vec3 grad_phi(vec3 p, float t) {
  const vec3 k1 = vec3(1.7, 2.3, 0.0);
  const vec3 k2 = vec3(0.0, 1.9, 2.6);
  const vec3 k3 = vec3(2.4, 0.0, 1.5);
  return k1 * cos(dot(k1, p) + 0.9 * t) + k2 * cos(dot(k2, p) - 0.7 * t) + k3 * cos(dot(k3, p) + 1.1 * t);
}

void main() {
  vec3 p = in_pos;
  vec3 v = in_vel;
  float dt = u_dt;
  float r = length(p) + 1e-5;
  vec3 n = p / r;
  vec3 tang = cross(n, grad_phi(p, u_time));
  vec3 acc = vec3(0.0);

  if (u_mode == 0) {                         // idle nebula: maintain wide volume
    p += v * dt;
    v *= exp(-1.0 * dt);
  } else if (u_mode == 1) {                  // swirling sphere shell
    float R = u_sphere_r * (0.93 + 0.07 * in_seed.w);
    float ang = in_seed.w * 6.2831853;
    vec3 axis = normalize(cross(in_seed.xyz, vec3(sin(ang), 0.37, cos(ang))));
    vec3 orbit = cross(axis, p) * ORBIT;
    float vr = dot(v, n);
    float centripetal = (dot(v, v) - vr * vr) / r;
    acc = (orbit + tang * SWIRL - v) * DAMP + n * ((R - r) * K_RAD - vr * RAD_DAMP - centripetal);
    v += in_seed.xyz * (u_kick * KICK);
  } else if (u_mode == 2) {                  // fly to the model, staggered, critically damped
    float delay = in_seed.w * STAGGER;
    float e = smoothstep(delay, delay + 0.35, u_form_t);
    float k = K_FORM * e;
    float s = clamp((u_explode - in_offset.w) / max(1.0 - in_offset.w, 1e-3), 0.0, 1.0);
    s = s * s * (3.0 - 2.0 * s);
    vec3 tgt = u_rot * ((in_target + in_offset.xyz * s - u_excenter * u_explode) * u_scale);
    if (abs(in_group - u_sel) < 0.5) tgt += u_pull;
    acc = k * (tgt - p) - 2.0 * sqrt(k) * v + (1.0 - e) * (tang * SWIRL - v * DAMP);
  } else {                                   // dissolve: blow outward
    acc = normalize(p * 0.5 + in_seed.xyz) * OUT_ACC + tang * 0.8 - v * 0.3;
  }
  if (u_mode != 0) {
    v += acc * dt;
    p += v * dt;
  }
  out_pos = p;
  out_vel = v;
}`;

export const DISCARD_FS = `#version 300 es
precision mediump float;
out vec4 f_color;
void main() { f_color = vec4(0.0); }`;

export const PARTICLE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_vel;
layout(location=2) in vec3 in_tint;
layout(location=3) in float in_group;
uniform mat4 u_mvp;
uniform vec3 u_color;
uniform float u_tint_mix;    // 0 = model colour (sphere), 1 = per-part colours (formed machine)
uniform float u_alpha;
uniform float u_point_px;    // point size in px at distance 1 (scaled with framebuffer height)
uniform float u_gain;
uniform float u_sel;         // selected part id: it glows, everything else dims (-100 = none)
uniform float u_cut_on;      // cut-away: hide everything beyond the plane x = u_cut
uniform float u_cut;
uniform float u_flow;        // travelling pulses of light (blood / air flow)
uniform float u_time;
uniform float u_nebula;      // 1 = fluid cosmic nebula, 0 = formed model / sphere
uniform int   u_state;       // 0 = Idle Nebula, 1 = Loading Sphere, 2 = Model
${MOUSE_GLSL}
out vec3 v_col;
out float v_a;
out float v_soft;
flat out int v_state;

void main() {
  if (u_cut_on > 0.5 && in_pos.x > u_cut) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 1.0;
    v_col = vec3(0.0);
    v_a = 0.0;
    v_soft = 2.0;
    v_state = u_state;
    return;
  }

  if (u_state == 0) {
    // =========================================================================
    // 1. THE IDLE STATE (First Page Nebula)
    // Blend Mode: gl.blendFunc(gl.SRC_ALPHA, gl.ONE) (Additive)
    // Dynamic noise-driven point sizes, vibrant Cyan/Purple/Gold gradient
    // =========================================================================
    v_state = 0;
    vec3 base = mix(u_color, in_tint, u_tint_mix);
    vec3 pWorld;
    vec3 pCol;
    float highlight;
    float sMask;
    calc_nebula(in_pos, u_time, u_nebula, base, pWorld, pCol, highlight, sMask);

    gl_Position = mouse_repel(u_mvp * vec4(pWorld, 1.0));

    // Volumetric Depth of Field:
    float zNorm = clamp((pWorld.z + 1.4) / 2.8, 0.0, 1.0);
    float zMistAlpha = mix(0.45, 1.30, smoothstep(0.05, 0.82, zNorm));
    float zBokehScale = mix(0.9, 2.8, pow(zNorm, 2.2));
    float zSoft = mix(3.2, 0.95, smoothstep(0.18, 0.88, zNorm));

    float pSeed = fract(sin(float(gl_VertexID) * 12.9898 + 78.233) * 43758.5453);
    float isStar = 0.0;
    if (pSeed > 0.985) {
      isStar = 1.0;
      float twinkle = 0.82 + 0.32 * sin(float(gl_VertexID) * 0.37 + u_time * 4.2);
      float baseStar = mix(2.8, 3.6, (pSeed - 0.985) / 0.015);
      gl_PointSize = baseStar * mix(0.9, 1.5, zNorm) * twinkle;
      v_soft = 1.6;
    } else {
      gl_PointSize = 1.2 * zBokehScale;
      v_soft = zSoft;
    }

    float bokehAlphaMod = mix(1.0, 0.80, smoothstep(0.70, 1.0, zNorm));
    v_a = u_alpha * zMistAlpha * bokehAlphaMod * mix(mix(0.65, 1.0, highlight), 1.6, isStar) * sMask;
    v_col = mix(pCol * 1.35, vec3(1.0, 1.0, 1.0) * 1.5, isStar * 0.90);

  } else if (u_state == 1) {
    // =========================================================================
    // 2. THE SPHERE STATE (Textured Loading)
    // Blend Mode: gl.blendFunc(gl.SRC_ALPHA, gl.ONE) (Additive)
    // Fixed medium point size 1.5, base cyan/blue color with low alpha 0.15
    // Overlapping lines build up into a luminous, dense, shimmering texture
    // =========================================================================
    v_state = 1;
    v_soft = 2.0;

    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = 1.5;

    v_col = vec3(0.8, 0.9, 1.0);
    v_a = 0.15;

  } else {
    // =========================================================================
    // 3. THE MODEL STATE (Brain, Heart, etc.)
    // Blend Mode: gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA) (Standard Alpha)
    // 80% fine base size 1.0, 15-20% shimmering sparkles (1.5 to 2.0) for metallic luster
    // =========================================================================
    v_state = 2;
    v_soft = 2.0;

    gl_Position = u_mvp * vec4(in_pos, 1.0);

    // 15-20% Sparkle Density: roughly 18% of particles scale to 1.5 - 2.0
    float pSeed = fract(sin(float(gl_VertexID) * 12.9898 + 78.233) * 43758.5453);
    if (pSeed > 0.82) {
      gl_PointSize = mix(1.5, 2.0, (pSeed - 0.82) / 0.18);
    } else {
      gl_PointSize = 1.0;
    }

    vec3 base = mix(u_color, in_tint, u_tint_mix);
    if (u_sel > -50.0) base *= abs(in_group - u_sel) < 0.5 ? 1.8 : 0.25;
    base *= 1.0 + u_flow * 1.6 * pow(max(0.0, sin(in_pos.y * 9.0 - u_time * 5.0)), 14.0);
    float edge = u_cut_on * (1.0 - smoothstep(0.0, 0.03, u_cut - in_pos.x));
    base = mix(base, vec3(1.0), edge * 0.7) * (1.0 + edge);

    v_col = base;
    v_a = u_alpha;
  }
}
`;

export const PARTICLE_FS = `#version 300 es
precision highp float;
in vec3 v_col;
in float v_a;
in float v_soft;
flat in int v_state;
uniform float u_gain;
out vec4 f_color;

void main() {
  if (v_state == 2) {
    // 3. THE MODEL STATE:
    // Standard alpha blending: subtle color lift + 20% silver shimmer for premium metallic finish
    vec3 boostedColor = v_col * 1.18;
    vec3 finalModelColor = mix(boostedColor, vec3(0.95, 0.95, 1.0), 0.20);
    f_color = vec4(finalModelColor, v_a);
    return;
  }

  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;

  if (v_state == 1) {
    // 2. THE SPHERE STATE:
    // Additive blending: base color (0.8, 0.9, 1.0) with low alpha (0.15)
    // Soft core so overlapping lines smoothly build up into a luminous, textured point cloud volume
    float glow = exp(-d2 * 2.5);
    f_color = vec4(v_col, v_a * glow);
    return;
  }

  // 1. THE IDLE STATE (First Page Nebula):
  // Additive glowing cosmic volume
  float glow = exp(-d2 * v_soft);
  f_color = vec4(v_col, v_a * glow);
}
`;

export const TRAIL_VS = `#version 300 es
precision highp float;
// Instanced LINES: 2 vertices per particle instance. Vertex 0 = head, vertex 1 = tail = pos - vel*len.
layout(location=0) in vec3 in_pos;   // per-instance
layout(location=1) in vec3 in_vel;   // per-instance
layout(location=2) in vec3 in_tint;  // per-instance
layout(location=3) in float in_group; // per-instance
uniform float u_sel;
uniform float u_cut_on;
uniform float u_cut;
uniform mat4 u_mvp;
uniform vec3 u_color;
uniform float u_tint_mix;
uniform float u_alpha;
uniform float u_trail;               // trail length in seconds of motion
uniform float u_time;
uniform float u_nebula;
${MOUSE_GLSL}
out vec3 v_col;
out float v_a;
void main() {
  float tail = float(gl_VertexID & 1);
  if (u_nebula > 0.001) {
    vec3 base = mix(mix(u_color, in_tint, u_tint_mix), vec3(1.0), 0.3);
    vec3 pHead, pCol;
    float hl, sMask;
    calc_nebula(in_pos, u_time, u_nebula, base, pHead, pCol, hl, sMask);
    vec3 p = pHead - in_vel * (u_trail * tail);
    gl_Position = mouse_repel(u_mvp * vec4(p, 1.0));
    float speed = length(in_vel);
    v_col = pCol;
    v_a = u_alpha * (1.0 - tail) * clamp(speed * 0.5 - 0.1, 0.0, 1.0) * sMask;
  } else {
    // Exact original model trail code
    vec3 p = in_pos - in_vel * (u_trail * tail);
    gl_Position = u_mvp * vec4(p, 1.0);
    float speed = length(in_vel);
    v_col = mix(mix(u_color, in_tint, u_tint_mix), vec3(1.0), 0.3);
    v_a = u_alpha * (1.0 - tail) * clamp(speed * 0.5 - 0.1, 0.0, 1.0);
  }
  if (u_sel > -50.0 && abs(in_group - u_sel) >= 0.5) v_a *= 0.25;
  if (u_cut_on > 0.5 && in_pos.x > u_cut) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_a = 0.0; }
}`;

export const TRAIL_FS = `#version 300 es
precision mediump float;
in vec3 v_col;
in float v_a;
out vec4 f_color;
void main() { f_color = vec4(v_col, v_a * 0.15); }`;

export const LINES_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 in_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }`;

export const LINES_FS = `#version 300 es
precision mediump float;
uniform vec3 u_color;
uniform float u_alpha;
out vec4 f_color;
void main() { f_color = vec4(u_color * u_alpha, 1.0); }`;

export const BG_VS = `#version 300 es
precision highp float;
// Full-screen triangle, no vertex buffer: draw 3 vertices.
out vec2 v_uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = vec2(p.x, 1.0 - p.y);            // video rows are top-down
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const BG_FS = `#version 300 es
precision mediump float;
uniform sampler2D u_tex;
uniform float u_dim;
uniform float u_has_tex;
uniform vec2 u_uv_scale;                  // object-fit: cover
uniform float u_mirror;                   // selfie view
in vec2 v_uv;
out vec4 f_color;
void main() {
  vec2 uv = (v_uv - 0.5) * u_uv_scale + 0.5;
  if (u_mirror > 0.5) uv.x = 1.0 - uv.x;
  vec3 cam = texture(u_tex, uv).rgb * u_dim;
  vec2 q = v_uv - vec2(0.62, 0.5);
  vec3 bg = vec3(0.012, 0.016, 0.03) + vec3(0.03, 0.045, 0.08) * exp(-dot(q, q) * 3.0);   // no camera: deep-space gradient
  f_color = vec4(mix(bg, cam, u_has_tex), 1.0);
}`;
