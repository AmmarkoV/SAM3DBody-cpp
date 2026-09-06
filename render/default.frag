#version 330 core
in  vec3 vNorm;
in  vec3 vViewNorm;
uniform vec3      uColor;       // mesh tint, 0-1 per channel (--mesh-color / --color)
uniform sampler2D uScene;       // the background camera image
uniform vec2      uResolution;  // viewport size, to map gl_FragCoord -> UV
uniform float     uShiny;       // 0 = matte (original look), 1 = full chrome
uniform float     uAlpha;       // mesh opacity (0 = invisible, 1 = opaque)
out vec4 fragColor;

void main() {
    vec3  N = normalize(vNorm);
    vec3  L = normalize(vec3(0.3, 0.8, 0.5));

    // Half-Lambert / wrap lighting instead of a hard clamp(dot(N,L),0,1).
    // The mesh only carries per-vertex normals, linearly interpolated per
    // pixel, so the geometric facets between triangles are real (dihedral
    // angles of 10-20+ degrees are common on a body-sized mesh).  A hard
    // N·L=0 terminator plus that faceted normal field is exactly what makes
    // individual triangles pop as visibly different shades ("Mach banding").
    // Wrapping the light around the surface and smoothing the ramp removes
    // the slope discontinuity at the terminator, which is what the eye
    // actually keys on — same cost as the old formula, just different math.
    float ndotl = dot(N, L);
    float wrap  = 0.15;
    float d = clamp((ndotl + wrap) / (1.0 + wrap), 0.0, 1.0);
    d = d * d * (3.0 - 2.0 * d);   // smoothstep ease, kills the remaining clamp kinks
    d = d * 0.65 + 0.35;           // same overall range as the old 0.3..1.0 ramp
    vec3  base = uColor * d;

    // ── Screen-space pseudo-reflection ("Silicon Dreams" chrome) ─────────────
    // Sample the scene behind the mesh, displaced by the view-space normal so
    // the body mirrors its surroundings.  The background texture has its V axis
    // flipped on upload, so flip gl_FragCoord.y to line the reflection up with
    // what's actually visible behind the fragment.
    vec3 vn = normalize(vViewNorm);
    vec2 uv = vec2(gl_FragCoord.x / uResolution.x,
                   1.0 - gl_FragCoord.y / uResolution.y);
    uv += vn.xy * 0.15;
    vec3 refl = texture(uScene, clamp(uv, 0.0, 1.0)).rgb;

    // Fresnel-ish term: more reflective at grazing angles (rim glints).
    float fres = pow(1.0 - clamp(abs(vn.z), 0.0, 1.0), 2.0);
    float k    = uShiny * mix(0.55, 1.0, fres);

    // Tint the reflection with uColor too (colored metal, not a plain
    // mirror) so --color stays visible at every --shiny level instead of
    // fading out to a neutral reflection as k -> 1.
    vec3 col = mix(base, refl * uColor, k);

    // Cheap ordered dither (one hash, a couple ALU ops) to break up the
    // 8-bit quantization bands that a smooth shading gradient otherwise
    // shows on a fixed-function framebuffer with no MSAA/sRGB.
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    col += (dither - 0.5) / 255.0;

    fragColor = vec4(col, uAlpha);
}
