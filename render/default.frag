#version 330 core
in  vec3 vNorm;
in  vec3 vViewNorm;
uniform vec3      uColor;       // mesh tint (set per-person from the renderer)
uniform sampler2D uScene;       // the background camera image
uniform vec2      uResolution;  // viewport size, to map gl_FragCoord -> UV
uniform float     uShiny;       // 0 = matte (original look), 1 = full chrome
uniform float     uAlpha;       // mesh opacity (0 = invisible, 1 = opaque)
out vec4 fragColor;

void main() {
    vec3  N = normalize(vNorm);
    vec3  L = normalize(vec3(0.3, 0.8, 0.5));
    float d = clamp(dot(N, L), 0.0, 1.0) * 0.7 + 0.3;
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

    vec3 col = mix(base, refl, k);
    fragColor = vec4(col, uAlpha);
}
