#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
uniform mat4 uView;            // upper-3x3 used to take normals into view space
out vec3 vNorm;
out vec3 vViewNorm;            // view-space normal, for screen-space reflection
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vNorm     = aNorm;
    vViewNorm = mat3(uView) * aNorm;
}
