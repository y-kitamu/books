#version 140

in vec3 iPosition;
in vec2 iTexCoords;

out vec2 TexCoords;

void main() {
    float PI = 3.141592653589793;
    TexCoords = iTexCoords;
    float x = sin(iPosition.x * PI) * cos(iPosition.y * PI / 2);
    float y = sin(iPosition.y * PI);
    float z = -cos(iPosition.x * PI) * cos(iPosition.y * PI / 2);
    gl_Position = vec4(x, y, z, 1.0);
}
