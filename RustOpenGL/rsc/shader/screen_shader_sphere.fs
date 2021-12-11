#version 140

in vec2 TexCoords;

uniform sampler2D uScreenTexture;

void main() {
    gl_FragColor = texture(uScreenTexture, TexCoords);
}
