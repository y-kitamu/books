const drawCanvas = () => {
  /*========== Create a WebGL Context ==========*/
  const root = document.getElementById("root") as HTMLDivElement;
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.setAttribute("width", `${root.clientWidth}`);
  canvas.setAttribute("height", `${root.clientHeight}`);
  const gl = canvas?.getContext("webgl");
  if (!gl) {
    console.log("WebGL unavailable");
    return;
  }
  /*========== Define and Store the Geometry ==========*/
  /*====== Define front-face vertices ======*/
  // prettier-ignore
  const squares = [
    // front face
    -0.3, -0.3, -0.3,
    0.3, -0.3, -0.3,
    0.3, 0.3, -0.3,
      -0.3, -0.3, -0.3,
      -0.3,  0.3, -0.3,
    0.3, 0.3, -0.3,
    // back face
      -0.2, -0.2, 0.3,
    0.4, -0.2, 0.3,
    0.4, 0.4, 0.3,
      -0.2, -0.2, 0.3,
      -0.2, 0.4, 0.3,
    0.4, 0.4, 0.3,
  ];
  /*====== Define front-face buffer ======*/
  // prepare buffer data (vbo : データをGPUのbuffer上に置く)
  const origBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, origBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(squares), gl.STATIC_DRAW);

  /*========== Shaders ==========*/
  /*====== Define shader source ======*/
  const vsSource = `
attribute vec4 aPosition;

void main() {
gl_Position = aPosition;
}
`;

  const fsSource = `
precision mediump float;
void main () {
gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;
  /*====== Create shaders ======*/
  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  if (!vertexShader || !fragmentShader) {
    console.log("Failed to create shader.");
    return;
  }
  /*====== Compile shaders ======*/
  gl.shaderSource(vertexShader, vsSource);
  gl.compileShader(vertexShader);
  gl.shaderSource(fragmentShader, fsSource);
  gl.compileShader(fragmentShader);
  /*====== Create shader program ======*/
  const program = gl.createProgram();
  if (!program) {
    console.log("Failed to create gl program.");
    return;
  }
  /*====== Link shader program ======*/
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.useProgram(program);
  /*====== Connect the attribute with the vertex shader =======*/
  // prepare vao (vboのデータに意味をつける)
  const pointsAttributeLocation = gl.getAttribLocation(program, "aPosition");
  gl.vertexAttribPointer(pointsAttributeLocation, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pointsAttributeLocation);
  /*========== Drawing ========== */
  /*====== Draw the points to the screen ======*/
  // clear canvas
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  // draw
  gl.drawArrays(gl.TRIANGLES, 0, 12);
  // gl.drawArrays(gl.LINE_LOOP, 0, 12);
};

drawCanvas();
