const vsSource = `
attribute vec4 vertex_points;

void main() {
gl_Position = vertex_points;
}
`;

const fsSource = `
precision mediump float;
void main () {
gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;

const drawCanvas = () => {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const root = document.getElementById("root") as HTMLDivElement;
  console.log(root.clientWidth, root.clientHeight);
  canvas.setAttribute("width", `${root.clientWidth}`);
  canvas.setAttribute("height", `${root.clientHeight}`);
  const gl = canvas?.getContext("webgl");

  if (!gl) {
    console.log("WebGL unavailable");
    return;
  }
  // clear canvas
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // create Shader
  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  if (!vertexShader || !fragmentShader) {
    console.log("Failed to create shader.");
    return;
  }
  gl.shaderSource(vertexShader, vsSource);
  gl.compileShader(vertexShader);
  gl.shaderSource(fragmentShader, fsSource);
  gl.compileShader(fragmentShader);
  //
  const program = gl.createProgram();
  if (!program) {
    console.log("Failed to create gl program.");
    return;
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  console.log("Finish shader compile!");

  // prepare buffer data (vbo : データをGPUのbuffer上に置く)
  const coordinates = [-0.7, 0.7, -0.7, 0.0, 0.7, 0.0];
  const pointsBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(coordinates), gl.STATIC_DRAW);

  // prepare vao (vboのデータに意味をつける)
  const pointsAttributeLocation = gl.getAttribLocation(
    program,
    "vertex_points"
  );
  gl.vertexAttribPointer(pointsAttributeLocation, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pointsAttributeLocation);

  // draw
  gl.useProgram(program);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  console.log(gl.drawingBufferWidth, gl.drawingBufferHeight);
};

drawCanvas();
