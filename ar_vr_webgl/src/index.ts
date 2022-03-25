import { mat4 } from "gl-matrix";
import * as THREE from "three";

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
    // top face
    -0.3, 0.3, -0.3,
    0.3, 0.3, -0.3,
    -0.2, 0.4,  0.3,
    0.4, 0.4,  0.3,
    0.3, 0.3, -0.3,
    -0.2, 0.4,  0.3,
  ];

  // prettier-ignore
  const squareColors = [
    0.0,  0.0,  1.0,  1.0,
    0.0,  0.0,  1.0,  1.0,
    0.0,  0.0,  1.0,  1.0,
    0.0,  0.0,  1.0,  1.0,
    0.0,  0.0,  1.0,  1.0,
    0.0,  0.0,  1.0,  1.0,
    1.0,  0.0,  0.0,  1.0,
    1.0,  0.0,  0.0,  1.0,
    1.0,  0.0,  0.0,  1.0,
    1.0,  0.0,  0.0,  1.0,
    1.0,  0.0,  0.0,  1.0,
    1.0,  0.0,  0.0,  1.0,
    0.0,  1.0,  0.0,  1.0,
    0.0,  1.0,  0.0,  1.0,
    0.0,  1.0,  0.0,  1.0,
    0.0,  1.0,  0.0,  1.0,
    0.0,  1.0,  0.0,  1.0,
    0.0,  1.0,  0.0,  1.0,
  ];
  /*====== Define front-face buffer ======*/
  // prepare buffer data (vbo : データをGPUのbuffer上に置く)
  const origBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, origBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(squares), gl.STATIC_DRAW);

  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array(squareColors),
    gl.STATIC_DRAW
  );

  /*========== Shaders ==========*/
  /*====== Define shader source ======*/
  const vsSource = `
attribute vec4 aPosition;
attribute vec4 aVertexColor;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
varying lowp vec4 vColor;

void main() {
gl_Position = uProjectionMatrix * uModelViewMatrix * aPosition;
vColor = aVertexColor;
}
`;

  const fsSource = `
varying lowp vec4 vColor;
void main () {
gl_FragColor = vColor;
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
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
    alert(
      "An error occurred compiling the shaders : " +
        gl.getShaderInfoLog(vertexShader)
    );
    gl.deleteShader(vertexShader);
    return;
  }
  gl.shaderSource(fragmentShader, fsSource);
  gl.compileShader(fragmentShader);
  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
    alert(
      "An error occurred compiling the shaders : " +
        gl.getShaderInfoLog(fragmentShader)
    );
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    return;
  }
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

  let then = 0.0;
  let cubeRotation = 0.0;
  const render = (now: number) => {
    now *= 0.001;
    let deltaTime = now - then;
    then = now;
    /*====== Connect the attribute with the vertex shader =======*/
    // prepare vao (vboのデータに意味をつける)
    const pointsAttributeLocation = gl.getAttribLocation(program, "aPosition");
    gl.bindBuffer(gl.ARRAY_BUFFER, origBuffer);
    gl.vertexAttribPointer(pointsAttributeLocation, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(pointsAttributeLocation);

    const colorAttributeLocation = gl.getAttribLocation(
      program,
      "aVertexColor"
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.vertexAttribPointer(colorAttributeLocation, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(colorAttributeLocation);

    /*====== Uniform variables ======*/
    const modelMatrixLocation = gl.getUniformLocation(
      program,
      "uModelViewMatrix"
    );
    const modelViewMatrix = mat4.create();
    mat4.translate(
      modelViewMatrix, // destination matrix
      modelViewMatrix, // matrix to translate
      [0.0, 0.0, -2.0]
    ); // amount to translate
    mat4.rotate(
      modelViewMatrix,
      modelViewMatrix,
      cubeRotation,
      [0.0, 0.0, 1.0]
    );
    mat4.rotate(
      modelViewMatrix,
      modelViewMatrix,
      cubeRotation,
      [0.0, 1.0, 0.0]
    );
    gl.uniformMatrix4fv(modelMatrixLocation, false, modelViewMatrix);

    const projMatrixLocation = gl.getUniformLocation(
      program,
      "uProjectionMatrix"
    );
    const fieldOfView = (45 * Math.PI) / 180; // in radians
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 0.1;
    const zFar = 100.0;
    const projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);
    gl.uniformMatrix4fv(projMatrixLocation, false, projectionMatrix);

    /*========== Drawing ========== */
    /*====== Draw the points to the screen ======*/
    // clear canvas
    gl.clearColor(0, 0, 0, 0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    // draw
    gl.drawArrays(gl.TRIANGLES, 0, 18);
    // gl.drawArrays(gl.LINE_LOOP, 0, 12);
    cubeRotation += deltaTime;
    requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
};

const drawCanvasWithThree = () => {
  // create the context
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const gl = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
  });
  // create and set the camera
  const angleOfView = 55;
  const aspectRatio = canvas.clientWidth / canvas.clientHeight;
  const nearPlane = 0.1;
  const farPlane = 100;
  const camera = new THREE.PerspectiveCamera(
    angleOfView,
    aspectRatio,
    nearPlane,
    farPlane
  );
  camera.position.set(0, 8, 30);
  // create the scene
  const scene = new THREE.Scene();
  // add fog later...
  // GEOMETRY
  const cubeSize = 4;
  const cubeGeometry = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
  // Create the upright plane
  // Create the cube
  // Create the Sphere
  // MATERIALS and TEXTURES
  const cubeMaterial = new THREE.MeshPhongMaterial({ color: "pink" });
  const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
  cube.position.set(cubeSize + 1, cubeSize + 1, 0);
  scene.add(cube);

  //LIGHTS
  const color = 0xffffff;
  const intensity = 1;
  const light = new THREE.DirectionalLight(color, intensity);
  scene.add(light); // MESH
  // DRAW
  const draw = () => {
    if (resizeGLToDisplaySize(gl)) {
      const canvas = gl.domElement;
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
    }
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    cube.rotation.z += 0.01;
    gl.render(scene, camera);
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  // SET ANIMATION LOOP
  // UPDATE RESIZE
};

const resizeGLToDisplaySize = (gl: THREE.WebGLRenderer) => {
  const canvas = gl.domElement;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const needResize = canvas.width != width || canvas.height != height;
  if (needResize) {
    gl.setSize(width, height, false);
  }
  return needResize;
};

drawCanvasWithThree();
