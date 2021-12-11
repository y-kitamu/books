use std::mem;
use std::os::raw::c_void;
use std::path::Path;
use std::time::Duration;

use c_str_macro::c_str;
use cgmath::perspective;
use cgmath::prelude::SquareMatrix;
use gl::types::{GLfloat, GLsizei, GLsizeiptr};
use imgui::im_str;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use RustOpenGL::frame_buffer::FrameBuffer;
use RustOpenGL::image_manager::ImageManager;
use RustOpenGL::shader::Shader;
use RustOpenGL::vertex::Vertex;

#[allow(dead_code)]
type Point3 = cgmath::Point3<f32>;
#[allow(dead_code)]
type Vector3 = cgmath::Vector3<f32>;
#[allow(dead_code)]
type Matrix4 = cgmath::Matrix4<f32>;

const WINDOW_WIDTH: u32 = 900;
const WINDOW_HEIGHT: u32 = 480;
const FLOAT_NUM: usize = 8;
const VERTEX_NUM: usize = 36;
const BUF_LEN: usize = FLOAT_NUM * VERTEX_NUM;

enum ShaderMode {
    General,
    Sphere,
    Bloom,
    RetroTV,
}

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    {
        let gl_attr = video_subsystem.gl_attr();
        gl_attr.set_context_profile(sdl2::video::GLProfile::Core);
        gl_attr.set_context_version(3, 1);
        let (major, minor) = gl_attr.context_version();
        println!("OK : init OpenGL: version = {}.{}", major, minor);
    }

    let window = video_subsystem
        .window("SDL", WINDOW_WIDTH, WINDOW_HEIGHT)
        .opengl()
        .position_centered()
        .build()
        .unwrap();

    let _gl_context = window.gl_create_context().unwrap();
    gl::load_with(|s| video_subsystem.gl_get_proc_address(s) as _);

    let mut shader_mode = ShaderMode::General;
    let screen_shader = Shader::new("rsc/shader/screen_shader.vs", "rsc/shader/screen_shader.fs");
    let screen_shader_sphere = Shader::new(
        "rsc/shader/screen_shader_sphere.vs",
        "rsc/shader/screen_shader_sphere.fs",
    );
    let screen_shader_bloom = Shader::new(
        "rsc/shader/screen_shader_bloom.vs",
        "rsc/shader/screen_shader_bloom.fs",
    );
    let screen_shader_retro_tv = Shader::new(
        "rsc/shader/screen_shader_retro_tv.vs",
        "rsc/shader/screen_shader_retro_tv.fs",
    );

    let frame_buffer = FrameBuffer::new(WINDOW_WIDTH, WINDOW_HEIGHT);
    let vertex_vec = new_screen_vertex_vec(-1.0, -1.0, 1.0, 1.0, 20);

    let screen_vertex = Vertex::new(
        (vertex_vec.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
        vertex_vec.as_ptr() as *const c_void,
        gl::STATIC_DRAW,
        vec![gl::FLOAT, gl::FLOAT],
        vec![3, 2],
        5 * mem::size_of::<GLfloat>() as GLsizei,
        20 * 20 * 2 * 3,
    );

    let mut depth_test_frame: bool = true;
    let mut blend_frame: bool = true;
    let mut wireframe_frame: bool = false;
    let mut culling_frame: bool = true;

    let mut image_manager = ImageManager::new();
    image_manager.load_image(Path::new("rsc/image/surface.png"), "surface", true);

    let shader = Shader::new("rsc/shader/shader.vs", "rsc/shader/shader.fs");

    #[rustfmt::skip]
    let buffer_array: [f32; BUF_LEN] = [
        //1
        0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0,

        0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0,

        // 2
        0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0,

        0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.1, 0.0, -1.0, 0.0, 1.0, 1.0,

        // 3
        0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,

        0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,

        // 4
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,

        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,

        // 5
        1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,

        1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,

        // 6
        0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0,

        0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0,
    ];

    let vertex = Vertex::new(
        (BUF_LEN * mem::size_of::<GLfloat>()) as GLsizeiptr,
        buffer_array.as_ptr() as *const c_void,
        gl::STATIC_DRAW,
        vec![gl::FLOAT, gl::FLOAT, gl::FLOAT],
        vec![3, 3, 2],
        8 * mem::size_of::<GLfloat>() as GLsizei,
        VERTEX_NUM as i32,
    );

    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);

    let mut imgui_sdl2_context = imgui_sdl2::ImguiSdl2::new(&mut imgui_context, &window);
    let renderer = imgui_opengl_renderer::Renderer::new(&mut imgui_context, |s| {
        video_subsystem.gl_get_proc_address(s) as _
    });

    let mut depth_test: bool = true;
    let mut blend: bool = true;
    let mut wireframe: bool = false;
    let mut culling: bool = true;
    let mut camera_x: f32 = 2.0f32;
    let mut camera_y: f32 = -2.0f32;
    let mut camera_z: f32 = 2.0f32;
    let mut alpha: f32 = 1.0f32;

    let mut material_specular: Vector3 = Vector3 {
        x: 0.2,
        y: 0.2,
        z: 0.2,
    };
    let mut material_shininess: f32 = 0.1f32;
    let mut light_direction: Vector3 = Vector3 {
        x: 1.0,
        y: 1.0,
        z: 0.0,
    };
    let mut ambient: Vector3 = Vector3 {
        x: 0.3,
        y: 0.3,
        z: 0.3,
    };
    let mut diffuse: Vector3 = Vector3 {
        x: 0.5,
        y: 0.5,
        z: 0.5,
    };
    let mut specular: Vector3 = Vector3 {
        x: 0.2,
        y: 0.2,
        z: 0.2,
    };

    let surface_texture_id = image_manager.get_texture_id("surface");
    let start_time = std::time::Instant::now();

    let mut debug_window_mode = true;
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            imgui_sdl2_context.handle_event(&mut imgui_context, &event);
            if imgui_sdl2_context.ignore_event(&event) {
                continue;
            }

            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::D),
                    repeat: false,
                    ..
                } => {
                    debug_window_mode = !debug_window_mode;
                    println!("debug mode: {}", debug_window_mode);
                }
                _ => {}
            }
        }

        unsafe {
            frame_buffer.bind_as_frame_buffer();

            if depth_test {
                gl::Enable(gl::DEPTH_TEST);
            } else {
                gl::Disable(gl::DEPTH_TEST);
            }

            if blend {
                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            } else {
                gl::Disable(gl::BLEND);
            }

            if wireframe {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
            } else {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
            }

            if culling {
                gl::Enable(gl::CULL_FACE);
            } else {
                gl::Disable(gl::CULL_FACE);
            }

            gl::Viewport(0, 0, WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32);

            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            let model_matrix = Matrix4::identity();
            let view_matrix = Matrix4::look_at_rh(
                Point3 {
                    x: camera_x,
                    y: camera_y,
                    z: camera_z,
                },
                Point3 {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
                Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            );
            let projection_matrix: Matrix4 = perspective(
                cgmath::Deg(45.0f32),
                WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32,
                0.1,
                100.0,
            );

            shader.use_program();
            shader.set_mat4(c_str!("uModel"), &model_matrix);
            shader.set_mat4(c_str!("uView"), &view_matrix);
            shader.set_mat4(c_str!("uProjection"), &projection_matrix);
            shader.set_float(c_str!("uAlpha"), alpha);
            shader.set_vec3(c_str!("uViewPosition"), camera_x, camera_y, camera_z);
            shader.set_vector3(c_str!("uMaterial.specular"), &material_specular);
            shader.set_float(c_str!("uMaterial.shininess"), material_shininess);
            shader.set_vector3(c_str!("light.direction"), &light_direction);
            shader.set_vector3(c_str!("uLight.ambient"), &ambient);
            shader.set_vector3(c_str!("uLight.diffuse"), &diffuse);
            shader.set_vector3(c_str!("uLight.specular"), &specular);

            gl::BindTexture(gl::TEXTURE_2D, surface_texture_id as u32);
            vertex.draw();
            gl::BindTexture(gl::TEXTURE_2D, 0);

            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

            if depth_test_frame {
                gl::Enable(gl::DEPTH_TEST);
            } else {
                gl::Disable(gl::DEPTH_TEST);
            }

            if blend_frame {
                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            } else {
                gl::Disable(gl::BLEND);
            }

            if wireframe_frame {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
            } else {
                gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
            }

            if culling_frame {
                gl::Enable(gl::CULL_FACE);
            } else {
                gl::Disable(gl::CULL_FACE);
            }

            gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            frame_buffer.bind_as_texture();

            match shader_mode {
                ShaderMode::General => {
                    screen_shader.use_program();
                }
                ShaderMode::Sphere => {
                    screen_shader_sphere.use_program();
                }
                ShaderMode::Bloom => {
                    screen_shader_bloom.use_program();
                }
                ShaderMode::RetroTV => {
                    screen_shader_retro_tv.use_program();
                    screen_shader_retro_tv.set_float(c_str!("uScreenHeight"), WINDOW_HEIGHT as f32);
                    let now_time = std::time::Instant::now();
                    screen_shader_retro_tv
                        .set_float(c_str!("uTime"), (now_time - start_time).as_secs_f32());
                }
            }

            screen_vertex.draw();
            gl::BindTexture(gl::TEXTURE_2D, 0);

            imgui_sdl2_context.prepare_frame(
                imgui_context.io_mut(),
                &window,
                &event_pump.mouse_state(),
            );

            let ui = imgui_context.frame();
            imgui::Window::new(im_str!("Information"))
                .size([300.0, 450.0], imgui::Condition::FirstUseEver)
                .position([10.0, 10.0], imgui::Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text(im_str!("OpenGL Test App ver1.0"));
                    ui.separator();
                    ui.text(im_str!("FPS : {:.1}", ui.io().framerate));
                    let display_size = ui.io().display_size;
                    ui.text(format!(
                        "Display Size: ({:.1}, {:.1})",
                        display_size[0], display_size[1]
                    ));
                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(format!(
                        "Mouse Positioin : ({:.1}, {:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));

                    ui.separator();

                    ui.checkbox(im_str!("Depth Test"), &mut depth_test);
                    ui.checkbox(im_str!("Blend"), &mut blend);
                    ui.checkbox(im_str!("Wireframe"), &mut wireframe);
                    ui.checkbox(im_str!("Culling"), &mut culling);

                    ui.separator();

                    imgui::Slider::new(im_str!("Camera X"))
                        .range(-5.0..=5.0)
                        .build(&ui, &mut camera_x);
                    imgui::Slider::new(im_str!("Camera Y"))
                        .range(-5.0..=5.0)
                        .build(&ui, &mut camera_y);
                    imgui::Slider::new(im_str!("Camera Z"))
                        .range(-5.0..=5.0)
                        .build(&ui, &mut camera_z);

                    ui.separator();

                    ui.text(im_str!("FBO Shader"));
                    if ui.button(im_str!("General"), [60.0, 20.0]) {
                        shader_mode = ShaderMode::General;
                    }
                    ui.same_line(80.0);
                    if ui.button(im_str!("Sphere"), [60.0, 20.0]) {
                        shader_mode = ShaderMode::Sphere;
                    }
                    ui.same_line(150.0);
                    if ui.button(im_str!("Bloom"), [60.0, 20.0]) {
                        shader_mode = ShaderMode::Bloom;
                    }
                    ui.same_line(220.0);
                    if ui.button(im_str!("RetroTV"), [60.0, 20.0]) {
                        shader_mode = ShaderMode::RetroTV;
                    }

                    ui.checkbox(im_str!("Depth Test for FBO"), &mut depth_test_frame);
                    ui.checkbox(im_str!("Blend for FBO"), &mut blend_frame);
                    ui.checkbox(im_str!("Wireframe for FBO"), &mut wireframe_frame);
                    ui.checkbox(im_str!("Culling for FBO"), &mut culling_frame);
                });

            imgui::Window::new(im_str!("Light"))
                .size([300.0, 450.0], imgui::Condition::FirstUseEver)
                .position([600.0, 10.0], imgui::Condition::FirstUseEver)
                .build(&ui, || {
                    imgui::Slider::new(im_str!("Alpha"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut alpha);

                    ui.separator();

                    imgui::Slider::new(im_str!("Matrerial Specular X"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut material_specular.x);
                    imgui::Slider::new(im_str!("Matrerial Specular Y"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut material_specular.y);
                    imgui::Slider::new(im_str!("Matrerial Specular Z"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut material_specular.z);

                    imgui::Slider::new(im_str!("Material Shininess"))
                        .range(0.0..=2.0)
                        .build(&ui, &mut material_shininess);

                    ui.separator();

                    imgui::Slider::new(im_str!("Direction X"))
                        .range(-1.0..=1.0)
                        .build(&ui, &mut light_direction.x);
                    imgui::Slider::new(im_str!("Direction Y"))
                        .range(-1.0..=1.0)
                        .build(&ui, &mut light_direction.y);
                    imgui::Slider::new(im_str!("Direction Z"))
                        .range(-1.0..=1.0)
                        .build(&ui, &mut light_direction.z);

                    ui.separator();

                    imgui::Slider::new(im_str!("Ambient R"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut ambient.x);
                    imgui::Slider::new(im_str!("Ambient G"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut ambient.y);
                    imgui::Slider::new(im_str!("Ambient B"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut ambient.z);

                    ui.separator();

                    imgui::Slider::new(im_str!("Diffuse R"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut diffuse.x);
                    imgui::Slider::new(im_str!("Diffuse G"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut diffuse.y);
                    imgui::Slider::new(im_str!("Diffuse B"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut diffuse.z);

                    ui.separator();

                    imgui::Slider::new(im_str!("Specular R"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut specular.x);
                    imgui::Slider::new(im_str!("Specular G"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut specular.y);
                    imgui::Slider::new(im_str!("Specular B"))
                        .range(0.0..=1.0)
                        .build(&ui, &mut specular.z);
                });
            imgui_sdl2_context.prepare_render(&ui, &window);
            if debug_window_mode {
                renderer.render(ui);
            }

            window.gl_swap_window();
        }
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}

fn new_screen_vertex_vec(
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    division: i32,
) -> std::vec::Vec<f32> {
    let mut vertex_vec: std::vec::Vec<f32> = std::vec::Vec::new();

    for x in 0..division {
        for y in 0..division {
            let l = left + (right - left) / division as f32 * x as f32;
            let r = left + (right - left) / division as f32 * (x + 1) as f32;
            let t = top + (bottom - top) / division as f32 * y as f32;
            let b = top + (bottom - top) / division as f32 * (y + 1) as f32;

            let lc = 1.0 / division as f32 * x as f32;
            let rc = 1.0 / division as f32 * (x + 1) as f32;
            let tc = 1.0 / division as f32 * y as f32;
            let bc = 1.0 / division as f32 * (y + 1) as f32;

            vertex_vec.extend([l, t, 0.0, lc, tc].iter().cloned());
            vertex_vec.extend([r, t, 0.0, rc, tc].iter().cloned());
            vertex_vec.extend([l, b, 0.0, lc, bc].iter().cloned());
            vertex_vec.extend([l, b, 0.0, lc, bc].iter().cloned());
            vertex_vec.extend([r, t, 0.0, rc, tc].iter().cloned());
            vertex_vec.extend([r, b, 0.0, rc, bc].iter().cloned());
        }
    }
    vertex_vec
}
