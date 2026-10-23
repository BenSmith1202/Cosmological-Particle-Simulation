// Interactive point viewer: a 2D x-y slice or a 3D orbit of speed-coloured points, drawn with wgpu.
// Settings (run, colours, glow) live in a panel (S), and are saved to settings.json as they change.
//
// Controls: left-drag rotates (orbit) or pans (slice), right-drag pans (orbit), scroll zooms, Tab switches view,
// Home resets the view, F freezes the speed colour range, S opens the settings panel, Esc quits.
// Runs on the main thread (winit requirement). The simulation runs on a worker thread.

use std::path::PathBuf;
use std::sync::{mpsc::TryRecvError, Arc};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::appearance::{self, Appearance, ColorBy, ColorStop, MAX_STOPS};
use crate::frame::Frame;
use crate::camera::{self, Camera, Mat4, Orbit};
use crate::exposure::{self, Adapter};
use crate::glow::{Glow, GlowParams, ADDITIVE, HDR_FORMAT};
use crate::runner::{RunSettings, Runner, MAX_PARTICLES_PER_SIDE, MIN_PARTICLES_PER_SIDE};
use crate::store::{self, Saved};
use crate::lens::{self, Lens, LensParams};
use crate::lens_fft::{self, LensFft, LensFftParams};
use crate::volume::{GasParams, Volume};

// Scroll zoom per line of wheel movement in the 2D slice view.
const ZOOM_PER_LINE: f32 = 1.1;
// Fraction of particles, from the slow end, that clamp to the coldest colour.
const COLD_PERCENTILE: f32 = 0.10;

#[derive(Clone, Copy, PartialEq, Debug)]
enum ViewMode {
    Slice,
    Orbit,
}

// Uniform block. Layout must match `struct View` in shader.wgsl (256 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    view_proj: [[f32; 4]; 4],
    color_min: f32,
    color_max: f32,
    stop_count: u32,
    point_sigma: f32,
    viewport: [f32; 2],
    // How far between the previous and current sim frame to draw, 0..1.
    alpha: f32,
    // Periodic box side, so interpolation takes the short way across the boundary.
    box_size: f32,
    // 0 = speed, 1 = density.
    color_by: u32,
    rsd_axis: u32,
    // 0 turns the redshift-space shift off.
    rsd_strength: f32,
    // 0 turns depth fog off.
    fog_strength: f32,
    // Depth that fog leaves unchanged (the orbit distance).
    fog_ref: f32,
    // Multi-plane lensing of the particles: 0 = off, else the exaggeration. Depth range and slab count.
    lens_strength: f32,
    lens_near: f32,
    lens_far: f32,
    lens_slabs: u32,
    aspect: f32,
    _pad: [f32; 2],
    stops: [[f32; 4]; MAX_STOPS],
}

// One record per particle: a vertex in point mode, an instance in splat mode. Holds the previous and current sim
// frame, and the shader draws in between. Stride 64 bytes, matching the attributes in `particle_pipeline`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleVertex {
    prev_pos: [f32; 3],
    prev_speed: f32,
    pos: [f32; 3],
    speed: f32,
    // Redshift-space shift (comoving distance) and 1 + Î´, for the previous and current frame.
    prev_rsd: [f32; 3],
    prev_density: f32,
    rsd: [f32; 3],
    density: f32,
}

// The optional layers for one frame. `volume` is set whenever the density grid is needed (gas or lensing).
struct Layers {
    volume: Option<GasParams>,
    gas: bool,
    lens: Option<LensParams>,
    // Set when high-quality lensing is on (it replaces the direct deflection for the background).
    hq: Option<HqLens>,
}

// High-quality lensing for one frame: the GPU inputs, and the map size and slab count to configure.
struct HqLens {
    params: LensFftParams,
    k: u32,
    slabs: u32,
}

// What egui produced for this frame, handed to the renderer.
struct UiOutput {
    primitives: Vec<egui::ClippedPrimitive>,
    textures: egui::TexturesDelta,
    pixels_per_point: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum PanelTab {
    Simulation,
    Colors,
    Render,
    Layers,
}

// The settings panel. `draft` is what the Simulation tab edits. Apply sends it to the runner.
struct PanelState {
    open: bool,
    tab: PanelTab,
    draft: RunSettings,
}

// Natural-log range of a quantity (speed or density) over a frame. The low end is the COLD_PERCENTILE value, so the
// lowest particles clamp to the coldest colour instead of stretching the scale. The high end is the maximum, so the
// hottest ones still reach the end of the ramp.
fn log_range(speeds: &[f32]) -> (f32, f32) {
    if speeds.is_empty() {
        return (0.0, 1.0);
    }
    let mut logs: Vec<f32> = speeds.iter().map(|s| s.max(1e-12).ln()).collect();
    let k = ((logs.len() - 1) as f32 * COLD_PERCENTILE) as usize;
    let (_, &mut lo, _) = logs.select_nth_unstable_by(k, f32::total_cmp);
    let hi = logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    if hi - lo < 1e-6 {
        return (lo, lo + 1.0);
    }
    (lo, hi)
}

// How each particle is drawn. Both footprints add into the HDR scene.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Footprint {
    // One vertex per particle in a point list: one pixel each.
    Point,
    // One instance per particle, six vertices each (a quad), with a Gaussian in the fragment shader.
    Splat,
}

fn particle_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    footprint: Footprint,
) -> wgpu::RenderPipeline {
    // Per-particle data steps once per vertex for points, and once per instance (quad) for splats.
    let (step_mode, topology, vs, fs) = match footprint {
        Footprint::Point => (wgpu::VertexStepMode::Vertex, wgpu::PrimitiveTopology::PointList, "vs_point", "fs_point"),
        Footprint::Splat => (wgpu::VertexStepMode::Instance, wgpu::PrimitiveTopology::TriangleList, "vs_splat", "fs_splat"),
    };
    // Layout matches `ParticleIn` in shader.wgsl: previous then current position and speed (locations 0-3),
    // then previous and current redshift shift and density (locations 4-7).
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ParticleVertex>() as wgpu::BufferAddress,
        step_mode,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32,
            2 => Float32x3,
            3 => Float32,
            4 => Float32x3,
            5 => Float32,
            6 => Float32x3,
            7 => Float32,
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particles"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            buffers: &[Some(vertex_layout)],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                // Additive, so overlaps brighten and draw order doesn't matter (no sorting or depth buffer needed).
                blend: Some(ADDITIVE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

// The particle shader's bind group: the view uniform, and the high-quality lensing deflection texture.
fn particle_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    lens_fft: &LensFft,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(lens_fft.deflection_view()) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

// Everything that touches the GPU, and the egui context that draws on top of the points.
struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Both add into the HDR scene. Point size 0 selects `pipeline_point`, anything larger `pipeline_splat`.
    pipeline_point: wgpu::RenderPipeline,
    pipeline_splat: wgpu::RenderPipeline,
    // HDR scene, bloom, and tone map. Every frame goes through it.
    glow: Glow,
    // Density grid: the gas glow, and the convergence for lensing.
    volume: Volume,
    // Lensed background galaxies, drawn into the HDR scene before the particles when lensing is on.
    lens: Lens,
    // High-quality lensing: FFT deflection per depth slab. Always exists (small until turned on), because the
    // particle shader always binds its deflection texture.
    lens_fft: LensFft,
    uniform_buf: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    lens_sampler: wgpu::Sampler,
    // The particles' bind group: view uniform, deflection texture, sampler. Rebuilt when lens_fft's texture is.
    bind_group: wgpu::BindGroup,
    bind_group_generation: u64,
    vertex_buf: wgpu::Buffer,
    vertex_capacity: usize,
    point_count: u32,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl Gpu {
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no compatible GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() }))
            .expect("request device");

        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Deflection per depth slab, sampled in the vertex shader to move lensed particles.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view uniform"),
            contents: bytemuck::bytes_of(&ViewUniform::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let lens_fft = LensFft::new(&device, &queue, config.width as f32 / config.height.max(1) as f32);
        let lens_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lens deflection"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = particle_bind_group(&device, &bind_group_layout, &uniform_buf, &lens_fft, &lens_sampler);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline_point = particle_pipeline(&device, &pipeline_layout, &shader, Footprint::Point);
        let pipeline_splat = particle_pipeline(&device, &pipeline_layout, &shader, Footprint::Splat);

        let glow = Glow::new(&device, config.format, config.width, config.height);
        let volume = Volume::new(&device, config.width, config.height);
        let lens = Lens::new(&device);

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points"),
            size: std::mem::size_of::<ParticleVertex>() as wgpu::BufferAddress,
            // STORAGE: high-quality lensing reads the particles in a compute pass.
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, egui_wgpu::RendererOptions::default());

        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline_point,
            pipeline_splat,
            glow,
            volume,
            lens,
            lens_fft,
            uniform_buf,
            bind_group_layout,
            lens_sampler,
            bind_group,
            bind_group_generation: 0,
            vertex_buf,
            vertex_capacity: 0,
            point_count: 0,
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.glow.resize(&self.device, width, height);
        self.volume.resize(&self.device, width, height);
    }

    // Copies a pair of sim frames into the vertex buffer, growing it if the count increased.
    // `prev` and `cur` must hold the same particles in the same order.
    fn upload(&mut self, prev: &Frame, cur: &Frame) {
        let vertices: Vec<ParticleVertex> = (0..cur.positions.len())
            .map(|i| ParticleVertex {
                prev_pos: prev.positions[i],
                prev_speed: prev.speeds[i],
                pos: cur.positions[i],
                speed: cur.speeds[i],
                prev_rsd: prev.rsd[i],
                prev_density: prev.densities[i],
                rsd: cur.rsd[i],
                density: cur.densities[i],
            })
            .collect();
        if vertices.len() > self.vertex_capacity {
            self.vertex_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("points"),
                size: (vertices.len() * std::mem::size_of::<ParticleVertex>()) as wgpu::BufferAddress,
                // STORAGE: high-quality lensing reads the particles in a compute pass.
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.vertex_capacity = vertices.len();
        }
        self.queue
            .write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&vertices));
        self.point_count = vertices.len() as u32;
    }

    // Adds every particle into the HDR scene. `clear` empties the scene first. Without it, the particles add onto
    // whatever is already there (the lensed background).
    fn draw_particles(&self, encoder: &mut wgpu::CommandEncoder, footprint: Footprint, clear: bool) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.glow.scene_view(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: if clear { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if self.point_count > 0 {
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            match footprint {
                Footprint::Point => {
                    pass.set_pipeline(&self.pipeline_point);
                    // One vertex per particle, one pixel each.
                    pass.draw(0..self.point_count, 0..1);
                }
                Footprint::Splat => {
                    pass.set_pipeline(&self.pipeline_splat);
                    // Six vertices (one quad) per instance, one instance per particle.
                    pass.draw(0..6, 0..self.point_count);
                }
            }
        }
    }

    // Draws the particles into the HDR scene, runs bloom and the tone map onto the window, then the egui panel on top.
    fn render(&mut self, view: &ViewUniform, ui: &mut UiOutput, glow: GlowParams, footprint: Footprint, layers: Layers) {
        for (id, deltas) in ui.textures.set.drain() {
            for delta in &deltas {
                self.egui_renderer.update_texture(&self.device, &self.queue, id, delta);
            }
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            // Timeout or occluded: skip this frame.
            _ => return,
        };
        let target = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(view));

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui.pixels_per_point,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        let ui_buffers = self
            .egui_renderer
            .update_buffers(&self.device, &self.queue, &mut encoder, &ui.primitives, &screen);

        if let Some(v) = &layers.volume {
            self.volume.prepare(&self.queue, v);
        }
        // High-quality lensing: deflection per depth slab, from the particles, by FFT. It also feeds the background.
        if let Some(hq) = &layers.hq {
            let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
            self.lens_fft.configure(&self.device, &self.queue, hq.k, hq.slabs, aspect);
            self.lens_fft.encode(&self.device, &mut encoder, &self.queue, &self.vertex_buf, &hq.params);
        }
        // The particle shader binds the deflection texture, which `configure` may have just recreated.
        if self.bind_group_generation != self.lens_fft.generation() {
            self.bind_group = particle_bind_group(&self.device, &self.bind_group_layout, &self.uniform_buf, &self.lens_fft, &self.lens_sampler);
            self.bind_group_generation = self.lens_fft.generation();
        }
        // Lensed background, which fills the scene. Direct mode: the convergence from the density grid first.
        let lensing = layers.lens.is_some();
        if let Some(l) = &layers.lens {
            let fft_deflection = if layers.hq.is_some() {
                Some(self.lens_fft.background_view())
            } else {
                self.lens.set_size(&self.device, l.size);
                self.volume.encode_kappa(&mut encoder, self.lens.kappa_view());
                None
            };
            self.lens.encode(&self.device, &mut encoder, &self.queue, self.glow.scene_view(), l, fft_deflection);
        }
        self.draw_particles(&mut encoder, footprint, !lensing);
        // Gas goes into the scene before the bloom and the meter, so it glows and counts towards exposure like stars.
        if layers.gas && layers.volume.is_some() {
            self.volume.encode_gas(&mut encoder, self.glow.scene_view());
        }
        self.glow.encode(&mut encoder, &self.queue, &target, glow);

        {
            // A second pass that keeps the points underneath and draws the panel over them.
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &ui.primitives, &screen);
        }

        for id in &ui.textures.free {
            self.egui_renderer.free_texture(id);
        }

        let mut buffers = ui_buffers;
        buffers.push(encoder.finish());
        self.queue.submit(buffers);
        // The meter's copy is now submitted, so its readback can be mapped.
        self.glow.after_submit();
        self.queue.present(frame);
    }
}

// Each new gap between frame arrivals moves the interval estimate this fraction of the way.
const INTERVAL_SMOOTHING: f32 = 0.3;

// Smoothed time between sim frames, in seconds. The first gap is taken as-is.
fn next_interval(current: Option<f32>, gap: f32) -> f32 {
    match current {
        Some(i) => i + (gap - i) * INTERVAL_SMOOTHING,
        None => gap,
    }
}

// How far to draw between the previous and current frame: 0 at the moment the current one arrived, 1 one interval
// later, then held at 1 until the next frame. Before two frames have arrived there is no interval, so show the latest.
fn interp_alpha(elapsed: f32, interval: Option<f32>) -> f32 {
    match interval {
        Some(i) if i > 0.0 => (elapsed / i).clamp(0.0, 1.0),
        _ => 1.0,
    }
}

fn lerp_range(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

// Interpolation between sim frames. The view draws between the previous and current frame, so it runs one
// frame interval behind the simulation, in exchange for smooth motion.
#[derive(Default)]
struct Playback {
    // The newest frame. It becomes the "previous" frame when the next one arrives.
    current: Option<Frame>,
    // When `current` arrived.
    arrived: Option<Instant>,
    // Smoothed time between frame arrivals. None until two frames of the same run have arrived.
    interval: Option<f32>,
    // Colour ranges of the previous frame, one per quantity (indexed by ColorBy). Interpolated like the positions,
    // so colours don't jump.
    prev_ranges: [(f32, f32); 2],
}

impl Playback {
    fn alpha(&self) -> f32 {
        let elapsed = self.arrived.map_or(0.0, |t| t.elapsed().as_secs_f32());
        interp_alpha(elapsed, self.interval)
    }
}

struct App {
    runner: Runner,
    box_size: f32,
    gpu: Option<Gpu>,
    mode: ViewMode,
    slice: Option<Camera>,
    orbit: Option<Orbit>,
    // Latest frame's redshift, shown in the title.
    redshift: f32,
    playback: Playback,
    // Auto exposure: the eased value, the latest target from the meter, and when it was last eased.
    exposure: Adapter,
    exposure_target: Option<f32>,
    last_redraw: Instant,
    // Colour ranges (natural log) of the current frame: speed, then density.
    ranges: [(f32, f32); 2],
    // Set by F: keep the colour range fixed at this value.
    frozen_range: Option<(f32, f32)>,
    cursor: [f32; 2],
    // Held mouse buttons: left rotates (orbit) or pans (slice), right pans in orbit.
    left_down: bool,
    right_down: bool,
    panel: PanelState,
    appearance: Appearance,
    // What was last written to disk. Changes are saved when this differs from the current state.
    last_saved: Saved,
    save_path: PathBuf,
}

impl App {
    fn viewport(&self) -> [f32; 2] {
        let c = &self.gpu.as_ref().expect("gpu").config;
        [c.width as f32, c.height as f32]
    }

    // The active camera's view-projection. The particles use it, and the gas uses its inverse.
    fn view_proj(&self) -> Mat4 {
        let viewport = self.viewport();
        match self.mode {
            ViewMode::Slice => self.slice.expect("slice camera").view_proj(viewport),
            ViewMode::Orbit => self.orbit.expect("orbit camera").view_proj(viewport),
        }
    }

    // The optional layers for this frame. Both gas and lensing need the density grid and the inverse camera; if the
    // camera can't be inverted, both are skipped.
    // The grid is in real space, so with the redshift-space view on, gas and particles are offset along that axis.
    fn layers(&self) -> Layers {
        let a = self.appearance.sanitized();
        let inv = camera::inverse(&self.view_proj());
        // The density grid is needed for gas, and for direct-mode lensing (high quality bins the particles instead).
        let volume = if a.gas || (a.lens && !a.lens_hq) { self.volume_params(inv) } else { None };
        let hq = match (a.lens && a.lens_hq, inv) {
            (true, Some(inv_view_proj)) => Some(self.hq_lens(inv_view_proj)),
            _ => None,
        };
        let lens = match (a.lens, inv) {
            (true, Some(inv_view_proj)) => {
                let viewport = self.viewport();
                Some(LensParams {
                    inv_view_proj,
                    aspect: viewport[0] / viewport[1],
                    strength: a.lens_strength,
                    galaxy_brightness: a.lens_galaxies,
                    show_kappa: a.lens_show_kappa,
                    ortho: self.mode == ViewMode::Slice,
                    box_size: self.box_size,
                    size: a.lens_resolution,
                })
            }
            _ => None,
        };
        Layers {
            gas: a.gas && volume.is_some(),
            volume,
            lens,
            hq,
        }
    }

    // Depth range of the box as the slabs divide it: clip w of the nearest and furthest box corners in the orbit
    // view, world z across the box in the slice view.
    fn lens_depth_range(&self, view_proj: &Mat4) -> (f32, f32) {
        if self.mode == ViewMode::Slice {
            return (0.0, self.box_size);
        }
        let l = self.box_size;
        let (mut near, mut far) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..8 {
            let c = [(i & 1) as f32 * l, ((i >> 1) & 1) as f32 * l, ((i >> 2) & 1) as f32 * l];
            // Clip w is the fourth row of the matrix (column-major m[c][r]) dotted with (c, 1).
            let w = view_proj[0][3] * c[0] + view_proj[1][3] * c[1] + view_proj[2][3] * c[2] + view_proj[3][3];
            near = near.min(w);
            far = far.max(w);
        }
        // Inside the box, part of it is behind the camera. Keep the range positive and non-empty.
        let near = near.max(1e-3);
        (near, far.max(near + 1e-3))
    }

    // Inputs for high-quality lensing this frame. See `Params` in lensfft.wgsl.
    fn hq_lens(&self, inv_view_proj: Mat4) -> HqLens {
        let a = self.appearance.sanitized();
        let view_proj = self.view_proj();
        let viewport = self.viewport();
        let aspect = viewport[0] / viewport[1];
        let k = a.lens_hq_resolution;
        let count = self.gpu.as_ref().map_or(0, |g| g.point_count);
        let ortho = self.mode == ViewMode::Slice;
        let (depth_near, depth_far) = self.lens_depth_range(&view_proj);
        let l = self.box_size;
        // Texel area per depthÂ², and the visible screen height at the typical lens distance (both in world units).
        let (area_coeff, r_ref, visible_height) = if ortho {
            let ppu = self.slice.expect("slice camera").pixels_per_unit;
            let (w, h) = (viewport[0] / ppu, viewport[1] / ppu);
            ((w / k as f32) * (h / k as f32), 1.0, h)
        } else {
            let t = (camera::ORBIT_FOV_Y / 2.0).tan();
            let r_ref = 0.5 * (depth_near + depth_far);
            (4.0 * t * t * aspect / (k * k) as f32, r_ref, 2.0 * r_ref * t)
        };
        // One particle spacing at the lens distance, as a fraction of the screen height.
        let spacing = l / (count.max(1) as f32).cbrt();
        HqLens {
            params: LensFftParams {
                view_proj,
                inv_view_proj,
                particle_count: count,
                depth_near,
                depth_far,
                box_size: l,
                alpha: self.playback.alpha(),
                aspect,
                smoothing: spacing / visible_height,
                weight_scale: l * l / count.max(1) as f32,
                area_coeff,
                ortho,
                r_ref,
            },
            k,
            slabs: a.lens_slabs,
        }
    }

    fn volume_params(&self, inv: Option<Mat4>) -> Option<GasParams> {
        let a = self.appearance.sanitized();
        let (stops, stop_count) = a.uniform_stops();
        Some(GasParams {
            inv_view_proj: inv?,
            box_size: self.box_size,
            alpha: self.playback.alpha(),
            intensity: a.gas_intensity,
            steps: a.gas_steps,
            low: a.gas_low,
            high: a.gas_high,
            stops,
            stop_count,
        })
    }

    fn view_uniform(&self) -> ViewUniform {
        let viewport = self.viewport();
        let view_proj = self.view_proj();
        let (color_min, color_max) = self.display_range();
        let (stops, stop_count) = self.appearance.uniform_stops();
        let a = self.appearance.sanitized();
        // Fog only makes sense with depth: the orbit view's perspective. The slice view is flat.
        let orbit = (self.mode == ViewMode::Orbit).then(|| self.orbit.expect("orbit camera"));
        let lens_range = self.lens_depth_range(&view_proj);
        ViewUniform {
            view_proj,
            color_min,
            color_max,
            stop_count,
            point_sigma: a.point_size,
            viewport,
            alpha: self.playback.alpha(),
            box_size: self.box_size,
            color_by: a.color_by as u32,
            rsd_axis: a.rsd_axis as u32,
            rsd_strength: if a.rsd { a.rsd_strength } else { 0.0 },
            fog_strength: if a.fog && orbit.is_some() { a.fog_strength } else { 0.0 },
            fog_ref: orbit.map_or(0.0, |o| o.distance),
            // Particles are lensed only in the orbit view: the slab weights need distances from the camera.
            lens_strength: if a.lens && a.lens_hq && a.lens_particles && orbit.is_some() { a.lens_strength } else { 0.0 },
            lens_near: lens_range.0,
            lens_far: lens_range.1,
            lens_slabs: a.lens_slabs,
            aspect: viewport[0] / viewport[1],
            _pad: [0.0; 2],
            stops,
        }
    }

    // The colour range on screen now: frozen, or interpolated between the previous and current frame.
    fn display_range(&self) -> (f32, f32) {
        self.frozen_range
            .unwrap_or_else(|| {
                let q = self.appearance.color_by as usize;
                lerp_range(self.playback.prev_ranges[q], self.ranges[q], self.playback.alpha())
            })
    }

    // Exposure for this frame: auto exposure times its bias, or the manual value.
    fn current_exposure(&self) -> f32 {
        let a = self.appearance.sanitized();
        if a.auto_exposure {
            self.exposure.exposure() * a.exposure_bias
        } else {
            a.exposure
        }
    }

    // Bloom and tone-map settings for this frame. Sanitized, so slider values the GPU shouldn't see never reach it.
    fn glow_params(&self) -> GlowParams {
        let a = self.appearance.sanitized();
        GlowParams {
            bloom: a.glow,
            strength: a.glow_strength,
            radius: a.glow_radius,
            exposure: self.current_exposure(),
            meter: a.auto_exposure,
            hue_preserving: a.hue_preserving,
        }
    }

    // Point size 0 is one-pixel points. Anything else is a splat.
    fn footprint(&self) -> Footprint {
        if self.appearance.sanitized().point_size == 0.0 {
            Footprint::Point
        } else {
            Footprint::Splat
        }
    }

    // Restarts the run with the panel's settings, and shows the sanitized values back in the panel.
    fn restart(&mut self) {
        self.runner.restart(self.panel.draft.clone());
        self.panel.draft = self.runner.settings.clone();
        self.ranges = [(0.0, 1.0); 2];
        self.frozen_range = None;
        self.redshift = self.panel.draft.z_start as f32;
        // The new run's particles don't correspond to the old ones, and its pace may differ, so start fresh.
        // Exposure carries over, so the view doesn't flash while the meter catches up.
        self.playback = Playback::default();
    }

    // Writes the current settings if they differ from what is on disk. Failures are reported, not fatal.
    fn autosave(&mut self) {
        let current = Saved {
            run: self.runner.settings.clone(),
            appearance: self.appearance.clone(),
        };
        if current == self.last_saved {
            return;
        }
        match store::save(&self.save_path, &current) {
            Ok(()) => self.last_saved = current,
            Err(err) => eprintln!("could not save {}: {err}", self.save_path.display()),
        }
    }
}

// Draws the settings panel. Returns true when the user asks to apply the run settings.
// `exposure_now` is the exposure in use this frame, shown in the Render tab.
fn settings_panel(ctx: &egui::Context, panel: &mut PanelState, appearance: &mut Appearance, exposure_now: f32) -> bool {
    let mut apply = false;
    egui::Window::new("Settings")
        .open(&mut panel.open)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut panel.tab, PanelTab::Simulation, "Simulation");
                ui.selectable_value(&mut panel.tab, PanelTab::Colors, "Colours");
                ui.selectable_value(&mut panel.tab, PanelTab::Render, "Render");
                ui.selectable_value(&mut panel.tab, PanelTab::Layers, "Layers");
            });
            ui.separator();
            match panel.tab {
                PanelTab::Simulation => apply = simulation_tab(ui, &mut panel.draft),
                PanelTab::Colors => colors_tab(ui, &mut appearance.stops),
                PanelTab::Render => render_tab(ui, appearance, exposure_now),
                PanelTab::Layers => layers_tab(ui, appearance),
            }
        });
    apply
}

fn simulation_tab(ui: &mut egui::Ui, draft: &mut RunSettings) -> bool {
    ui.add(
        egui::Slider::new(&mut draft.particles_per_side, MIN_PARTICLES_PER_SIDE..=MAX_PARTICLES_PER_SIDE)
            .text("particles per side"),
    );
    ui.label(format!(
        "{} particles, mesh {}Â³",
        draft.particles_per_side.pow(3),
        2 * draft.particles_per_side
    ));
    ui.add(egui::DragValue::new(&mut draft.seed).prefix("seed "));
    ui.separator();
    ui.add(egui::DragValue::new(&mut draft.z_start).speed(1.0).range(10.0..=2000.0).prefix("start z "));
    ui.add(egui::DragValue::new(&mut draft.z_end).speed(0.05).range(-0.95..=1000.0).prefix("end z "));
    ui.add(egui::Slider::new(&mut draft.steps, 16..=5000).text("steps"));
    ui.add(egui::Slider::new(&mut draft.pacing_ms, 0..=500).text("pause per step (ms)"));
    ui.separator();
    ui.checkbox(&mut draft.treepm, "TreePM short-range force");
    ui.add_enabled(
        draft.treepm,
        egui::Slider::new(&mut draft.theta, 0.1..=1.0).text("opening angle"),
    );
    ui.add_enabled(
        draft.treepm,
        egui::Slider::new(&mut draft.softening_scale, 0.0..=0.5).text("softening (Ã— spacing)"),
    );
    ui.separator();
    let apply = ui.button("Apply and restart").clicked();
    ui.label("Changes take effect when you apply.");
    apply
}

fn colors_tab(ui: &mut egui::Ui, stops: &mut Vec<ColorStop>) {
    // Preview: the ramp as a strip, sampled across the speed range. Stops are linear and the shader interpolates in linear
    // space, so the strip does the same, then encodes to sRGB through egui::Rgba (linear) -> Color32 (sRGB).
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(200.0), 18.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let segments = 96;
    for i in 0..segments {
        let t = (i as f32 + 0.5) / segments as f32;
        let c = appearance::color_at(stops, t);
        let x0 = rect.left() + rect.width() * i as f32 / segments as f32;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / segments as f32;
        let strip = egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom()));
        painter.rect_filled(strip, 0.0, egui::Color32::from(egui::Rgba::from_rgb(c[0], c[1], c[2])));
    }
    ui.add_space(4.0);

    let count = stops.len();
    let mut remove = None;
    for (i, stop) in stops.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut stop.color);
            ui.add(egui::Slider::new(&mut stop.position, 0.0..=1.0).text(format!("stop {}", i + 1)));
            if ui.add_enabled(count > 2, egui::Button::new("remove")).clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        stops.remove(i);
    }

    ui.horizontal(|ui| {
        if ui.add_enabled(count < MAX_STOPS, egui::Button::new("add stop")).clicked() {
            let position = 0.5;
            stops.push(ColorStop {
                position,
                color: appearance::color_at(stops, position),
            });
        }
        if ui.button("reset colours").clicked() {
            *stops = Appearance::default().stops;
        }
    });
    ui.label("Position is a fraction of the speed range: 0 = coldest, 1 = fastest.");
}

fn layers_tab(ui: &mut egui::Ui, a: &mut Appearance) {
    ui.horizontal(|ui| {
        ui.label("Colour by");
        ui.selectable_value(&mut a.color_by, ColorBy::Speed, "speed");
        ui.selectable_value(&mut a.color_by, ColorBy::Density, "density");
    });
    ui.separator();

    ui.checkbox(&mut a.rsd, "Redshift-space view")
        .on_hover_text("Place particles where a redshift survey would: shifted along the line of sight by their velocity. Clusters stretch into fingers pointing along that axis.");
    ui.add_enabled_ui(a.rsd, |ui| {
        ui.horizontal(|ui| {
            ui.label("line of sight");
            ui.selectable_value(&mut a.rsd_axis, 0, "x");
            ui.selectable_value(&mut a.rsd_axis, 1, "y");
            ui.selectable_value(&mut a.rsd_axis, 2, "z");
        });
        ui.add(egui::Slider::new(&mut a.rsd_strength, 0.0..=5.0).text("strength (1 = physical)"));
    });
    ui.separator();

    ui.checkbox(&mut a.fog, "Depth fog (orbit view)");
    ui.add_enabled(a.fog, egui::Slider::new(&mut a.fog_strength, 0.0..=5.0).text("fog strength"));
    ui.separator();

    ui.checkbox(&mut a.gas, "Volumetric gas")
        .on_hover_text("A glow ray-marched through the density grid. It shows the dark-matter density, drawn to look like gas.");
    ui.add_enabled_ui(a.gas, |ui| {
        ui.add(egui::Slider::new(&mut a.gas_intensity, 0.0..=20.0).logarithmic(true).text("intensity"));
        ui.add(egui::Slider::new(&mut a.gas_low, -1.0..=3.5).text("glow from log10(1+Î´)"));
        ui.add(egui::Slider::new(&mut a.gas_high, -0.9..=4.0).text("brightest at log10(1+Î´)"));
        ui.add(
            egui::Slider::new(&mut a.gas_steps, appearance::MIN_GAS_STEPS..=appearance::MAX_GAS_STEPS)
                .text("quality (steps)"),
        );
    });
    ui.separator();

    ui.checkbox(&mut a.lens, "Gravitational lensing")
        .on_hover_text("Background galaxies seen through the box's mass. Clusters bend their light into arcs. The bending is exaggerated: at this scale the real deflection would be under a pixel.");
    ui.add_enabled_ui(a.lens, |ui| {
        ui.add(egui::Slider::new(&mut a.lens_strength, 0.0..=50.0).logarithmic(true).text("strength (exaggeration)"));
        ui.add(egui::Slider::new(&mut a.lens_galaxies, 0.0..=10.0).logarithmic(true).text("background brightness"));
        ui.checkbox(&mut a.lens_hq, "High quality (FFT, depth slabs)")
            .on_hover_text("For faster GPUs. Mass comes from the particles (not the density grid), the bending is computed by FFT, and the box is split into depth slabs so each slab is bent only by the mass in front of it.");
        if a.lens_hq {
            ui.horizontal(|ui| {
                ui.label("resolution");
                for size in lens_fft::LENS_HQ_SIZES {
                    ui.selectable_value(&mut a.lens_hq_resolution, size, size.to_string());
                }
            });
            ui.add(egui::Slider::new(&mut a.lens_slabs, lens_fft::MIN_SLABS..=lens_fft::MAX_SLABS).text("depth slabs"));
            ui.checkbox(&mut a.lens_particles, "Lens the particles too")
                .on_hover_text("Bend each particle by the mass in front of it. Orbit view only: the slice view has no camera distance to weight by.");
        } else {
            ui.add(
                egui::Slider::new(&mut a.lens_resolution, lens::MIN_LENS_SIZE..=lens::MAX_LENS_SIZE)
                    .step_by(16.0)
                    .text("mass map resolution"),
            )
            .on_hover_text("Detail of the projected mass and the bending. The bending cost grows with the fourth power: on a laptop iGPU, 64 ran at about 65 fps and 128 at about 32. Above about 128 the density grid limits detail.");
        }
        ui.checkbox(&mut a.lens_show_kappa, "Show projected mass")
            .on_hover_text("Tint the background warm where mass is concentrated and blue in voids.");
    });
}

fn render_tab(ui: &mut egui::Ui, appearance: &mut Appearance, exposure_now: f32) {
    // Point size 0 means one-pixel points. The checkbox makes that explicit instead of hiding it at the slider's end.
    let mut pixel_points = appearance.point_size == 0.0;
    if ui.checkbox(&mut pixel_points, "One-pixel points").changed() {
        appearance.point_size = if pixel_points { 0.0 } else { Appearance::default().point_size };
    }
    if !pixel_points {
        ui.add(
            egui::Slider::new(&mut appearance.point_size, appearance::MIN_SPLAT_SIGMA..=appearance::MAX_SPLAT_SIGMA)
                .text("point size (px)"),
        );
    }
    ui.separator();
    ui.checkbox(&mut appearance.auto_exposure, "Auto exposure");
    if appearance.auto_exposure {
        // Bias scales the automatic value: 2 is one stop brighter, 0.5 one stop darker.
        ui.add(
            egui::Slider::new(&mut appearance.exposure_bias, 0.1..=10.0)
                .logarithmic(true)
                .text("exposure bias"),
        );
    } else {
        ui.add(
            egui::Slider::new(&mut appearance.exposure, 0.1..=20.0)
                .logarithmic(true)
                .text("exposure"),
        );
    }
    ui.label(format!("exposure now: {exposure_now:.2}"));
    ui.checkbox(&mut appearance.hue_preserving, "Preserve hue")
        .on_hover_text("On: bright areas keep their colour. Off: bright areas clip towards white, like an overexposed photo.");
    ui.separator();
    ui.checkbox(&mut appearance.glow, "Bloom");
    let glow_on = appearance.glow;
    ui.add_enabled(glow_on, egui::Slider::new(&mut appearance.glow_strength, 0.0..=5.0).text("bloom strength"));
    ui.add_enabled(glow_on, egui::Slider::new(&mut appearance.glow_radius, 0.5..=2.0).text("bloom spread"));
    ui.separator();
    // Not strictly true on integrated graphics: the GPU and CPU share one power budget, so heavy rendering can slow the sim.
    ui.label("These settings only change rendering. On integrated graphics, heavy rendering (gas, bloom) shares the power budget with the simulation and can slow it.");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Celestial Sim   (S: settings)")
            .with_inner_size(LogicalSize::new(900.0, 900.0));
        let window = Arc::new(event_loop.create_window(attributes).expect("create window"));
        let size = window.inner_size();
        let gpu = Gpu::new(window);
        self.slice = Some(Camera::fit_box(self.box_size, [size.width as f32, size.height as f32]));
        self.orbit = Some(Orbit::fit_box(self.box_size));
        self.gpu = Some(gpu);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Take the newest frame and drop any older ones the simulation queued up.
        let mut latest = None;
        loop {
            match self.runner.frames.try_recv() {
                Ok(frame) => latest = Some(frame),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        if let (Some(frame), Some(gpu)) = (latest, self.gpu.as_mut()) {
            let now = Instant::now();
            let ranges = [log_range(&frame.speeds), log_range(&frame.densities)];
            // The old current frame becomes the previous one, if it holds the same particles. After a restart it
            // doesn't, so the first frame of a run is drawn against itself.
            let prev = self
                .playback
                .current
                .take()
                .filter(|p| p.positions.len() == frame.positions.len());
            match &prev {
                Some(p) => {
                    gpu.upload(p, &frame);
                    self.playback.prev_ranges = self.ranges;
                    if let Some(t) = self.playback.arrived {
                        let gap = now.duration_since(t).as_secs_f32();
                        self.playback.interval = Some(next_interval(self.playback.interval, gap));
                    }
                }
                None => {
                    gpu.upload(&frame, &frame);
                    self.playback.prev_ranges = ranges;
                }
            }
            self.ranges = ranges;
            // The grid is only in frames sent while the gas is on. Without one, forget the old grids, so turning the
            // gas back on later starts from fresh data.
            match &frame.grid {
                Some(grid) => gpu.volume.upload(&gpu.device, &gpu.queue, grid),
                None => gpu.volume.reset(),
            }
            self.redshift = frame.redshift;
            self.playback.arrived = Some(now);
            self.playback.current = Some(frame);
            gpu.window.set_title(&format!(
                "Celestial Sim   z = {:.2}   {} particles   (S: settings)",
                self.redshift, gpu.point_count
            ));
        }
        if let Some(gpu) = self.gpu.as_mut() {
            // A brightness reading from a frame or two ago, if it has arrived. Only the target is set here.
            // The exposure eases towards it on each redraw.
            if let Some(reading) = gpu.glow.poll_meter(&gpu.device) {
                if let Some(target) = reading.target(exposure::KEY) {
                    self.exposure_target = Some(target);
                }
            }
            gpu.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // egui sees every event first. If it consumes one (a click on the panel), the camera ignores it.
        let consumed = match self.gpu.as_mut() {
            Some(gpu) => gpu.egui_state.on_window_event(&gpu.window, &event).consumed,
            None => false,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_redraw).as_secs_f32();
                self.last_redraw = now;
                if self.appearance.auto_exposure {
                    if let Some(target) = self.exposure_target {
                        self.exposure.update(target, dt);
                    }
                }
                let exposure_now = self.current_exposure();
                // The sim only builds the density grid while it's wanted (for gas or lensing). Cheap to set every frame.
                let a = &self.appearance;
                self.runner.set_want_grid(a.gas || (a.lens && !a.lens_hq));
                let layers = self.layers();
                let uniform = self.view_uniform();
                let glow = self.glow_params();
                let footprint = self.footprint();
                let mut apply = false;
                if let Some(gpu) = self.gpu.as_mut() {
                    let ctx = gpu.egui_ctx.clone();
                    let raw = gpu.egui_state.take_egui_input(&gpu.window);
                    let panel = &mut self.panel;
                    let appearance = &mut self.appearance;
                    let full = ctx.run_ui(raw, |ui| {
                        apply = settings_panel(ui.ctx(), panel, appearance, exposure_now);
                    });
                    gpu.egui_state.handle_platform_output(&gpu.window, full.platform_output);
                    let mut ui = UiOutput {
                        primitives: ctx.tessellate(full.shapes, full.pixels_per_point),
                        textures: full.textures_delta,
                        pixels_per_point: full.pixels_per_point,
                    };
                    gpu.render(&uniform, &mut ui, glow, footprint, layers);
                    ui.textures.clear();
                }
                if apply {
                    self.restart();
                }
                self.autosave();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32, position.y as f32];
                let delta = [p[0] - self.cursor[0], p[1] - self.cursor[1]];
                if !consumed {
                    let viewport_h = self.viewport()[1];
                    match self.mode {
                        ViewMode::Slice => {
                            if self.left_down || self.right_down {
                                if let Some(camera) = self.slice.as_mut() {
                                    camera.pan(delta);
                                }
                            }
                        }
                        ViewMode::Orbit => {
                            if let Some(orbit) = self.orbit.as_mut() {
                                if self.left_down {
                                    orbit.rotate(delta);
                                } else if self.right_down {
                                    orbit.pan(delta, viewport_h);
                                }
                            }
                        }
                    }
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                // A press over the panel never starts a drag. A release always ends one, even over the panel.
                let allowed = !down || !consumed;
                if allowed {
                    match button {
                        MouseButton::Left => self.left_down = down,
                        MouseButton::Right => self.right_down = down,
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !consumed {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                    };
                    match self.mode {
                        ViewMode::Slice => {
                            let viewport = self.viewport();
                            let cursor = self.cursor;
                            if let Some(camera) = self.slice.as_mut() {
                                camera.zoom_at(cursor, viewport, ZOOM_PER_LINE.powf(lines));
                            }
                        }
                        ViewMode::Orbit => {
                            if let Some(orbit) = self.orbit.as_mut() {
                                orbit.zoom(lines);
                            }
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !consumed => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Tab) => {
                        self.mode = match self.mode {
                            ViewMode::Slice => ViewMode::Orbit,
                            ViewMode::Orbit => ViewMode::Slice,
                        };
                    }
                    Key::Named(NamedKey::Home) => {
                        let viewport = self.viewport();
                        match self.mode {
                            ViewMode::Slice => self.slice = Some(Camera::fit_box(self.box_size, viewport)),
                            ViewMode::Orbit => self.orbit = Some(Orbit::fit_box(self.box_size)),
                        }
                    }
                    Key::Character(ref c) if c.eq_ignore_ascii_case("f") => {
                        // Freeze the range that is on screen right now, which may be mid-interpolation.
                        self.frozen_range = match self.frozen_range {
                            Some(_) => None,
                            None => Some(self.display_range()),
                        };
                    }
                    Key::Character(ref c) if c.eq_ignore_ascii_case("s") => {
                        self.panel.open = !self.panel.open;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

// Loads the saved settings, starts the first run, opens the window, and runs until it is closed.
// Blocks the calling (main) thread.
pub fn run(box_size: f32, save_path: PathBuf) {
    let saved = store::load(&save_path);
    let event_loop = EventLoop::new().expect("create event loop");
    let runner = Runner::start(saved.run.clone());
    let appearance = saved.appearance.clone();
    let last_saved = Saved {
        run: runner.settings.clone(),
        appearance: appearance.clone(),
    };
    let mut app = App {
        runner,
        box_size,
        gpu: None,
        mode: ViewMode::Orbit,
        slice: None,
        orbit: None,
        redshift: 0.0,
        playback: Playback::default(),
        // Start from the manual exposure, so the first frames aren't wildly off before the meter reports.
        exposure: Adapter::new(saved.appearance.sanitized().exposure),
        exposure_target: None,
        last_redraw: Instant::now(),
        ranges: [(0.0, 1.0); 2],
        frozen_range: None,
        cursor: [0.0, 0.0],
        left_down: false,
        right_down: false,
        panel: PanelState {
            open: false,
            tab: PanelTab::Simulation,
            draft: saved.run.clone(),
        },
        appearance,
        last_saved,
        save_path,
    };
    event_loop.run_app(&mut app).expect("event loop");
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same as `lerp_periodic` in shader.wgsl.
    fn lerp_periodic(a: [f32; 3], b: [f32; 3], t: f32, l: f32) -> [f32; 3] {
        [0, 1, 2].map(|k| {
            let d = (b[k] - a[k]) - l * ((b[k] - a[k]) / l).round();
            let p = a[k] + d * t;
            p - l * (p / l).floor()
        })
    }

    #[test]
    fn interpolation_hits_both_frames_at_the_ends() {
        let (a, b) = ([10.0, 20.0, 30.0], [12.0, 18.0, 31.0]);
        assert_eq!(lerp_periodic(a, b, 0.0, 100.0), a);
        assert_eq!(lerp_periodic(a, b, 1.0, 100.0), b);
        assert_eq!(lerp_periodic(a, b, 0.5, 100.0), [11.0, 19.0, 30.5]);
    }

    #[test]
    fn interpolation_crosses_the_boundary_the_short_way() {
        // 99.5 -> 0.5 in a box of 100 is a 1-unit move across the edge. Halfway is on the edge (0), not at 50.
        let mid = lerp_periodic([99.5, 50.0, 50.0], [0.5, 50.0, 50.0], 0.5, 100.0);
        assert!(mid[0] < 0.01 || mid[0] > 99.99, "mid {mid:?}");
        let quarter = lerp_periodic([99.5, 50.0, 50.0], [0.5, 50.0, 50.0], 0.25, 100.0);
        assert!((quarter[0] - 99.75).abs() < 1e-4, "quarter {quarter:?}");
        // And the other direction.
        let back = lerp_periodic([0.5, 0.0, 0.0], [99.5, 0.0, 0.0], 0.25, 100.0);
        assert!((back[0] - 0.25).abs() < 1e-4, "back {back:?}");
    }

    #[test]
    fn alpha_runs_from_zero_to_one_over_the_interval() {
        assert_eq!(interp_alpha(0.0, Some(0.25)), 0.0);
        assert!((interp_alpha(0.125, Some(0.25)) - 0.5).abs() < 1e-6);
        // Holds at the current frame if the next one is late.
        assert_eq!(interp_alpha(10.0, Some(0.25)), 1.0);
        // No interval yet (first frame of a run): show it as-is.
        assert_eq!(interp_alpha(0.0, None), 1.0);
    }

    #[test]
    fn interval_starts_at_the_first_gap_and_smooths_after() {
        assert_eq!(next_interval(None, 0.3), 0.3);
        let i = next_interval(Some(0.2), 0.3);
        assert!(i > 0.2 && i < 0.3);
        // Repeated gaps converge on that gap.
        let mut i = Some(0.1);
        for _ in 0..50 {
            i = Some(next_interval(i, 0.25));
        }
        assert!((i.unwrap() - 0.25).abs() < 1e-4);
    }

    #[test]
    fn particle_record_matches_vertex_layout() {
        // Eight attributes (position, speed, redshift shift, density, each previous and current). 64 bytes, no padding.
        assert_eq!(std::mem::size_of::<ParticleVertex>(), 64);
    }

    #[test]
    fn log_range_maps_extremes() {
        let (lo, hi) = log_range(&[0.5, 2.0, 8.0]);
        assert!((lo - 0.5f32.ln()).abs() < 1e-6);
        assert!((hi - 8.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn log_range_clamps_slow_tail_to_cold_end() {
        // 95% of particles are at speed 1, with 5% very slow. The cold end should sit at speed 1, not at the slow tail.
        let mut speeds = vec![1.0f32; 95];
        speeds.extend([1e-4f32; 5]);
        let (lo, hi) = log_range(&speeds);
        assert!(lo.abs() < 1e-6, "lo {lo}");
        assert!(hi > lo);
    }

    #[test]
    fn log_range_handles_uniform_and_empty_frames() {
        let (lo, hi) = log_range(&[1.0, 1.0]);
        assert!(hi > lo);
        assert_eq!(log_range(&[]), (0.0, 1.0));
    }

    #[test]
    fn view_uniform_matches_shader_layout() {
        // The WGSL `View` struct is 272 bytes: a mat4x4, five 16-byte rows of scalars, and eight vec4 stops.
        assert_eq!(std::mem::size_of::<ViewUniform>(), 272);
    }

    // Same weight as `fs_splat` in shader.wgsl: a normalized 2D Gaussian evaluated at a pixel-centre offset.
    fn splat_weight(dx: f32, dy: f32, sigma: f32) -> f32 {
        let s2 = sigma * sigma;
        (-(dx * dx + dy * dy) / (2.0 * s2)).exp() / (std::f32::consts::TAU * s2)
    }

    // Total brightness a splat deposits, summing over the pixel centres its quad (Â±3Ïƒ) covers.
    // (cx, cy) is the particle centre within its pixel, in [0, 1).
    fn splat_total(cx: f32, cy: f32, sigma: f32) -> f32 {
        let reach = (3.0 * sigma).ceil() as i32 + 1;
        let mut sum = 0.0;
        for py in -reach..=reach {
            for px in -reach..=reach {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                if dx.abs() <= 3.0 * sigma && dy.abs() <= 3.0 * sigma {
                    sum += splat_weight(dx, dy, sigma);
                }
            }
        }
        sum
    }

    #[test]
    fn splat_brightness_is_the_same_at_every_subpixel_position() {
        // This is what removes the shimmer: as a particle slides across a pixel, its total brightness can't change.
        // At the minimum size, check a grid of sub-pixel positions. All should sum to 1 within 1%.
        let sigma = appearance::MIN_SPLAT_SIGMA;
        for i in 0..10 {
            for j in 0..10 {
                let total = splat_total(i as f32 / 10.0, j as f32 / 10.0, sigma);
                assert!((total - 1.0).abs() < 0.01, "sigma {sigma} at ({i}, {j})/10: total {total}");
            }
        }
    }

    #[test]
    fn narrow_splats_would_shimmer() {
        // Why the minimum exists: at 0.3 px, the total depends strongly on where the centre sits in the pixel.
        let centred = splat_total(0.5, 0.5, 0.3);
        let corner = splat_total(0.0, 0.0, 0.3);
        assert!((centred - corner).abs() > 0.1, "centred {centred}, corner {corner}");
    }

    #[test]
    fn shader_is_valid_wgsl() {
        // Parse and validate the shaders without a GPU, so shader errors show up in `cargo test`.
        for source in [
            include_str!("shader.wgsl"),
            include_str!("post.wgsl"),
            include_str!("meter.wgsl"),
            include_str!("volume.wgsl"),
            include_str!("lens.wgsl"),
            include_str!("fft.wgsl"),
            include_str!("lensfft.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("WGSL parse");
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .expect("WGSL validation");
        }
    }
}
