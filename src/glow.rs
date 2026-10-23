// HDR scene, mip-chain bloom, and tone map. Particles are always drawn additively into the HDR scene, and every frame
// is tone-mapped onto the window, so turning bloom off doesn't change the base image. The passes (see post.wgsl):
//   downsample scene -> mip0 -> ... -> mipN, upsample with additive blending mipN -> ... -> mip0, composite.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::exposure::Reading;

// Half-float target, so bright overlapping particles can exceed 1.0 before the tone map.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// The chain stops at this many levels, or when the next level would be smaller than MIN_LEVEL_SIZE.
const MAX_LEVELS: usize = 7;
const MIN_LEVEL_SIZE: u32 = 8;

// Per-pass uniform. Layout must match `struct Post` in post.wgsl (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostParams {
    texel: [f32; 2],
    radius: f32,
    strength: f32,
    exposure: f32,
    level_scale: f32,
    hue_preserving: f32,
    _pad: f32,
}

// The controls that change while the view runs.
#[derive(Clone, Copy, Debug)]
pub struct GlowParams {
    // Run the bloom passes. When false, only the tone map runs.
    pub bloom: bool,
    pub strength: f32,
    pub radius: f32,
    // Final exposure for the tone map: the manual value, or auto exposure times its bias.
    pub exposure: f32,
    // Measure scene brightness for auto exposure this frame (when a measurement isn't already in flight).
    pub meter: bool,
    // Hue-preserving tone map, or per-channel (bright colours clip towards white).
    pub hue_preserving: bool,
}

// Readback states, shared with the map_async callback (which may run on another thread).
const MAP_PENDING: u8 = 0;
const MAP_DONE: u8 = 1;
const MAP_FAILED: u8 = 2;

// Where the brightness reading is in its round trip to the CPU. Only one is in flight at a time,
// because a buffer can't be copied into while it is mapped.
#[derive(Clone, Copy, PartialEq, Debug)]
enum MeterState {
    Idle,
    // The copy into the readback buffer is in this frame's commands, which aren't submitted yet.
    Copied,
    // Submitted and map_async requested. Waiting for the GPU to finish.
    Mapping,
}

// GPU brightness meter (meter.wgsl), and the buffers that bring its 8-byte result back to the CPU.
struct Meter {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    // Reads bloom level 0. Rebuilt on resize, since that texture is recreated.
    bind_group: wgpu::BindGroup,
    storage: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: MeterState,
    mapped: Arc<AtomicU8>,
}

// Size of the meter's result: two f32s.
const READING_BYTES: wgpu::BufferAddress = 8;

impl Meter {
    fn new(device: &wgpu::Device, level0: &wgpu::TextureView) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("meter.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meter"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // textureLoad only, no sampling, so filterability doesn't matter.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meter"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meter"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_meter"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meter result"),
            size: READING_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meter readback"),
            size: READING_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Self::bind(device, &layout, level0, &storage);
        Self {
            pipeline,
            layout,
            bind_group,
            storage,
            readback,
            state: MeterState::Idle,
            mapped: Arc::new(AtomicU8::new(MAP_PENDING)),
        }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        level0: &wgpu::TextureView,
        storage: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meter"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(level0),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
            ],
        })
    }

    // Measures level 0 and queues the copy to the readback buffer. Skipped while a reading is already in flight.
    fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.state != MeterState::Idle {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("meter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.storage, 0, &self.readback, 0, READING_BYTES);
        self.state = MeterState::Copied;
    }

    // Asks for the readback buffer to be mapped, once the commands that fill it have been submitted.
    fn after_submit(&mut self) {
        if self.state != MeterState::Copied {
            return;
        }
        self.mapped.store(MAP_PENDING, Ordering::Release);
        let flag = Arc::clone(&self.mapped);
        self.readback.map_async(wgpu::MapMode::Read, .., move |result| {
            flag.store(if result.is_ok() { MAP_DONE } else { MAP_FAILED }, Ordering::Release);
        });
        self.state = MeterState::Mapping;
    }

    // Returns the reading once the GPU has finished, and frees the buffer for the next one. Never blocks.
    fn poll(&mut self, device: &wgpu::Device) -> Option<Reading> {
        if self.state != MeterState::Mapping {
            return None;
        }
        // Lets wgpu run finished map callbacks. Poll (not Wait), so a slow GPU never stalls the frame.
        let _ = device.poll(wgpu::PollType::Poll);
        match self.mapped.load(Ordering::Acquire) {
            MAP_DONE => {
                // Decoded byte by byte, so the mapped memory's alignment doesn't matter.
                let reading = self.readback.get_mapped_range(..).ok().map(|bytes| {
                    let f = |i: usize| f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
                    Reading {
                        sum_log: f(0),
                        lit: f(4),
                    }
                });
                self.readback.unmap();
                self.state = MeterState::Idle;
                reading
            }
            MAP_FAILED => {
                self.state = MeterState::Idle;
                None
            }
            _ => None,
        }
    }
}

// Sizes of the bloom levels for a window: the first is half the window, then each halves again.
// There is always at least one level, even for a tiny window.
pub fn bloom_level_sizes(width: u32, height: u32) -> Vec<[u32; 2]> {
    let mut size = [(width / 2).max(1), (height / 2).max(1)];
    let mut sizes = vec![size];
    while sizes.len() < MAX_LEVELS {
        let next = [size[0] / 2, size[1] / 2];
        if next[0].min(next[1]) < MIN_LEVEL_SIZE {
            break;
        }
        sizes.push(next);
        size = next;
    }
    sizes
}

// One post-processing pass: what it reads (bind group), its uniform, and the texel size of its source.
struct Pass {
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    // Texel size of the texture this pass samples.
    src_texel: [f32; 2],
}

// Everything that depends on the window size. Rebuilt on resize.
struct Targets {
    scene: wgpu::TextureView,
    levels: Vec<wgpu::TextureView>,
    // down[i] samples the scene (i = 0) or level i-1, and writes level i.
    down: Vec<Pass>,
    // up[i] samples level i+1 and adds into level i.
    up: Vec<Pass>,
    // Samples the scene and level 0, and writes the window.
    composite: Pass,
}

pub struct Glow {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    targets: Targets,
    meter: Meter,
}

impl Glow {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(3),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let down = post_pipeline(device, &pipeline_layout, &shader, "fs_down", HDR_FORMAT, None);
        // The upsample adds onto the level's downsampled content, so each level ends up with all the blurrier ones.
        let up = post_pipeline(device, &pipeline_layout, &shader, "fs_up", HDR_FORMAT, Some(ADDITIVE));
        let composite = post_pipeline(device, &pipeline_layout, &shader, "fs_composite", output_format, None);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let targets = build_targets(device, &layout, &sampler, width, height);
        let meter = Meter::new(device, &targets.levels[0]);
        Self {
            layout,
            sampler,
            down,
            up,
            composite,
            targets,
            meter,
        }
    }

    // Recreates the scene and bloom textures for a new window size, and points the meter at the new level 0.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.targets = build_targets(device, &self.layout, &self.sampler, width, height);
        self.meter.bind_group = Meter::bind(device, &self.meter.layout, &self.targets.levels[0], &self.meter.storage);
    }

    // Call after submitting the frame's commands, so a queued brightness reading can start its trip to the CPU.
    pub fn after_submit(&mut self) {
        self.meter.after_submit();
    }

    // The latest brightness reading, if one has arrived since the last call. Never blocks.
    pub fn poll_meter(&mut self, device: &wgpu::Device) -> Option<Reading> {
        self.meter.poll(device)
    }

    // The HDR target the particles are drawn into.
    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.targets.scene
    }

    // Runs the bloom chain (if enabled), the brightness meter (if asked), and writes the tone-mapped result to `out`.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, out: &wgpu::TextureView, p: GlowParams) {
        let t = &self.targets;
        let levels = t.levels.len();
        let write = |pass: &Pass, params: PostParams| queue.write_buffer(&pass.uniform, 0, bytemuck::bytes_of(&params));
        let base = PostParams {
            texel: [0.0; 2],
            radius: p.radius,
            strength: 0.0,
            exposure: p.exposure,
            level_scale: 1.0 / levels as f32,
            hue_preserving: if p.hue_preserving { 1.0 } else { 0.0 },
            _pad: 0.0,
        };

        // Level 0 is needed by the bloom, and by the meter. The rest of the downsample chain only by the bloom.
        let down_count = if p.bloom { t.down.len() } else if p.meter { 1 } else { 0 };
        for (i, pass) in t.down.iter().enumerate().take(down_count) {
            write(pass, PostParams { texel: pass.src_texel, ..base });
            run_pass(encoder, "bloom down", &t.levels[i], &self.down, &pass.bind_group, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        }
        // Measure before the upsample adds the blurred levels into level 0, so the reading is the scene itself.
        if p.meter {
            self.meter.encode(encoder);
        }
        if p.bloom {
            // Smallest first, so each level receives the already-accumulated level below it.
            for (i, pass) in t.up.iter().enumerate().rev() {
                write(pass, PostParams { texel: pass.src_texel, ..base });
                run_pass(encoder, "bloom up", &t.levels[i], &self.up, &pass.bind_group, wgpu::LoadOp::Load);
            }
        }

        let strength = if p.bloom { p.strength } else { 0.0 };
        write(&t.composite, PostParams { texel: t.composite.src_texel, strength, ..base });
        run_pass(encoder, "composite", out, &self.composite, &t.composite.bind_group, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
    }
}

// Adds source to destination. Used for the particles and the bloom upsample.
pub const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

// Builds the window-sized textures, and a uniform and bind group for every pass that reads them.
fn build_targets(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
) -> Targets {
    let scene_size = [width.max(1), height.max(1)];
    let sizes = bloom_level_sizes(scene_size[0], scene_size[1]);
    let scene = target(device, "scene", scene_size);
    let levels: Vec<wgpu::TextureView> = sizes.iter().map(|&s| target(device, "bloom level", s)).collect();

    // `bloom` is only read by the composite. Other passes bind their source there too, as a placeholder that
    // is never the texture being written (writing and reading one texture in a pass is rejected).
    let make_pass = |src: &wgpu::TextureView, src_size: [u32; 2], bloom: &wgpu::TextureView| {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post uniform"),
            size: std::mem::size_of::<PostParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom),
                },
            ],
        });
        Pass {
            bind_group,
            uniform,
            src_texel: [1.0 / src_size[0] as f32, 1.0 / src_size[1] as f32],
        }
    };

    let down = (0..levels.len())
        .map(|i| {
            if i == 0 {
                make_pass(&scene, scene_size, &scene)
            } else {
                make_pass(&levels[i - 1], sizes[i - 1], &levels[i - 1])
            }
        })
        .collect();
    let up = (0..levels.len().saturating_sub(1))
        .map(|i| make_pass(&levels[i + 1], sizes[i + 1], &levels[i + 1]))
        .collect();
    let composite = make_pass(&scene, scene_size, &levels[0]);

    Targets {
        scene,
        levels,
        down,
        up,
        composite,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn target(device: &wgpu::Device, label: &str, size: [u32; 2]) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn post_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    fragment: &str,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_full"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn run_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same as `tone_map_hue` in post.wgsl.
    fn tone_map(c: [f32; 3]) -> [f32; 3] {
        let m = c[0].max(c[1]).max(c[2]);
        if m <= 0.0 {
            return [0.0; 3];
        }
        let s = (1.0 - (-m).exp()) / m;
        c.map(|x| x * s)
    }

    #[test]
    fn tone_map_keeps_hue_and_never_clips() {
        for &c in &[[0.2, 0.1, 0.05], [5.0, 1.0, 0.2], [100.0, 40.0, 1.0], [0.0, 0.0, 3.0]] {
            let t = tone_map(c);
            // At most 1. Very bright input reaches exactly 1.0 in f32, which is the ceiling, not a clip.
            assert!(t.iter().all(|&x| (0.0..=1.0).contains(&x)), "{c:?} -> {t:?}");
            // Channel ratios are unchanged, so the hue is the same.
            let m = c[0].max(c[1]).max(c[2]);
            let tm = t[0].max(t[1]).max(t[2]);
            for k in 0..3 {
                assert!((c[k] / m - t[k] / tm).abs() < 1e-5, "{c:?} -> {t:?}");
            }
        }
        assert_eq!(tone_map([0.0; 3]), [0.0; 3]);
    }

    // Same as `tone_map_white` in post.wgsl.
    fn tone_map_white(c: [f32; 3]) -> [f32; 3] {
        c.map(|x| 1.0 - (-x).exp())
    }

    #[test]
    fn per_channel_tone_map_bleaches_bright_colours_towards_white() {
        // The bright end of the default ramp, orange, at increasing exposure. Per channel, the weakest channel (blue)
        // catches up with the strongest, so the colour heads towards white. The hue-preserving map keeps the ratio.
        let orange = [1.0, 0.55, 0.1];
        let bright = orange.map(|x| x * 20.0);
        let white = tone_map_white(bright);
        let kept = tone_map(bright);
        assert!(white[2] / white[0] > 0.8, "per-channel blue/red {}", white[2] / white[0]);
        assert!((kept[2] / kept[0] - 0.1).abs() < 1e-4, "hue-preserving blue/red {}", kept[2] / kept[0]);
    }

    #[test]
    fn tone_map_is_monotone_in_brightness() {
        let mut prev = 0.0;
        for i in 1..100 {
            let v = tone_map([i as f32 * 0.1, 0.0, 0.0])[0];
            assert!(v > prev);
            prev = v;
        }
    }

    #[test]
    fn level_sizes_halve_and_stop_before_tiny() {
        // 900 px: 450, 225, 112, 56, 28, 14. The next (7) is below the minimum.
        let sizes = bloom_level_sizes(900, 900);
        let widths: Vec<u32> = sizes.iter().map(|s| s[0]).collect();
        assert_eq!(widths, vec![450, 225, 112, 56, 28, 14]);
    }

    #[test]
    fn level_sizes_cap_and_never_empty() {
        assert_eq!(bloom_level_sizes(100_000, 100_000).len(), MAX_LEVELS);
        // A tiny window still gets one level, so the composite always has a bloom texture to read.
        assert_eq!(bloom_level_sizes(1, 1), vec![[1, 1]]);
    }

    #[test]
    fn post_params_match_shader_layout() {
        assert_eq!(std::mem::size_of::<PostParams>(), 32);
    }
}
