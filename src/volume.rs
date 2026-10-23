// GPU side of the density grid (volume.wgsl): the grids as 3D textures, and the passes that march rays through them.
// The gas glow (marched at half size, then added into the HDR scene) and the lensing convergence both use them.

use bytemuck::{Pod, Zeroable};

use crate::appearance::MAX_STOPS;
use crate::camera::Mat4;
use crate::frame::DensityGrid;
use crate::glow::{ADDITIVE, HDR_FORMAT};
use crate::lens::KAPPA_FORMAT;

// 16-bit float is filterable on every adapter. 32-bit float filtering is an optional feature, often missing on iGPUs.
const GRID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
// Largest value a 16-bit float holds is 65504. Denser cells are clamped below it rather than becoming infinity.
const MAX_GRID_VALUE: f32 = 60_000.0;

// Uniform block. Layout must match `struct Volume` in volume.wgsl (224 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VolumeUniform {
    inv_view_proj: [[f32; 4]; 4],
    box_size: f32,
    alpha: f32,
    intensity: f32,
    steps: u32,
    low: f32,
    high: f32,
    stop_count: u32,
    _pad: f32,
    stops: [[f32; 4]; MAX_STOPS],
}

// Settings for one frame of marching (gas and convergence share them).
pub struct GasParams {
    pub inv_view_proj: Mat4,
    pub box_size: f32,
    pub alpha: f32,
    pub intensity: f32,
    pub steps: u32,
    pub low: f32,
    pub high: f32,
    pub stops: [[f32; 4]; MAX_STOPS],
    pub stop_count: u32,
}

// The previous and current density grid. Each upload overwrites the older one and makes it current,
// so the shader can blend between them like the particles.
struct Grids {
    n: usize,
    textures: [wgpu::Texture; 2],
    // Index of the current grid. The other is the previous one.
    current: usize,
    // march_groups[k] reads texture 1 - k as previous and texture k as current.
    march_groups: [wgpu::BindGroup; 2],
}

pub struct Volume {
    march_layout: wgpu::BindGroupLayout,
    add_layout: wgpu::BindGroupLayout,
    march: wgpu::RenderPipeline,
    add: wgpu::RenderPipeline,
    kappa: wgpu::RenderPipeline,
    grid_sampler: wgpu::Sampler,
    glow_sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    // Half-size HDR target for the march, and the bind group that reads it for the add pass.
    glow: wgpu::TextureView,
    add_group: wgpu::BindGroup,
    grids: Option<Grids>,
}

impl Volume {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume"),
            source: wgpu::ShaderSource::Wgsl(include_str!("volume.wgsl").into()),
        });

        let texture_3d = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume march"),
            entries: &[
                texture_3d(0),
                texture_3d(1),
                sampler_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let add_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume add"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(5),
            ],
        });

        let pipeline = |layout: &wgpu::BindGroupLayout, fragment: &str, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(fragment),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_full"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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
        };
        let march = pipeline(&march_layout, "fs_march", HDR_FORMAT, None);
        // Added on top of the particles already in the scene.
        let add = pipeline(&add_layout, "fs_add", HDR_FORMAT, Some(ADDITIVE));
        let kappa = pipeline(&march_layout, "fs_kappa", KAPPA_FORMAT, None);

        // The density field is periodic, so the grid repeats at the box faces instead of clamping. A sample near one
        // face blends with the cells on the opposite face, which is where its neighbours really are.
        let grid_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("density grid"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let glow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gas glow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume uniform"),
            size: std::mem::size_of::<VolumeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let glow = glow_target(device, width, height);
        let add_group = add_bind_group(device, &add_layout, &glow, &glow_sampler);
        Self {
            march_layout,
            add_layout,
            march,
            add,
            kappa,
            grid_sampler,
            glow_sampler,
            uniform,
            glow,
            add_group,
            grids: None,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.glow = glow_target(device, width, height);
        self.add_group = add_bind_group(device, &self.add_layout, &self.glow, &self.glow_sampler);
    }

    // Forgets the grids, so the next upload starts fresh instead of blending from a stale frame.
    // Called while the gas is off, and when a grid of a new size arrives.
    pub fn reset(&mut self) {
        self.grids = None;
    }

    // Uploads a new density grid as the current one. The first upload (or one of a new size) fills both textures,
    // so there is nothing to blend from.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid: &DensityGrid) {
        let bits: Vec<u16> = grid
            .data
            .iter()
            .map(|&v| half::f16::from_f32(v.clamp(0.0, MAX_GRID_VALUE)).to_bits())
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bits);

        if self.grids.as_ref().is_none_or(|g| g.n != grid.n) {
            let textures = [grid_texture(device, grid.n), grid_texture(device, grid.n)];
            for t in &textures {
                write_grid(queue, t, grid.n, bytes);
            }
            let views = textures.each_ref().map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
            let march_groups = [
                self.march_bind_group(device, &views[1], &views[0]),
                self.march_bind_group(device, &views[0], &views[1]),
            ];
            self.grids = Some(Grids {
                n: grid.n,
                textures,
                current: 0,
                march_groups,
            });
            return;
        }
        let g = self.grids.as_mut().expect("grids exist");
        g.current = 1 - g.current;
        write_grid(queue, &g.textures[g.current], grid.n, bytes);
    }

    // Writes this frame's marching settings. Call before `encode_gas` or `encode_kappa`.
    pub fn prepare(&self, queue: &wgpu::Queue, p: &GasParams) {
        let u = VolumeUniform {
            inv_view_proj: p.inv_view_proj,
            box_size: p.box_size,
            alpha: p.alpha,
            intensity: p.intensity,
            steps: p.steps,
            low: p.low,
            high: p.high,
            stop_count: p.stop_count,
            _pad: 0.0,
            stops: p.stops,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    // Marches the glow into the half-size target, then adds it into `scene`. Does nothing until a grid arrives.
    pub fn encode_gas(&self, encoder: &mut wgpu::CommandEncoder, scene: &wgpu::TextureView) {
        let Some(g) = &self.grids else {
            return;
        };
        fullscreen_pass(encoder, "gas march", &self.glow, &self.march, &g.march_groups[g.current], wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        fullscreen_pass(encoder, "gas add", scene, &self.add, &self.add_group, wgpu::LoadOp::Load);
    }

    // Projected overdensity per pixel of `target` (the lensing convergence). Clears to zero (no lensing) until a
    // grid arrives, so stale convergence from an earlier run never lenses the background.
    pub fn encode_kappa(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        match &self.grids {
            Some(g) => fullscreen_pass(encoder, "kappa", target, &self.kappa, &g.march_groups[g.current], wgpu::LoadOp::Clear(wgpu::Color::BLACK)),
            None => clear_pass(encoder, target),
        }
    }

    fn march_bind_group(&self, device: &wgpu::Device, prev: &wgpu::TextureView, cur: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume march"),
            layout: &self.march_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(prev),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(cur),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.grid_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        })
    }
}

fn grid_texture(device: &wgpu::Device, n: usize) -> wgpu::Texture {
    let n = n as u32;
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("density grid"),
        size: wgpu::Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: n,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: GRID_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

// Rows of n texels (2 bytes each), n rows per z slice, n slices. The grid is already x fastest.
fn write_grid(queue: &wgpu::Queue, texture: &wgpu::Texture, n: usize, bytes: &[u8]) {
    let n32 = n as u32;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(n32 * 2),
            rows_per_image: Some(n32),
        },
        wgpu::Extent3d {
            width: n32,
            height: n32,
            depth_or_array_layers: n32,
        },
    );
}

// The glow is smooth, so half the window size is enough, and it is a quarter of the cost of full size.
fn glow_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("gas glow"),
            size: wgpu::Extent3d {
                width: (width / 2).max(1),
                height: (height / 2).max(1),
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

fn add_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    glow: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume add"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(glow),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn clear_pass(encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn fullscreen_pass(
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

    #[test]
    fn volume_uniform_matches_shader_layout() {
        assert_eq!(std::mem::size_of::<VolumeUniform>(), 224);
    }

    #[test]
    fn half_floats_round_trip_and_dense_cells_stay_finite() {
        // 1 + δ values the grid holds, from voids to dense cores. Half precision keeps them to about 0.1%.
        for v in [0.05f32, 1.0, 37.5, 4000.0] {
            let back = half::f16::from_bits(half::f16::from_f32(v).to_bits()).to_f32();
            assert!((back - v).abs() / v < 1e-3, "{v} -> {back}");
        }
        // Above 65504 a half float overflows to infinity. The clamp keeps the densest cells finite.
        let clamped = half::f16::from_f32(1e6f32.clamp(0.0, MAX_GRID_VALUE)).to_f32();
        assert!(clamped.is_finite());
    }
}
