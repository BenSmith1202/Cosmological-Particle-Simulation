// GPU side of gravitational lensing (lens.wgsl): the convergence and deflection grids, and the passes that build the
// deflection and draw the lensed background. The convergence itself is marched by the volume (fs_kappa), since it
// reads the same density grids.

use bytemuck::{Pod, Zeroable};

use crate::camera::Mat4;
use crate::glow::HDR_FORMAT;

// Convergence: one signed value per texel. Deflection: αx, αy, and κ (for the projected-mass tint), the same
// layout the FFT path writes. Both 16-bit floats, so they are filterable.
pub const KAPPA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const DEFLECT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// Grid side, set by the quality slider. Every deflection texel sums over every convergence texel, so the cost is
// size⁴ texture loads per frame: 17 million at 64, 268 million at 128. Above about 128 the density grid the
// convergence comes from (at most 128 cells across the box) is the limit, so a finer lensing grid adds cost but
// little detail.
// Measured on the dev laptop (iGPU, 64k particles): lensing off 238 fps; on at 32, 64, 128: 69, 65, 32 fps.
// So about 10 ms is a fixed cost (most likely the full-resolution background galaxies, not profiled), and the
// size⁴ deflection only dominates near the top of the range.
pub const MIN_LENS_SIZE: u32 = 32;
pub const MAX_LENS_SIZE: u32 = 128;

// Uniform block. Layout must match `struct Lens` in lens.wgsl (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LensUniform {
    inv_view_proj: [[f32; 4]; 4],
    aspect: f32,
    strength: f32,
    galaxy_brightness: f32,
    show_kappa: f32,
    ortho: u32,
    box_size: f32,
    kappa_size: f32,
    _pad: f32,
}

pub struct LensParams {
    pub inv_view_proj: Mat4,
    pub aspect: f32,
    pub strength: f32,
    pub galaxy_brightness: f32,
    pub show_kappa: bool,
    pub ortho: bool,
    pub box_size: f32,
    // Grid side, MIN_LENS_SIZE..=MAX_LENS_SIZE.
    pub size: u32,
}

// The size-dependent part: both grids, and the bind groups that read them. Rebuilt when the resolution changes.
struct Grids {
    size: u32,
    kappa_view: wgpu::TextureView,
    deflect_view: wgpu::TextureView,
    // The deflection pass writes the deflection texture, so its bind group must not also bind it as an input.
    // Two bind groups: one without the deflection texture (for that pass), one with it (for the background).
    deflect_group: wgpu::BindGroup,
    background_group: wgpu::BindGroup,
}

pub struct Lens {
    deflect_layout: wgpu::BindGroupLayout,
    background_layout: wgpu::BindGroupLayout,
    deflect: wgpu::RenderPipeline,
    background: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    grids: Grids,
}

impl Lens {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lens.wgsl").into()),
        });

        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let deflect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens deflect"),
            entries: &[texture_entry(0), uniform_entry],
        });
        let background_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens background"),
            entries: &[
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform_entry,
            ],
        });

        let pipeline = |layout: &wgpu::BindGroupLayout, fragment: &str, format: wgpu::TextureFormat| {
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
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let deflect = pipeline(&deflect_layout, "fs_deflect", DEFLECT_FORMAT);
        // The background clears the scene: it is the first thing drawn, and the particles add on top.
        let background = pipeline(&background_layout, "fs_background", HDR_FORMAT);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lens"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens uniform"),
            size: std::mem::size_of::<LensUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grids = build_grids(device, &deflect_layout, &background_layout, &sampler, &uniform, 64);
        Self {
            deflect_layout,
            background_layout,
            deflect,
            background,
            sampler,
            uniform,
            grids,
        }
    }

    // Rebuilds both grids if the resolution changed. Cheap to call every frame.
    pub fn set_size(&mut self, device: &wgpu::Device, size: u32) {
        let size = size.clamp(MIN_LENS_SIZE, MAX_LENS_SIZE);
        if size != self.grids.size {
            self.grids = build_grids(device, &self.deflect_layout, &self.background_layout, &self.sampler, &self.uniform, size);
        }
    }

    // Where the volume writes the convergence each frame.
    pub fn kappa_view(&self) -> &wgpu::TextureView {
        &self.grids.kappa_view
    }

    // Builds the deflection from this frame's convergence, then draws the lensed background into `scene`
    // (clearing it). The particles must be drawn after this, adding on top. Call `set_size` first.
    // With `fft_deflection` (the high-quality path's background layer), the direct deflection pass is skipped
    // and the background uses that texture instead.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        scene: &wgpu::TextureView,
        p: &LensParams,
        fft_deflection: Option<&wgpu::TextureView>,
    ) {
        let g = &self.grids;
        let u = LensUniform {
            inv_view_proj: p.inv_view_proj,
            aspect: p.aspect,
            strength: p.strength,
            galaxy_brightness: p.galaxy_brightness,
            show_kappa: if p.show_kappa { 1.0 } else { 0.0 },
            ortho: p.ortho as u32,
            box_size: p.box_size,
            kappa_size: g.size as f32,
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        match fft_deflection {
            Some(view) => {
                // Made per frame: the FFT path's texture is rebuilt when its resolution changes.
                let group = background_bind_group(device, &self.background_layout, view, &self.sampler, &self.uniform);
                pass(encoder, "lens background", scene, &self.background, &group);
            }
            None => {
                pass(encoder, "lens deflect", &g.deflect_view, &self.deflect, &g.deflect_group);
                pass(encoder, "lens background", scene, &self.background, &g.background_group);
            }
        }
    }
}

fn build_grids(
    device: &wgpu::Device,
    deflect_layout: &wgpu::BindGroupLayout,
    background_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    size: u32,
) -> Grids {
    let grid = |label, format| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let kappa_view = grid("convergence", KAPPA_FORMAT);
    let deflect_view = grid("deflection", DEFLECT_FORMAT);

    let deflect_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lens deflect"),
        layout: deflect_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&kappa_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let background_group = background_bind_group(device, background_layout, &deflect_view, sampler, uniform);
    Grids {
        size,
        kappa_view,
        deflect_view,
        deflect_group,
        background_group,
    }
}

// The background reads one deflection texture: this lens's own, or the FFT path's background layer.
fn background_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    deflection: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lens background"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(deflection),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}

fn pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_uniform_matches_shader_layout() {
        assert_eq!(std::mem::size_of::<LensUniform>(), 96);
    }

    // Same sum as `fs_deflect` in lens.wgsl, on the CPU, for a square map (aspect 1).
    fn deflection(kappa: &[f32], n: usize, at: [f32; 2]) -> [f32; 2] {
        let cell = 1.0 / n as f32;
        let eps2 = 0.25 * 2.0 * cell * cell;
        let mut a = [0.0f32; 2];
        for j in 0..n {
            for i in 0..n {
                let there = [(i as f32 + 0.5) * cell - 0.5, (j as f32 + 0.5) * cell - 0.5];
                let d = [at[0] - there[0], at[1] - there[1]];
                let r2 = d[0] * d[0] + d[1] * d[1] + eps2;
                let k = kappa[j * n + i];
                a[0] += k * d[0] / r2;
                a[1] += k * d[1] / r2;
            }
        }
        [a[0] * cell * cell / std::f32::consts::PI, a[1] * cell * cell / std::f32::consts::PI]
    }

    #[test]
    fn point_mass_deflection_falls_as_one_over_distance_and_points_away() {
        // All mass in the centre texel: α = (1/π) m / r, pointing from the mass towards the image (so the image
        // sits further out than the source, which is what stretches background galaxies around a cluster).
        let n = 17;
        let mut kappa = vec![0.0; n * n];
        kappa[(n / 2) * n + n / 2] = 1.0;
        // Far enough out that the half-cell softening changes the ratio by about 1%.
        let near = deflection(&kappa, n, [0.3, 0.0]);
        let far = deflection(&kappa, n, [0.45, 0.0]);
        assert!(near[0] > 0.0 && near[1].abs() < 1e-9, "near {near:?}");
        let ratio = near[0] / far[0];
        assert!((ratio - 1.5).abs() < 0.03, "ratio {ratio}");
    }

    #[test]
    fn uniform_sheet_gives_no_deflection_at_its_centre() {
        // By symmetry, mass spread evenly pulls equally in every direction at the centre.
        let n = 16;
        let kappa = vec![1.0; n * n];
        let a = deflection(&kappa, n, [0.0, 0.0]);
        assert!(a[0].abs() < 1e-5 && a[1].abs() < 1e-5, "{a:?}");
    }

    #[test]
    fn deflection_of_a_fixed_mass_doesnt_depend_on_grid_resolution() {
        // The same total mass in the centre, on a coarse and a fine grid, should bend light by the same amount.
        // This is what lets the quality slider change detail without changing the overall strength.
        let at = [0.3, 0.0];
        let mass_at_centre = |n: usize| {
            let mut kappa = vec![0.0; n * n];
            // κ is a density: putting mass m in one cell of area 1/n² means κ = m·n².
            kappa[(n / 2) * n + n / 2] = (n * n) as f32;
            kappa
        };
        let coarse = deflection(&mass_at_centre(17), 17, at)[0];
        let fine = deflection(&mass_at_centre(33), 33, at)[0];
        assert!((coarse - fine).abs() / fine < 0.02, "coarse {coarse}, fine {fine}");
    }
}
