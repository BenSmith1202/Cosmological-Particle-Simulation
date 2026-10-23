// High-quality lensing: multi-plane (depth slabs), with deflections from FFT convolution instead of a direct sum.
//
// The deflection is a convolution of the convergence κ with the softened thin-lens kernel
//   G(d) = (1/π) d / (|d|² + ε²) · ΔA,
// the same kernel the direct sum in lens.wgsl uses. On a K×K map that is K⁴ work directly, or O(K² log K) with FFTs:
// α = IFFT(FFT(κ) · FFT(G)). Gx and Gy are packed as one complex field Gx + i·Gy, so a single inverse FFT returns
// αx in the real part and αy in the imaginary part (both are real fields, so nothing mixes).
// The map is zero-padded to N = 2K per side. An FFT convolution wraps around, so without padding, mass near one
// edge would bend light at the opposite edge.

use num_complex::Complex;

// Kernel FFT, H = FFT(Gx + i·Gy), on the padded N×N grid (N = 2k), row-major [y][x].
// Texels are aspect/k wide and 1/k tall in the isotropic screen units the shaders use.
pub fn kernel_hat(k: usize, aspect: f32) -> Vec<Complex<f32>> {
    let n = 2 * k;
    let (dx, dy) = (aspect / k as f32, 1.0 / k as f32);
    // Half-cell softening, the same as the direct sum, so a texel's own mass stays finite.
    let eps2 = 0.25 * (dx * dx + dy * dy);
    let scale = dx * dy / std::f32::consts::PI;
    let offset = |i: usize| if i < n / 2 { i as f32 } else { i as f32 - n as f32 };
    let mut g = vec![Complex::new(0.0, 0.0); n * n];
    for y in 0..n {
        for x in 0..n {
            let d = [offset(x) * dx, offset(y) * dy];
            let r2 = d[0] * d[0] + d[1] * d[1] + eps2;
            g[y * n + x] = Complex::new(d[0] / r2 * scale, d[1] / r2 * scale);
        }
    }
    fft2d(&mut g, n, false);
    g
}

// In-place 2D FFT of an n×n row-major grid, rows then columns. Unnormalized in both directions.
pub fn fft2d(data: &mut [Complex<f32>], n: usize, inverse: bool) {
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    for row in data.chunks_mut(n) {
        fft.process(row);
    }
    let mut column = vec![Complex::new(0.0, 0.0); n];
    for x in 0..n {
        for y in 0..n {
            column[y] = data[y * n + x];
        }
        fft.process(&mut column);
        for y in 0..n {
            data[y * n + x] = column[y];
        }
    }
}

// One FFT stage's parameters. Layout must match `struct Stage` in fft.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StageParams {
    n: u32,
    ns: u32,
    axis: u32,
    inverse: u32,
}

// Uniform entries read with dynamic offsets must start at multiples of this (wgpu's default limit).
const STAGE_STRIDE: u64 = 256;

// 2D FFTs of `maps` n×n complex maps on the GPU, between two storage buffers A and B. Results end in A,
// since a 2D transform is 2·log2(n) stages, an even number of ping-pongs.
pub struct GpuFft {
    pipeline: wgpu::ComputePipeline,
    n: u32,
    maps: u32,
    // log2(n): stages per axis.
    log_n: u32,
    // Every stage of a forward and an inverse transform, at STAGE_STRIDE apart. Constant for a given n.
    // Held so it lives as long as the bind groups that read it.
    _stages: wgpu::Buffer,
    a_to_b: wgpu::BindGroup,
    b_to_a: wgpu::BindGroup,
}

impl GpuFft {
    // `a` and `b` must each hold maps·n·n vec2<f32>. n must be a power of two.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, n: u32, maps: u32, a: &wgpu::Buffer, b: &wgpu::Buffer) -> Self {
        assert!(n.is_power_of_two() && n >= 2, "FFT size must be a power of two");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fft.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let stage_size = wgpu::BufferSize::new(std::mem::size_of::<StageParams>() as u64);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft"),
            entries: &[
                storage(0, true),
                storage(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // One buffer holds every stage. Each dispatch picks its entry by offset.
                        has_dynamic_offset: true,
                        min_binding_size: stage_size,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_fft"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Forward x stages, forward y stages, inverse x, inverse y: 4·log2(n) entries.
        let log_n = n.trailing_zeros();
        let mut bytes = vec![0u8; (4 * log_n) as usize * STAGE_STRIDE as usize];
        let mut i = 0;
        for inverse in [0, 1] {
            for axis in [0, 1] {
                for s in 0..log_n {
                    let p = StageParams { n, ns: 1 << s, axis, inverse };
                    let at = i * STAGE_STRIDE as usize;
                    bytes[at..at + 16].copy_from_slice(bytemuck::bytes_of(&p));
                    i += 1;
                }
            }
        }
        let stages = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft stages"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&stages, 0, &bytes);

        let bind = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &stages,
                            offset: 0,
                            size: stage_size,
                        }),
                    },
                ],
            })
        };
        let a_to_b = bind(a, b);
        let b_to_a = bind(b, a);
        Self { pipeline, n, maps, log_n, _stages: stages, a_to_b, b_to_a }
    }

    // Full 2D transform of every map in A (unnormalized). The result is back in A.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, inverse: bool) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(if inverse { "fft inverse" } else { "fft forward" }),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        let first = if inverse { 2 * self.log_n } else { 0 };
        let groups_x = (self.n / 2).div_ceil(64);
        for s in 0..2 * self.log_n {
            let group = if s % 2 == 0 { &self.a_to_b } else { &self.b_to_a };
            let offset = ((first + s) as u64 * STAGE_STRIDE) as u32;
            pass.set_bind_group(0, group, &[offset]);
            pass.dispatch_workgroups(groups_x, self.n, self.maps);
        }
    }
}

// Resolutions offered (the FFT needs powers of two), and the slab limit.
// At 512 with 8 slabs, each complex field buffer is 9 · 1024² · 8 bytes = 75 MB, under wgpu's default 128 MiB
// storage-binding limit. More slabs at 512 would exceed it.
pub const LENS_HQ_SIZES: [u32; 4] = [64, 128, 256, 512];
pub const MIN_SLABS: u32 = 2;
pub const MAX_SLABS: u32 = 8;
const DEFLECT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// Recompute the kernel when the window's aspect changes by more than this. The kernel FFT is a CPU job
// (tens of ms at 512), so it shouldn't rerun on every pixel of a window drag.
const ASPECT_TOLERANCE: f32 = 0.01;

// Uniform block. Layout must match `struct Params` in lensfft.wgsl (192 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LensFftUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    k: u32,
    n: u32,
    slabs: u32,
    count: u32,
    depth_near: f32,
    depth_far: f32,
    box_size: f32,
    alpha: f32,
    aspect: f32,
    smoothing: f32,
    weight_scale: f32,
    area_coeff: f32,
    ortho: u32,
    r_ref: f32,
    _pad0: f32,
    _pad1: f32,
}

// One frame's camera and simulation inputs. See `Params` in lensfft.wgsl for what each means.
pub struct LensFftParams {
    pub view_proj: crate::camera::Mat4,
    pub inv_view_proj: crate::camera::Mat4,
    pub particle_count: u32,
    pub depth_near: f32,
    pub depth_far: f32,
    pub box_size: f32,
    pub alpha: f32,
    pub aspect: f32,
    pub smoothing: f32,
    pub weight_scale: f32,
    pub area_coeff: f32,
    pub ortho: bool,
    pub r_ref: f32,
}

// Everything sized by the resolution and slab count. Rebuilt when either changes.
struct Sized {
    k: u32,
    slabs: u32,
    aspect: f32,
    counts: wgpu::Buffer,
    // Kept alive for the FFT and the bind groups that use them.
    _field_a: wgpu::Buffer,
    _field_b: wgpu::Buffer,
    kernel: wgpu::Buffer,
    fft: GpuFft,
    deflection_view: wgpu::TextureView,
    background_view: wgpu::TextureView,
    resolve_group: wgpu::BindGroup,
    multiply_group: wgpu::BindGroup,
    write_group: wgpu::BindGroup,
}

pub struct LensFft {
    bin: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    multiply: wgpu::ComputePipeline,
    write: wgpu::ComputePipeline,
    bin_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    multiply_layout: wgpu::BindGroupLayout,
    write_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sized: Sized,
    // Bumped whenever the deflection texture is recreated, so users of it can rebuild their bind groups.
    generation: u64,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl LensFft {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, aspect: f32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens fft"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lensfft.wgsl").into()),
        });
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        // Each pass's layout lists only the bindings it uses. Buffers declared read_write in the shader must be
        // read_write here too, even in passes that only read them.
        let bin_layout = layout("lens bin", &[uniform_entry(), storage_entry(1, true), storage_entry(2, false)]);
        let resolve_layout = layout("lens resolve", &[uniform_entry(), storage_entry(2, false), storage_entry(4, false)]);
        let multiply_layout = layout("lens multiply", &[uniform_entry(), storage_entry(4, false), storage_entry(5, true)]);
        let write_layout = layout(
            "lens write",
            &[
                uniform_entry(),
                storage_entry(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: DEFLECT_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        );
        let pipeline = |layout: &wgpu::BindGroupLayout, entry: &str| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bin = pipeline(&bin_layout, "cs_bin");
        let resolve = pipeline(&resolve_layout, "cs_resolve");
        let multiply = pipeline(&multiply_layout, "cs_multiply");
        let write = pipeline(&write_layout, "cs_write");
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens fft uniform"),
            size: std::mem::size_of::<LensFftUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Smallest configuration until high-quality lensing is turned on.
        let sized = build_sized(
            device, queue, &resolve_layout, &multiply_layout, &write_layout, &uniform, LENS_HQ_SIZES[0], MIN_SLABS, aspect,
        );
        Self {
            bin,
            resolve,
            multiply,
            write,
            bin_layout,
            resolve_layout,
            multiply_layout,
            write_layout,
            uniform,
            sized,
            generation: 0,
        }
    }

    // Rebuilds the buffers and textures if the resolution or slab count changed, and the kernel if the aspect did.
    pub fn configure(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, k: u32, slabs: u32, aspect: f32) {
        let s = &self.sized;
        let aspect_changed = (aspect - s.aspect).abs() > ASPECT_TOLERANCE * s.aspect;
        if k == s.k && slabs == s.slabs {
            if aspect_changed {
                write_kernel(queue, &s.kernel, k, aspect);
                self.sized.aspect = aspect;
            }
            return;
        }
        self.sized = build_sized(
            device, queue, &self.resolve_layout, &self.multiply_layout, &self.write_layout, &self.uniform, k, slabs, aspect,
        );
        self.generation += 1;
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    // All deflection layers: slab targets 0..slabs-1, then the background at layer `slabs`.
    pub fn deflection_view(&self) -> &wgpu::TextureView {
        &self.sized.deflection_view
    }

    // The background's layer alone, as a plain 2D texture.
    pub fn background_view(&self) -> &wgpu::TextureView {
        &self.sized.background_view
    }

    // The deflection texture itself, for tests that read it back.
    #[cfg(test)]
    fn deflection_texture(&self) -> wgpu::Texture {
        self.sized.background_view.texture().clone()
    }

    // Runs the whole chain for this frame. `particles` is the viewer's particle buffer (it needs STORAGE usage).
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        particles: &wgpu::Buffer,
        p: &LensFftParams,
    ) {
        let s = &self.sized;
        let n = 2 * s.k;
        let u = LensFftUniform {
            view_proj: p.view_proj,
            inv_view_proj: p.inv_view_proj,
            k: s.k,
            n,
            slabs: s.slabs,
            count: p.particle_count,
            depth_near: p.depth_near,
            depth_far: p.depth_far,
            box_size: p.box_size,
            alpha: p.alpha,
            aspect: p.aspect,
            smoothing: p.smoothing,
            weight_scale: p.weight_scale,
            area_coeff: p.area_coeff,
            ortho: p.ortho as u32,
            r_ref: p.r_ref,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        encoder.clear_buffer(&s.counts, 0, None);

        // The particle buffer is recreated when the count grows, so this bind group is made per frame (cheap).
        let bin_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens bin"),
            layout: &self.bin_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: s.counts.as_entire_binding() },
            ],
        });
        let targets = s.slabs + 1;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lens bin"), timestamp_writes: None });
            pass.set_pipeline(&self.bin);
            pass.set_bind_group(0, &bin_group, &[]);
            pass.dispatch_workgroups(p.particle_count.div_ceil(64).max(1), 1, 1);
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &s.resolve_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(8), n.div_ceil(8), targets);
        }
        s.fft.encode(encoder, false);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lens multiply"), timestamp_writes: None });
            pass.set_pipeline(&self.multiply);
            pass.set_bind_group(0, &s.multiply_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(8), n.div_ceil(8), targets);
        }
        s.fft.encode(encoder, true);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lens write"), timestamp_writes: None });
            pass.set_pipeline(&self.write);
            pass.set_bind_group(0, &s.write_group, &[]);
            pass.dispatch_workgroups(s.k.div_ceil(8), s.k.div_ceil(8), targets);
        }
    }
}

fn write_kernel(queue: &wgpu::Queue, buffer: &wgpu::Buffer, k: u32, aspect: f32) {
    let h: Vec<[f32; 2]> = kernel_hat(k as usize, aspect).iter().map(|c| [c.re, c.im]).collect();
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(&h));
}

#[allow(clippy::too_many_arguments)]
fn build_sized(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resolve_layout: &wgpu::BindGroupLayout,
    multiply_layout: &wgpu::BindGroupLayout,
    write_layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    k: u32,
    slabs: u32,
    aspect: f32,
) -> Sized {
    let n = 2 * k;
    let targets = slabs + 1;
    let buffer = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
    let storage = wgpu::BufferUsages::STORAGE;
    let counts = buffer("lens counts", (slabs * k * k * 4) as u64, storage | wgpu::BufferUsages::COPY_DST);
    let field_bytes = (targets * n * n * 8) as u64;
    let field_a = buffer("lens field a", field_bytes, storage);
    let field_b = buffer("lens field b", field_bytes, storage);
    let kernel = buffer("lens kernel", (n * n * 8) as u64, storage | wgpu::BufferUsages::COPY_DST);
    write_kernel(queue, &kernel, k, aspect);
    let fft = GpuFft::new(device, queue, n, targets, &field_a, &field_b);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lens deflection"),
        size: wgpu::Extent3d { width: k, height: k, depth_or_array_layers: targets },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEFLECT_FORMAT,
        // COPY_SRC lets tests read the deflection back.
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let deflection_view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let background_view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: slabs,
        array_layer_count: Some(1),
        ..Default::default()
    });

    let group = |layout, entries: &[wgpu::BindGroupEntry]| device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("lens fft"), layout, entries });
    let resolve_group = group(
        resolve_layout,
        &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: counts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: field_a.as_entire_binding() },
        ],
    );
    let multiply_group = group(
        multiply_layout,
        &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: field_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: kernel.as_entire_binding() },
        ],
    );
    let write_group = group(
        write_layout,
        &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: field_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&deflection_view) },
        ],
    );
    Sized {
        k,
        slabs,
        aspect,
        counts,
        _field_a: field_a,
        _field_b: field_b,
        kernel,
        fft,
        deflection_view,
        background_view,
        resolve_group,
        multiply_group,
        write_group,
    }
}

// Deflection of a k×k convergence map by FFT, the CPU version of what the GPU does. Returns (αx, αy) per texel.
#[cfg(test)]
pub fn deflection_fft(kappa: &[f32], k: usize, aspect: f32) -> Vec<[f32; 2]> {
    let n = 2 * k;
    let mut field = vec![Complex::new(0.0, 0.0); n * n];
    for y in 0..k {
        for x in 0..k {
            field[y * n + x] = Complex::new(kappa[y * k + x], 0.0);
        }
    }
    fft2d(&mut field, n, false);
    let h = kernel_hat(k, aspect);
    for (f, hk) in field.iter_mut().zip(&h) {
        *f *= hk;
    }
    fft2d(&mut field, n, true);
    let norm = 1.0 / (n * n) as f32;
    (0..k * k)
        .map(|i| {
            let v = field[(i / k) * n + i % k] * norm;
            [v.re, v.im]
        })
        .collect()
}

// One radix-2 Stockham stage, exactly as fft.wgsl does it, on a single line of length n.
// After stages ns = 1, 2, 4, ..., n/2 (ping-ponging src and dst), dst holds the FFT in natural order:
// Stockham reorders as it goes, so no bit-reversal pass is needed.
#[cfg(test)]
pub fn stockham_stage(src: &[Complex<f32>], dst: &mut [Complex<f32>], ns: usize, inverse: bool) {
    let n = src.len();
    let half = n / 2;
    for j in 0..half {
        let k = j % ns;
        let sign = if inverse { 1.0 } else { -1.0 };
        let angle = sign * 2.0 * std::f32::consts::PI * k as f32 / (2 * ns) as f32;
        let a = src[j];
        let b = src[j + half] * Complex::new(angle.cos(), angle.sin());
        let d = (j / ns) * ns * 2 + k;
        dst[d] = a + b;
        dst[d + ns] = a - b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic pseudo-random values in [-1, 1).
    fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    // The direct sum from lens.wgsl (fs_deflect), on the CPU, with a non-square aspect.
    fn deflection_direct(kappa: &[f32], k: usize, aspect: f32) -> Vec<[f32; 2]> {
        let (dx, dy) = (aspect / k as f32, 1.0 / k as f32);
        let eps2 = 0.25 * (dx * dx + dy * dy);
        let scale = dx * dy / std::f32::consts::PI;
        let mut out = vec![[0.0f32; 2]; k * k];
        for yi in 0..k {
            for xi in 0..k {
                let mut a = [0.0f32; 2];
                for yj in 0..k {
                    for xj in 0..k {
                        let d = [(xi as f32 - xj as f32) * dx, (yi as f32 - yj as f32) * dy];
                        let r2 = d[0] * d[0] + d[1] * d[1] + eps2;
                        let kv = kappa[yj * k + xj];
                        a[0] += kv * d[0] / r2;
                        a[1] += kv * d[1] / r2;
                    }
                }
                out[yi * k + xi] = [a[0] * scale, a[1] * scale];
            }
        }
        out
    }

    #[test]
    fn fft_deflection_equals_the_direct_sum() {
        // Same kernel, zero-padded: the FFT result must match the direct sum everywhere, edges included.
        // If padding were missing, mass near one edge would wrap to the other and the edges would disagree.
        let (k, aspect) = (16, 1.6);
        let kappa = noise(k * k, 3);
        let fast = deflection_fft(&kappa, k, aspect);
        let slow = deflection_direct(&kappa, k, aspect);
        let scale = slow.iter().fold(0.0f32, |m, a| m.max(a[0].abs()).max(a[1].abs()));
        for (i, (f, s)) in fast.iter().zip(&slow).enumerate() {
            for c in 0..2 {
                assert!((f[c] - s[c]).abs() < 1e-4 * scale, "texel {i} component {c}: fft {} direct {}", f[c], s[c]);
            }
        }
    }

    // A headless GPU device, or None on machines without one (GPU tests then skip).
    pub fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    // Copies `bytes` from a GPU buffer to the CPU, waiting for the GPU.
    pub fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(src, 0, &readback, 0, bytes);
        queue.submit(Some(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = readback.get_mapped_range(..).expect("mapped").to_vec();
        readback.unmap();
        data
    }

    #[test]
    fn gpu_fft_matches_rustfft() {
        // Runs fft.wgsl on the real GPU: two 32x32 maps forward, compared with rustfft. Then forward and inverse,
        // which should return the input times n² (both directions are unnormalized).
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter; skipping the GPU FFT check");
            return;
        };
        let (n, maps) = (32usize, 2usize);
        let len = n * n * maps;
        let re = noise(len, 21);
        let im = noise(len, 22);
        let input: Vec<[f32; 2]> = re.iter().zip(&im).map(|(&a, &b)| [a, b]).collect();
        let bytes = (len * 8) as u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let make = |label| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage, mapped_at_creation: false });
        let (a, b) = (make("a"), make("b"));
        let fft = GpuFft::new(&device, &queue, n as u32, maps as u32, &a, &b);

        let run = |inverse_too: bool| -> Vec<[f32; 2]> {
            queue.write_buffer(&a, 0, bytemuck::cast_slice(&input));
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            fft.encode(&mut encoder, false);
            if inverse_too {
                fft.encode(&mut encoder, true);
            }
            queue.submit(Some(encoder.finish()));
            bytemuck::cast_slice(&read_back(&device, &queue, &a, bytes)).to_vec()
        };

        let forward = run(false);
        for m in 0..maps {
            let mut want: Vec<Complex<f32>> = input[m * n * n..(m + 1) * n * n].iter().map(|v| Complex::new(v[0], v[1])).collect();
            fft2d(&mut want, n, false);
            let scale = want.iter().fold(0.0f32, |s, v| s.max(v.norm()));
            for (i, w) in want.iter().enumerate() {
                let g = forward[m * n * n + i];
                assert!(
                    (g[0] - w.re).abs() < 1e-4 * scale && (g[1] - w.im).abs() < 1e-4 * scale,
                    "map {m} texel {i}: gpu {g:?}, cpu {w}"
                );
            }
        }

        let round = run(true);
        let n2 = (n * n) as f32;
        for (g, v) in round.iter().zip(&input) {
            assert!((g[0] / n2 - v[0]).abs() < 1e-4 && (g[1] / n2 - v[1]).abs() < 1e-4, "round trip {g:?} vs {v:?}");
        }
    }

    // Reads every layer of the deflection texture: (αx, αy, κ) per texel, layer-major.
    fn read_deflection(device: &wgpu::Device, queue: &wgpu::Queue, lens: &LensFft) -> Vec<[f32; 3]> {
        let texture = lens.deflection_texture();
        let (k, layers) = (texture.width(), texture.depth_or_array_layers());
        let row = k * 8;
        let bytes = (row * k * layers) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deflection copy"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(k) },
            },
            wgpu::Extent3d { width: k, height: k, depth_or_array_layers: layers },
        );
        queue.submit(Some(encoder.finish()));
        let raw = read_back(device, queue, &buffer, bytes);
        let halves: &[u16] = bytemuck::cast_slice(&raw);
        halves
            .chunks(4)
            .map(|h| [0, 1, 2].map(|c| half::f16::from_bits(h[c]).to_f32()))
            .collect()
    }

    #[test]
    fn gpu_lensing_subtracts_the_mean_and_bends_light_away_from_a_clump() {
        // The whole chain on the GPU (bin, resolve, FFT, multiply, write), in the slice view, where depth is world z
        // and every slab counts fully towards the background.
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter; skipping the GPU lensing check");
            return;
        };
        let box_size = 100.0f32;
        let viewport = [800.0, 800.0];
        let camera = crate::camera::Camera::fit_box(box_size, viewport);
        let view_proj = camera.view_proj(viewport);
        let inv_view_proj = crate::camera::inverse(&view_proj).expect("invertible");
        let k = 64u32;

        // A uniform 32³ lattice (mean density), optionally plus a clump of 4000 extra particles at the box centre.
        let side = 32usize;
        let lattice: Vec<[f32; 3]> = (0..side * side * side)
            .map(|i| {
                let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
                [x, y, z].map(|c| (c as f32 + 0.5) * box_size / side as f32)
            })
            .collect();
        let run = |extra: usize| -> Vec<[f32; 3]> {
            let mut positions = lattice.clone();
            positions.extend((0..extra).map(|i| {
                let f = i as f32 / extra as f32;
                // A small ring of points around the centre, about 1 unit across.
                [50.0 + (f * 40.0).cos(), 50.0 + (f * 40.0).sin(), 50.0]
            }));
            // ParticleVertex layout: 16 floats per particle, position in floats 0-2 (previous) and 4-6 (current).
            let mut data = vec![0.0f32; positions.len() * 16];
            for (i, p) in positions.iter().enumerate() {
                data[i * 16..i * 16 + 3].copy_from_slice(p);
                data[i * 16 + 4..i * 16 + 7].copy_from_slice(p);
            }
            let particles = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particles"),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&particles, 0, bytemuck::cast_slice(&data));

            let mut lens = LensFft::new(&device, &queue, 1.0);
            lens.configure(&device, &queue, k, 2, 1.0);
            let world_w = viewport[0] / camera.pixels_per_unit;
            let params = LensFftParams {
                view_proj,
                inv_view_proj,
                particle_count: positions.len() as u32,
                depth_near: 0.0,
                depth_far: box_size,
                box_size,
                alpha: 1.0,
                aspect: 1.0,
                smoothing: 0.03,
                // Mean density from the lattice alone, so the clump is the overdensity.
                weight_scale: box_size * box_size / lattice.len() as f32,
                area_coeff: (world_w / k as f32).powi(2),
                ortho: true,
                r_ref: 1.0,
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            lens.encode(&device, &mut encoder, &queue, &particles, &params);
            queue.submit(Some(encoder.finish()));
            // The background is the last layer.
            let all = read_deflection(&device, &queue, &lens);
            all[(2 * k * k) as usize..].to_vec()
        };
        let at = |map: &[[f32; 3]], u: f32, v: f32| map[((v * k as f32) as u32 * k + (u * k as f32) as u32) as usize];

        // Uniform matter: the mean is subtracted, so the bending is small compared with the clump's below.
        let flat = run(0);
        let flat_max = flat.iter().fold(0.0f32, |m, a| m.max(a[0].abs()).max(a[1].abs()));

        let clump = run(4000);
        let right = at(&clump, 0.65, 0.5);
        let left = at(&clump, 0.35, 0.5);
        let further = at(&clump, 0.8, 0.5);
        // Light bends away from the clump on both sides: +x on the right, −x on the left.
        assert!(right[0] > 0.0 && left[0] < 0.0, "right {right:?}, left {left:?}");
        assert!(flat_max < 0.2 * right[0], "uniform matter bends too much: {flat_max} vs clump {}", right[0]);
        // Roughly 1/r: at twice the distance, about half the deflection.
        let ratio = right[0] / further[0];
        assert!((ratio - 2.0).abs() < 0.4, "1/r ratio {ratio}");
        // The recovered projected mass is positive at the clump and much smaller far from it.
        let core = at(&clump, 0.5, 0.5)[2];
        assert!(core > 0.0 && core > 5.0 * further[2].abs(), "κ at clump {core}, far {}", further[2]);
    }

    #[test]
    fn stockham_matches_rustfft_both_directions() {
        // The GPU shader runs these exact stages. Check them against rustfft before trusting the shader.
        let n = 64;
        let re = noise(n, 5);
        let im = noise(n, 9);
        let input: Vec<Complex<f32>> = re.iter().zip(&im).map(|(&a, &b)| Complex::new(a, b)).collect();
        for inverse in [false, true] {
            let mut a = input.clone();
            let mut b = vec![Complex::new(0.0, 0.0); n];
            let mut ns = 1;
            while ns < n {
                stockham_stage(&a, &mut b, ns, inverse);
                std::mem::swap(&mut a, &mut b);
                ns *= 2;
            }
            let mut want = input.clone();
            let mut planner = rustfft::FftPlanner::<f32>::new();
            let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
            fft.process(&mut want);
            for (g, w) in a.iter().zip(&want) {
                assert!((g - w).norm() < 1e-4, "inverse {inverse}: got {g}, want {w}");
            }
        }
    }
}
