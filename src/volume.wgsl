// Volumetric "gas": a glow ray-marched through the density grid (1 + δ, a 3D texture).
// Emission only: every sample adds light, nothing absorbs. That keeps it independent of draw order, so it can be
// added into the HDR scene before the bloom, the meter, and the tone map, with no need to know particle depths.
// Tradeoff: gas never hides particles or other gas behind it. It reads as a glow, not as smoke.
//
// Two passes: fs_march renders at half the window size (the glow is smooth, so this costs little), then fs_add
// upsamples it with bilinear filtering and adds it into the full-size scene.

// Layout must match VolumeUniform in volume.rs (224 bytes).
struct Volume {
    // Inverse of the camera's view-projection: clip space back to world space.
    inv_view_proj: mat4x4<f32>,
    box_size: f32,
    // How far from the previous to the current density grid, 0..1 (the same as the particles).
    alpha: f32,
    intensity: f32,
    steps: u32,
    // Glowing range, in log10(1 + δ).
    low: f32,
    high: f32,
    stop_count: u32,
    _pad: f32,
    // The particles' colour ramp, so gas and particles share a palette.
    stops: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var prev_grid: texture_3d<f32>;
@group(0) @binding(1) var cur_grid: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> vol: Volume;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VsOut {
    // One oversized triangle covers the whole screen.
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn ramp(t: f32) -> vec3<f32> {
    let n = vol.stop_count;
    if (t <= vol.stops[0].w) {
        return vol.stops[0].xyz;
    }
    for (var i = 1u; i < n; i = i + 1u) {
        let a = vol.stops[i - 1u];
        let b = vol.stops[i];
        if (t <= b.w) {
            return mix(a.xyz, b.xyz, (t - a.w) / max(b.w - a.w, 1e-6));
        }
    }
    return vol.stops[n - 1u].xyz;
}

// World-space point for a pixel at clip-space depth `z` (0 = near plane, 1 = far plane).
fn unproject(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let w = vol.inv_view_proj * ndc;
    return w.xyz / w.w;
}

// Interleaved gradient noise (Jimenez 2014): a cheap per-pixel value in [0, 1). Offsetting each ray's first sample
// by it turns the banding a fixed step pattern would leave into fine grain, which the upsample then smooths.
fn pixel_noise(px: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(px, vec2<f32>(0.06711056, 0.00583715))));
}

// A pixel's ray through the box [0, L]³. Points are near + dir·t, and the box spans t in [tmin, tmax].
struct Ray {
    near: vec3<f32>,
    dir: vec3<f32>,
    tmin: f32,
    tmax: f32,
};

fn box_ray(uv: vec2<f32>) -> Ray {
    let near = unproject(uv, 0.0);
    let dir = unproject(uv, 1.0) - near;
    let l = vol.box_size;
    // Slab method. t runs from 0 at the near plane to 1 at the far plane. Zero direction components are nudged
    // so the division stays finite.
    let safe = select(dir, vec3<f32>(1e-8), abs(dir) < vec3<f32>(1e-8));
    let t0 = (vec3<f32>(0.0) - near) / safe;
    let t1 = (vec3<f32>(l) - near) / safe;
    var r: Ray;
    r.near = near;
    r.dir = dir;
    r.tmin = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
    r.tmax = min(min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)), 1.0);
    return r;
}

// Density 1 + δ at a world point, blended between the previous and current grid.
fn density_at(p: vec3<f32>) -> f32 {
    // The texture's x, y, z are the world's x, y, z (frame.rs orders the grid that way), each scaled to [0, 1].
    let uvw = p / vol.box_size;
    // textureSampleLevel, not textureSample: it is allowed inside a loop with varying bounds.
    return mix(
        textureSampleLevel(prev_grid, samp, uvw, 0.0).r,
        textureSampleLevel(cur_grid, samp, uvw, 0.0).r,
        vol.alpha,
    );
}

@fragment
fn fs_march(in: VsOut) -> @location(0) vec4<f32> {
    let r = box_ray(in.uv);
    if (r.tmax <= r.tmin) {
        return vec4<f32>(0.0);
    }
    let n = vol.steps;
    let dt = (r.tmax - r.tmin) / f32(n);
    let jitter = pixel_noise(in.pos.xy);
    let range = vol.high - vol.low;
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < n; i = i + 1u) {
        let rho = density_at(r.near + r.dir * (r.tmin + (f32(i) + jitter) * dt));
        let u = clamp((log(max(rho, 1e-6)) / log(10.0) - vol.low) / range, 0.0, 1.0);
        // u² fades the glow in smoothly from the threshold, instead of starting at a hard edge.
        sum = sum + ramp(u) * (u * u);
    }
    // Each sample stands for a segment of length |dir|·dt. Measured in box lengths, the glow doesn't depend on
    // the step count (quality only changes the noise) or on the box's units.
    let segment = length(r.dir) * dt / vol.box_size;
    return vec4<f32>(sum * segment * vol.intensity, 0.0);
}

// Lensing convergence (up to a constant): projected overdensity along the ray, ∫ δ dl in box lengths.
// Positive through clusters and filaments, negative through voids (which slightly demagnify what's behind them).
// Rendered on a small grid (see lens.rs), since the deflection built from it is smooth.
@fragment
fn fs_kappa(in: VsOut) -> @location(0) vec4<f32> {
    let r = box_ray(in.uv);
    if (r.tmax <= r.tmin) {
        return vec4<f32>(0.0);
    }
    let n = vol.steps;
    let dt = (r.tmax - r.tmin) / f32(n);
    var sum = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        sum = sum + density_at(r.near + r.dir * (r.tmin + (f32(i) + 0.5) * dt)) - 1.0;
    }
    return vec4<f32>(sum * length(r.dir) * dt / vol.box_size, 0.0, 0.0, 0.0);
}

// The half-size glow, sampled with bilinear filtering and added into the full-size scene (additive blending).
// Bindings 4 and 5: one module can't declare two resources at the same group and binding, and this pass uses its own
// bind group layout, which only has these two entries.
@group(0) @binding(4) var glow_tex: texture_2d<f32>;
@group(0) @binding(5) var glow_samp: sampler;

@fragment
fn fs_add(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(glow_tex, glow_samp, in.uv).rgb, 0.0);
}
