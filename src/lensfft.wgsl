// Multi-plane lensing passes (with fft.wgsl in between). Per frame:
//   cs_bin      each particle adds its mass into its depth slab's projected-mass map (fixed-point atomics)
//   cs_resolve  per target, the lensing convergence: slabs in front of it, weighted by distance, mean subtracted.
//               Targets 1..S-1 are the slabs (slab 0 has nothing in front), target S is the background at infinity.
//   (forward FFT of every target)
//   cs_multiply times the kernel FFT (deflection of a point mass), and a Gaussian that smooths at particle spacing
//   (inverse FFT: αx in the real part, αy in the imaginary part)
//   cs_write    deflection into a texture array, one layer per target, plus κ = ½∇·α for the projected-mass tint
// See lens_fft.rs for the method and its tests.

// Layout must match LensFftUniform in lens_fft.rs (192 bytes).
struct Params {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // Map side, FFT side (2k, zero-padded), slab count, particle count.
    k: u32,
    n: u32,
    slabs: u32,
    count: u32,
    // Depth range of the box from the camera (clip w in perspective, world z in the slice view).
    depth_near: f32,
    depth_far: f32,
    box_size: f32,
    // Interpolation between sim frames, so the mass sits where the particles are drawn.
    alpha: f32,
    aspect: f32,
    // Gaussian smoothing length, in the same isotropic screen units as the deflection.
    smoothing: f32,
    // L² / N: one particle's mass per unit area, in convergence units, before dividing by the texel area.
    weight_scale: f32,
    // Texel area at depth 1 (perspective) or everywhere (slice view), in world units².
    area_coeff: f32,
    // 1 = slice view: no distances, so no slab weighting and no particle lensing.
    ortho: u32,
    // Typical lens distance, so the background's weights average to about 1.
    r_ref: f32,
    _pad0: f32,
    _pad1: f32,
};

// Same layout as ParticleVertex in viewer.rs: storage layout puts each f32 right after its vec3, 64 bytes in all.
struct Particle {
    prev_pos: vec3<f32>,
    prev_speed: f32,
    pos: vec3<f32>,
    speed: f32,
    prev_rsd: vec3<f32>,
    prev_density: f32,
    rsd: vec3<f32>,
    density: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
// Per slab, per texel: sum of particle weights in fixed point (blending f32 render targets is an optional feature,
// and 16-bit floats lose small additions once a texel's sum is large).
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
// (slabs + 1) complex N×N maps: convergence in, deflection out.
@group(0) @binding(4) var<storage, read_write> field: array<vec2<f32>>;
// FFT of the deflection kernel, N×N.
@group(0) @binding(5) var<storage, read> kernel: array<vec2<f32>>;
@group(0) @binding(6) var deflection: texture_storage_2d_array<rgba16float, write>;

// Fixed-point scale for the counts: 2^18 keeps 4e-6 of resolution and room for sums up to 16384.
const FIXED: f32 = 262144.0;
// A particle almost touching the camera covers a tiny texel area and would get an enormous weight.
const MAX_WEIGHT: f32 = 1000.0;
const TAU: f32 = 6.28318531;

fn lerp_periodic(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    // Same as shader.wgsl: the short way across the periodic boundary, then back into [0, L).
    let l = p.box_size;
    let d = (b - a) - l * round((b - a) / l);
    let q = a + d * t;
    return q - l * floor(q / l);
}

// Depth used to pick slabs: clip w (distance along the view direction) in perspective, world z in the slice view.
fn depth_of(world: vec3<f32>) -> f32 {
    if (p.ortho == 1u) {
        return world.z;
    }
    return (p.view_proj * vec4<f32>(world, 1.0)).w;
}

fn slab_of(depth: f32) -> u32 {
    let f = (depth - p.depth_near) / (p.depth_far - p.depth_near);
    return u32(clamp(floor(f * f32(p.slabs)), 0.0, f32(p.slabs - 1u)));
}

@compute @workgroup_size(64)
fn cs_bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    let q = particles[i];
    // Real-space position: lensing follows the real mass, even with the redshift-space view on.
    let world = lerp_periodic(q.prev_pos, q.pos, p.alpha);
    let clip = p.view_proj * vec4<f32>(world, 1.0);
    if (clip.w <= 1e-6) {
        return;
    }
    let ndc = clip.xy / clip.w;
    if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0) {
        return;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let tx = min(u32(uv.x * f32(p.k)), p.k - 1u);
    let ty = min(u32(uv.y * f32(p.k)), p.k - 1u);
    let s = slab_of(depth_of(world));
    // A texel at depth w covers area_coeff·w² of the plane through the particle (perspective), or area_coeff (slice).
    var area = p.area_coeff;
    if (p.ortho == 0u) {
        area = area * clip.w * clip.w;
    }
    let weight = min(p.weight_scale / max(area, 1e-12), MAX_WEIGHT);
    atomicAdd(&counts[(s * p.k + ty) * p.k + tx], u32(weight * FIXED + 0.5));
}

fn unproject(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let w = p.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    return w.xyz / w.w;
}

@compute @workgroup_size(8, 8, 1)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = p.n;
    let k = p.k;
    let x = gid.x;
    let y = gid.y;
    let t = gid.z;
    if (x >= n || y >= n || t > p.slabs) {
        return;
    }
    let out = (t * n + y) * n + x;
    // The padding stays zero. Slab 0 has nothing in front of it. In the slice view only the background is lensed.
    if (x >= k || y >= k || t == 0u || (p.ortho == 1u && t < p.slabs)) {
        field[out] = vec2<f32>(0.0);
        return;
    }

    // This texel's ray through the box (slab method), with t running 0..1 from the near to the far plane.
    let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(k);
    let near = unproject(uv, 0.0);
    let dir = unproject(uv, 1.0) - near;
    let l = p.box_size;
    let safe = select(dir, vec3<f32>(1e-8), abs(dir) < vec3<f32>(1e-8));
    let t0 = (vec3<f32>(0.0) - near) / safe;
    let t1 = (vec3<f32>(l) - near) / safe;
    let box_in = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
    let box_out = min(min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)), 1.0);
    // Depth is linear along the ray, so each slab is a t-interval.
    let d0 = depth_of(near);
    let dd = depth_of(near + dir) - d0;
    let ray_len = length(dir);

    let span = p.depth_far - p.depth_near;
    let rs = p.depth_near + span * (f32(t) + 0.5) / f32(p.slabs);
    var kappa = 0.0;
    for (var li = 0u; li < p.slabs; li = li + 1u) {
        // A slab target is lensed only by slabs in front of it. The background (t = slabs) sees all of them.
        if (t < p.slabs && li >= t) {
            break;
        }
        let s0 = p.depth_near + span * f32(li) / f32(p.slabs);
        let s1 = p.depth_near + span * f32(li + 1u) / f32(p.slabs);
        var ta = (s0 - d0) / dd;
        var tb = (s1 - d0) / dd;
        if (dd < 0.0) {
            let tmp = ta;
            ta = tb;
            tb = tmp;
        }
        // Length of ray inside both the box and this slab: the mean density's share of this texel's column.
        let len = max(min(tb, box_out) - max(ta, box_in), 0.0) * ray_len;
        let mass = f32(atomicLoad(&counts[(li * k + y) * k + x])) / FIXED;
        // Overdensity of the column, in box lengths: projected mass minus what the mean density would give.
        let kl = mass - len / l;

        // Lensing efficiency. A lens at distance r_l bends light from a source at r_s by r_l (r_s - r_l) / r_s, which
        // is zero for a lens next to the source and largest halfway. For a source at infinity it is r_l.
        // Divided by r_ref, so the background's weights average to about 1 and its strength matches the direct mode.
        let rl = 0.5 * (s0 + s1);
        var w = 1.0;
        if (p.ortho == 0u) {
            if (t == p.slabs) {
                w = rl / p.r_ref;
            } else {
                w = rl * (rs - rl) / (rs * p.r_ref);
            }
        }
        kappa = kappa + w * kl;
    }
    field[out] = vec2<f32>(kappa, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = p.n;
    let x = gid.x;
    let y = gid.y;
    if (x >= n || y >= n || gid.z > p.slabs) {
        return;
    }
    let idx = (gid.z * n + y) * n + x;
    // Signed frequency index, then angular wavenumber for texels aspect/k wide and 1/k tall.
    let fx = select(f32(x), f32(x) - f32(n), x >= n / 2u);
    let fy = select(f32(y), f32(y) - f32(n), y >= n / 2u);
    let kx = TAU * fx / (f32(n) * p.aspect / f32(p.k));
    let ky = TAU * fy / (f32(n) / f32(p.k));
    // Gaussian smoothing at particle spacing. Each particle is a point mass, and with the deflection exaggerated
    // hundreds of times, each would lens on its own. The simulation has no information below its particle spacing.
    let damp = exp(-0.5 * (kx * kx + ky * ky) * p.smoothing * p.smoothing);
    let h = kernel[y * n + x];
    let v = field[idx];
    field[idx] = vec2<f32>(v.x * h.x - v.y * h.y, v.x * h.y + v.y * h.x) * damp;
}

// Deflection at texel (x, y) of target t, normalized for the unnormalized inverse FFT.
fn alpha_at(t: u32, x: u32, y: u32) -> vec2<f32> {
    let n = p.n;
    return field[(t * n + y) * n + x] / f32(n * n);
}

@compute @workgroup_size(8, 8, 1)
fn cs_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = p.k;
    let x = gid.x;
    let y = gid.y;
    let t = gid.z;
    if (x >= k || y >= k || t > p.slabs) {
        return;
    }
    let a = alpha_at(t, x, y);
    // Since α = ∇ψ and ∇²ψ = 2κ, the convergence is ½∇·α. Recovered here (central differences), it is exactly the
    // smoothed mass doing the bending, so the "show projected mass" tint matches the lensing.
    let xl = select(x - 1u, x, x == 0u);
    let xr = min(x + 1u, k - 1u);
    let yl = select(y - 1u, y, y == 0u);
    let yr = min(y + 1u, k - 1u);
    let dx = p.aspect / f32(k);
    let dy = 1.0 / f32(k);
    let div = (alpha_at(t, xr, y).x - alpha_at(t, xl, y).x) / (f32(xr - xl) * dx)
        + (alpha_at(t, x, yr).y - alpha_at(t, x, yl).y) / (f32(yr - yl) * dy);
    textureStore(deflection, vec2<u32>(x, y), t, vec4<f32>(a, 0.5 * div, 0.0));
}
