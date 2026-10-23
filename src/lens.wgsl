// Gravitational lensing of a background of distant galaxies, by the matter in the box (thin-lens, screen space).
//
// fs_deflect: deflection α on a small grid from the convergence κ (projected overdensity, from volume.wgsl):
//   α(θ) = (1/π) Σ κ(θ') (θ − θ') / |θ − θ'|² ΔA
// This is the 2D thin-lens formula. The convergence's real amplitude would give deflections under a pixel at this
// scale, so lens.strength exaggerates it. The pattern (where arcs form, which way they bend) is the physical one.
//
// fs_background: the lens equation β = θ − α. The background seen at pixel θ is the part of the sky at β, so galaxies
// near massive structures stretch into arcs. It fills the HDR scene before the particles are added on top.

// Layout must match LensUniform in lens.rs (96 bytes).
struct Lens {
    inv_view_proj: mat4x4<f32>,
    // Window width / height, so distances are measured the same way in x and y.
    aspect: f32,
    strength: f32,
    galaxy_brightness: f32,
    // 1 = tint the background by convergence (warm where mass is, blue in voids).
    show_kappa: f32,
    // 1 = slice view (orthographic): the background is a plane, not a sky.
    ortho: u32,
    box_size: f32,
    // Convergence and deflection grid side, in texels.
    kappa_size: f32,
    _pad: f32,
};

@group(0) @binding(0) var kappa: texture_2d<f32>;
@group(0) @binding(1) var deflection: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> lens: Lens;

// Background galaxies per unit of cube-face coordinate (each face spans -1..1, so 160 cells across 90°).
const SKY_CELLS: f32 = 80.0;
// Background galaxies across the box, in the slice view.
const SLICE_CELLS: f32 = 120.0;
const PI: f32 = 3.14159265;

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

// Screen position with x scaled by the aspect ratio, so the deflection formula's distances are isotropic.
fn iso(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((uv.x - 0.5) * lens.aspect, uv.y - 0.5);
}

@fragment
fn fs_deflect(in: VsOut) -> @location(0) vec4<f32> {
    let n = u32(lens.kappa_size);
    let here = iso(in.uv);
    let cell = vec2<f32>(lens.aspect, 1.0) / f32(n);
    // Softening of half a cell, so each texel's own mass (at zero distance) stays finite.
    let eps2 = 0.25 * dot(cell, cell);
    var a = vec2<f32>(0.0);
    for (var j = 0u; j < n; j = j + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            let k = textureLoad(kappa, vec2<u32>(i, j), 0).r;
            let d = here - iso((vec2<f32>(f32(i), f32(j)) + 0.5) / f32(n));
            a = a + k * d / (dot(d, d) + eps2);
        }
    }
    // The third channel carries this texel's convergence, so the background reads deflection and projected mass from
    // one texture. The high-quality (FFT) path fills the same channel, so the background works with either.
    return vec4<f32>(a * (cell.x * cell.y) / PI, textureLoad(kappa, vec2<u32>(in.pos.xy), 0).r, 0.0);
}

// PCG hash (O'Neill): a well-mixed 32-bit value from any 32-bit input. Enough to scatter galaxies with no pattern.
fn hash(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Next random number in [0, 1) from a running seed.
fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = hash(*seed);
    return f32(*seed) / 4294967296.0;
}

// Faint distant galaxies on a 2D grid of cells, `p` in cell units. Each cell may hold one small elliptical galaxy with
// its own position, size, brightness (mostly faint, a few bright), shape, angle, and colour. Neighbouring cells are
// checked too, so a galaxy near a cell edge isn't cut off. `face` keeps cube faces from repeating each other.
fn galaxies(p: vec2<f32>, face: u32) -> vec3<f32> {
    let base = floor(p);
    var sum = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let c = base + vec2<f32>(f32(dx), f32(dy));
            var seed = hash(bitcast<u32>(i32(c.x)) ^ hash(bitcast<u32>(i32(c.y)) ^ hash(face)));
            // 60% of cells hold a galaxy.
            if (rand(&seed) > 0.6) {
                continue;
            }
            let center = c + vec2<f32>(rand(&seed), rand(&seed));
            let size = mix(0.06, 0.22, rand(&seed) * rand(&seed));
            let bright = pow(rand(&seed), 3.0);
            let elong = 1.0 - 0.6 * rand(&seed);
            let angle = rand(&seed) * PI;
            let color = mix(vec3<f32>(1.0, 0.82, 0.6), vec3<f32>(0.6, 0.75, 1.0), rand(&seed));
            let d = p - center;
            let cs = vec2<f32>(cos(angle), sin(angle));
            let q = vec2<f32>(dot(d, cs), dot(d, vec2<f32>(-cs.y, cs.x))) / vec2<f32>(size, size * elong);
            sum = sum + color * bright * exp(-2.0 * dot(q, q));
        }
    }
    return sum;
}

// The sky in a direction: the direction picks a cube face, and its position on that face picks the galaxy cells.
// This keeps galaxies evenly sized everywhere, unlike latitude/longitude cells that squeeze at the poles.
fn sky(dir: vec3<f32>) -> vec3<f32> {
    let a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        return galaxies(dir.yz / a.x * SKY_CELLS, select(1u, 0u, dir.x > 0.0));
    }
    if (a.y >= a.z) {
        return galaxies(dir.xz / a.y * SKY_CELLS, select(3u, 2u, dir.y > 0.0));
    }
    return galaxies(dir.xy / a.z * SKY_CELLS, select(5u, 4u, dir.z > 0.0));
}

fn unproject(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let w = lens.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    return w.xyz / w.w;
}

@fragment
fn fs_background(in: VsOut) -> @location(0) vec4<f32> {
    // Deflection in isotropic units, back to uv: x was scaled by the aspect ratio, so undo it.
    let lensed = textureSample(deflection, samp, in.uv);
    let alpha = lensed.xy * lens.strength;
    let src = in.uv - vec2<f32>(alpha.x / lens.aspect, alpha.y);

    var color: vec3<f32>;
    if (lens.ortho == 1u) {
        // Slice view: a flat background plane under the box, in world x-y.
        let w = unproject(src, 0.5);
        color = galaxies(w.xy / lens.box_size * SLICE_CELLS, 6u);
    } else {
        // Orbit view: a sky at infinity, so it turns with the camera but doesn't move as the camera moves.
        color = sky(normalize(unproject(src, 1.0) - unproject(src, 0.0)));
    }
    color = color * lens.galaxy_brightness;

    if (lens.show_kappa > 0.5) {
        let k = lensed.z * lens.strength;
        color = color + select(vec3<f32>(0.05, 0.1, 0.25) * -k, vec3<f32>(0.3, 0.15, 0.05) * k, k > 0.0) * 0.5;
    }
    return vec4<f32>(color, 0.0);
}
