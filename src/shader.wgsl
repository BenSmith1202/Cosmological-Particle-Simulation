// Particle renderer. Every particle adds its colour into an HDR scene (additive blending), so overlaps brighten and
// draw order doesn't matter. Two footprints:
//   - points: one pixel per particle (vs_point / fs_point), the crispest and cheapest.
//   - splats: an instanced quad with a normalized Gaussian (vs_splat / fs_splat), which moves smoothly between pixels.
// Each particle carries the same total brightness in both modes: 1 pixel at full colour, or a Gaussian that sums to 1.
// Positions and speeds are interpolated between the previous and current sim frame, so motion is smooth even though
// the simulation only delivers a few frames per second.

// Layout must match ViewUniform in viewer.rs (272 bytes).
struct View {
    // Camera projection (2D slice or 3D orbit), applied to world positions.
    view_proj: mat4x4<f32>,
    // Natural log of the range of the coloured quantity (speed or density) that maps to the colour ramp.
    color_min: f32,
    color_max: f32,
    // How many entries of `stops` are in use.
    stop_count: u32,
    // Splat Gaussian sigma, in pixels. Unused in point mode.
    point_sigma: f32,
    // Window size in pixels, to turn a pixel-sized splat into clip-space offsets.
    viewport: vec2<f32>,
    // How far from the previous to the current frame to draw, 0..1.
    alpha: f32,
    // Periodic box side, for interpolating across the boundary.
    box_size: f32,
    // 0 = colour by speed, 1 = colour by density.
    color_by: u32,
    // Redshift-space line of sight: 0 = x, 1 = y, 2 = z.
    rsd_axis: u32,
    // Redshift-space shift multiplier. 0 = off, 1 = physical.
    rsd_strength: f32,
    // Depth fog strength. 0 = off (always 0 in the slice view).
    fog_strength: f32,
    // Depth (clip w) that fog leaves unchanged: the orbit centre. Nearer is brighter, further is dimmer.
    fog_ref: f32,
    // Multi-plane lensing of the particles. 0 = off. Otherwise the exaggeration, as for the background.
    lens_strength: f32,
    // Depth range (clip w) the slabs divide, and how many slabs.
    lens_near: f32,
    lens_far: f32,
    lens_slabs: u32,
    // Window width / height, to turn the deflection's isotropic units back into screen coordinates.
    aspect: f32,
    _pad0: f32,
    _pad1: f32,
    // Colour stops, sorted by position: linear rgb in xyz, position (0..1 along the speed range) in w.
    stops: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> view: View;
// Deflection per depth slab (lens_fft.rs). Layer s bends particles in slab s by the mass in front of them.
@group(0) @binding(1) var lens_deflection: texture_2d_array<f32>;
@group(0) @binding(2) var lens_samp: sampler;

// Piecewise-linear ramp through the user's stops. Below the first stop or above the last, the end colour holds.
fn ramp(t: f32) -> vec3<f32> {
    let n = view.stop_count;
    if (t <= view.stops[0].w) {
        return view.stops[0].xyz;
    }
    for (var i = 1u; i < n; i = i + 1u) {
        let a = view.stops[i - 1u];
        let b = view.stops[i];
        if (t <= b.w) {
            let f = (t - a.w) / max(b.w - a.w, 1e-6);
            return mix(a.xyz, b.xyz, f);
        }
    }
    return view.stops[n - 1u].xyz;
}

// Interpolates a position in the periodic box, the short way round. A particle that crossed the boundary between
// frames (e.g. 99.5 -> 0.5 in a box of 100) moves 1 unit across the edge, instead of streaking 99 units back
// through the box. Mirrored by `lerp_periodic` in viewer.rs tests.
fn lerp_periodic(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    let l = view.box_size;
    // Shortest displacement: wrap the difference into [-L/2, L/2].
    let d = (b - a) - l * round((b - a) / l);
    let p = a + d * t;
    // Back into [0, L).
    return p - l * floor(p / l);
}

// Colour by log of the chosen quantity, so a wide range stays readable.
fn quantity_color(q: f32) -> vec3<f32> {
    let t = clamp((log(max(q, 1e-12)) - view.color_min) / (view.color_max - view.color_min), 0.0, 1.0);
    return ramp(t);
}

// Per-particle inputs, shared by both footprints: previous and current frame of each attribute.
struct ParticleIn {
    @location(0) prev_pos: vec3<f32>,
    @location(1) prev_speed: f32,
    @location(2) pos: vec3<f32>,
    @location(3) speed: f32,
    @location(4) prev_rsd: vec3<f32>,
    @location(5) prev_density: f32,
    @location(6) rsd: vec3<f32>,
    @location(7) density: f32,
};

// Where a particle is drawn this frame, and its colour before fog.
struct Placed {
    world: vec3<f32>,
    color: vec3<f32>,
};

fn place(p: ParticleIn) -> Placed {
    var world = lerp_periodic(p.prev_pos, p.pos, view.alpha);
    if (view.rsd_strength > 0.0) {
        // Keep only the line-of-sight component of the shift, using a mask so no dynamic vector indexing is needed.
        let axis = vec3<f32>(f32(view.rsd_axis == 0u), f32(view.rsd_axis == 1u), f32(view.rsd_axis == 2u));
        let shift = mix(p.prev_rsd, p.rsd, view.alpha);
        world = world + axis * dot(shift, axis) * view.rsd_strength;
        // A large shift can leave the box. Wrap it back, as the survey's periodic box would.
        world = world - view.box_size * floor(world / view.box_size);
    }
    var q = mix(p.prev_speed, p.speed, view.alpha);
    if (view.color_by == 1u) {
        q = mix(p.prev_density, p.density, view.alpha);
    }
    var out: Placed;
    out.world = world;
    out.color = quantity_color(q);
    return out;
}

// Moves a projected particle by gravitational lensing from the slabs in front of it. The lens equation puts the
// image of a source at its true position plus the deflection, evaluated (weak lensing) at that position.
fn lensed(clip: vec4<f32>) -> vec4<f32> {
    if (view.lens_strength <= 0.0 || clip.w <= 0.0) {
        return clip;
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let f = (clip.w - view.lens_near) / (view.lens_far - view.lens_near);
    let slab = i32(clamp(floor(f * f32(view.lens_slabs)), 0.0, f32(view.lens_slabs - 1u)));
    // textureSampleLevel: the only texture sampling allowed in a vertex shader.
    let a = textureSampleLevel(lens_deflection, lens_samp, uv, slab, 0.0).xy * view.lens_strength;
    // Isotropic units to NDC: x was scaled by the aspect ratio, and uv's y points down while NDC's points up.
    // Times w, so the shift survives the perspective divide.
    let offset = vec2<f32>(2.0 * a.x / view.aspect, -2.0 * a.y) * clip.w;
    return vec4<f32>(clip.xy + offset, clip.zw);
}

// Depth fog from the clip-space w, which is the distance along the view direction in the perspective view.
// Particles at the orbit centre are unchanged. Clamped so a camera inside the box doesn't make near ones explode.
fn fog(w: f32) -> f32 {
    if (view.fog_strength <= 0.0) {
        return 1.0;
    }
    return clamp(exp(-view.fog_strength * (w - view.fog_ref) / view.box_size), 0.0, 4.0);
}

// ---- One-pixel points ----

struct PointOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_point(p: ParticleIn) -> PointOut {
    let placed = place(p);
    var out: PointOut;
    out.clip = lensed(view.view_proj * vec4<f32>(placed.world, 1.0));
    out.color = placed.color * fog(out.clip.w);
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 0.0);
}

// ---- Gaussian splats ----

struct SplatOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    // This fragment's offset from the particle centre, in pixels.
    @location(1) offset_px: vec2<f32>,
};

// Two triangles per quad, in unit-square coordinates.
const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_splat(@builtin(vertex_index) vertex: u32, p: ParticleIn) -> SplatOut {
    let placed = place(p);
    let center = lensed(view.view_proj * vec4<f32>(placed.world, 1.0));
    // ±3σ holds 99.7% of the Gaussian, so the quad edge cuts off almost nothing.
    let offset_px = QUAD[vertex] * 3.0 * view.point_sigma;
    // Pixels to clip space: 2/viewport per pixel, times w so the size survives the perspective divide.
    // The splat stays the same size on screen at any distance; only its centre is projected.
    let offset_clip = offset_px * 2.0 / view.viewport * center.w;

    var out: SplatOut;
    out.clip = center + vec4<f32>(offset_clip, 0.0, 0.0);
    out.color = placed.color * fog(center.w);
    out.offset_px = offset_px;
    return out;
}

@fragment
fn fs_splat(in: SplatOut) -> @location(0) vec4<f32> {
    // Normalized 2D Gaussian: integrates to 1 over the plane. Sampled at pixel centres, the sum stays within 0.2%
    // of 1 for σ ≥ 0.6 px, wherever the centre falls inside a pixel. That is what removes the shimmer.
    let s2 = view.point_sigma * view.point_sigma;
    let w = exp(-dot(in.offset_px, in.offset_px) / (2.0 * s2)) / (6.2831853 * s2);
    return vec4<f32>(in.color * w, 0.0);
}
