// Mip-chain bloom (after Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", 2014),
// and the tone-mapped composite. Every pass draws one fullscreen triangle with the same bind group layout.
//
//   scene ──13-tap──> mip0 ──13-tap──> mip1 ... ──> mipN      (downsample: each level is half the previous)
//   mipN ──tent, add──> mipN-1 ... ──tent, add──> mip0         (upsample: each level gains the blurrier ones below)
//   scene + mip0 · strength / levels ──tone map──> window
//
// Summing every level gives a tight core plus wide soft halos, which a single blur can't.

struct Post {
    // Size of one texel in the texture being sampled (`src`).
    texel: vec2<f32>,
    // Tent-filter spread in the upsample, in texels of `src`.
    radius: f32,
    // How strongly the bloom is added back (composite only).
    strength: f32,
    // Brightness multiplier before the tone map (composite only).
    exposure: f32,
    // 1 / number of bloom levels, so the summed levels don't brighten with window size (composite only).
    level_scale: f32,
    // 1 = hue-preserving tone map, 0 = per-channel, which bleaches bright colours to white (composite only).
    hue_preserving: f32,
    _pad: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> post: Post;
// The finished bloom (mip0). Only the composite reads it. Other passes bind `src` here as a placeholder.
@group(0) @binding(3) var bloom: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VsOut {
    // One oversized triangle covers the whole screen: (-1, 1), (3, 1), (-1, -1) in clip space.
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn tap(uv: vec2<f32>, dx: f32, dy: f32) -> vec3<f32> {
    return textureSample(src, samp, uv + vec2<f32>(dx, dy) * post.texel).rgb;
}

// 13 bilinear taps over a 6x6 source neighbourhood, for a half-size output. Every source texel near the output
// texel is read, so a one-pixel particle always reaches the bloom (the old 4-tap downsample skipped some).
// The taps form five overlapping 4x4-texel boxes, each the average of its four taps: the centre box (j, k, l, m)
// weighted 0.5, and four corner boxes (a b d e, b c e f, d e g h, e f h i) weighted 0.125 each. Weights sum to 1.
@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let a = tap(uv, -2.0, -2.0);
    let b = tap(uv, 0.0, -2.0);
    let c = tap(uv, 2.0, -2.0);
    let d = tap(uv, -2.0, 0.0);
    let e = tap(uv, 0.0, 0.0);
    let f = tap(uv, 2.0, 0.0);
    let g = tap(uv, -2.0, 2.0);
    let h = tap(uv, 0.0, 2.0);
    let i = tap(uv, 2.0, 2.0);
    let j = tap(uv, -1.0, -1.0);
    let k = tap(uv, 1.0, -1.0);
    let l = tap(uv, -1.0, 1.0);
    let m = tap(uv, 1.0, 1.0);
    let sum = e * 0.125
        + (a + c + g + i) * 0.03125
        + (b + d + f + h) * 0.0625
        + (j + k + l + m) * 0.125;
    return vec4<f32>(sum, 1.0);
}

// 3x3 tent filter on the smaller level, written into the next larger level with additive blending.
// `radius` spreads the taps. Up to about 2 texels they still overlap, so there are no gaps.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let r = post.radius;
    var sum = tap(uv, 0.0, 0.0) * 4.0;
    sum += (tap(uv, -r, 0.0) + tap(uv, r, 0.0) + tap(uv, 0.0, -r) + tap(uv, 0.0, r)) * 2.0;
    sum += tap(uv, -r, -r) + tap(uv, r, -r) + tap(uv, -r, r) + tap(uv, r, r);
    return vec4<f32>(sum / 16.0, 1.0);
}

// Hue-preserving tone map. The curve 1 - exp(-x) is applied to the brightest channel, and all three channels are
// scaled by the same factor. Ratios between channels (hue and saturation) are kept, and since the curve stays below 1,
// no channel can clip. Per-channel mapping would bend bright colours towards white and shift their hue.
// Tradeoff: very hot cores keep their colour instead of bleaching to white. Mirrored by `tone_map` in glow.rs.
fn tone_map_hue(c: vec3<f32>) -> vec3<f32> {
    let m = max(c.r, max(c.g, c.b));
    if (m <= 0.0) {
        return vec3<f32>(0.0);
    }
    return c * ((1.0 - exp(-m)) / m);
}

// Per-channel tone map: each channel saturates on its own, so a bright orange (1.0, 0.55, 0.1) heads towards white
// as the green and blue channels catch up with red. Looks like an overexposed photo. Mirrored by `tone_map_white`.
fn tone_map_white(c: vec3<f32>) -> vec3<f32> {
    return 1.0 - exp(-c);
}

// Adds the bloom to the scene, applies exposure, then tone-maps onto the window.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(src, samp, in.uv).rgb;
    // With bloom off, mip0 holds a stale frame. Skip it rather than multiply by 0, which would turn an inf into NaN.
    // `post.strength` is a uniform, so this branch is uniform and textureSample is allowed inside it.
    var glow = vec3<f32>(0.0);
    if (post.strength > 0.0) {
        glow = textureSample(bloom, samp, in.uv).rgb * post.level_scale;
    }
    let c = (scene + glow * post.strength) * post.exposure;
    if (post.hue_preserving > 0.5) {
        return vec4<f32>(tone_map_hue(c), 1.0);
    }
    return vec4<f32>(tone_map_white(c), 1.0);
}
