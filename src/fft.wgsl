// Radix-2 Stockham FFT on the GPU: one dispatch per stage, over every line of every map at once.
// A 2D transform is log2(n) stages along x (rows), then log2(n) along y (columns), ping-ponging between two buffers.
// Stockham reorders as it goes, so the output is in natural order with no bit-reversal pass. Unnormalized.
// Mirrored on the CPU by `stockham_stage` in lens_fft.rs, which is checked against rustfft.

// Layout must match StageParams in lens_fft.rs (16 bytes). Read at a dynamic offset, one entry per stage.
struct Stage {
    // Transform length (the map is n×n).
    n: u32,
    // Size of the sub-transforms this stage combines: 1, 2, 4, ..., n/2.
    ns: u32,
    // 0 = along x (rows), 1 = along y (columns).
    axis: u32,
    // 1 = inverse (positive exponent).
    inverse: u32,
};

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> stage: Stage;

// x: butterfly within a line (n/2 of them). y: which line. z: which map.
@compute @workgroup_size(64)
fn cs_fft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = stage.n;
    let half = n / 2u;
    let j = gid.x;
    if (j >= half) {
        return;
    }
    // Element t of this line is at base + t·stride: consecutive along x, n apart along y.
    let map_base = gid.z * n * n;
    var base = map_base + gid.y * n;
    var stride = 1u;
    if (stage.axis == 1u) {
        base = map_base + gid.y;
        stride = n;
    }

    let ns = stage.ns;
    let k = j % ns;
    var angle = -6.28318531 * f32(k) / f32(2u * ns);
    if (stage.inverse == 1u) {
        angle = -angle;
    }
    let a = src[base + j * stride];
    let bb = src[base + (j + half) * stride];
    let tw = vec2<f32>(cos(angle), sin(angle));
    // Complex multiply b · e^{i·angle}.
    let b = vec2<f32>(bb.x * tw.x - bb.y * tw.y, bb.x * tw.y + bb.y * tw.x);
    let d = (j / ns) * ns * 2u + k;
    dst[base + d * stride] = a + b;
    dst[base + (d + ns) * stride] = a - b;
}
