// Brightness meter for auto exposure: the log-average brightness of lit pixels, over the first bloom level.
// That level is a half-size downsample that reads every scene pixel, so no star is skipped, and it has far fewer
// texels to visit than the full scene. One workgroup of 256 threads strides over all texels, then sums in shared memory.

@group(0) @binding(0) var src: texture_2d<f32>;

struct Reading {
    // Sum of ln(brightness) over lit texels.
    sum_log: f32,
    // Number of lit texels.
    lit: f32,
};
@group(0) @binding(1) var<storage, read_write> reading: Reading;

// Texels dimmer than this count as background. The view is black except where particles land.
const LIT_THRESHOLD: f32 = 1e-4;
const THREADS: u32 = 256u;

var<workgroup> sums: array<f32, 256>;
var<workgroup> counts: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_meter(@builtin(local_invocation_index) li: u32) {
    let size = textureDimensions(src);
    let n = size.x * size.y;
    var s = 0.0;
    var c = 0.0;
    for (var i = li; i < n; i = i + THREADS) {
        let rgb = textureLoad(src, vec2<u32>(i % size.x, i / size.x), 0).rgb;
        // Brightest channel, the same quantity the tone map works on.
        let m = max(rgb.r, max(rgb.g, rgb.b));
        if (m > LIT_THRESHOLD) {
            s = s + log(m);
            c = c + 1.0;
        }
    }
    sums[li] = s;
    counts[li] = c;
    workgroupBarrier();

    // Tree reduction: halve the active threads each round until thread 0 holds the totals.
    for (var stride = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            sums[li] = sums[li] + sums[li + stride];
            counts[li] = counts[li] + counts[li + stride];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        reading.sum_log = sums[0];
        reading.lit = counts[0];
    }
}
