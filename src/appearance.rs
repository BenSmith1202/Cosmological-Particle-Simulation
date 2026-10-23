// How the points look: the speed colour ramp, the point footprint, bloom, and exposure.
// Stops are placed by fraction of the frame's speed range: 0 is the coldest colour, 1 is the fastest.

use serde::{Deserialize, Serialize};

use crate::real::Real;

// The GPU uniform holds this many stops at most.
pub const MAX_STOPS: usize = 8;
const MIN_STOPS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorStop {
    // Fraction of the speed range, in [0, 1].
    pub position: Real,
    // Linear RGB. The window surface is sRGB, so the GPU encodes this on output.
    // egui's colour picker also edits linear RGB, so picked colours match the render.
    pub color: [Real; 3],
}

// The allowed high-quality lensing size closest to `size` (the FFT needs a power of two).
fn nearest_hq_size(size: u32) -> u32 {
    *crate::lens_fft::LENS_HQ_SIZES
        .iter()
        .min_by_key(|&&s| s.abs_diff(size))
        .expect("at least one size")
}

// Converts one sRGB-encoded channel (what colour pickers and hex codes show) to linear light.
pub fn srgb_to_linear(c: Real) -> Real {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// Splat width (Gaussian sigma, in pixels). 0 means one-pixel points. Anything else is kept in this range:
// below 0.6 px the splat no longer sums to the same brightness at every sub-pixel position, so it would shimmer again.
pub const MIN_SPLAT_SIGMA: Real = 0.6;
pub const MAX_SPLAT_SIGMA: Real = 3.0;

// What the colour ramp maps: peculiar speed, or local density (the classic cosmic-web look).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ColorBy {
    #[default]
    Speed,
    Density,
}

// Gas ray-march steps per pixel. More is smoother and slower.
pub const MIN_GAS_STEPS: u32 = 16;
pub const MAX_GAS_STEPS: u32 = 256;

// `serde(default)` fills any field missing from an older settings.json with its default, instead of failing the load.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    pub stops: Vec<ColorStop>,
    // Point footprint: Gaussian sigma in pixels, or 0 for one-pixel points.
    pub point_size: Real,
    // Bloom on or off. Off still renders in HDR with the same tone map, so the base image doesn't change.
    pub glow: bool,
    // How strongly the bloom is added back.
    pub glow_strength: Real,
    // Tent-filter spread in the bloom upsample, in texels of each level. The mip chain supplies the large scales.
    pub glow_radius: Real,
    // Brightness multiplier before the tone map, used when auto exposure is off.
    pub exposure: Real,
    // Auto exposure: follows the measured brightness of lit pixels, so zooming or structure growth don't change the look.
    pub auto_exposure: bool,
    // Multiplies the auto exposure. 1 is the default key, 2 is one stop brighter.
    pub exposure_bias: Real,
    // Tone map keeps hue (bright colours stay coloured), or works per channel (bright colours clip to white).
    pub hue_preserving: bool,

    pub color_by: ColorBy,
    // Redshift-space view: shift particles along one axis by their velocity, as a redshift survey would place them.
    pub rsd: bool,
    // Line-of-sight axis: 0 = x, 1 = y, 2 = z.
    pub rsd_axis: usize,
    // 1 is the physical shift. Larger exaggerates it.
    pub rsd_strength: Real,
    // Depth fog in the orbit view: near particles brighter, far ones dimmer.
    pub fog: bool,
    pub fog_strength: Real,
    // Volumetric gas: a glow ray-marched through the density grid.
    pub gas: bool,
    pub gas_intensity: Real,
    // Density range that glows, in log10(1 + δ). Below `gas_low` is invisible, and `gas_high` is the brightest.
    pub gas_low: Real,
    pub gas_high: Real,
    pub gas_steps: u32,
    // Gravitational lensing of a background of distant galaxies. Strength exaggerates the deflection, since the
    // physical one would be under a pixel at this scale.
    pub lens: bool,
    pub lens_strength: Real,
    pub lens_galaxies: Real,
    // Tint the background by the convergence (projected mass), to see what is doing the lensing.
    pub lens_show_kappa: bool,
    // Side of the projected-mass and deflection grids. More is more detail, at a cost of size⁴ (see lens.rs).
    pub lens_resolution: u32,
    // High-quality lensing (lens_fft.rs): FFT deflection, mass from the particles, depth slabs. Off by default,
    // since it is the most expensive layer.
    pub lens_hq: bool,
    // Map side in high-quality mode: one of LENS_HQ_SIZES.
    pub lens_hq_resolution: u32,
    pub lens_slabs: u32,
    // Also bend the simulated particles by the mass in front of them (orbit view only).
    pub lens_particles: bool,
}

impl Default for Appearance {
    // The ramp from dark blue (coldest) through purple, magenta, and orange to yellow-white (fastest), evenly spaced.
    // Written in sRGB, as the colours look on screen, and converted to linear for storage.
    fn default() -> Self {
        let srgb = [
            [0.02, 0.03, 0.25],
            [0.40, 0.10, 0.55],
            [0.95, 0.15, 0.75],
            [1.00, 0.55, 0.10],
            [1.00, 0.97, 0.75],
        ];
        let stops = srgb
            .iter()
            .enumerate()
            .map(|(i, &c)| ColorStop {
                position: i as Real / (srgb.len() - 1) as Real,
                color: c.map(srgb_to_linear),
            })
            .collect();
        Self {
            stops,
            point_size: 0.8,
            glow: false,
            glow_strength: 1.0,
            glow_radius: 1.0,
            // A splat spreads one particle's brightness over several pixels (peak 1/(2πσ²) ≈ 0.25 at σ = 0.8),
            // so isolated particles need more exposure than one-pixel points did.
            exposure: 3.0,
            auto_exposure: true,
            exposure_bias: 1.0,
            hue_preserving: true,
            color_by: ColorBy::Speed,
            rsd: false,
            rsd_axis: 1,
            rsd_strength: 1.0,
            fog: false,
            fog_strength: 1.5,
            gas: false,
            gas_intensity: 1.0,
            // δ ≈ 1 (twice the mean) starts to glow, δ ≈ 100 is the brightest: filaments and halos, not voids.
            gas_low: 0.3,
            gas_high: 2.0,
            gas_steps: 64,
            lens: false,
            lens_strength: 2.0,
            lens_galaxies: 1.0,
            lens_show_kappa: false,
            lens_resolution: 64,
            lens_hq: false,
            lens_hq_resolution: 128,
            lens_slabs: 8,
            lens_particles: true,
        }
    }
}

impl Appearance {
    // Keeps the stops sorted and within limits, and the glow values in usable ranges.
    pub fn sanitized(&self) -> Self {
        let mut stops: Vec<ColorStop> = self
            .stops
            .iter()
            .map(|s| ColorStop {
                position: s.position.clamp(0.0, 1.0),
                color: s.color.map(|c| c.clamp(0.0, 1.0)),
            })
            .collect();
        stops.sort_by(|a, b| a.position.total_cmp(&b.position));
        stops.truncate(MAX_STOPS);
        if stops.len() < MIN_STOPS {
            return Self {
                stops: Appearance::default().stops,
                ..self.sanitized_glow()
            };
        }
        Self {
            stops,
            ..self.sanitized_glow()
        }
    }

    fn sanitized_glow(&self) -> Self {
        // Zero (or less) selects one-pixel points. Small non-zero sizes are raised to the minimum splat.
        let point_size = if self.point_size <= 0.0 {
            0.0
        } else {
            self.point_size.clamp(MIN_SPLAT_SIGMA, MAX_SPLAT_SIGMA)
        };
        Self {
            stops: Vec::new(),
            point_size,
            glow: self.glow,
            glow_strength: self.glow_strength.clamp(0.0, 5.0),
            glow_radius: self.glow_radius.clamp(0.5, 2.0),
            exposure: self.exposure.clamp(0.1, 20.0),
            auto_exposure: self.auto_exposure,
            exposure_bias: self.exposure_bias.clamp(0.1, 10.0),
            hue_preserving: self.hue_preserving,
            color_by: self.color_by,
            rsd: self.rsd,
            rsd_axis: self.rsd_axis.min(2),
            rsd_strength: self.rsd_strength.clamp(0.0, 5.0),
            fog: self.fog,
            fog_strength: self.fog_strength.clamp(0.0, 5.0),
            gas: self.gas,
            gas_intensity: self.gas_intensity.clamp(0.0, 20.0),
            gas_low: self.gas_low.clamp(-1.0, 3.5),
            // Keep a usable range above the low end, so the shader never divides by zero.
            gas_high: self.gas_high.clamp(self.gas_low.clamp(-1.0, 3.5) + 0.1, 4.0),
            gas_steps: self.gas_steps.clamp(MIN_GAS_STEPS, MAX_GAS_STEPS),
            lens: self.lens,
            lens_strength: self.lens_strength.clamp(0.0, 50.0),
            lens_galaxies: self.lens_galaxies.clamp(0.0, 10.0),
            lens_show_kappa: self.lens_show_kappa,
            lens_resolution: self.lens_resolution.clamp(crate::lens::MIN_LENS_SIZE, crate::lens::MAX_LENS_SIZE),
            lens_hq: self.lens_hq,
            lens_hq_resolution: nearest_hq_size(self.lens_hq_resolution),
            lens_slabs: self.lens_slabs.clamp(crate::lens_fft::MIN_SLABS, crate::lens_fft::MAX_SLABS),
            lens_particles: self.lens_particles,
        }
    }

    // Stops for the shader: sorted by position, padded to MAX_STOPS by repeating the last one, plus the real count.
    // Each entry is (r, g, b, position).
    pub fn uniform_stops(&self) -> ([[Real; 4]; MAX_STOPS], u32) {
        let mut stops = self.sanitized().stops;
        stops.sort_by(|a, b| a.position.total_cmp(&b.position));
        let count = stops.len();
        let mut out = [[0.0; 4]; MAX_STOPS];
        for i in 0..MAX_STOPS {
            let s = stops[i.min(count - 1)];
            out[i] = [s.color[0], s.color[1], s.color[2], s.position];
        }
        (out, count as u32)
    }
}

// Colour at fraction `t` of the speed range, interpolating between sorted stops. Matches the shader's ramp.
pub fn color_at(stops: &[ColorStop], t: Real) -> [Real; 3] {
    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| a.position.total_cmp(&b.position));
    let Some(first) = sorted.first() else {
        return [0.0; 3];
    };
    if t <= first.position {
        return first.color;
    }
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if t <= b.position {
            let f = (t - a.position) / (b.position - a.position).max(1e-6);
            return [0, 1, 2].map(|k| a.color[k] + (b.color[k] - a.color[k]) * f);
        }
    }
    sorted.last().map_or([0.0; 3], |s| s.color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_to_linear_matches_reference_points() {
        // Ends are fixed, and mid-grey 0.5 in sRGB is about 0.214 in linear light.
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb_to_linear(0.5) - 0.21404).abs() < 1e-4);
        // The linear segment and the power segment meet at the threshold.
        let below = srgb_to_linear(0.04045);
        let above = srgb_to_linear(0.04046);
        assert!((above - below).abs() < 1e-5);
    }

    #[test]
    fn default_stops_are_evenly_spaced_and_sorted() {
        let a = Appearance::default();
        assert_eq!(a.stops.len(), 5);
        assert_eq!(a.stops[0].position, 0.0);
        assert_eq!(a.stops[4].position, 1.0);
        assert!(a.stops.windows(2).all(|w| w[0].position <= w[1].position));
    }

    #[test]
    fn sanitize_sorts_clamps_and_limits_stops() {
        let stops = (0..12)
            .map(|i| ColorStop {
                position: 1.5 - i as Real * 0.2,
                color: [2.0, -1.0, 0.5],
            })
            .collect();
        let a = Appearance { stops, ..Appearance::default() }.sanitized();
        assert_eq!(a.stops.len(), MAX_STOPS);
        assert!(a.stops.windows(2).all(|w| w[0].position <= w[1].position));
        assert!(a.stops.iter().all(|s| (0.0..=1.0).contains(&s.position)));
        assert_eq!(a.stops[0].color, [1.0, 0.0, 0.5]);
    }

    #[test]
    fn too_few_stops_fall_back_to_the_default_ramp() {
        let a = Appearance {
            stops: vec![ColorStop { position: 0.5, color: [1.0, 1.0, 1.0] }],
            ..Appearance::default()
        }
        .sanitized();
        assert_eq!(a.stops, Appearance::default().stops);
    }

    #[test]
    fn color_at_matches_stops_and_interpolates() {
        let stops = vec![
            ColorStop { position: 0.0, color: [0.0, 0.0, 0.0] },
            ColorStop { position: 1.0, color: [1.0, 1.0, 1.0] },
        ];
        assert_eq!(color_at(&stops, 0.0), [0.0, 0.0, 0.0]);
        assert_eq!(color_at(&stops, 1.0), [1.0, 1.0, 1.0]);
        let mid = color_at(&stops, 0.5);
        assert!(mid.iter().all(|c| (c - 0.5).abs() < 1e-6));
    }

    #[test]
    fn point_size_zero_is_pixel_mode_and_others_are_clamped() {
        let size = |s: Real| Appearance { point_size: s, ..Appearance::default() }.sanitized().point_size;
        assert_eq!(size(0.0), 0.0);
        assert_eq!(size(-1.0), 0.0);
        assert_eq!(size(0.1), MIN_SPLAT_SIGMA);
        assert_eq!(size(1.5), 1.5);
        assert_eq!(size(99.0), MAX_SPLAT_SIGMA);
    }

    #[test]
    fn missing_fields_in_saved_json_take_defaults() {
        // An appearance saved before point_size existed should still load, with the default size.
        let old = r#"{"stops":[],"glow":true,"glow_strength":1.0,"glow_radius":1.0,"exposure":2.0}"#;
        let a: Appearance = serde_json::from_str(old).expect("old appearance json");
        assert_eq!(a.point_size, Appearance::default().point_size);
        assert!(a.glow);
    }

    #[test]
    fn hq_lensing_sizes_snap_to_powers_of_two() {
        let size = |r: u32| Appearance { lens_hq_resolution: r, ..Appearance::default() }.sanitized().lens_hq_resolution;
        assert_eq!(size(128), 128);
        assert_eq!(size(200), 256);
        assert_eq!(size(1), 64);
        assert_eq!(size(100_000), 512);
        let slabs = Appearance { lens_slabs: 50, ..Appearance::default() }.sanitized().lens_slabs;
        assert_eq!(slabs, crate::lens_fft::MAX_SLABS);
    }

    #[test]
    fn gas_range_stays_ordered() {
        let a = Appearance { gas_low: 2.0, gas_high: 1.0, ..Appearance::default() }.sanitized();
        assert!(a.gas_high > a.gas_low);
        let a = Appearance { rsd_axis: 9, gas_steps: 1, ..Appearance::default() }.sanitized();
        assert_eq!(a.rsd_axis, 2);
        assert_eq!(a.gas_steps, MIN_GAS_STEPS);
    }

    #[test]
    fn uniform_stops_pad_and_count() {
        let (stops, count) = Appearance::default().uniform_stops();
        assert_eq!(count, 5);
        // Unused slots repeat the last stop, so the shader can read any index safely.
        assert_eq!(stops[7], stops[4]);
        assert_eq!(stops[4][3], 1.0);
    }
}
