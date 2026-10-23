// Auto exposure: picks an exposure from the GPU's brightness measurement, and eases towards it over time.
//
// The measurement (meter.wgsl) is the log-average brightness of lit pixels only. A star field is mostly black, so an
// average over all pixels would be tiny and blow out the stars. Averaging logs (a geometric mean) keeps a few very
// bright cores from dominating. The target puts that typical lit pixel at KEY before the tone map.

// Typical lit-pixel brightness after exposure, before the tone map. The bias slider scales it.
// Calibrated by measurement: at the start of the default run (64k particles, point size 0.8, 900 px window), the meter
// read a geometric mean of about 0.04, because the faint blurred edges of stars count as lit. 0.12 / 0.04 gives
// exposure 3, the manual default, so switching auto exposure on at that point doesn't jump the brightness.
pub const KEY: f32 = 0.12;
// Time constant for easing towards the target, in seconds. Shorter reacts faster, but flickers more.
pub const ADAPT_SECONDS: f32 = 0.5;
const MIN_EXPOSURE: f32 = 0.05;
const MAX_EXPOSURE: f32 = 1000.0;

// One reading from the meter: the sum of log brightness over lit pixels, and how many pixels were lit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub sum_log: f32,
    pub lit: f32,
}

impl Reading {
    // Exposure that puts the geometric mean of lit pixels at `key`. None when nothing is lit (an empty view),
    // so the exposure holds instead of shooting to the maximum.
    pub fn target(&self, key: f32) -> Option<f32> {
        if self.lit < 1.0 {
            return None;
        }
        let mean_log = self.sum_log / self.lit;
        Some((key / mean_log.exp()).clamp(MIN_EXPOSURE, MAX_EXPOSURE))
    }
}

// Eases exposure towards a target. Works in log space, so going from 1 to 2 takes as long as from 4 to 8.
#[derive(Clone, Copy, Debug)]
pub struct Adapter {
    log_exposure: f32,
}

impl Adapter {
    pub fn new(exposure: f32) -> Self {
        Self {
            log_exposure: exposure.clamp(MIN_EXPOSURE, MAX_EXPOSURE).ln(),
        }
    }

    pub fn exposure(&self) -> f32 {
        self.log_exposure.exp()
    }

    // Moves a fraction 1 - e^(-dt/τ) of the way to `target`. This is frame-rate independent: two steps of dt
    // land in the same place as one step of 2·dt.
    pub fn update(&mut self, target: f32, dt: f32) {
        let k = 1.0 - (-dt.max(0.0) / ADAPT_SECONDS).exp();
        let goal = target.clamp(MIN_EXPOSURE, MAX_EXPOSURE).ln();
        self.log_exposure += (goal - self.log_exposure) * k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_puts_the_geometric_mean_at_the_key() {
        // Four lit pixels of brightness 0.1: geometric mean 0.1, so exposure = key / 0.1.
        let r = Reading { sum_log: 4.0 * 0.1f32.ln(), lit: 4.0 };
        let t = r.target(KEY).expect("lit pixels");
        assert!((t - KEY / 0.1).abs() < 1e-4, "target {t}");
    }

    #[test]
    fn empty_view_gives_no_target() {
        assert_eq!(Reading { sum_log: 0.0, lit: 0.0 }.target(KEY), None);
    }

    #[test]
    fn brighter_views_get_less_exposure() {
        let dim = Reading { sum_log: 0.01f32.ln(), lit: 1.0 }.target(KEY).unwrap();
        let bright = Reading { sum_log: 1.0f32.ln(), lit: 1.0 }.target(KEY).unwrap();
        assert!(bright < dim);
    }

    #[test]
    fn adapter_converges_and_is_frame_rate_independent() {
        let mut one = Adapter::new(1.0);
        let mut two = Adapter::new(1.0);
        one.update(8.0, 0.2);
        two.update(8.0, 0.1);
        two.update(8.0, 0.1);
        assert!((one.exposure() - two.exposure()).abs() < 1e-4);

        // 10 s is 20 time constants, which closes all but e^-20 of the log-distance.
        let mut a = Adapter::new(1.0);
        for _ in 0..600 {
            a.update(8.0, 1.0 / 60.0);
        }
        assert!((a.exposure() - 8.0).abs() < 0.01, "exposure {}", a.exposure());
    }

    #[test]
    fn adapter_eases_rather_than_jumps() {
        let mut a = Adapter::new(1.0);
        a.update(100.0, 1.0 / 60.0);
        assert!(a.exposure() > 1.0 && a.exposure() < 2.0, "exposure {}", a.exposure());
    }
}
