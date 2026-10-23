// Zel'dovich initial conditions for a periodic box.
//
// Units: length in h⁻¹Mpc, time in 1/H0. So k is in h Mpc⁻¹ and the σ8 radius is 8 directly.
// The field is generated at z = 0 (D = 1) and then scaled back to a_init by D(a_init).

use num_complex::Complex;
use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::cosmology::Cosmology;
use crate::fft::{wavenumber, Fft3};
use crate::real::Real;

// Planck 2018 values, from memory. Verify before relying on them for precision work.
pub const N_S: Real = 0.965;
pub const SIGMA8: Real = 0.811;
// Top-hat radius for σ8, in h⁻¹Mpc.
pub const SIGMA8_RADIUS: Real = 8.0;

// P(k) = amplitude · k^n_s · T²(k), in (h⁻¹Mpc)³ with k in h Mpc⁻¹.
pub struct PowerSpectrum {
    pub amplitude: Real,
    pub n_s: Real,
}

impl PowerSpectrum {
    pub fn power(&self, cosmo: &Cosmology, k: Real) -> Real {
        if k == 0.0 {
            return 0.0;
        }
        let t = cosmo.bbks_transfer(k);
        self.amplitude * k.powf(self.n_s) * t * t
    }

    // σ(R)² = (1/2π²) ∫ dk k² P(k) W²(kR), integrated in ln k with Simpson's rule.
    pub fn sigma(&self, cosmo: &Cosmology, radius: Real) -> Real {
        let steps = 8192;
        let (ln_lo, ln_hi) = ((1e-4 as Real).ln(), (1e3 as Real).ln());
        let h = (ln_hi - ln_lo) / steps as Real;
        let integrand = |ln_k: Real| {
            let k = ln_k.exp();
            let w = tophat(k * radius);
            k.powi(3) * self.power(cosmo, k) * w * w
        };
        let mut sum = integrand(ln_lo) + integrand(ln_hi);
        for i in 1..steps {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += weight * integrand(ln_lo + i as Real * h);
        }
        let integral = sum * h / 3.0;
        (integral / (2.0 * std::f64::consts::PI.powi(2) as Real)).sqrt()
    }

    // Spectrum with the given tilt, scaled so that σ(R) = sigma8 at z = 0.
    // σ scales as sqrt(amplitude), so compute σ for amplitude 1 and rescale.
    pub fn normalized(cosmo: &Cosmology, n_s: Real, sigma8: Real, radius: Real) -> Self {
        let unit = PowerSpectrum { amplitude: 1.0, n_s };
        let sigma_unit = unit.sigma(cosmo, radius);
        PowerSpectrum {
            amplitude: (sigma8 / sigma_unit).powi(2),
            n_s,
        }
    }
}

// Spherical top-hat window in Fourier space, W(x) = 3 (sin x - x cos x) / x³.
// Uses 1 at small x, where the direct formula cancels badly.
fn tophat(x: Real) -> Real {
    if x < 1e-4 {
        return 1.0;
    }
    3.0 * (x.sin() - x * x.cos()) / (x * x * x)
}

// Colored Fourier modes F_k of the z = 0 density field. F_k = N · δ_k, with N = n³ cells.
// Real-space white noise (unit variance per cell) is transformed, then each mode is scaled by sqrt(N P / V).
// That makes E|δ_k|² = P(k) / V, the standard discrete normalization.
pub fn colored_modes(n: usize, box_size: Real, cosmo: &Cosmology, ps: &PowerSpectrum, seed: u64) -> Vec<Complex<Real>> {
    let n3 = n * n * n;
    let volume = box_size * box_size * box_size;
    let fft = Fft3::new(n);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut modes: Vec<Complex<Real>> = (0..n3)
        .map(|_| Complex::new(StandardNormal.sample(&mut rng), 0.0))
        .collect();
    fft.forward(&mut modes);

    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let idx = (ix * n + iy) * n + iz;
                let k = [
                    wavenumber(ix, n, box_size),
                    wavenumber(iy, n, box_size),
                    wavenumber(iz, n, box_size),
                ];
                let k_mag = k.iter().map(|q| q * q).sum::<Real>().sqrt();
                // The mean density contrast is zero, so the k = 0 mode is dropped.
                modes[idx] = if k_mag == 0.0 {
                    Complex::new(0.0, 0.0)
                } else {
                    modes[idx] * (n3 as Real * ps.power(cosmo, k_mag) / volume).sqrt()
                };
            }
        }
    }
    modes
}

pub struct Particles {
    pub positions: Vec<[Real; 3]>,
    // Momentum p = a² dx/dt, the variable the integrator evolves (Phase 5).
    pub momenta: Vec<[Real; 3]>,
}

// Zel'dovich initial conditions on an n³ lattice, starting at scale factor `a_init`.
// Displacement: x = q + D(a) Ψ(q), with ∇·Ψ = -δ. Momentum: p = a² f E D Ψ, where f = dlnD/dlna.
pub fn zeldovich(
    n: usize,
    box_size: Real,
    cosmo: &Cosmology,
    ps: &PowerSpectrum,
    a_init: Real,
    seed: u64,
) -> Particles {
    let n3 = n * n * n;
    let h = box_size / n as Real;
    let fft = Fft3::new(n);
    let modes = colored_modes(n, box_size, cosmo, ps, seed);

    // Ψ_k = i k δ_k / k². In the unnormalized-FFT form (which is N · Ψ_k), that is i k F / k².
    let mut psi: [Vec<Real>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for axis in 0..3 {
        let mut psi_k = vec![Complex::new(0.0, 0.0); n3];
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let idx = (ix * n + iy) * n + iz;
                    let k = [
                        wavenumber(ix, n, box_size),
                        wavenumber(iy, n, box_size),
                        wavenumber(iz, n, box_size),
                    ];
                    let k2: Real = k.iter().map(|q| q * q).sum();
                    if k2 > 0.0 {
                        psi_k[idx] = modes[idx] * Complex::new(0.0, k[axis]) / k2;
                    }
                }
            }
        }
        fft.inverse(&mut psi_k);
        psi[axis] = psi_k.iter().map(|c| c.re / n3 as Real).collect();
    }

    // Scale the z = 0 displacement back to a_init.
    let d = cosmo.growth(a_init);
    let f = cosmo.growth_rate(a_init);
    let e = cosmo.e(a_init);
    let momentum_scale = a_init * a_init * f * e * d;

    let mut positions = Vec::with_capacity(n3);
    let mut momenta = Vec::with_capacity(n3);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let idx = (ix * n + iy) * n + iz;
                let q = [ix as Real * h, iy as Real * h, iz as Real * h];
                let mut x = [0.0; 3];
                let mut p = [0.0; 3];
                for axis in 0..3 {
                    // Wrap into [0, L) periodically.
                    x[axis] = (q[axis] + d * psi[axis][idx]).rem_euclid(box_size);
                    p[axis] = momentum_scale * psi[axis][idx];
                }
                positions.push(x);
                momenta.push(p);
            }
        }
    }
    Particles { positions, momenta }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn bbks_transfer_is_one_at_large_scales() {
        let c = Cosmology::lcdm_default();
        assert!((c.bbks_transfer(1e-6) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn normalized_spectrum_gives_sigma8() {
        // Recomputing σ(8) from the normalized spectrum should return sigma8 back.
        let c = Cosmology::lcdm_default();
        let ps = PowerSpectrum::normalized(&c, N_S, SIGMA8, SIGMA8_RADIUS);
        let s8 = ps.sigma(&c, SIGMA8_RADIUS);
        assert!((s8 - SIGMA8).abs() / SIGMA8 < 1e-6, "σ8 = {s8}");
    }

    #[test]
    fn growth_rate_matches_omega_m_approximation_today() {
        // f ≈ Ωm^0.55 at z = 0 (Linder's approximation, accurate to about 0.5%). Ωm^0.55 ≈ 0.530.
        let c = Cosmology::lcdm_default();
        let f = c.growth_rate(1.0);
        assert!((f - c.omega_m.powf(0.55)).abs() < 0.01, "f = {f}");
    }

    #[test]
    fn measured_power_matches_input() {
        // Bin the measured power of the colored modes, V |δ_k|² with δ_k = F/N, and compare each bin with the average input P.
        // Only bins with many modes are checked. A bin's relative sampling error is about sqrt(2/N_modes), so the tolerance is 3σ.
        let n = 32;
        let box_size = 100.0;
        let c = Cosmology::lcdm_default();
        let ps = PowerSpectrum::normalized(&c, N_S, SIGMA8, SIGMA8_RADIUS);
        let modes = colored_modes(n, box_size, &c, &ps, 42);

        let n3 = (n * n * n) as Real;
        let volume = box_size * box_size * box_size;
        let kf = 2.0 * PI / box_size;
        // Bin by integer shell |m| = round(sqrt(mx² + my² + mz²)), skipping Nyquist-containing modes.
        let mut measured = vec![0.0; n];
        let mut expected = vec![0.0; n];
        let mut count = vec![0usize; n];
        let signed = |i: usize| if i <= n / 2 { i as i64 } else { i as i64 - n as i64 };
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let m = [signed(ix), signed(iy), signed(iz)];
                    if m.iter().any(|&q| q.abs() == (n / 2) as i64) {
                        continue;
                    }
                    let m2: i64 = m.iter().map(|q| q * q).sum();
                    let shell = (m2 as f64).sqrt().round() as usize;
                    if shell == 0 || shell >= n {
                        continue;
                    }
                    let idx = (ix * n + iy) * n + iz;
                    let p_meas = volume * modes[idx].norm_sqr() / (n3 * n3);
                    let k_mode = kf * (m2 as Real).sqrt();
                    measured[shell] += p_meas;
                    expected[shell] += ps.power(&c, k_mode);
                    count[shell] += 1;
                }
            }
        }

        let mut checked = 0;
        for shell in 1..n {
            if count[shell] < 200 {
                continue;
            }
            let n_modes = count[shell] as Real;
            let meas = measured[shell] / n_modes;
            let exp = expected[shell] / n_modes;
            let tol = 3.0 * (2.0 / n_modes).sqrt();
            let rel = (meas - exp).abs() / exp;
            assert!(rel < tol, "shell {shell}: measured {meas:.4e}, expected {exp:.4e}, rel {rel:.3}, tol {tol:.3}");
            checked += 1;
        }
        assert!(checked >= 3, "too few well-sampled bins: {checked}");
    }
}
