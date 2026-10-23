// FFT Poisson solver for the periodic comoving potential (units H0 = 1).
//
// Derivation: physical ∇²Φ = 4πG a² ρ̄ δ, with ρ̄ = ρc0 Ωm / a³.
// Using 4πG ρc0 = (3/2) H0² gives ∇²φ = (3/2) Ωm δ / a.
// In k-space: φ_k = -(3/2) Ωm δ_k / (a k²), and the acceleration is g = -∇φ, so g_k = -i k φ_k.

use num_complex::Complex;

use crate::cosmology::Cosmology;
use crate::fft::{wavenumber, Fft3};
use crate::mesh::Mesh;
use crate::real::Real;

pub struct PoissonSolver {
    // Cells per side. Must match the mesh passed to `acceleration`.
    n: usize,
    // Periodic box side length.
    box_size: Real,
    fft: Fft3,
}

impl PoissonSolver {
    pub fn new(n: usize, box_size: Real) -> Self {
        Self {
            n,
            box_size,
            fft: Fft3::new(n),
        }
    }

    // Acceleration -∇φ on the grid, for density contrast `delta` at scale factor `a`.
    // With r_split > 0, the potential is filtered by exp(-k² r_s²), which leaves the short-range force to the tree.
    pub fn acceleration(&self, delta: &Mesh, cosmo: &Cosmology, a: Real, r_split: Real) -> [Mesh; 3] {
        let n = self.n;
        assert_eq!(delta.n, n, "mesh size must match the solver");

        // Forward FFT of the density contrast.
        let mut delta_k: Vec<Complex<Real>> = delta
            .cells
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect();
        self.fft.forward(&mut delta_k);

        // One k-space buffer per acceleration component.
        let prefactor = 1.5 * cosmo.omega_m / a;
        let mut accel_k = [
            vec![Complex::new(0.0, 0.0); n * n * n],
            vec![Complex::new(0.0, 0.0); n * n * n],
            vec![Complex::new(0.0, 0.0); n * n * n],
        ];
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let idx = (ix * n + iy) * n + iz;
                    let k = [
                        wavenumber(ix, n, self.box_size),
                        wavenumber(iy, n, self.box_size),
                        wavenumber(iz, n, self.box_size),
                    ];
                    let k2: Real = k.iter().map(|q| q * q).sum();
                    // The k = 0 mode is the mean density, which is zero for a density contrast. No force.
                    if k2 == 0.0 {
                        continue;
                    }
                    // φ_k = -prefactor δ_k / k², then g_k = -i k φ_k.
                    let filter = (-k2 * r_split * r_split).exp();
                    let phi = delta_k[idx] * (-prefactor / k2 * filter);
                    for axis in 0..3 {
                        accel_k[axis][idx] = phi * Complex::new(0.0, -k[axis]);
                    }
                }
            }
        }

        // Inverse FFT each component. The inverse is unnormalized, so divide by n³.
        let scale = 1.0 / (n * n * n) as Real;
        let to_mesh = |mut comp: Vec<Complex<Real>>| {
            self.fft.inverse(&mut comp);
            Mesh {
                n,
                box_size: self.box_size,
                cells: comp.iter().map(|c| c.re * scale).collect(),
            }
        };
        let [ax, ay, az] = accel_k;
        [to_mesh(ax), to_mesh(ay), to_mesh(az)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Mesh filled with f(x) at each cell centre, x = (i + 0.5) h along the first axis.
    fn mesh_from_fn(n: usize, box_size: Real, f: impl Fn(Real) -> Real) -> Mesh {
        let h = box_size / n as Real;
        let mut mesh = Mesh::new(n, box_size);
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    mesh.cells[(ix * n + iy) * n + iz] = f((ix as Real + 0.5) * h);
                }
            }
        }
        mesh
    }

    fn max_abs(mesh: &Mesh) -> Real {
        mesh.cells.iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    #[test]
    fn plane_wave_force_matches_analytic() {
        // δ = cos(k0 x) with k0 = 2π (mode m = 1). Analytically:
        // φ = -(c/k0²) cos(k0 x), and g_x = -∇φ = -(c/k0) sin(k0 x), with c = (3/2) Ωm / a.
        let n = 16;
        let box_size = 1.0;
        let a = 0.5;
        let cosmo = Cosmology::lcdm_default();
        let k0 = 2.0 * PI;
        let c = 1.5 * cosmo.omega_m / a;

        let delta = mesh_from_fn(n, box_size, |x| (k0 * x).cos());
        let solver = PoissonSolver::new(n, box_size);
        let [gx, gy, gz] = solver.acceleration(&delta, &cosmo, a, 0.0);

        let expected = mesh_from_fn(n, box_size, |x| -(c / k0) * (k0 * x).sin());
        let err_x = gx
            .cells
            .iter()
            .zip(&expected.cells)
            .fold(0.0, |m: Real, (g, e)| m.max((g - e).abs()));
        let amplitude = c / k0;
        assert!(err_x < 1e-5 * amplitude, "x error {err_x}");
        assert!(max_abs(&gy) < 1e-5 * amplitude, "y should be zero");
        assert!(max_abs(&gz) < 1e-5 * amplitude, "z should be zero");
    }

    #[test]
    fn uniform_density_contrast_gives_no_force() {
        // A constant δ is pure k = 0 mode, which the solver drops.
        let n = 8;
        let cosmo = Cosmology::lcdm_default();
        let delta = mesh_from_fn(n, 1.0, |_| 1.0);
        let solver = PoissonSolver::new(n, 1.0);
        let [gx, gy, gz] = solver.acceleration(&delta, &cosmo, 1.0, 0.0);
        assert!(max_abs(&gx) < 1e-5);
        assert!(max_abs(&gy) < 1e-5);
        assert!(max_abs(&gz) < 1e-5);
    }

    #[test]
    fn force_scales_inversely_with_scale_factor() {
        // g ∝ 1/a, so doubling a halves the acceleration.
        let n = 16;
        let cosmo = Cosmology::lcdm_default();
        let delta = mesh_from_fn(n, 1.0, |x| (2.0 * PI * x).cos());
        let solver = PoissonSolver::new(n, 1.0);
        let [g_early, _, _] = solver.acceleration(&delta, &cosmo, 0.5, 0.0);
        let [g_late, _, _] = solver.acceleration(&delta, &cosmo, 1.0, 0.0);
        let ratio = max_abs(&g_early) / max_abs(&g_late);
        assert!((ratio - 2.0).abs() < 1e-4, "ratio {ratio}");
    }
}
