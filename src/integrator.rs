// Kick-drift-kick leapfrog in the scale factor, for comoving particles in a periodic box.
//
// State: comoving positions x and momentum p = a² dx/dt. Units: H0 = 1.
// Equations of motion, from ẍ + 2Hẋ = -(1/a²)∇φ:
//   dp/dt = g, with g = -∇φ from the Poisson solver,
//   dx = p da / (a³ E)   (drift),
//   dp = g da / (a E)    (kick).
// Each step is split at the time midpoint a_m, so the first and second half-kicks cover equal time.

use rayon::prelude::*;

use crate::cosmology::Cosmology;
use crate::mesh::Mesh;
use crate::poisson::PoissonSolver;
use crate::real::Real;
use crate::tree::{self, ForceSplit};

// Simpson intervals for the kick and drift integrals within a step.
const STEP_INTEGRAL_STEPS: usize = 64;
// Bisection iterations to find the time-midpoint a_m.
const MIDPOINT_ITERATIONS: usize = 60;

#[cfg(target_arch = "wasm32")]
struct PendingStep {
    a1: Real,
    k2: Real,
    acc: Vec<[Real; 3]>,
    delta: Mesh,
    tree: tree::BuiltTree,
    next_i: usize,
    stack: Vec<u32>,
}

pub struct Simulation {
    cosmo: Cosmology,
    box_size: Real,
    mesh_n: usize,
    solver: PoissonSolver,
    pub positions: Vec<[Real; 3]>,
    pub momenta: Vec<[Real; 3]>,
    pub a: Real,
    // Acceleration at the current positions. Reused as the first half-kick of the next step.
    accel: Vec<[Real; 3]>,
    // TreePM split. None means pure PM.
    split: Option<ForceSplit>,
    // Density contrast δ on the mesh at the current positions, from the last force calculation.
    // Kept so the viewer's density colouring and gas don't need a second deposit.
    delta: Mesh,
    #[cfg(target_arch = "wasm32")]
    pending: Option<PendingStep>,
}

impl Simulation {
    pub fn new(
        cosmo: Cosmology,
        box_size: Real,
        mesh_n: usize,
        positions: Vec<[Real; 3]>,
        momenta: Vec<[Real; 3]>,
        a: Real,
        split: Option<ForceSplit>,
    ) -> Self {
        let mut sim = Self {
            cosmo,
            box_size,
            mesh_n,
            solver: PoissonSolver::new(mesh_n, box_size),
            positions,
            momenta,
            a,
            accel: Vec::new(),
            split,
            delta: Mesh::new(mesh_n, box_size),
            #[cfg(target_arch = "wasm32")]
            pending: None,
        };
        (sim.accel, sim.delta) = sim.accelerations(sim.a);
        sim
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_wasm(
        cosmo: Cosmology,
        box_size: Real,
        mesh_n: usize,
        positions: Vec<[Real; 3]>,
        momenta: Vec<[Real; 3]>,
        a: Real,
        split: Option<ForceSplit>,
    ) -> Self {
        let mut sim = Self {
            cosmo,
            box_size,
            mesh_n,
            solver: PoissonSolver::new(mesh_n, box_size),
            positions,
            momenta,
            a,
            accel: Vec::new(),
            split,
            delta: Mesh::new(mesh_n, box_size),
            pending: None,
        };
        let delta = sim.density_contrast();
        let r_split = sim.split.map_or(0.0, |s| s.r_split);
        let [gx, gy, gz] = sim.solver.acceleration(&delta, &sim.cosmo, a, r_split);
        sim.accel = sim
            .positions
            .iter()
            .map(|&x| [gx.interpolate(x), gy.interpolate(x), gz.interpolate(x)])
            .collect();
        sim.delta = delta;
        sim
    }

    // Density contrast δ = cell mass / mean cell mass - 1, from CIC deposit of unit-mass particles.
    pub fn density_contrast(&self) -> Mesh {
        let mut mesh = Mesh::new(self.mesh_n, self.box_size);
        mesh.deposit(&self.positions, 1.0);
        let cells = (self.mesh_n * self.mesh_n * self.mesh_n) as Real;
        let mean = self.positions.len() as Real / cells;
        for c in mesh.cells.iter_mut() {
            *c = *c / mean - 1.0;
        }
        mesh
    }

    // Density contrast at the current positions (as of the last step).
    pub fn delta(&self) -> &Mesh {
        &self.delta
    }

    // E(a) = H(a)/H0 at the current scale factor.
    pub fn expansion_rate(&self) -> Real {
        self.cosmo.e(self.a)
    }

    // Gravitational acceleration at each particle: PM (filtered if TreePM is on), plus the tree's short-range force.
    // Also returns the density contrast mesh the PM force was solved from.
    fn accelerations(&self, a: Real) -> (Vec<[Real; 3]>, Mesh) {
        let delta = self.density_contrast();
        let r_split = self.split.map_or(0.0, |s| s.r_split);
        let [gx, gy, gz] = self.solver.acceleration(&delta, &self.cosmo, a, r_split);
        let mut acc: Vec<[Real; 3]> = self
            .positions
            .par_iter()
            .map(|&x| [gx.interpolate(x), gy.interpolate(x), gz.interpolate(x)])
            .collect();
        if let Some(split) = &self.split {
            // Newtonian coupling for unit particles in the mean density: K = (3/2) Ωm L³ / (4π a N).
            let n = self.positions.len() as Real;
            let k = 1.5 * self.cosmo.omega_m * self.box_size.powi(3) / (4.0 * std::f32::consts::PI * a * n);
            let short = tree::accelerations(&self.positions, self.box_size, split, k);
            for (g, s) in acc.iter_mut().zip(short) {
                for c in 0..3 {
                    g[c] += s[c];
                }
            }
        }
        (acc, delta)
    }

    // Advance from the current a to a_end in `steps` steps, evenly spaced in ln a.
    #[cfg(test)]
    pub fn run(&mut self, a_end: Real, steps: usize) {
        let a_start = self.a;
        for i in 0..steps {
            let a1 = a_start * (a_end / a_start).powf((i + 1) as Real / steps as Real);
            self.step_to(a1);
        }
    }

    // One KDK step from the current a to a1, starting from the cached acceleration.
    pub fn step_to(&mut self, a1: Real) {
        let a0 = self.a;
        let am = self.midpoint(a0, a1);

        let k1 = self.kick_factor(a0, am);
        kick(&mut self.momenta, &self.accel, k1);
        let d = self.drift_factor(a0, a1);
        drift(&mut self.positions, &self.momenta, d, self.box_size);
        (self.accel, self.delta) = self.accelerations(a1);
        let k2 = self.kick_factor(am, a1);
        kick(&mut self.momenta, &self.accel, k2);
        self.a = a1;
    }

    // Advances a KDK step in time-sliced chunks on wasm32 so the browser UI stays at 60 FPS.
    // Returns true when the step to `a1` has completed.
    #[cfg(target_arch = "wasm32")]
    pub fn step_to_incremental(&mut self, a1: Real, budget: web_time::Duration) -> bool {
        let started = web_time::Instant::now();
        if self.pending.is_none() {
            let a0 = self.a;
            let am = self.midpoint(a0, a1);
            let k1 = self.kick_factor(a0, am);
            kick(&mut self.momenta, &self.accel, k1);
            let d = self.drift_factor(a0, a1);
            drift(&mut self.positions, &self.momenta, d, self.box_size);
            let k2 = self.kick_factor(am, a1);
            let delta = self.density_contrast();
            let r_split = self.split.map_or(0.0, |s| s.r_split);
            let [gx, gy, gz] = self.solver.acceleration(&delta, &self.cosmo, a1, r_split);
            let acc: Vec<[Real; 3]> = self
                .positions
                .iter()
                .map(|&x| [gx.interpolate(x), gy.interpolate(x), gz.interpolate(x)])
                .collect();
            if self.split.is_none() {
                self.accel = acc;
                self.delta = delta;
                kick(&mut self.momenta, &self.accel, k2);
                self.a = a1;
                return true;
            }
            let tree = tree::Tree::build(&self.positions, self.box_size).into_built();
            self.pending = Some(PendingStep {
                a1,
                k2,
                acc,
                delta,
                tree,
                next_i: 0,
                stack: Vec::with_capacity(64),
            });
            if started.elapsed() >= budget {
                return false;
            }
        }

        if let (Some(p), Some(split)) = (self.pending.as_mut(), self.split.as_ref()) {
            let n = self.positions.len();
            let k = 1.5 * self.cosmo.omega_m * self.box_size.powi(3) / (4.0 * std::f32::consts::PI * p.a1 * n as Real);
            while p.next_i < n {
                let end = (p.next_i + 256).min(n);
                for i in p.next_i..end {
                    let s = p.tree.accel_on(&self.positions, i, split, k, &mut p.stack);
                    for c in 0..3 {
                        p.acc[i][c] += s[c];
                    }
                }
                p.next_i = end;
                if p.next_i < n && started.elapsed() >= budget {
                    return false;
                }
            }
        }

        if let Some(p) = self.pending.take() {
            self.accel = p.acc;
            self.delta = p.delta;
            kick(&mut self.momenta, &self.accel, p.k2);
            self.a = p.a1;
        }
        true
    }

    #[allow(dead_code)]
    fn _unused_end(&self) {
    }

    // ∫ da / (a E): the time-to-momentum factor for a kick.
    fn kick_factor(&self, a0: Real, a1: Real) -> Real {
        self.integrate(a0, a1, |a| 1.0 / (a * self.cosmo.e(a)))
    }

    // ∫ da / (a³ E): the time-to-position factor for a drift.
    fn drift_factor(&self, a0: Real, a1: Real) -> Real {
        self.integrate(a0, a1, |a| 1.0 / (a * a * a * self.cosmo.e(a)))
    }

    // Scale factor at the time midpoint of [a0, a1]: the point where the kick factor is half the total.
    fn midpoint(&self, a0: Real, a1: Real) -> Real {
        let half = self.kick_factor(a0, a1) / 2.0;
        let (mut lo, mut hi) = (a0, a1);
        for _ in 0..MIDPOINT_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            if self.kick_factor(a0, mid) < half {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    // Simpson's rule on [lo, hi].
    fn integrate(&self, lo: Real, hi: Real, f: impl Fn(Real) -> Real) -> Real {
        let n = STEP_INTEGRAL_STEPS;
        let h = (hi - lo) / n as Real;
        let mut sum = f(lo) + f(hi);
        for i in 1..n {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += weight * f(lo + i as Real * h);
        }
        sum * h / 3.0
    }
}

// p += g · factor for every particle.
fn kick(momenta: &mut [[Real; 3]], g: &[[Real; 3]], factor: Real) {
    for (p, gi) in momenta.iter_mut().zip(g) {
        for axis in 0..3 {
            p[axis] += gi[axis] * factor;
        }
    }
}

// x += p · factor for every particle, wrapped into [0, box_size).
fn drift(positions: &mut [[Real; 3]], momenta: &[[Real; 3]], factor: Real, box_size: Real) {
    for (x, p) in positions.iter_mut().zip(momenta) {
        for axis in 0..3 {
            x[axis] = (x[axis] + p[axis] * factor).rem_euclid(box_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::Particles;
    use std::f32::consts::PI;

    // Plane wave along x: δ_lin = A cos(k x) at z = 0, starting at a_init with Zel'dovich displacements.
    // Lattice of n³ particles at q = i h, box side `box_size`.
    fn plane_wave(n: usize, box_size: Real, amp: Real, cosmo: &Cosmology, a_init: Real) -> Particles {
        let k = 2.0 * PI / box_size;
        let h = box_size / n as Real;
        let d = cosmo.growth(a_init);
        let f = cosmo.growth_rate(a_init);
        let e = cosmo.e(a_init);
        let mut positions = Vec::new();
        let mut momenta = Vec::new();
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let q = [ix as Real * h, iy as Real * h, iz as Real * h];
                    // Ψ_x = -(A/k) sin(kq) gives δ = -∇·Ψ = A cos(kq).
                    let psi_x = -(amp / k) * (k * q[0]).sin();
                    let x = [(q[0] + d * psi_x).rem_euclid(box_size), q[1], q[2]];
                    let p = [a_init * a_init * f * e * d * psi_x, 0.0, 0.0];
                    positions.push(x);
                    momenta.push(p);
                }
            }
        }
        Particles { positions, momenta }
    }

    // Amplitude of the cos(kx) mode of δ on the mesh: 2 · mean(δ cos(kx)).
    fn mode_amplitude(sim: &Simulation, box_size: Real) -> Real {
        let delta = sim.density_contrast();
        let n = delta.n;
        let h = box_size / n as Real;
        let k = 2.0 * PI / box_size;
        let mut sum = 0.0;
        for ix in 0..n {
            let c = ((ix as Real + 0.5) * h * k).cos();
            for iy in 0..n {
                for iz in 0..n {
                    sum += delta.cells[(ix * n + iy) * n + iz] * c;
                }
            }
        }
        2.0 * sum / (n * n * n) as Real
    }

    #[test]
    fn linear_growth_matches_growth_factor() {
        // Amplitude ratio between a_final and a_init should match D(a_final)/D(a_init). A plane wave is exact
        // Zel'dovich in 1D, so the only residual is discretization. The particle count sets it: 16 particles
        // on a 32 mesh gives +4.5%, 32 particles on a 32 mesh gives -0.4%. Refining the mesh alone does not converge.
        let box_size = 1.0;
        let cosmo = Cosmology::lcdm_default();
        let (a_init, a_final) = (0.02, 0.1);
        let amp = 0.5;
        let particles = plane_wave(32, box_size, amp, &cosmo, a_init);
        let mut sim = Simulation::new(cosmo, box_size, 32, particles.positions, particles.momenta, a_init, None);

        let before = mode_amplitude(&sim, box_size);
        sim.run(a_final, 40);
        let after = mode_amplitude(&sim, box_size);

        let expected = sim.cosmo.growth(a_final) / sim.cosmo.growth(a_init);
        let ratio = after / before;
        let rel = (ratio - expected).abs() / expected;
        assert!(rel < 0.01, "growth ratio {ratio:.5}, expected {expected:.5}, rel {rel:.5}");
    }

    #[test]
    fn steps_converge() {
        // Running with 10 steps and with 40 should give nearly the same final positions.
        let box_size = 1.0;
        let cosmo = Cosmology::lcdm_default();
        let (a_init, a_final) = (0.02, 0.1);
        let amp = 0.5;
        let run = |steps: usize| {
            let p = plane_wave(16, box_size, amp, &cosmo, a_init);
            let mut sim = Simulation::new(Cosmology::lcdm_default(), box_size, 32, p.positions, p.momenta, a_init, None);
            sim.run(a_final, steps);
            sim.positions
        };
        let coarse = run(10);
        let fine = run(40);
        let max_diff = coarse
            .iter()
            .zip(&fine)
            .flat_map(|(a, b)| (0..3).map(move |k| {
                let d = (a[k] - b[k]).abs();
                // Positions are periodic, so take the shorter way around the box.
                d.min(box_size - d)
            }))
            .fold(0.0, Real::max);
        assert!(max_diff < 1e-3 * box_size, "max position difference {max_diff:e}");
    }

    #[test]
    fn total_momentum_stays_near_zero() {
        // A symmetric plane wave has no net momentum, and the leapfrog should keep it that way to round-off.
        let box_size = 1.0;
        let cosmo = Cosmology::lcdm_default();
        let p = plane_wave(16, box_size, 0.5, &cosmo, 0.02);
        let mut sim = Simulation::new(cosmo, box_size, 32, p.positions, p.momenta, 0.02, None);
        sim.run(0.1, 20);
        let total: [Real; 3] = sim.momenta.iter().fold([0.0; 3], |mut s, p| {
            for k in 0..3 {
                s[k] += p[k];
            }
            s
        });
        let scale: Real = sim.momenta.iter().map(|p| p[0].abs()).sum();
        for k in 0..3 {
            assert!(total[k].abs() < 1e-3 * scale, "momentum axis {k}: {:e}", total[k]);
        }
    }

    #[test]
    fn treepm_pair_force_matches_newton() {
        // Two particles 3 units apart in a 100-unit box. Their total TreePM force should match the Newtonian
        // pair force K/d², with K = (3/2) Ωm L³ / (4π a N), up to the small periodic-image correction.
        let box_size = 100.0;
        let cosmo = Cosmology::lcdm_default();
        let (a, n) = (1.0, 2.0);
        let omega_m = cosmo.omega_m;
        let mesh_n = 64;
        // Softening is kept well below the 3-unit separation. The default would be 2.5 units for two particles.
        let split = ForceSplit { softening: 0.01, ..ForceSplit::standard(box_size, mesh_n, 2) };
        let positions = vec![[50.0, 50.0, 50.0], [53.0, 50.0, 50.0]];
        let momenta = vec![[0.0; 3]; 2];
        let sim = Simulation::new(cosmo, box_size, mesh_n, positions, momenta, a, Some(split));

        let (acc, _) = sim.accelerations(a);
        let k = 1.5 * omega_m * box_size.powi(3) / (4.0 * PI * a * n);
        let newton = k / 9.0;
        // Particle 0 is pulled towards +x, particle 1 towards −x.
        assert!(acc[0][0] > 0.0 && acc[1][0] < 0.0);
        let rel0 = (acc[0][0] - newton).abs() / newton;
        let rel1 = (acc[1][0] + newton).abs() / newton;
        assert!(rel0 < 0.05, "particle 0: {} vs Newton {newton} (rel {rel0})", acc[0][0]);
        assert!(rel1 < 0.05, "particle 1: {} vs Newton {newton} (rel {rel1})", acc[1][0]);
    }
}
