// Flat ΛCDM background: expansion rate and linear growth factor.
// Radiation is ignored (fine for z <= 50, per Conversation.md:412).

use crate::real::Real;

// RK4 steps for the growth ODE, from a = 1e-3 to the target a.
const GROWTH_STEPS: usize = 4000;

pub struct Cosmology {
    // Radiation (photons plus massless neutrinos), matter, and dark energy, as fractions of today's critical density.
    pub omega_r: Real,
    pub omega_m: Real,
    pub omega_l: Real,
    // Hubble constant today, km/s/Mpc.
    pub h0: Real,
}

impl Cosmology {
    // Parameters from Conversation.md:404-410. Radiation is Ωr h² ≈ 4.15e-5 (photons plus massless neutrinos,
    // from memory). Dark energy takes up the rest, so the universe stays flat.
    pub fn lcdm_default() -> Self {
        let omega_r = 9.14e-5;
        Self {
            omega_r,
            omega_m: 0.315,
            omega_l: 1.0 - 0.315 - omega_r,
            h0: 67.4,
        }
    }

    // E(a) = H(a)/H0 = sqrt(Ωr a^-4 + Ωm a^-3 + ΩΛ). Flat, so no curvature term.
    pub fn e(&self, a: Real) -> Real {
        (self.omega_r / (a * a * a * a) + self.omega_m / (a * a * a) + self.omega_l).sqrt()
    }

    // H(a) in km/s/Mpc.
    #[cfg(test)]
    pub fn hubble(&self, a: Real) -> Real {
        self.h0 * self.e(a)
    }

    // Linear growth δ(a) by integrating the growth ODE in ln a, with unnormalized starting values.
    //   δ'' + (2 + dlnE/dlna) δ' = (3/2) Ωm δ / (a³ E²),   ' = d/dlna.
    // Radiation changes the expansion, so the closed-form integral D ∝ E ∫ da/(aE)³ is no longer exact.
    // The integral is exact only for matter plus Λ, so the ODE is used instead.
    // Returns (δ, dδ/dlna) at a_end.
    fn growth_state(&self, a_end: Real) -> (Real, Real) {
        // Start deep in the radiation era. With radiation, use the Meszaros growing mode D = 1 + 3y/2,
        // with y = a/a_eq and a_eq = Ωr/Ωm. Its derivative in ln a is 3y/2. Without radiation, D ∝ a.
        let a_start: Real = 1e-3;
        let (mut d, mut dl) = if self.omega_r > 0.0 {
            let y = a_start * self.omega_m / self.omega_r;
            (1.0 + 1.5 * y, 1.5 * y)
        } else {
            (a_start, a_start)
        };

        let rhs = |la: Real, d: Real, dl: Real| -> (Real, Real) {
            let a = la.exp();
            let e2 = self.omega_r / (a * a * a * a) + self.omega_m / (a * a * a) + self.omega_l;
            let dln_e = 0.5 * (-4.0 * self.omega_r / (a * a * a * a) - 3.0 * self.omega_m / (a * a * a)) / e2;
            let source = 1.5 * self.omega_m / (a * a * a * e2) * d;
            (dl, -(2.0 + dln_e) * dl + source)
        };

        // RK4 in ln a. Each step's time is computed from the start, not accumulated, so f32 rounding doesn't drift.
        let la0 = a_start.ln();
        let h = (a_end.ln() - la0) / GROWTH_STEPS as Real;
        for step in 0..GROWTH_STEPS {
            let la = la0 + step as Real * h;
            let (k1d, k1l) = rhs(la, d, dl);
            let (k2d, k2l) = rhs(la + 0.5 * h, d + 0.5 * h * k1d, dl + 0.5 * h * k1l);
            let (k3d, k3l) = rhs(la + 0.5 * h, d + 0.5 * h * k2d, dl + 0.5 * h * k2l);
            let (k4d, k4l) = rhs(la + h, d + h * k3d, dl + h * k3l);
            d += h / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
            dl += h / 6.0 * (k1l + 2.0 * k2l + 2.0 * k3l + k4l);
        }
        (d, dl)
    }

    // Linear growth factor, normalized so D(1) = 1. Scales as a in the early matter-dominated era.
    pub fn growth(&self, a: Real) -> Real {
        self.growth_state(a).0 / self.growth_state(1.0).0
    }

    // Growth rate f(a) = dlnD/dlna. The ODE state holds dδ/dlna directly, so no finite difference is needed.
    pub fn growth_rate(&self, a: Real) -> Real {
        let (d, dl) = self.growth_state(a);
        dl / d
    }

    // BBKS transfer function (Bardeen et al. 1986), dark-matter only, no baryon correction.
    // `k` is in h/Mpc. The shape parameter is Γ = Ωm h, so q = k / Γ.
    pub fn bbks_transfer(&self, k: Real) -> Real {
        let h = self.h0 / 100.0;
        let q = k / (self.omega_m * h);
        if q == 0.0 {
            return 1.0;
        }
        let poly = 1.0 + 3.89 * q + (16.1 * q).powi(2) + (5.46 * q).powi(3) + (6.71 * q).powi(4);
        // ln_1p keeps precision at small q, where 1 + 2.34q would round in f32.
        (2.34 * q).ln_1p() / (2.34 * q) * poly.powf(-0.25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_rate_is_one_today() {
        let c = Cosmology::lcdm_default();
        assert!((c.e(1.0) - 1.0).abs() < 1e-5);
        assert!((c.hubble(1.0) - 67.4).abs() < 1e-5);
    }

    #[test]
    fn growth_is_one_today() {
        let c = Cosmology::lcdm_default();
        assert!((c.growth(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn growth_scales_as_a_early() {
        // Without radiation, D/a is constant at early times, with corrections O(a³).
        // (With radiation, D/a changes by a few percent between a = 0.01 and 0.02, so use a radiation-free cosmology here.)
        let c = Cosmology { omega_r: 0.0, ..Cosmology::lcdm_default() };
        let r1 = c.growth(1e-3) / 1e-3;
        let r2 = c.growth(2e-3) / 2e-3;
        assert!((r1 - r2).abs() / r1 < 1e-4);
    }

    #[test]
    fn growth_matches_reference_at_half() {
        // Reference from an independent Python integral (same formula, 20000-point Simpson), with radiation included.
        // Solved from the growth ODE (Meszaros start). Without radiation the closed form gives 0.6067538.
        let c = Cosmology::lcdm_default();
        assert!((c.growth(0.5) - 0.6068044).abs() < 1e-4);
    }
}
