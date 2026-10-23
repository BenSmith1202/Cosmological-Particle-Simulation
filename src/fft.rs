// 3D FFT on an n³ grid stored as (ix * n + iy) * n + iz, built from 1D rustfft transforms.

use std::sync::Arc;

use num_complex::Complex;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

use crate::real::Real;

pub struct Fft3 {
    n: usize,
    forward: Arc<dyn Fft<Real>>,
    inverse: Arc<dyn Fft<Real>>,
}

impl Fft3 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<Real>::new();
        Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    // Unnormalized forward transform: F_k = Σ_x f(x) e^{-ikx}.
    pub fn forward(&self, data: &mut [Complex<Real>]) {
        self.transform(data, self.forward.as_ref());
    }

    // Unnormalized inverse transform: f(x) = Σ_k F_k e^{ikx}. Divide by n³ for the true inverse.
    pub fn inverse(&self, data: &mut [Complex<Real>]) {
        self.transform(data, self.inverse.as_ref());
    }

    // Transforms every line along each axis. The z lines are contiguous, so they are transformed in place.
    // The x and y lines are strided, so each set is gathered into a contiguous buffer, transformed, and scattered back.
    // Lines run in parallel, and each worker reuses one scratch buffer instead of allocating per line.
    fn transform(&self, data: &mut [Complex<Real>], fft: &dyn Fft<Real>) {
        let n = self.n;
        let scratch_len = fft.get_inplace_scratch_len();
        let new_scratch = || vec![Complex::new(0.0, 0.0); scratch_len];

        data.par_chunks_mut(n)
            .for_each_init(new_scratch, |scratch, line| fft.process_with_scratch(line, scratch));

        for axis in [1, 0] {
            let mut lines = vec![Complex::new(0.0, 0.0); n * n * n];
            {
                let src: &[Complex<Real>] = data;
                lines
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(ab, line)| {
                        let (a, b) = (ab / n, ab % n);
                        for (t, v) in line.iter_mut().enumerate() {
                            *v = src[flat_index(n, axis, a, b, t)];
                        }
                    });
            }
            lines
                .par_chunks_mut(n)
                .for_each_init(new_scratch, |scratch, line| fft.process_with_scratch(line, scratch));
            for ab in 0..n * n {
                let (a, b) = (ab / n, ab % n);
                for t in 0..n {
                    data[flat_index(n, axis, a, b, t)] = lines[ab * n + t];
                }
            }
        }
    }
}

// Flat index of position `t` along `axis` (0 = x, 1 = y, 2 = z), with (a, b) fixed on the other two axes.
fn flat_index(n: usize, axis: usize, a: usize, b: usize, t: usize) -> usize {
    match axis {
        0 => (t * n + a) * n + b,
        1 => (a * n + t) * n + b,
        _ => (a * n + b) * n + t,
    }
}

// Signed angular wavenumber for grid index `i` in a periodic box of side `box_size`.
// The Nyquist mode of an even grid is 0, because a real derivative can't be represented there.
pub fn wavenumber(i: usize, n: usize, box_size: Real) -> Real {
    let two_pi_over_l = 2.0 * std::f64::consts::PI as Real / box_size;
    if n % 2 == 0 && i == n / 2 {
        0.0
    } else if i <= n / 2 {
        i as Real * two_pi_over_l
    } else {
        (i as Real - n as Real) * two_pi_over_l
    }
}
