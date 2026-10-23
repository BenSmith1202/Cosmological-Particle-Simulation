// Cloud-in-cell (CIC) mass deposit and field interpolation on a periodic cubic grid.
// Each particle is spread over the 8 nearest grid points with trilinear weights.

use crate::real::Real;

pub struct Mesh {
    // Cells per side.
    pub n: usize,
    // Side length of the periodic box, in the same units as particle positions.
    pub box_size: Real,
    // n³ values, indexed as (ix * n + iy) * n + iz.
    pub cells: Vec<Real>,
}

impl Mesh {
    pub fn new(n: usize, box_size: Real) -> Self {
        Self {
            n,
            box_size,
            cells: vec![0.0; n * n * n],
        }
    }

    fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.n + iy) * self.n + iz
    }

    // The 8 (cell index, weight) pairs that a position touches, with periodic wrap.
    fn stencil(&self, x: [Real; 3]) -> [(usize, Real); 8] {
        let n = self.n as i64;
        let h = self.box_size / self.n as Real;
        let mut base = [0i64; 3];
        let mut frac = [0.0; 3];
        for k in 0..3 {
            // Cell i is centred at (i + 0.5) h, so shift by half a cell before flooring.
            let s = x[k] / h - 0.5;
            let i0 = s.floor();
            base[k] = i0 as i64;
            frac[k] = s - i0;
        }
        let mut out = [(0, 0.0); 8];
        let mut c = 0;
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let w = axis_weight(frac[0], dx)
                        * axis_weight(frac[1], dy)
                        * axis_weight(frac[2], dz);
                    // rem_euclid gives the periodic wrap, including for negative indices.
                    let ix = (base[0] + dx).rem_euclid(n) as usize;
                    let iy = (base[1] + dy).rem_euclid(n) as usize;
                    let iz = (base[2] + dz).rem_euclid(n) as usize;
                    out[c] = (self.index(ix, iy, iz), w);
                    c += 1;
                }
            }
        }
        out
    }

    // Clears the grid, then deposits `mass` per particle.
    pub fn deposit(&mut self, positions: &[[Real; 3]], mass: Real) {
        self.cells.fill(0.0);
        for &p in positions {
            for (idx, w) in self.stencil(p) {
                self.cells[idx] += mass * w;
            }
        }
    }

    // Trilinear interpolation of the grid field at a position.
    pub fn interpolate(&self, x: [Real; 3]) -> Real {
        self.stencil(x)
            .iter()
            .map(|&(idx, w)| self.cells[idx] * w)
            .sum()
    }
}

// Weight for one axis: 1 - f for the lower cell (d = 0), f for the upper cell (d = 1).
fn axis_weight(f: Real, d: i64) -> Real {
    if d == 0 { 1.0 - f } else { f }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic pseudo-random positions in [0, L), so the tests need no rand dependency.
    fn scattered_positions(count: usize, box_size: Real) -> Vec<[Real; 3]> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as Real / (1u64 << 53) as Real * box_size
        };
        (0..count).map(|_| [next(), next(), next()]).collect()
    }

    #[test]
    fn deposit_conserves_total_mass() {
        let mut mesh = Mesh::new(16, 1.0);
        let positions = scattered_positions(5000, 1.0);
        mesh.deposit(&positions, 0.25);
        let total: Real = mesh.cells.iter().sum();
        // f32 accumulation over 5000 deposits: allow a relative error of 1e-5.
        assert!((total - 0.25 * 5000.0).abs() < 1e-5 * 0.25 * 5000.0);
    }

    #[test]
    fn particle_at_cell_centre_fills_one_cell() {
        let n = 8;
        let box_size = 1.0;
        let h = box_size / n as Real;
        let mut mesh = Mesh::new(n, box_size);
        // Centre of cell (2, 3, 4).
        let p = [2.5 * h, 3.5 * h, 4.5 * h];
        mesh.deposit(&[p], 1.0);
        assert!((mesh.cells[mesh.index(2, 3, 4)] - 1.0).abs() < 1e-5);
        let total: Real = mesh.cells.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert_eq!(mesh.cells.iter().filter(|&&v| v > 1e-12).count(), 1);
    }

    #[test]
    fn lattice_deposits_uniform_grid() {
        // One particle at the centre of every cell gives a uniform grid of value `mass`.
        let n = 8;
        let box_size = 2.0;
        let h = box_size / n as Real;
        let mut positions = Vec::new();
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    positions.push([
                        (ix as Real + 0.5) * h,
                        (iy as Real + 0.5) * h,
                        (iz as Real + 0.5) * h,
                    ]);
                }
            }
        }
        let mut mesh = Mesh::new(n, box_size);
        mesh.deposit(&positions, 3.0);
        for &v in &mesh.cells {
            assert!((v - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn particle_on_boundary_wraps_to_opposite_face() {
        // A particle just below L splits between the last cell and the first, through the periodic wrap.
        let n = 4;
        let box_size = 1.0;
        let h = box_size / n as Real;
        let mut mesh = Mesh::new(n, box_size);
        // Exactly on the face between cell n-1 and cell 0 (x = L).
        mesh.deposit(&[[box_size, 0.5 * h, 0.5 * h]], 1.0);
        assert!((mesh.cells[mesh.index(n - 1, 0, 0)] - 0.5).abs() < 1e-5);
        assert!((mesh.cells[mesh.index(0, 0, 0)] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn interpolating_constant_field_returns_constant() {
        let mut mesh = Mesh::new(8, 1.0);
        mesh.cells.fill(2.5);
        for p in scattered_positions(200, 1.0) {
            assert!((mesh.interpolate(p) - 2.5).abs() < 1e-5);
        }
    }

    #[test]
    fn interpolation_reproduces_cell_value_at_centre() {
        let n = 8;
        let box_size = 1.0;
        let h = box_size / n as Real;
        let mut mesh = Mesh::new(n, box_size);
        let idx = mesh.index(5, 1, 6);
        mesh.cells[idx] = 7.0;
        let v = mesh.interpolate([5.5 * h, 1.5 * h, 6.5 * h]);
        assert!((v - 7.0).abs() < 1e-5);
    }
}
