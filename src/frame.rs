// What the simulation thread sends the viewer after each step, and the conversions that build it.

use crate::mesh::Mesh;
use crate::real::Real;

pub struct Frame {
    pub redshift: f32,
    pub positions: Vec<[f32; 3]>,
    // Peculiar speed |p| / a, in the same units as the box.
    pub speeds: Vec<f32>,
    // Local density in units of the mean, 1 + δ, sampled from the PM mesh at each particle (CIC).
    pub densities: Vec<f32>,
    // Redshift-space shift: each particle's peculiar velocity as a comoving distance, v / (aH). See `rsd_shift`.
    pub rsd: Vec<[f32; 3]>,
    // Density grid for the volumetric gas. Only built while the viewer asks for it, since it can be tens of MB.
    pub grid: Option<DensityGrid>,
}

// 1 + δ on an n³ grid, ordered x fastest, then y, then z (the order a 3D texture expects).
// Cell i spans [i, i+1)·L/n, so texel centres line up with the mesh's cell centres at (i + 0.5)·L/n.
pub struct DensityGrid {
    pub n: usize,
    pub data: Vec<f32>,
}

// The gas texture is at most this many cells per side. 128³ in 16-bit floats is 4 MB per frame.
pub const MAX_GRID: usize = 128;

// Converts the mesh's δ to a texture-ordered 1 + δ grid, averaging f³ blocks so the side is at most MAX_GRID.
// The mesh is indexed (ix·n + iy)·n + iz (z fastest), and the output is x fastest, so this also transposes.
// If n isn't a multiple of f, the last partial block on each side is dropped. With the mesh at twice the
// particle count per side, n is even and f is 1 or 2, so that never happens in practice.
pub fn downsample_grid(delta: &Mesh) -> DensityGrid {
    let n = delta.n;
    let f = n.div_ceil(MAX_GRID).max(1);
    let m = n / f;
    let norm = 1.0 / (f * f * f) as f32;
    let mut data = vec![0.0f32; m * m * m];
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                let mut sum = 0.0;
                for dx in 0..f {
                    for dy in 0..f {
                        for dz in 0..f {
                            let (ix, iy, iz) = (x * f + dx, y * f + dy, z * f + dz);
                            sum += delta.cells[(ix * n + iy) * n + iz];
                        }
                    }
                }
                data[(z * m + y) * m + x] = 1.0 + sum * norm;
            }
        }
    }
    DensityGrid { n: m, data }
}

// Redshift-space shift for one particle: v_pec / (aH), in comoving box units.
// p = a² dx/dt, so v_pec = a dx/dt = p / a, and dividing by aH (H = E in H0 = 1 units) gives p / (a² E).
// A survey that measures distance by redshift places the particle this much further along the line of sight.
pub fn rsd_shift(p: [Real; 3], a: Real, e: Real) -> [f32; 3] {
    let s = 1.0 / (a * a * e);
    [p[0] * s, p[1] * s, p[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_grids_keep_every_cell_and_transpose() {
        // n = 4 needs no downsampling. Put a value in mesh cell (ix, iy, iz) = (1, 2, 3).
        let mut mesh = Mesh::new(4, 10.0);
        mesh.cells[(1 * 4 + 2) * 4 + 3] = 5.0;
        let g = downsample_grid(&mesh);
        assert_eq!(g.n, 4);
        // In texture order that is x = 1, y = 2, z = 3: index (z·m + y)·m + x.
        assert_eq!(g.data[(3 * 4 + 2) * 4 + 1], 6.0);
        // Every other cell is mean density (δ = 0).
        assert_eq!(g.data.iter().filter(|&&v| v == 1.0).count(), 63);
    }

    #[test]
    fn large_grids_average_blocks() {
        // 200 per side becomes 100, each texel the mean of a 2x2x2 block.
        let mut mesh = Mesh::new(200, 100.0);
        // One block (cells 0..2 on each axis) with δ = 8 in one cell: the block mean is 1, so 1 + δ = 2.
        mesh.cells[0] = 8.0;
        let g = downsample_grid(&mesh);
        assert_eq!(g.n, 100);
        assert!((g.data[0] - 2.0).abs() < 1e-6);
        assert!((g.data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rsd_shift_is_velocity_over_a_h() {
        // At a = 0.5 and E = 2, p = (1, 0, 0): v = p/a = 2, aH = 1, so the shift is 2.
        let s = rsd_shift([1.0, 0.0, 0.0], 0.5, 2.0);
        assert!((s[0] - 2.0).abs() < 1e-6 && s[1] == 0.0 && s[2] == 0.0);
    }
}
