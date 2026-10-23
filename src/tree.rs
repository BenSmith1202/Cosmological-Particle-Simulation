// Short-range gravity for TreePM: a Barnes–Hut octree with the Gaussian-split kernel.
//
// The PM mesh supplies the long-range force, filtered by exp(-k² r_s²). This module supplies the rest:
//   S(r) = erfc(u) + (2u/√π) e^{-u²},  u = r / (2 r_s),
// multiplied by the softened Newtonian force, and only out to r_cut. Periodicity uses the minimum image.

use libm::erfcf;
use rayon::prelude::*;

use crate::real::Real;

const LEAF_SIZE: usize = 8;
const MAX_DEPTH: u32 = 40;
const NO_CHILD: u32 = u32::MAX;
const SQRT_3: Real = 1.732_050_8;
// 2/√π
const TWO_OVER_SQRT_PI: Real = 1.128_379_2;

// Force split and tolerances for TreePM.
#[derive(Clone, Copy, Debug)]
pub struct ForceSplit {
    // Gaussian smoothing scale of the PM force. The short-range force supplies the remainder.
    pub r_split: Real,
    // The short-range force is dropped beyond this distance.
    pub r_cut: Real,
    // Plummer softening length for the short-range force.
    pub softening: Real,
    // Barnes–Hut opening angle: a node with size / distance < theta is treated as one mass.
    pub theta: Real,
}

impl ForceSplit {
    // Defaults: r_s = 1.25 mesh cells, r_cut = 4.5 r_s, softening = 0.05 mean spacing, θ = 0.5.
    #[cfg(test)]
    pub fn standard(box_size: Real, mesh_n: usize, particles_per_side: usize) -> Self {
        Self::with(box_size, mesh_n, particles_per_side, 0.05, 0.5)
    }

    // Same split as `standard`, with the softening (as a fraction of mean spacing) and opening angle set explicitly.
    pub fn with(box_size: Real, mesh_n: usize, particles_per_side: usize, softening_scale: Real, theta: Real) -> Self {
        let cell = box_size / mesh_n as Real;
        let r_split = 1.25 * cell;
        let spacing = box_size / particles_per_side as Real;
        Self {
            r_split,
            r_cut: 4.5 * r_split,
            softening: softening_scale * spacing,
            theta,
        }
    }
}

// Short-range fraction S(r): the part of the Newtonian force that the PM mesh does not carry.
pub fn short_range_factor(r: Real, r_split: Real) -> Real {
    let u = r / (2.0 * r_split);
    erfcf(u) + TWO_OVER_SQRT_PI * u * (-u * u).exp()
}

// Acceleration on a target from a source of `mass` at displacement `d` = target - source.
// `k` is the Newtonian coupling, so a pair at distance r has magnitude k·mass/r² before the split and softening.
fn pair(d: [Real; 3], mass: Real, k: Real, split: &ForceSplit) -> [Real; 3] {
    let r2 = norm2(d);
    let s = r2 + split.softening * split.softening;
    let f = -k * mass * short_range_factor(r2.sqrt(), split.r_split) / (s * s.sqrt());
    [d[0] * f, d[1] * f, d[2] * f]
}

#[derive(Clone, Copy)]
struct Node {
    // Cube centre and half side length.
    center: [Real; 3],
    half: Real,
    // Centre of mass (unit particle masses).
    com: [Real; 3],
    count: u32,
    children: [u32; 8],
    // Range of `Tree::order` covered by this node.
    start: u32,
    end: u32,
}

pub struct Tree<'a> {
    positions: &'a [[Real; 3]],
    box_size: Real,
    // Particle indices, grouped so that each node's particles are contiguous.
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl<'a> Tree<'a> {
    pub fn build(positions: &'a [[Real; 3]], box_size: Real) -> Self {
        let n = positions.len();
        let mut tree = Self {
            positions,
            box_size,
            order: (0..n as u32).collect(),
            nodes: Vec::with_capacity(2 * n / LEAF_SIZE + 1),
        };
        // The root cube covers the whole box. Positions are wrapped into [0, L), so nothing falls outside.
        let half = box_size / 2.0;
        tree.build_node(0, n, [half; 3], half, 0);
        tree
    }

    // Builds the node for order[start..end] and recurses into non-empty octants. Returns the node's index.
    fn build_node(&mut self, start: usize, end: usize, center: [Real; 3], half: Real, depth: u32) -> u32 {
        let idx = self.nodes.len() as u32;
        let positions = self.positions;

        let mut com = [0.0; 3];
        for &j in &self.order[start..end] {
            let p = positions[j as usize];
            for k in 0..3 {
                com[k] += p[k];
            }
        }
        let count = (end - start) as Real;
        for c in com.iter_mut() {
            *c /= count;
        }
        self.nodes.push(Node {
            center,
            half,
            com,
            count: (end - start) as u32,
            children: [NO_CHILD; 8],
            start: start as u32,
            end: end as u32,
        });

        if end - start <= LEAF_SIZE || depth >= MAX_DEPTH {
            return idx;
        }

        // Counting sort of this node's particles into the eight octants.
        let octant = |p: [Real; 3]| {
            (p[0] >= center[0]) as usize | (((p[1] >= center[1]) as usize) << 1) | (((p[2] >= center[2]) as usize) << 2)
        };
        let mut counts = [0usize; 8];
        for &j in &self.order[start..end] {
            counts[octant(positions[j as usize])] += 1;
        }
        let mut offsets = [0usize; 8];
        let mut running = start;
        for o in 0..8 {
            offsets[o] = running;
            running += counts[o];
        }
        let mut next = offsets;
        let mut sorted = vec![0u32; end - start];
        for &j in &self.order[start..end] {
            let o = octant(positions[j as usize]);
            sorted[next[o] - start] = j;
            next[o] += 1;
        }
        self.order[start..end].copy_from_slice(&sorted);

        let quarter = half / 2.0;
        for o in 0..8 {
            if counts[o] == 0 {
                continue;
            }
            let child_center = [0, 1, 2].map(|k| {
                center[k] + if (o >> k) & 1 == 1 { quarter } else { -quarter }
            });
            let child = self.build_node(offsets[o], offsets[o] + counts[o], child_center, quarter, depth + 1);
            self.nodes[idx as usize].children[o] = child;
        }
        idx
    }

    // Displacement a - b, wrapped to the nearest periodic image.
    fn min_image(&self, a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
        let l = self.box_size;
        let mut d = [0.0; 3];
        for k in 0..3 {
            let x = a[k] - b[k];
            d[k] = x - l * (x / l).round();
        }
        d
    }

    // Short-range acceleration on particle `i`, walking the tree with an explicit stack.
    fn accel_on(&self, i: usize, split: &ForceSplit, k: Real, stack: &mut Vec<u32>) -> [Real; 3] {
        let xi = self.positions[i];
        let cut2 = split.r_cut * split.r_cut;
        let mut acc = [0.0; 3];
        stack.clear();
        stack.push(0);
        while let Some(n) = stack.pop() {
            let node = self.nodes[n as usize];

            // Skip a cube that lies entirely beyond the cutoff.
            let to_centre = self.min_image(node.center, xi);
            if norm2(to_centre).sqrt() - SQRT_3 * node.half > split.r_cut {
                continue;
            }

            if node.children[0] == NO_CHILD {
                // Leaf: sum over its particles directly.
                for &j in &self.order[node.start as usize..node.end as usize] {
                    let j = j as usize;
                    if j == i {
                        continue;
                    }
                    let d = self.min_image(xi, self.positions[j]);
                    if norm2(d) > cut2 {
                        continue;
                    }
                    add_to(&mut acc, pair(d, 1.0, k, split));
                }
            } else {
                // Internal node: use its centre of mass if it is far enough away, otherwise descend.
                let to_com = self.min_image(node.com, xi);
                let r = norm2(to_com).sqrt();
                if 2.0 * node.half < split.theta * r {
                    let d = [-to_com[0], -to_com[1], -to_com[2]];
                    add_to(&mut acc, pair(d, node.count as Real, k, split));
                } else {
                    for &c in &node.children {
                        if c != NO_CHILD {
                            stack.push(c);
                        }
                    }
                }
            }
        }
        acc
    }
}

// Short-range accelerations for every particle. `k` is the Newtonian coupling, K = (3/2) Ωm L³ / (4π a N).
pub fn accelerations(positions: &[[Real; 3]], box_size: Real, split: &ForceSplit, k: Real) -> Vec<[Real; 3]> {
    let tree = Tree::build(positions, box_size);
    (0..positions.len())
        .into_par_iter()
        .map_init(
            || Vec::with_capacity(64),
            |stack, i| tree.accel_on(i, split, k, stack),
        )
        .collect()
}

fn add_to(acc: &mut [Real; 3], v: [Real; 3]) {
    for k in 0..3 {
        acc[k] += v[k];
    }
}

fn norm2(v: [Real; 3]) -> Real {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic pseudo-random positions in [0, L).
    fn scattered(count: usize, box_size: Real, seed: u64) -> Vec<[Real; 3]> {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as Real / (1u64 << 24) as Real * box_size
        };
        (0..count).map(|_| [next(), next(), next()]).collect()
    }

    // Reference O(N²) short-range sum with the same kernel and cutoff.
    fn brute_force(positions: &[[Real; 3]], box_size: Real, split: &ForceSplit, k: Real) -> Vec<[Real; 3]> {
        let tree = Tree::build(positions, box_size);
        (0..positions.len())
            .map(|i| {
                let mut acc = [0.0; 3];
                for j in 0..positions.len() {
                    if j == i {
                        continue;
                    }
                    let d = tree.min_image(positions[i], positions[j]);
                    if norm2(d) > split.r_cut * split.r_cut {
                        continue;
                    }
                    add_to(&mut acc, pair(d, 1.0, k, split));
                }
                acc
            })
            .collect()
    }

    fn max_norm(v: &[[Real; 3]]) -> Real {
        v.iter().fold(0.0, |m, a| m.max(norm2(*a).sqrt()))
    }

    fn max_diff(a: &[[Real; 3]], b: &[[Real; 3]]) -> Real {
        a.iter()
            .zip(b)
            .fold(0.0, |m, (x, y)| m.max(norm2([x[0] - y[0], x[1] - y[1], x[2] - y[2]]).sqrt()))
    }

    #[test]
    fn short_range_factor_limits() {
        // S(0) = 1: the full Newtonian force at zero separation. S → 0 at large separation.
        assert!((short_range_factor(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!(short_range_factor(10.0, 1.0) < 1e-6);
        // Monotone decreasing in between.
        let mut prev = 1.0;
        for step in 1..=50 {
            let s = short_range_factor(step as Real * 0.1, 1.0);
            assert!(s <= prev + 1e-7);
            prev = s;
        }
    }

    #[test]
    fn tree_with_theta_zero_matches_brute_force() {
        // θ = 0 never accepts a node, so the tree must reproduce the pair sum exactly (up to f32 round-off).
        let box_size = 10.0;
        let split = ForceSplit { r_split: 0.5, r_cut: 2.0, softening: 0.05, theta: 0.0 };
        let positions = scattered(400, box_size, 7);
        let fast = accelerations(&positions, box_size, &split, 1.0);
        let slow = brute_force(&positions, box_size, &split, 1.0);
        let rel = max_diff(&fast, &slow) / max_norm(&slow);
        assert!(rel < 1e-5, "relative difference {rel:e}");
    }

    #[test]
    fn tree_with_theta_half_is_close_to_brute_force() {
        // The centre-of-mass approximation should stay within about 2% of the exact sum at θ = 0.5.
        let box_size = 10.0;
        let split = ForceSplit { r_split: 0.5, r_cut: 2.5, softening: 0.05, theta: 0.5 };
        let positions = scattered(3000, box_size, 11);
        let fast = accelerations(&positions, box_size, &split, 1.0);
        let slow = brute_force(&positions, box_size, &split, 1.0);
        let rel = max_diff(&fast, &slow) / max_norm(&slow);
        assert!(rel < 0.02, "relative difference {rel:e}");
    }

    #[test]
    fn periodic_pair_interacts_across_the_boundary() {
        // Two particles near opposite faces are close through the boundary, so they must attract across it.
        let box_size = 10.0;
        let split = ForceSplit { r_split: 0.5, r_cut: 2.0, softening: 0.05, theta: 0.5 };
        let positions = [[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]];
        let acc = accelerations(&positions, box_size, &split, 1.0);
        // Particle 0 is pulled towards −x (towards particle 1 through the boundary).
        assert!(acc[0][0] < 0.0, "acc {:?}", acc[0]);
        assert!(acc[1][0] > 0.0, "acc {:?}", acc[1]);
    }
}
