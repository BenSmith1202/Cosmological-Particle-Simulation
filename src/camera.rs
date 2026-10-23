// Camera math for the viewer: a 2D slice camera and a 3D orbit camera, both producing a view-projection matrix.
// Matrices are column-major, as WGSL expects: m[column][row].

use crate::real::Real;

pub type Mat4 = [[Real; 4]; 4];

// Product a * b.
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

// Depth slope of the slice view's projection. Small, so the whole box sits near depth 0.5.
const SLICE_DEPTH_SCALE: Real = 1e-4;

// Inverse of a 4x4 matrix, by cofactors. None if it is singular. Used to turn screen pixels back into world rays.
pub fn inverse(m: &Mat4) -> Option<Mat4> {
    // Flatten column-major m[c][r] into a[c * 4 + r].
    let a: Vec<Real> = m.iter().flatten().copied().collect();
    let mut inv = [0.0; 16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if det.abs() < 1e-30 {
        return None;
    }
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = inv[c * 4 + r] / det;
        }
    }
    Some(out)
}

// Right-handed view matrix looking from `eye` towards `target`.
fn look_at(eye: [Real; 3], target: [Real; 3], up: [Real; 3]) -> Mat4 {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

// Right-handed perspective projection, mapping depth to [0, 1] as wgpu expects.
fn perspective(fovy: Real, aspect: Real, near: Real, far: Real) -> Mat4 {
    let t = 1.0 / (fovy / 2.0).tan();
    [
        [t / aspect, 0.0, 0.0, 0.0],
        [0.0, t, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn sub(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [Real; 3], b: [Real; 3]) -> Real {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [Real; 3]) -> [Real; 3] {
    let n = dot(v, v).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

// 2D slice camera: a world point at the centre of the window, and a zoom level.
// Looks straight down the z axis, so it shows the x–y plane.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub center: [Real; 2],
    pub pixels_per_unit: Real,
}

impl Camera {
    // Shows the whole box with a margin.
    pub fn fit_box(box_size: Real, viewport: [Real; 2]) -> Self {
        let min_dim = viewport[0].min(viewport[1]);
        Self {
            center: [box_size / 2.0, box_size / 2.0],
            pixels_per_unit: 0.9 * min_dim / box_size,
        }
    }

    // World point under pixel `px` (origin top-left, y down).
    pub fn world_at(&self, px: [Real; 2], viewport: [Real; 2]) -> [Real; 2] {
        let rel = Self::rel_to_centre(px, viewport);
        [
            self.center[0] + rel[0] / self.pixels_per_unit,
            self.center[1] + rel[1] / self.pixels_per_unit,
        ]
    }

    // Drag by `delta_px`: the content follows the cursor, so the camera moves the opposite way.
    pub fn pan(&mut self, delta_px: [Real; 2]) {
        self.center[0] -= delta_px[0] / self.pixels_per_unit;
        // Screen y points down and world y points up.
        self.center[1] += delta_px[1] / self.pixels_per_unit;
    }

    // Zoom by `factor`, keeping the world point under `px` fixed on screen.
    pub fn zoom_at(&mut self, px: [Real; 2], viewport: [Real; 2], factor: Real) {
        let anchor = self.world_at(px, viewport);
        self.pixels_per_unit *= factor;
        let rel = Self::rel_to_centre(px, viewport);
        self.center = [
            anchor[0] - rel[0] / self.pixels_per_unit,
            anchor[1] - rel[1] / self.pixels_per_unit,
        ];
    }

    // Orthographic view-projection: world (x, y) maps to clip space. Depth is 0.5 + z · SLICE_DEPTH_SCALE, which keeps
    // any z within ±5000 inside the [0, 1] depth range (nothing is clipped), and keeps the matrix invertible, so the
    // gas ray-march can turn pixels back into rays.
    pub fn view_proj(&self, viewport: [Real; 2]) -> Mat4 {
        let sx = 2.0 * self.pixels_per_unit / viewport[0];
        let sy = 2.0 * self.pixels_per_unit / viewport[1];
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, SLICE_DEPTH_SCALE, 0.0],
            [-sx * self.center[0], -sy * self.center[1], 0.5, 1.0],
        ]
    }

    // Pixel offset from the viewport centre, with y flipped to point up.
    fn rel_to_centre(px: [Real; 2], viewport: [Real; 2]) -> [Real; 2] {
        [px[0] - viewport[0] / 2.0, -(px[1] - viewport[1] / 2.0)]
    }
}

// 3D orbit camera: it circles `target` at `distance`, at the given yaw (around y) and pitch (up from the x–z plane).
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub target: [Real; 3],
    pub yaw: Real,
    pub pitch: Real,
    pub distance: Real,
}

// Vertical field of view of the orbit camera, in radians. The lensing needs it to know texel sizes at each depth.
pub const ORBIT_FOV_Y: Real = 45.0 * std::f32::consts::PI / 180.0;

// Pitch is kept just short of straight up or down, so the view never flips.
const MAX_PITCH: Real = 1.55;
// Scroll zoom per line of wheel movement.
const ZOOM_PER_LINE: Real = 0.9;

impl Orbit {
    // Starts from a slightly elevated view of the whole box.
    pub fn fit_box(box_size: Real) -> Self {
        Self {
            target: [box_size / 2.0; 3],
            yaw: 0.6,
            pitch: 0.35,
            distance: 2.5 * box_size,
        }
    }

    pub fn eye(&self) -> [Real; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [
            self.target[0] + self.distance * cp * sy,
            self.target[1] + self.distance * sp,
            self.target[2] + self.distance * cp * cy,
        ]
    }

    // Drag in pixels: horizontal drag turns around y, vertical drag tilts the view.
    pub fn rotate(&mut self, delta_px: [Real; 2]) {
        self.yaw -= delta_px[0] * 0.005;
        self.pitch = (self.pitch + delta_px[1] * 0.005).clamp(-MAX_PITCH, MAX_PITCH);
    }

    // Moves the orbit centre across the screen, scaled to the distance, so the content follows the cursor.
    pub fn pan(&mut self, delta_px: [Real; 2], viewport_h: Real) {
        let eye = self.eye();
        let forward = normalize(sub(self.target, eye));
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        // At 45° vertical field of view, the visible height at the orbit centre is 2·tan(22.5°)·distance.
        let scale = 2.0 * (22.5f32).to_radians().tan() * self.distance / viewport_h;
        for k in 0..3 {
            self.target[k] += (-delta_px[0] * right[k] + delta_px[1] * up[k]) * scale;
        }
    }

    pub fn zoom(&mut self, lines: Real) {
        self.distance = (self.distance * ZOOM_PER_LINE.powf(lines)).clamp(1e-3, 1e5);
    }

    // Perspective view-projection looking at the orbit centre.
    pub fn view_proj(&self, viewport: [Real; 2]) -> Mat4 {
        let aspect = viewport[0] / viewport[1];
        // Near and far planes scale with the orbit distance, so depth precision holds at any zoom.
        let near = self.distance * 0.01;
        let far = self.distance * 10.0;
        let view = look_at(self.eye(), self.target, [0.0, 1.0, 0.0]);
        mul(&perspective(ORBIT_FOV_Y, aspect, near, far), &view)
    }
}

// Applies a view-projection matrix to a point, returning clip coordinates. Used by tests.
#[cfg(test)]
fn project(m: &Mat4, p: [Real; 3]) -> [Real; 4] {
    let v = [p[0], p[1], p[2], 1.0];
    let mut out = [0.0; 4];
    for r in 0..4 {
        out[r] = (0..4).map(|c| m[c][r] * v[c]).sum();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEWPORT: [Real; 2] = [800.0, 600.0];

    #[test]
    fn viewport_centre_maps_to_camera_centre() {
        let cam = Camera::fit_box(100.0, VIEWPORT);
        let w = cam.world_at([VIEWPORT[0] / 2.0, VIEWPORT[1] / 2.0], VIEWPORT);
        assert!((w[0] - cam.center[0]).abs() < 1e-4);
        assert!((w[1] - cam.center[1]).abs() < 1e-4);
    }

    #[test]
    fn zoom_keeps_cursor_point_fixed() {
        // The world point under the cursor must not move when zooming about the cursor.
        let mut cam = Camera::fit_box(100.0, VIEWPORT);
        let cursor = [150.0, 420.0];
        let before = cam.world_at(cursor, VIEWPORT);
        cam.zoom_at(cursor, VIEWPORT, 3.0);
        let after = cam.world_at(cursor, VIEWPORT);
        assert!((before[0] - after[0]).abs() < 1e-3);
        assert!((before[1] - after[1]).abs() < 1e-3);
    }

    #[test]
    fn zoom_in_increases_scale() {
        let mut cam = Camera::fit_box(100.0, VIEWPORT);
        let scale = cam.pixels_per_unit;
        cam.zoom_at([400.0, 300.0], VIEWPORT, 2.0);
        assert!((cam.pixels_per_unit - 2.0 * scale).abs() < 1e-4);
    }

    #[test]
    fn dragging_right_moves_content_right() {
        // Dragging right by 50 px should move the world point under the cursor 50 px to the right on screen.
        let mut cam = Camera::fit_box(100.0, VIEWPORT);
        let start = [300.0, 300.0];
        let world_before = cam.world_at(start, VIEWPORT);
        cam.pan([50.0, 0.0]);
        // The same world point is now 50 px further right, so it sits at `start.x` in the moved view.
        let world_at_new_cursor = cam.world_at([start[0] + 50.0, start[1]], VIEWPORT);
        assert!((world_before[0] - world_at_new_cursor[0]).abs() < 1e-3);
    }

    #[test]
    fn slice_projection_matches_world_at() {
        // The orthographic matrix should send the camera centre to the middle of the viewport.
        let cam = Camera::fit_box(100.0, VIEWPORT);
        let clip = project(&cam.view_proj(VIEWPORT), [cam.center[0], cam.center[1], 0.0]);
        assert!(clip[0].abs() < 1e-5 && clip[1].abs() < 1e-5);
    }

    #[test]
    fn orbit_target_projects_to_screen_centre() {
        let orbit = Orbit::fit_box(100.0);
        let clip = project(&orbit.view_proj(VIEWPORT), orbit.target);
        assert!(clip[0].abs() < 1e-4 && clip[1].abs() < 1e-4, "clip {clip:?}");
        // The target is in front of the camera: positive depth in clip space and w > 0.
        assert!(clip[3] > 0.0);
    }

    // Max |(m · inv) − I| over all entries.
    fn identity_error(m: &Mat4) -> Real {
        let inv = inverse(m).expect("invertible");
        let p = mul(m, &inv);
        let mut err: Real = 0.0;
        for c in 0..4 {
            for r in 0..4 {
                let want = if c == r { 1.0 } else { 0.0 };
                err = err.max((p[c][r] - want).abs());
            }
        }
        err
    }

    #[test]
    fn both_view_projections_are_invertible() {
        // The gas ray-march unprojects pixels with these inverses, so both cameras must invert accurately.
        assert!(identity_error(&Orbit::fit_box(100.0).view_proj(VIEWPORT)) < 1e-3);
        assert!(identity_error(&Camera::fit_box(100.0, VIEWPORT).view_proj(VIEWPORT)) < 1e-3);
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        assert!(inverse(&[[0.0; 4]; 4]).is_none());
    }

    #[test]
    fn slice_view_keeps_the_whole_box_in_depth_range() {
        let m = Camera::fit_box(100.0, VIEWPORT).view_proj(VIEWPORT);
        for z in [0.0, 50.0, 100.0] {
            let depth = project(&m, [50.0, 50.0, z])[2];
            assert!((0.0..=1.0).contains(&depth), "z {z} -> depth {depth}");
        }
    }

    #[test]
    fn orbit_eye_is_at_distance() {
        let orbit = Orbit::fit_box(100.0);
        let e = orbit.eye();
        let d = dot(sub(e, orbit.target), sub(e, orbit.target)).sqrt();
        assert!((d - orbit.distance).abs() < 1e-3);
    }

    #[test]
    fn rotation_keeps_target_in_view_and_clamps_pitch() {
        let mut orbit = Orbit::fit_box(100.0);
        orbit.rotate([123.0, -4000.0]);
        assert!(orbit.pitch >= -MAX_PITCH && orbit.pitch <= MAX_PITCH);
        let clip = project(&orbit.view_proj(VIEWPORT), orbit.target);
        assert!(clip[0].abs() < 1e-4 && clip[1].abs() < 1e-4);
    }

    #[test]
    fn zoom_in_moves_eye_closer() {
        let mut orbit = Orbit::fit_box(100.0);
        let before = orbit.distance;
        orbit.zoom(3.0);
        assert!(orbit.distance < before);
    }

    #[test]
    fn pan_moves_target_with_cursor() {
        // Dragging right should move the target to the left, so the content appears to follow the cursor.
        let mut orbit = Orbit::fit_box(100.0);
        let before = orbit.target;
        orbit.pan([10.0, 0.0], VIEWPORT[1]);
        let forward = normalize(sub(before, orbit.eye()));
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let moved = dot(sub(orbit.target, before), right);
        assert!(moved < 0.0, "moved {moved}");
    }
}
