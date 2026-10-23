// Runs the simulation on a worker thread, and restarts it with new settings when the viewer asks.
// The viewer owns the window, so it stays on the main thread. Frames cross a one-slot channel.

use serde::{Deserialize, Serialize};

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::cosmology::Cosmology;
use crate::ics::{zeldovich, PowerSpectrum, N_S, SIGMA8, SIGMA8_RADIUS};
use crate::integrator::Simulation;
use crate::real::Real;
use crate::tree::ForceSplit;
use crate::frame::{downsample_grid, rsd_shift, Frame};
use rayon::prelude::*;

// Box side, in h⁻¹Mpc. Fixed for now: changing it changes the scale of the whole run.
pub const BOX: Real = 100.0;

// Particle lattice limits per side. 100³ = 1M particles uses about 0.7 GB of RAM, and its 32 MB vertex buffer is well
// under wgpu's default 256 MiB buffer limit (about 200³). The real cost is time: roughly 25 minutes for 512 steps.
pub const MIN_PARTICLES_PER_SIDE: usize = 5;
pub const MAX_PARTICLES_PER_SIDE: usize = 200;

// Everything the settings panel can change. Changes take effect on the next restart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunSettings {
    // Start and end redshift. Negative z is the future.
    pub z_start: Real,
    pub z_end: Real,
    pub steps: usize,
    pub particles_per_side: usize,
    pub seed: u64,
    // TreePM short-range force on or off. Off means pure PM.
    pub treepm: bool,
    // Barnes–Hut opening angle.
    pub theta: Real,
    // Softening as a fraction of the mean particle spacing.
    pub softening_scale: Real,
    // Pause after each step, in milliseconds, so the formation can be watched more slowly.
    pub pacing_ms: u32,
}

impl Default for RunSettings {
    // The current defaults: z = 500 → −0.9, 40³ particles, TreePM with θ = 0.5, no pause.
    fn default() -> Self {
        Self {
            z_start: 500.0,
            z_end: -0.9,
            steps: 512,
            particles_per_side: 40,
            seed: 1,
            treepm: true,
            theta: 0.5,
            softening_scale: 0.05,
            pacing_ms: 0,
        }
    }
}

impl RunSettings {
    // Clamps every field into a range the simulation can run. Keeps the panel from producing a broken run.
    pub fn sanitized(&self) -> Self {
        let z_start = self.z_start.clamp(10.0, 2000.0);
        Self {
            z_start,
            z_end: self.z_end.clamp(-0.95, z_start - 1.0),
            steps: self.steps.clamp(16, 5000),
            particles_per_side: self.particles_per_side.clamp(MIN_PARTICLES_PER_SIDE, MAX_PARTICLES_PER_SIDE),
            seed: self.seed,
            treepm: self.treepm,
            theta: self.theta.clamp(0.1, 1.0),
            softening_scale: self.softening_scale.clamp(0.0, 0.5),
            pacing_ms: self.pacing_ms.min(2000),
        }
    }
}

// Flags shared between the viewer and the worker thread.
struct Control {
    // Set to ask the worker to exit at its next step.
    stop: AtomicBool,
    pacing_ms: AtomicU32,
    // Set by the viewer while the volumetric gas is on. The density grid is only built and sent then.
    want_grid: AtomicBool,
}

pub struct Runner {
    pub frames: Receiver<Frame>,
    pub settings: RunSettings,
    control: Arc<Control>,
    handle: Option<JoinHandle<()>>,
}

impl Runner {
    // Starts a run on a worker thread. The settings are sanitized first.
    pub fn start(settings: RunSettings) -> Self {
        let settings = settings.sanitized();
        // One slot: if the viewer hasn't taken the last frame, the worker drops the new one rather than waiting.
        let (tx, rx) = sync_channel(1);
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            pacing_ms: AtomicU32::new(settings.pacing_ms),
            want_grid: AtomicBool::new(false),
        });
        let worker_control = Arc::clone(&control);
        let worker_settings = settings.clone();
        let handle = std::thread::spawn(move || run_simulation(tx, &worker_settings, &worker_control));
        Self {
            frames: rx,
            settings,
            control,
            handle: Some(handle),
        }
    }

    // Stops the current run and starts a new one. The old worker exits at its next step.
    pub fn restart(&mut self, settings: RunSettings) {
        self.control.stop.store(true, Ordering::Relaxed);
        *self = Runner::start(settings);
    }

    // Asks for (or stops) the density grid in each frame. Takes effect at the worker's next step.
    pub fn set_want_grid(&self, want: bool) {
        self.control.want_grid.store(want, Ordering::Relaxed);
    }

    // Waits for the run to finish. Used in headless mode.
    pub fn join(mut self) {
        if let Some(handle) = self.handle.take() {
            handle.join().expect("simulation thread");
        }
    }
}

fn run_simulation(frames: SyncSender<Frame>, settings: &RunSettings, control: &Control) {
    let cosmo = Cosmology::lcdm_default();
    let a_start = 1.0 / (1.0 + settings.z_start);
    let a_end = 1.0 / (1.0 + settings.z_end);
    let ps = PowerSpectrum::normalized(&cosmo, N_S, SIGMA8, SIGMA8_RADIUS);

    let n = settings.particles_per_side;
    // The mesh is twice the particle count per side. With an even multiple, the lattice deposits uniformly.
    let mesh_n = 2 * n;
    let init = zeldovich(n, BOX, &cosmo, &ps, a_start, settings.seed);
    let split = settings
        .treepm
        .then(|| ForceSplit::with(BOX, mesh_n, n, settings.softening_scale, settings.theta));
    let mut sim = Simulation::new(cosmo, BOX, mesh_n, init.positions, init.momenta, a_start, split);

    let steps = settings.steps;
    let started = Instant::now();
    for i in 1..=steps {
        if control.stop.load(Ordering::Relaxed) {
            return;
        }
        // Evenly spaced in ln a, matching Simulation::run.
        let a1 = a_start * (a_end / a_start).powf(i as Real / steps as Real);
        sim.step_to(a1);
        let _ = frames.try_send(snapshot(&sim, control.want_grid.load(Ordering::Relaxed)));
        // Max peculiar speed is a quick check for numerical blow-up: it should stay bounded.
        let max_speed = sim
            .momenta
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() / sim.a)
            .fold(0.0, Real::max);
        println!(
            "step {i}/{steps}  z = {:.2}  max speed {:.3}  elapsed {:.1} s",
            1.0 / sim.a - 1.0,
            max_speed,
            started.elapsed().as_secs_f64()
        );
        let pause = control.pacing_ms.load(Ordering::Relaxed);
        if pause > 0 {
            std::thread::sleep(Duration::from_millis(pause as u64));
        }
    }
}

// Converts the simulation state to render units. Speed is the peculiar velocity |p| / a.
// Density comes from the mesh the last force step deposited, so no extra deposit is needed.
fn snapshot(sim: &Simulation, want_grid: bool) -> Frame {
    let delta = sim.delta();
    let e = sim.expansion_rate();
    Frame {
        redshift: (1.0 / sim.a - 1.0) as f32,
        positions: sim
            .positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect(),
        speeds: sim
            .momenta
            .iter()
            .map(|p| ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() / sim.a) as f32)
            .collect(),
        // CIC deposits only positive mass, so 1 + δ is never negative.
        densities: sim.positions.par_iter().map(|&x| (1.0 + delta.interpolate(x)).max(0.0)).collect(),
        rsd: sim.momenta.iter().map(|&p| rsd_shift(p, sim.a, e)).collect(),
        grid: want_grid.then(|| downsample_grid(delta)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_current_run() {
        let d = RunSettings::default();
        assert_eq!(d.particles_per_side, 40);
        assert_eq!(d.z_start, 500.0);
        assert_eq!(d.z_end, -0.9);
        assert!(d.treepm);
    }

    #[test]
    fn sanitize_keeps_defaults_and_clamps_bad_values() {
        assert_eq!(RunSettings::default().sanitized(), RunSettings::default());
        let bad = RunSettings {
            particles_per_side: 500,
            steps: 1,
            z_start: 0.0,
            z_end: 5.0,
            theta: 9.0,
            ..RunSettings::default()
        };
        let s = bad.sanitized();
        assert_eq!(s.particles_per_side, MAX_PARTICLES_PER_SIDE);
        assert_eq!(s.steps, 16);
        assert_eq!(s.z_start, 10.0);
        // The end must stay below the start.
        assert!(s.z_end < s.z_start);
        assert_eq!(s.theta, 1.0);
    }
}
