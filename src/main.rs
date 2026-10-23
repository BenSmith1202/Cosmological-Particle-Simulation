// Cosmological particle-mesh simulation: Zel'dovich initial conditions, TreePM gravity, live viewer.
// Pass `--no-viz` to run headless (prints progress only). Runs are configured in the viewer's settings panel (S).
// Pass `--settings <path>` to load and save settings somewhere other than settings.json (used for test runs).

mod appearance;
mod camera;
mod cosmology;
mod exposure;
mod fft;
mod frame;
mod glow;
mod ics;
mod integrator;
mod lens;
mod lens_fft;
mod mesh;
mod poisson;
mod real;
mod runner;
mod store;
mod tree;
mod viewer;
mod volume;

use std::path::PathBuf;

use runner::{Runner, BOX};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let headless = args.iter().any(|arg| arg == "--no-viz");
    let settings_path = settings_path(&args);

    if headless {
        // No window: run the saved settings to completion and print progress.
        let saved = store::load(&settings_path);
        Runner::start(saved.run).join();
    } else {
        // The viewer must own the main thread. It starts the first run, and closing the window ends the program.
        viewer::run(BOX as f32, settings_path);
    }
}

// The path after `--settings`, or settings.json in the working directory.
fn settings_path(args: &[String]) -> PathBuf {
    args.iter()
        .position(|a| a == "--settings")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(store::SETTINGS_FILE))
}
