// Saves the run settings and appearance to `settings.json`, so they survive between runs.
// A missing or unreadable file falls back to the defaults, and the app still starts.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::appearance::Appearance;
use crate::runner::RunSettings;

pub const SETTINGS_FILE: &str = "settings.json";

// `serde(default)` lets a file from an older version load, filling in anything it lacks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Saved {
    pub run: RunSettings,
    pub appearance: Appearance,
}

impl Saved {
    // Sanitized values, so a hand-edited file can't break the run or the view.
    pub fn sanitized(&self) -> Self {
        Self {
            run: self.run.sanitized(),
            appearance: self.appearance.sanitized(),
        }
    }
}

// Loads from `path`. Any problem (missing file, bad JSON) gives the defaults, with a warning if the file existed.
pub fn load(path: &Path) -> Saved {
    match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Saved>(&text) {
            Ok(saved) => saved.sanitized(),
            Err(err) => {
                eprintln!("could not parse {}: {err}; using defaults", path.display());
                Saved::default()
            }
        },
        Err(_) => Saved::default(),
    }
}

pub fn save(path: &Path, saved: &Saved) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(saved).map_err(std::io::Error::other)?;
    std::fs::write(path, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let mut saved = Saved::default();
        saved.run.particles_per_side = 33;
        saved.run.treepm = false;
        saved.appearance.glow = true;
        saved.appearance.glow_strength = 2.5;
        let text = serde_json::to_string(&saved).expect("serialize");
        let back: Saved = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, saved);
    }

    #[test]
    fn missing_file_gives_defaults() {
        let path = std::env::temp_dir().join("cosmo_pm_missing_settings_test.json");
        let _ = std::fs::remove_file(&path);
        assert_eq!(load(&path), Saved::default());
    }

    #[test]
    fn bad_file_gives_defaults() {
        let path = std::env::temp_dir().join("cosmo_pm_bad_settings_test.json");
        std::fs::write(&path, "{ not json").expect("write test file");
        assert_eq!(load(&path), Saved::default());
        let _ = std::fs::remove_file(&path);
    }
}
