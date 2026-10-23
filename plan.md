# Plan: Laptop-scale Rust cosmological PM simulator

Source: `Conversation.md` (GPT chat export, 10/14/2026). Line refs below point into that file.

## 0. Status

- Project scaffold: `Cargo.toml`, `.gitignore`, and a placeholder `src/main.rs` (empty `main`, no development code yet).
- Crate versions in `Cargo.toml` were checked against crates.io on 10/14/2026.
- **Done: Rust toolchain.** rustup 1.29.1 via winget; stable `rustc 1.99.0` / `cargo 1.99.0`. MSVC Build Tools 2026 (14.51) and Windows SDK 10.0.26100 were already present.
- **Done: VS Code `rust-analyzer`** (v0.3.3081). `CodeLLDB` not installed (you chose rust-analyzer only).
- **Done: `cargo build` passes** after a Windows Defender exclusion for the `target\` folder (section 0a).

### 0a. Build blocker (resolved)

- Windows Defender blocked the `build-script-build.exe` files that cargo creates for each dependency with a build script.
- Resolved by a folder exclusion for `target\` added by the user in Windows Security.
- Verified: `cargo build` completes for all dependencies, including wgpu 30 and winit 0.30 (48 s, dev profile).

## 1. Goal and scope

Build a CPU-only Rust dark-matter N-body simulator in a periodic, expanding flat ΛCDM box. Start at z≈50, evolve to z=0 using Particle-Mesh (PM) gravity. Target hardware: the Core Ultra 7 255U laptop with 32 GB RAM, no CUDA.

Starting configuration, from the chat (`Conversation.md:1102-1134`):

- Box: 100 h⁻¹ Mpc, periodic
- Particles: 39³ = 59,319 (3× the earlier 27³; earlier 64³ was the laptop-scale start)
- PM grid: 78³ (twice the particle count per side)
- z: 500 → 0, 512 steps (the run was changed from z = 50 to make the cosmic time period 10× longer; the step count was kept)
- Initial conditions: Zel'dovich, with BBKS power spectrum

**Decided:** cosmology path, not the JS sandbox port. The JS code is not part of this project.

## 2. Why PM and not Barnes–Hut (for this project)

- Periodic boxes are PM's natural setting. The FFT handles periodicity directly. Barnes–Hut needs extra machinery for periodic images (`Conversation.md:1185-1194`).
- Easier to validate. PM has analytic checks (see Phase 3).
- **Tradeoff:** PM resolution is limited by the grid spacing. Halo interiors get smeared below about 1–2 cells (`Conversation.md:1093`). TreePM fixes this later.

## 3. Dependencies

`Cargo.toml` pins these (latest stable on crates.io, 10/14/2026). The chat suggested older versions (`Conversation.md:1461-1470`). The versions are consistent with each other:

| Crate | Version | Use |
|---|---|---|
| `rand` | 0.10 | random Gaussian field |
| `rand_distr` | 0.6 | Normal distribution (needs rand 0.10) |
| `rustfft` | 6.4 | 3D FFT, pure Rust, no vcpkg |
| `num-complex` | 0.4 | complex grid values |
| `rayon` | 1.12 | CPU parallelism |
| `wgpu` | 30 | GPU rendering for the viewer |
| `winit` | 0.30 | window and input (wgpu 30 and winit 0.30 both use raw-window-handle 0.6) |
| `pollster` | 1 | blocks on wgpu's async setup |
| `bytemuck` | 1 | casts vertex data to bytes |

- MSRV: wgpu 30 needs Rust 1.87. `Cargo.toml` sets `rust-version = "1.87"` and edition 2024.
- The chat says not to install FFTW or MKL yet (`Conversation.md:1321-1386`). Switch only if profiling shows the FFT is the bottleneck.
- **Guess, to confirm with `cargo build`:** that `wgpu` 30 and `winit` 0.30 link cleanly on Windows. I checked the dependency metadata, not a build.

## 4. Module layout

Start smaller than the chat's full layout (`Conversation.md:1198-1209`). Add files only when a phase needs them:

```text
src/
  main.rs          -- config, starts sim thread and viewer, timing output
  real.rs          -- `pub type Real = f64;` (one place to switch precision)
  cosmology.rs     -- H(a), growth factor D(a), BBKS transfer function, units
  particles.rs     -- positions/velocities, wrap to box
  mesh.rs          -- CIC deposit and interpolation
  poisson.rs       -- FFT Poisson solve, gradient in k-space
  fft.rs           -- 3D FFT on n³ grids (shared by Poisson and ICs)
  ics.rs           -- BBKS power spectrum, σ8 normalization, Zel'dovich ICs
  integrator.rs    -- KDK leapfrog in a, with cached acceleration
  viewer.rs        -- winit window, wgpu pipeline, camera (pan/zoom)
  shader.wgsl      -- instanced point quads, speed colour ramp
  io.rs            -- snapshot writer (binary), metadata
tests/             -- validation tests per phase
```

**Tradeoff (precision):** `Real` as a type alias costs nothing now and makes the f64 → f32 switch a one-line change. f64 keeps the cosmology integrals and long-time error clean. The laptop run is affordable in f64. Revisit with timing data.

## 5. Phases

Each phase ends with a check you can run. Don't start the next phase until the current check passes.

### Phase 0: Environment and skeleton
- Done: rustup installed, `cargo build` passes with wgpu and winit compiled.
- **Check:** project builds and runs an empty main. Passed (debug build; release not yet run).

### Phase 1: Cosmology background (complete)
- Implement `E(a) = H(a)/H0 = sqrt(Ωm/a³ + ΩΛ)` for flat ΛCDM. Use Ωm=0.315, ΩΛ=0.685, H0=67.4 (`Conversation.md:404-410`). Ignore radiation at z≤50 (`Conversation.md:412`).
- Implement the linear growth factor D(a) by numerical integration. Normalize it to D(1)=1.
- **Done.** Four unit tests pass (`cargo test`): E(1)=1 and H(1)=67.4, D(1)=1, D/a constant at early times, and D(0.5)=0.6067538.
- **Reference value:** from an independent Python integral (20000-point Simpson, same formula). Cross-checked against the Carroll–Press–Turner (1992) closed form, which gives 0.607620, about 0.14% off. The reference is hard-coded in the test with a comment.
- **Tests are inline** (`#[cfg(test)]`), not in `tests/`. This is a binary crate, so `tests/` can't import its modules. Revisit if we add a `lib.rs`.

### Phase 2: Mesh deposit and interpolation (complete)
- Cloud-in-cell (CIC) mass deposit onto the periodic grid. CIC interpolation back to particles.
- **Done.** Six unit tests pass (`cargo test`, 10 total with Phase 1).
- **Check wording changed:** the original "lattice gives zero net force" needs a force, which doesn't exist until Phase 3. In this phase the lattice check is that a particle at each cell centre deposits a uniform grid.
- **Checks:** total mass is conserved for 5000 scattered particles. A particle at a cell centre fills exactly one cell. A particle on the x = L face splits 50/50 across the periodic wrap. A constant field interpolates to that constant. Interpolation at a cell centre returns the cell value.

### Phase 3: FFT Poisson solver (complete, with one check deferred)
- Forward FFT of the density contrast δ. Divide by −k² in Fourier space. Inverse FFT for φ. Take the gradient with i·k to get the force field.
- Poisson equation in comoving units: `∇²φ = (3/2)·Ωm·δ / a`, in units where H0=1 and box length=1. **Derivation is from memory, not checked. Re-derive it before trusting absolute force amplitudes.** The chat's form `∇²Φ = 4πG a² ρ̄ δ` (`Conversation.md:610`) is the same equation in physical units.
- **Done.** Three unit tests pass (13 total in the crate). The prefactor was re-derived (see `src/poisson.rs` header), and the sign and prefactor are confirmed by the plane-wave test.
- **Checks done:** a plane wave δ = cos(2πx) gives the analytic force g = -(c/k0) sin(k0 x) to 1e-10 relative, with c = 1.5 Ωm / a. A constant δ gives zero force (k=0 dropped). Force scales as 1/a.
- **Not done: direct-sum comparison.** A brute-force direct sum must include periodic images, which needs an Ewald sum. Its own derivation and tests would be a separate piece of work, so I'm deferring this. Revisit before Phase 5 if the integrator's force needs an independent check.
- **Nyquist choice:** the Nyquist mode is zeroed in the gradient, since a real derivative can't be represented there (`src/poisson.rs`).

### Phase 4: Initial conditions (complete)
- **Done.** `src/ics.rs`. BBKS transfer function in `cosmology.rs`, power spectrum normalized to σ8 = 0.811 at R = 8 h⁻¹Mpc, Gaussian field, and Zel'dovich displacements and momenta.
- **Units:** length in h⁻¹Mpc, time in 1/H0. Box 100 h⁻¹Mpc (`Conversation.md:1114`). k is in h Mpc⁻¹, so the σ8 radius is 8 in code units.
- **Momentum:** p = a² dx/dt = a² f E D Ψ, derived from ẋ = f H D Ψ. The sign was checked by the growth test in Phase 5.
- **Tests:** BBKS limit; σ8 round-trip; f(1) ≈ Ωm^0.55 (within 0.01); measured P(k) matches the input in every shell with ≥ 200 modes, within 3σ (`measured_power_matches_input`).
- **Decision: σ8 = 0.811 and n_s = 0.965** are from memory of Planck 2018. Verify before any precision work.
- **Open:** the BBKS transfer has no baryon correction (dark matter only, as the chat recommends).

### Phase 5: Integrator (complete)
- **Done.** `src/integrator.rs`. KDK leapfrog in a, with kick and drift factors integrated numerically. Each step is split at the time midpoint a_m, found by bisection. `step_to` advances one step and caches the acceleration.
- **Equations:** p = a² dx/dt, dp/dt = g, dx = p da / (a³E), dp = g da / (aE). These follow from ẍ + 2Hẋ = -(1/a²)∇φ.
- **Tests (3 pass):**
  - *Total momentum:* stays below 1e-10 of the momentum scale, for a symmetric plane wave.
  - *Step convergence:* 10 vs 40 steps differ by less than 1e-3 of the box.
  - *Linear growth:* amplitude ratio over a = 0.02 → 0.1 matches D(a_f)/D(a_i) to within 1%. The sign is positive, so the Zel'dovich sign is right.
- **Growth ratio investigation.** The first attempt used 16 particles on a 32 mesh and gave +4.5%. A temporary diagnostic (since removed) found:

  | Particles | Mesh | Ratio error |
  |---|---|---|
  | 16 | 32 | +4.5% |
  | 16 | 64 | +6.6% (lattice sits on cell edges, so the deposit is non-uniform) |
  | 32 | 32 | −0.4% |
  | 32 | 64 | +1.9% |
  | 32 | 128 | +1.6% |
  | 64 | 128 | +0.4% |

  Time step (200 vs 40 steps) and amplitude (0.05 vs 0.5) made no difference, so the error is spatial. It tracks the particle count more than the mesh, which I did not expect. The test uses the 32 / 32 case.
  **Open:** I can't yet explain why the 64 and 128 meshes come out high. The sign goes against my estimate of CIC smoothing, which would make growth slower. I'd like to understand this before relying on absolute growth rates at high resolution.
- **Momentum check is a property check.** PM with CIC isn't exactly momentum-conserving, so I assert the residual is tiny only for this symmetric case.

### Phase 6: Visualization window (built, visual check pending)
- **Built.** `src/viewer.rs`, `src/shader.wgsl`. wgpu 30 + winit 0.30. The sim runs on a worker thread and sends frames through a 1-slot `sync_channel`. If the viewer is busy, the sim drops the frame and keeps going. The viewer runs on the main thread.
- **Points:** single pixels, drawn as a `PointList`. Earlier versions used instanced quads with a circle mask. Those cost four vertices per particle and gave no visual benefit at this density.
- **Speed colour:** |p| / a (peculiar speed), log-scaled, mapped through a 5-stop ramp (dark blue → purple → magenta → orange → yellow-white). The range auto-fits each frame. **F** freezes it.
- **Camera:** 2D x–y slice. Left-drag pans. Scroll zooms about the cursor. **Home** resets. **Esc** quits.
- **Tests (7 pass):** camera maps the viewport centre to the camera centre; zoom keeps the cursor point fixed; zoom-in scales correctly; drag direction is right; speed range handles extremes and uniform frames; the WGSL shader parses and validates with naga, with no GPU needed.
- **Runtime check:** a release build runs the viewer for 40 s with no panic. GPU setup succeeds on this machine, and frames flow (steps 1–113 reached).
- **Not checked:** I can't see the window from here. Whether the points look correct, and whether the colours and zoom feel right, needs a visual check on your side.
- **Timing, before optimization:** 64³ particles on a 128³ mesh took about 0.35 s per step in release.
- **Optimization pass (20k particles):**
  - Run configuration is now 27³ ≈ 20k particles on a 54³ mesh. The mesh is twice the particle count per side, which keeps the lattice deposit uniform.
  - Headless release run: 512 steps in 7.4 s (about 14 ms per step). Before the FFT and parallelism changes it was 9.3 s.
  - FFT lines run in parallel (rayon), and each worker reuses one scratch buffer. The old version allocated per line.
  - Force interpolation is parallel.
  - `[profile.dev] opt-level = 3`, so `cargo run` is fast too. Previously `cargo run` built unoptimized.
  - Points are one pixel each, not quads. This removes three quarters of the vertex work in the viewer.
- **Dependency added:** `naga = "30"` as a dev-dependency, for the shader test. It is the same naga version wgpu uses.
- **Warnings:** `Cosmology::hubble` and `Simulation::run` are unused in the binary (`run` is used by tests). Left in place for now.

### Phase 7: Output and inspection
- Binary snapshot: positions, velocities, a, box size, parameters. Save at z = 10, 5, 2, 1, 0 (`Conversation.md:1160`).
- Optional Python script (numpy/matplotlib, `Conversation.md:1416-1438`) to plot a density slice and P(k) from snapshots.
- **Check:** slices show filaments and voids growing with time. P(k) at z=0 is larger at small scales than at z=50.

### Phase 8: Timing on the laptop
- Run the 64³ / 128³ configuration in release mode, plugged in, in performance mode (`Conversation.md:1023`).
- Record per-step time and total wall time. **The chat's "minutes to an hour" is a guess** (`Conversation.md:1002`). Measure it.
- Try 128³ particles on a 256³ grid only if the 64³ run is correct and fast enough (`Conversation.md:1018-1025`).

### Phase 9: Later (not scheduled)
- 2LPT initial conditions (`Conversation.md:548-559`).
- CLASS/CAMB power-spectrum tables.
- Halo finder (Friends-of-Friends) (`Conversation.md:1170`).
- TreePM short-range force, or a CUDA port (`Conversation.md:1172-1179`). Only after Phases 0–8 are validated.

## 6. Validation strategy

- Unit tests in `tests/` for each phase check listed above.
- Keep a `validation.md` log: which check passed, with the numbers. Reproducibility matters more than speed here.
- Use a fixed random seed for initial conditions so runs are comparable.

## 7. Risks and things the chat got wrong or left open

- **Numbers are estimates, not measurements.** The GPU throughput and laptop runtime figures in the chat (`Conversation.md:95-101`, `1002`) are not benchmarks. Treat them as hypotheses.
- **Poisson prefactor and velocity factor** must be re-derived (Phases 3–4).
- **Small box limits.** A 100 h⁻¹ Mpc box with 64³ particles undersamples the largest modes. Results are qualitative, not precise (`Conversation.md:1089-1090`).
- **Windows GPU stack.** wgpu on the Intel iGPU is untested here. If the viewer fails to start, the sim should still run headless. Keep the viewer optional behind a flag.
- **Development tool.** You plan to build this with "GPT Luna" to save tokens (`Conversation.md:1490-1503`). This plan stands alone. Give it the whole file so the phase checks come along.

## 8. Viewer decisions to confirm

1. **2D slice first** (x–y, depth ignored), with 3D rotation later. Recommended.
2. **Speed scale:** log-scaled, auto-ranged to the current frame, or a fixed range you set. Recommended: auto-range with a key to freeze it.
3. **Viewer optional** behind a `--no-viz` flag, so the sim runs headless. Recommended.

## 9. Needs your approval before I install anything

Outside published Rust libraries, so I did not install these:

1. **Rust toolchain via rustup** (`https://rustup.rs`, or `winget install Rustlang.Rustup`). Required for any build. Approve the default stable toolchain?
2. **VS Code extension `rust-analyzer`.** Recommended for editing. Optional, but I'd install it.
3. **VS Code extension `CodeLLDB`.** Optional, for debugging.

Rust crates in `Cargo.toml` are published libraries, so `cargo build` can fetch them once rustup is installed.

### TreePM and f32 (added after Phase 8 was planned)
- **Particles:** 47³ = 103,823 (about 100k). Mesh 94³, twice the particle count per side.
- **Precision:** `Real = f32` everywhere. Test tolerances were loosened to f32 levels, and one real fix came out of it: `bbks_transfer` now uses `ln_1p`, since 1 + 2.34q rounds at small q.
- **Long range:** the PM potential is filtered by exp(−k² r_s²). `src/poisson.rs`.
- **Short range:** Barnes–Hut octree (`src/tree.rs`), θ = 0.5. Kernel S(r) = erfc(u) + (2u/√π)e^{−u²}, u = r/(2 r_s), times the Plummer-softened Newtonian force. Cutoff r_cut = 4.5 r_s. Minimum-image periodicity.
- **Parameters:** r_s = 1.25 mesh cells ≈ 1.33 h⁻¹Mpc. r_cut ≈ 6.0. Softening = 0.05 mean spacing ≈ 0.11.
- **Tests (33 pass):** tree with θ = 0 matches the brute-force pair sum; θ = 0.5 is within 2%; periodic pairs attract across the boundary; a two-particle TreePM force matches the Newtonian K/d² within 5%. That last one checks that the tree and mesh force normalizations agree.
- **Timing:** a full 512-step run (z = −0.9 end, current `Z_END`) takes about 192 s, roughly 0.37 s per step. Most of the cost is the short-range pair sum over about 100 neighbours per particle. A tabulated S(r) or a tighter cutoff would cut it.
- **Stability:** maximum peculiar speed follows the earlier curve (28.3 at z = −0.36, against 29.6 before). No blow-ups.

### Particle count and radiation (latest)
- **Particles:** 40³ = 64,000 (about 60% of the 104k run). Mesh 80³.
- **Radiation:** Ωr = 9.14e-5 (from memory; Ωr h² ≈ 4.15e-5 for photons plus massless neutrinos). Dark energy takes the remainder, so the universe stays flat. Expansion rate E(a) includes radiation.
- **Growth factor is now from the ODE, not the closed form.** The closed form D ∝ E∫da/(aE)³ is exact only for matter plus Λ. With radiation, it was 2% off the ODE over a = 0.02 → 0.1. The ODE starts at a = 10⁻³ from the Meszaros growing mode, D = 1 + 3y/2 with y = a/a_eq. Without radiation it reproduces the closed form to six digits (checked in Python).
- **Growth rate** comes directly from the ODE state, dδ/dlna. The earlier finite-difference version was wrong in f32, because the accumulated time drifted.
- **Reference values:** D(0.5)/D(1) = 0.6068044 with radiation (0.6067538 without).
- **Tests:** 33 pass. The "D ∝ a early" test uses a radiation-free cosmology, since with radiation D/a changes by a few percent between a = 0.01 and 0.02.
- **Run:** 118 s headless. Maximum speed stays bounded (18.3 at z = −0.9). The initial speeds are higher than before, since the growth rate changed.

### Orbital camera (latest)
- **Two views in one window.** Tab switches between the 2D x–y slice and the 3D orbit. The orbit starts active.
- **Orbit controls:** left-drag rotates, right-drag pans the centre, scroll zooms towards the centre. Home resets.
- **Choices:** perspective projection, 45° field of view. Zoom is towards the orbit centre, not the cursor, since cursor-anchored zoom is harder in 3D. There is no depth buffer, so draw order decides overlaps. The points are still single pixels.
- **Code:** camera maths is in src/camera.rs (matrices, orbit and slice cameras), with tests. The uniform is now a 4x4 view-projection matrix (80 bytes).
- **Tests:** 39 pass. Camera tests cover the target projecting to screen centre, eye distance, pitch clamping, zoom, and pan direction.
- **Runtime:** the viewer launches and runs for 30 s without panic. Visual check still needed.
- **Open:** depth-aware drawing, cursor-anchored zoom in 3D, and a depth-based fade if the overlap looks cluttered.
- **Colour ramp (latest):** dark blue (coldest) → purple → magenta → orange → yellow-white (fastest). Set in src/shader.wgsl.

### Settings panel (latest)
- **Toggle:** S opens and closes a floating panel over the view. Starts closed.
- **Settings:** particles per side (10–60), seed, start and end redshift, steps, pause per step, TreePM on/off, opening angle, softening.
- **Apply and restart:** stops the worker thread at its next step, and starts a fresh run with the new settings on a new thread. Settings are clamped to valid ranges first.
- **Input:** clicks and drags on the panel don't move the camera. egui consumes them first.
- **Code:** src/runner.rs (settings, worker thread, restart), src/viewer.rs (egui integration and panel). egui 0.36 is used because it matches wgpu 30 and winit 0.30.
- **Tests:** 41 pass. Settings clamping is tested. The panel isn't covered by tests.
- **Runtime:** the viewer launches and runs for 30 s with the panel code in place. Not yet checked by eye, and not yet checked by applying a restart.
- **Open:** pause and steps changes need a restart. Pause could be made live later.

### Colour threshold (latest)
- **Measured:** the speed distribution is concentrated between about 0.4 and 3 units, while the fastest particles reach 15–27. The old colour range started at the slowest particle, which put most of the bulk in the purple–magenta part of the ramp.
- **Change:** the cold end of the range is now the 10th-percentile speed, so the slowest 10% clamp to dark blue. The hot end is still the fastest particle. Set in src/viewer.rs as COLD_PERCENTILE.
- **Test:** the cold end is checked when 5% of the particles are very slow.

### Colour stops, persistence, and glow (latest)
- **Colour stops:** up to 8 stops, each a colour and a position as a fraction of the speed range (0 = coldest, 1 = fastest). Edited in the Colours tab, with a live strip preview. Passed to the shader as a uniform array, so colours change without a restart. Set in src/appearance.rs and src/shader.wgsl.
- **Persistence:** run settings and appearance are saved to settings.json in the working directory whenever they change, and loaded at start. Missing or bad files fall back to defaults. settings.json is in .gitignore. Uses serde and serde_json. Set in src/store.rs.
- **Glow and bloom (off by default):** points are added into an Rgba16Float scene, downsampled to 1/4 size, blurred with a 9-tap Gaussian (across, then down), and composited with a 1 - exp(-exposure x) tone map. Controls: strength, radius, exposure, in the Render tab. Set in src/glow.rs and src/post.wgsl.
- **Cost:** rendering only. The simulation thread is unaffected, and frames drop while the window is busy.
- **Bug found and fixed:** the downsample pass bound its target as an input, which wgpu rejects. Validated by running with glow on, then turned back off by default.
- **Tests:** 50 pass, including shader validation for both shaders and settings round-trip and bad-file fallback.
- **Open:** I haven't looked at the glow on screen yet, and I haven't tried the autosave with a real panel edit (no way to click from here). The first thing to check is whether the glow looks right at the default strength.

### Visuals rework (in progress)
- **Review findings:** downsample skips the inner 2x2 of each 4x4 block (bloom flicker); blur taps spaced by radius leave gaps (ghost copies); single-scale bloom; colour space mismatch (surface is Bgra8UnormSrgb); glow on/off use different pipelines; brightness depends on zoom; 1-px points shimmer; motion is steppy at about 4 sim frames/s.
- **Agreed plan:** F (colour space) -> A+B+C (always-HDR, soft splats, mip-chain bloom) -> E (frame interpolation) -> D (auto exposure, on by default, toggleable) -> G (hue-preserving tone map). Keep a 1-px point mode.
- **F done:** stops are linear RGB everywhere (shader, egui picker). Default ramp is written in sRGB and converted with srgb_to_linear in src/appearance.rs. The Colours-tab preview now encodes through egui::Rgba -> Color32, so it matches the render. Existing settings.json stops are unchanged; press reset colours to get the new defaults. 51 tests pass.
- **A, B, C done:**
  - A: particles always add into the Rgba16Float scene, and every frame is tone-mapped (1 - exp(-exposure x)). Bloom off just skips the bloom passes, so toggling it no longer changes the base image. The old alpha-blended path is gone.
  - B: point size 0 = one-pixel points (PointList). Otherwise each particle is an instanced quad (+-3 sigma) with a normalized Gaussian, so its total brightness is the same as a pixel point. Sigma is clamped to 0.6..3 px: below 0.6 the sum over pixel centres depends on sub-pixel position (tested: within 1% at 0.6, off by more than 10% at 0.3). Splats stay a fixed size on screen in orbit view.
  - C: mip-chain bloom (Jimenez 2014). 13-tap downsample from the scene through up to 7 half-size levels (stops below 8 px), then 3x3 tent upsample added into each larger level, smallest first. Composite divides by the level count. No Karis average, since it would dim isolated stars. Bloom spread is the tent spacing, clamped 0.5..2 texels so taps never leave gaps.
  - Defaults: point size 0.8, exposure 3.0 (splat peak is 1/(2 pi sigma^2) = 0.25), bloom strength 1, spread 1. Settings structs use serde(default), so older settings.json files load with new fields filled in.
  - Tests: 58 pass. New: splat normalization across sub-pixel positions, bloom level sizes, uniform layouts (224 and 32 bytes), point-size sanitizing, old-JSON loading.
  - Runtime: splats without bloom, splats with bloom, and pixel points with bloom each run 15 s without GPU validation errors. Not yet checked by eye.
- **Next:** E (frame interpolation), then D (auto exposure), then G (hue-preserving tone map).
- **E, D, G done:**
  - E (interpolation): each particle record holds the previous and current frame (32 bytes). The vertex shader blends by alpha = time since the current frame arrived / smoothed frame interval (30% exponential smoothing of arrival gaps), held at 1 until the next frame. lerp_periodic takes the short way across the box boundary. The colour range is interpolated too. A restart resets playback, since the new run's particles don't correspond. Display runs one interval behind the sim (measured 0.15-0.17 s at 64k particles).
  - D (auto exposure, on by default): a compute pass (meter.wgsl) takes the log-average of the brightest channel over lit texels (> 1e-4) of bloom level 0, measured before the upsample adds into it. The 8-byte result is copied to a MAP_READ buffer and mapped asynchronously (one in flight; polled, never waited on). The CPU eases exposure towards KEY / geometric-mean in log space with a 0.5 s time constant (frame-rate independent). KEY = 0.12, calibrated so the start of the default run matches the old manual exposure of 3 (the first guess, 0.3, measured as exposure 7.5). Exposure bias multiplies the auto value. Manual exposure remains when auto is off.
  - G (tone map): 1 - exp(-x) on the brightest channel, all channels scaled by the same factor. Hue and saturation are kept and nothing exceeds 1. Hot cores no longer bleach to white.
  - Tests: 70 pass, including periodic interpolation, alpha and interval timing, exposure target and adaptation, tone-map hue preservation, and naga validation of all three shaders.
  - Runtime: splats + bloom + auto, pixel points + auto, and splats with manual exposure each run 20 s without GPU errors. Meter readings return about 120 times per second with auto on, and none with it off. Not yet checked by eye.
- **Tone map toggle:** Render tab "Preserve hue" (default on). Off uses the per-channel 1 - exp(-x), so bright colours clip towards white. Passed to the composite in the spare slot of the Post uniform (still 32 bytes). 71 tests pass; both settings run without GPU errors.

### Layers (latest)
- **--settings <path>:** load and save another settings file. All runtime tests now use a temp file, so settings.json is never touched. Test scripts no longer force-close a running cosmo_pm.exe.
- **Colour by density:** the integrator keeps the δ mesh from each force step (no second deposit). Each frame carries 1 + δ per particle (CIC). Same ramp and percentile log range as speed, one range per quantity, both interpolated.
- **Redshift-space view:** each frame carries p / (a² E) per particle (peculiar velocity as a comoving distance). The shader adds its line-of-sight component times a strength (1 = physical), then wraps into the box. The axis is fixed (x/y/z, default y) rather than the camera direction, since fingers of God pointing at the camera can't be seen.
- **Depth fog:** orbit view only. Brightness x exp(-k (w - orbit distance) / L), clamped to [0, 4].
- **Volumetric gas:** emission-only ray march (no absorption, so order-independent) through the δ grid as an R16Float 3D texture, with repeat addressing because the field is periodic. The grid is only built while gas is on (want_grid flag to the sim thread), downsampled to at most 128³, transposed to x-fastest. Marched at half resolution with interleaved-gradient-noise jitter, then added into the HDR scene before bloom and the meter. Previous and current grids are blended by the same alpha as the particles. Emission is u² x ramp(u), u = position of log10(1 + δ) in [low, high], scaled by path length in box units.
- **Camera:** 4x4 inverse added. The slice projection now has a small depth slope (1e-4) so it can be inverted.
- **Particle record** is 64 bytes, view uniform 256 bytes. 80 tests pass. Gas at 40³, gas at 80³ (downsampled grid), and every layer at once run without GPU errors. Not yet checked by eye.
- **Note:** on integrated graphics, heavy rendering seemed to slow the sim (109 vs 145 steps in 25 s, one sample). The Render tab note was corrected.
- **Known limit:** gas is real-space density, so with the redshift-space view on, gas and particles are offset along the line of sight.

### Gravitational lensing (latest)
- **Method:** thin lens in screen space. Convergence κ ∝ ∫ δ dl along each pixel's ray (fs_kappa in volume.wgsl, 64x64, reusing the gas's density grids). Deflection α(θ) = (1/π) Σ κ (θ − θ')/|θ − θ'|² ΔA, by direct sum on the GPU (64⁴ ≈ 17M loads), with half-cell softening. Background drawn with the lens equation β = θ − α, then the particles add on top.
- **Background:** procedural distant galaxies (PCG-hashed cells: position, size, brightness, ellipticity, angle, colour). On cube faces of the view direction in orbit view (a sky at infinity), on a flat plane in the slice view.
- **Honest limits:** the physical deflection (≈30-50 arcsec for the biggest halos) would be under a pixel at the default view, so strength exaggerates it. The pattern is physical, the size is not. Single lens plane, mass outside the screen ignored, distance factors folded into strength.
- **Settings (Layers tab):** lensing on/off, strength, background brightness, show projected mass (warm = mass, blue = voids).
- **Tests:** 83 pass, including point-mass deflection falling as 1/r and pointing away from the mass, and zero deflection at the centre of a uniform sheet. Lensing alone, with the κ tint, and with gas plus bloom each run 20 s without GPU errors (built into target/check, since the user's copy was running). The slice-view lensing path is only validated by naga, not run (the view mode isn't a saved setting).
- **Lensing resolution slider:** 32-128 in steps of 16 (default 64), sizing both the projected-mass and deflection grids, rebuilt on change. Test: the same mass bends light by the same amount at two resolutions. Measured frame rate (iGPU, 64k particles): lensing off 238 fps; on at 32 / 64 / 128: 69 / 65 / 32 fps. About 10 ms of lensing is a fixed cost (most likely the full-resolution background galaxies, not profiled). Going past 128 would need FFT deflection and particle-based mass, since the density grid caps detail there.

### High-quality lensing: FFT + depth slabs (latest, off by default)
- **Mass from particles:** cs_bin projects each particle (real space, interpolated) into its depth slab's map with weight L²/(N·texel area at its depth), accumulated as fixed-point u32 atomics (2^18), since f32 blending is optional and f16 loses small additions.
- **Convergence per target:** cs_resolve subtracts the mean column (analytic ray length inside box ∩ slab) and weights lens slabs by r_l (r_s - r_l) / (r_s r_ref). Targets: slabs 1..S-1 (slab 0 has nothing in front) plus the background (weight r_l / r_ref). Slice view: background only, unweighted.
- **Deflection by FFT:** zero-padded to 2K, kernel FFT(Gx + i Gy) precomputed on the CPU (rebuilt on resolution change, or aspect change over 1%), one inverse FFT gives αx + i αy. Gaussian smoothing at one particle spacing on screen (the sim's real resolution; without it each particle would lens on its own). GPU Stockham radix-2 FFT (fft.wgsl), dynamic-offset stage uniforms, results back in buffer A.
- **Output:** Rgba16Float texture array, a layer per target: (αx, αy, κ = ½∇·α). The particle vertex shader moves each particle by its slab's layer (image = source + α). The background uses the last layer. Direct mode now also writes κ in the third channel, so the background shader is shared.
- **Settings:** Layers > lensing > "High quality (FFT, depth slabs)", resolution 64/128/256/512 (default 128), slabs 2-8 (default 8), "lens the particles too". 512 with 8 slabs fits the default 128 MiB storage-binding limit.
- **Tests (89 pass):** FFT deflection equals the direct sum including edges (CPU). Stockham equals rustfft (CPU). GPU FFT equals rustfft and round-trips (on the real GPU). End-to-end GPU chain: uniform lattice gives little bending (mean subtraction), a central clump bends outward on both sides, 1/r ratio within 20%, positive κ at the clump.
- **Measured (iGPU, 64k particles):** direct lensing 54 fps; high quality 128 / 256 / 512: 29 / 9.2 / 2.5 fps (about 4x per doubling, consistent with K² log K). No GPU errors in any mode, including with gas and bloom. Not checked by eye. Slice-view high-quality path not run (view mode isn't a setting).
- **Possible speed-up later:** a shared-memory FFT (each line's stages in workgroup memory) would cut global memory traffic, which likely dominates on an iGPU.

