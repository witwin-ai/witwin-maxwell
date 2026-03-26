# WiTwin Maxwell — Differentiable Electromagnetic Solver

[![PyPI](https://img.shields.io/pypi/v/witwin-maxwell)](https://pypi.org/project/witwin-maxwell/) [![Downloads](https://img.shields.io/pypi/dm/witwin-maxwell)](https://pypi.org/project/witwin-maxwell/) ![Code Size](https://img.shields.io/github/languages/code-size/Asixa/witwin-platform)[![License](https://img.shields.io/github/license/Asixa/witwin-platform)](COPYING)

WiTwin Maxwell is a differentiable full-wave electromagnetic solver with a **PyTorch-native interface** and **Slang-compiled CUDA kernels** at its core. The FDTD field-update loops run as hand-written GPU kernels compiled through [Slang](https://shader-slang.com/slang/), while the entire workflow — scene definition, simulation dispatch, result access, and automatic differentiation — stays inside standard PyTorch through `SceneModule`, `MaterialRegion`, and adjoint backward support.

The main public solver workflow today is:

- `FDTD`: Slang/CUDA Yee-grid time-domain solver with monitor extraction, multi-frequency DFT sampling, and differentiable adjoint support

Frequency-domain `FDFD` support is coming soon.

## Get Started

Python 3.10+ and an NVIDIA GPU are required.
This package depends on the base `witwin` package.

```bash
pip install witwin[maxwell]
```

## Public API

The main user-facing API is intentionally small:

- `Scene`: domain, grid, boundaries, structures, sources, monitors, ports, and differentiable material regions
- `Simulation`: runtime configuration through `Simulation.fdtd(...)`
- `Result`: structured field access, material tensors, monitor payloads, plotting, stats, and save support

For module-style inverse-design workflows, define a `SceneModule`, implement `to_scene()`, and pass that module directly into `Simulation`.

`Scene` stays declarative. Solver-sized Yee-grid coordinates, material tensors, and related runtime state are compiled during `Simulation.prepare()` / `Simulation.run()` and live on the internal solver scene rather than the public `Scene` object.

## Support Matrix

| Area | Currently supported | Notes |
| --- | --- | --- |
| Solvers | `Simulation.fdtd(...)` | FDTD supports time stepping and single- or multi-frequency DFT extraction. `Simulation.fdfd(...)` is coming soon. |
| Sources | `PointDipole`, `PlaneWave`, `GaussianBeam`, `ModeSource`, `TFSF` | `PlaneWave` / `GaussianBeam` support soft injection and `TFSF(...)`. `ModeSource` is still experimental. |
| Source time | `CW`, `GaussianPulse`, `RickerWavelet` | Shared waveform vocabulary across public source APIs. |
| Boundaries | `none`, `pml`, `periodic`, `bloch`, `pec`, `pmc` | Per-axis and per-face mixed layouts are available through `BoundarySpec.faces(...)`. |
| Materials | Isotropic `eps_r`, `mu_r`, `sigma_e`; `Debye`, `Drude`, `Lorentz`; `DiagonalTensor3`; `MaterialRegion` | `sigma_e` is the public frequency-domain conductivity path. `MaterialRegion` is the most direct differentiable design primitive. |
| Geometry | `Box`, `Sphere`, `Cylinder`, `Ellipsoid`, `Cone`, `Pyramid`, `Prism`, `Torus`, `HollowBox`, `Mesh` | Geometry and `Structure` primitives are re-exported through `witwin.maxwell`. |
| Monitors | `PointMonitor`, `PlaneMonitor`, `FluxMonitor`, `ModeMonitor` | Frequency selection is available through `Result.at(...)`. |
| Ports | `ModePort` | First-class modal port object; still experimental. |
| Results | `result.E`, `result.H`, `result.materials`, `Result.monitor(...)`, `Result.save(...)` | Structured field and material access stay torch-native. |
| Postprocess | Equivalent currents, Stratton-Chu propagation, near-to-far transform, directivity, bistatic RCS, S-parameters, modal overlap | Use `witwin.maxwell.postprocess`. |
| Differentiable workflows | `SceneModule`, `MaterialRegion`, supported trainable geometry parameters, FDTD adjoint backward | Public backward support currently targets trainable inputs that flow into the prepared-scene material tensors compiled from `Scene`. |

For the exhaustive user-visible capability inventory, see [`FEATURE_LIST.md`](FEATURE_LIST.md).

## Minimal Differentiable Example

The example below uses a point source and a dielectric cube, plots a vertical electric-field slice, and backpropagates to the cube position.

```python
import torch
import witwin.maxwell as mw

# Train the cube position directly.
box_x = torch.tensor(0.10, device="cuda", requires_grad=True)
box_position = torch.stack((box_x, box_x.new_tensor(0.0), box_x.new_tensor(0.06)))

# Build a minimal scene: one dielectric cube and one point dipole.
scene = mw.Scene(
    domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
    grid=mw.GridSpec.uniform(0.12),
    boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
    device="cuda",
    subpixel_samples=5,
)
scene.add_structure(
    mw.Structure(
        name="cube",
        geometry=mw.Box(position=box_position, size=(0.18, 0.18, 0.18)),
        material=mw.Material(eps_r=20.0),
    )
)
scene.add_source(
    mw.PointDipole(
        position=(0.0, 0.0, -0.06),
        polarization="Ez",
        width=0.04,
        source_time=mw.GaussianPulse(
            frequency=1.0e9,
            fwidth=0.25e9,
            amplitude=50.0,
        ),
    )
)
scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))

sim = mw.Simulation.fdtd(
    scene,
    frequencies=[1.0e9],
    run_time=mw.TimeConfig(time_steps=32),
    spectral_sampler=mw.SpectralSampler(window="none"),
    full_field_dft=True,
)

# Run the simulation and backpropagate from a probe value.
result = sim.run()
probe = result.monitor("probe")["data"]
loss = torch.abs(probe) ** 2
loss.backward()

print("probe =", probe)
print("loss =", float(loss.detach().item()))
print("d(loss)/d(box_x) =", box_x.grad)

# Plot a vertical field slice at y = 0.
result.plot.field(axis="y", position=0.0, component="abs", field_log_scale=True)
```

This example is intentionally minimal and does not require wrapping the scene in a class.

The current public backward path is for trainable inputs that contribute to compiled material tensors, such as `MaterialRegion.density` and supported trainable geometry parameters. Parameters that only affect source placement or other non-material branches are not yet part of the public adjoint path.

## Development and Validation

Common local commands:

```bash
python -m pytest tests
python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py
python -m pytest tests/boundaries/cpml/test_fdtd_cpml.py tests/monitors/observers/test_fdtd_observers.py
python -m pytest tests/gradients/test_fdtd_adjoint_bridge.py
python -m benchmark
python -m benchmark dipole_vacuum
python -m benchmark planewave_vacuum
```

Benchmark assets live under:

- `benchmark/scenes/`
- `benchmark/cache/`
- `benchmark/plots/`
- `benchmark/RESULTS.md`

## Current Notes

- Core Maxwell workflows are GPU/CUDA-first.
- `Simulation.fdfd(...)` is coming soon.
- Prefer `DiagonalTensor3` for anisotropic materials. Full rotated/off-diagonal `Tensor3x3` support is not implemented yet.
- The public differentiable path currently focuses on trainable inputs that affect compiled material tensors.
- `ModeSource`, `ModeMonitor`, and `ModePort` are available, but they are still marked experimental.

## License

GPL-3.0-or-later. See `COPYING` for the full license text.

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
