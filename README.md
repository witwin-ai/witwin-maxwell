# WiTwin Maxwell — Differentiable Electromagnetic Solver

[![PyPI](https://img.shields.io/pypi/v/witwin-maxwell)](https://pypi.org/project/witwin-maxwell/) [![Downloads](https://img.shields.io/pypi/dm/witwin-maxwell)](https://pypi.org/project/witwin-maxwell/) ![Code Size](https://img.shields.io/github/languages/code-size/Asixa/witwin-platform) [![License](https://img.shields.io/badge/license-dual--licensed-blue)](https://witwin.ai/license)

WiTwin Maxwell is a differentiable full-wave electromagnetic solver with a **PyTorch-native interface** and a native CUDA FDTD runtime at its core. The FDTD field-update loops run as hand-written GPU kernels shipped in prebuilt platform wheels, while the entire workflow — scene definition, simulation dispatch, result access, and automatic differentiation — stays inside standard PyTorch through `SceneModule`, `MaterialRegion`, and adjoint backward support.

The public solver workflow is `Scene -> Simulation -> Result` for both runtimes:

- `FDTD`: native-CUDA Yee-grid time-domain solver with CPML, multi-frequency DFT sampling, material dispersion and nonlinearity, and differentiable adjoint support
- `FDFD` (limited/experimental): CUDA-only sparse single-frequency solver for linear isotropic or diagonal-electric media with `none`/PML boundaries. It does not yet match FDTD's material, boundary, nonuniform-grid, source, monitor, or adjoint coverage.

## Get Started

CPython 3.10-3.14, PyTorch 2.10 or newer, and an NVIDIA GPU are supported.
This package depends on the base `witwin` package.

```bash
pip install witwin-maxwell
```

## Prebuilt CUDA Support

Release wheels are built for Linux x86_64 and Windows x86_64 with CUDA 12.8. Each wheel carries one Python-independent native FDTD library using the LibTorch Stable ABI introduced for this surface in PyTorch 2.10. The same wheel and native binary are CI load-tested with PyTorch 2.10/cu128, 2.11/cu128, and 2.12/cu126 across CPython 3.10-3.14; no Torch-minor-specific binary selection or JIT rebuild is required. The fat binaries contain native code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, plus compute 12.0 PTX. This includes native coverage for RTX 2080-class Turing GPUs and current data-center and RTX/RTX PRO Blackwell families.

Linux wheels target `manylinux_2_35_x86_64`. The installed NVIDIA driver must support the CUDA 12.x runtime supplied by PyTorch; the CUDA toolkit is only needed for source/JIT builds.

For full CUDA 12.8 and Blackwell support, use at least driver 570.26 on Linux or 570.65 on Windows. Pre-Blackwell systems can use NVIDIA's CUDA 12.x minor-version compatibility floor (525.60.13 on Linux or 528.33 on Windows), subject to NVIDIA's compatibility-mode feature limits.

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
| Solvers | `Simulation.fdtd(...)`; limited `Simulation.fdfd(...)` | FDTD is the primary, full-featured runtime. FDFD is a CUDA-only sparse single-frequency path with isotropic/diagonal-electric, dispersive/conductive, `none`/PML, iterative/direct-solver, and basic adjoint support; nonuniform grids, magnetic/nonlinear/full-tensor media, in-domain PEC, symmetry, and broader FDTD parity are not implemented. Both return `Result`. |
| Sources | `PointDipole`, `PlaneWave`, `GaussianBeam`, `AstigmaticGaussianBeam`, `ModeSource`, `TFSF`, uniform/custom current and custom field sources | Soft and TFSF launch paths are available. `TFSF.slab(...)` supports any normal axis, CW or broadband waveforms, and periodic/Bloch grating layouts. `ModeSource` remains experimental. |
| Source time | `CW`, `GaussianPulse`, `RickerWavelet`, `CustomSourceTime` | Shared waveform vocabulary across public source APIs, including sampled or callable custom temporal signals. |
| Boundaries | `none`, `pml`, `periodic`, `bloch`, `pec`, `pmc` | Per-axis and per-face mixed layouts are available through `BoundarySpec.faces(...)`, including x/y Bloch plus z PML for grating FDTD workflows. |
| Materials | Isotropic and tensor electric/magnetic media; conductive, Debye, Drude, Lorentz, Sellmeier, gain, nonlinear, modulated, perturbation, custom dispersive, `Medium2D`, `Graphene`, and `LossyMetalMedium` | Compatible material effects compose in the same compiled Yee model. `MaterialRegion` is the primary differentiable design primitive; specialized combinations and adjoint limits are listed in `FEATURE_LIST.md`. |
| Geometry and grids | `Box`, `Sphere`, `Cylinder`, `Ellipsoid`, `Cone`, `Pyramid`, `Prism`, `Torus`, `HollowBox`, `Mesh`, `PolySlab`, custom/automatic/nonuniform grids, mesh overrides, and subpixel averaging | Shared geometry and `Structure` primitives are re-exported through `witwin.maxwell`; `Scene` owns device placement and compilation. |
| Monitors | Point, plane, finite-plane, flux, time-domain, material/permittivity, mode, diffraction, dipole-emission, and closed-surface monitors | Frequency selection is available through `Result.at(...)`; closed surfaces feed equivalent-current and near-to-far postprocessing. |
| Ports | `ModePort` | First-class modal port with S-parameter/modal-overlap workflows, including broadband, lossy, anisotropic, and bent-waveguide forward modes; still experimental. |
| Results | `result.E`, `result.H`, `result.materials`, `Result.monitor(...)`, `Result.save(...)` | Structured field and material access stay torch-native. |
| Postprocess | Equivalent currents, background-aware/curved-surface Stratton-Chu propagation, near-to-far transform, directivity, bistatic RCS, S-parameters, and modal overlap | Use `witwin.maxwell.postprocess`. |
| Differentiable workflows | `SceneModule`, `MaterialRegion`, trainable material/geometry/source inputs, native-CUDA FDTD adjoint backward | Native reverse kernels cover standard/CPML/Bloch fields and supported conductive, dispersive, anisotropic, nonlinear, TFSF, and multi-source compositions. Explicit capability guards reject unsupported gradients. |
| Interoperability | Tidy3D scene export and GDS geometry import | Tidy3D export covers grids, boundaries, common geometry, broad material/source/monitor families, and validated SI/unit-convention conversions. |

For the exhaustive user-visible capability inventory, see [`FEATURE_LIST.md`](FEATURE_LIST.md). The numerical conventions and lessons from the Maxwell-vs-Tidy3D validation campaign are recorded in [`docs/validation/tidy3d-numerical-alignment-0.3.0.md`](docs/validation/tidy3d-numerical-alignment-0.3.0.md).

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

Benchmark assets live under `benchmark/scenes/`, `benchmark/cache/`, `benchmark/plots/`, and `benchmark/RESULTS.md`.

The release validation suite combines unit/API tests, native CUDA parity and adjoint tests, and Maxwell-vs-Tidy3D numerical comparisons. Cross-solver comparisons use common physical-domain coordinates, external PML on both sides, coordinate-aligned complex field slices, and solver-independent scalar observables. See the [0.3.0 numerical-alignment notes](docs/validation/tidy3d-numerical-alignment-0.3.0.md) for the conventions, visual diagnosis workflow, and known residuals.

## Current Notes

- Core Maxwell workflows are GPU/CUDA-first.
- FDFD is available but remains limited/experimental. It is suitable for supported linear single-frequency scenes; use FDTD for nonuniform grids, magnetic or nonlinear media, full off-diagonal anisotropy, in-domain PEC, symmetry, and the broader source/monitor/adjoint feature set.
- `bloch_wavevector="auto"` is supported for fixed-angle CW TFSF grating slabs; broadband automatic Bloch phase requests are rejected.
- Full off-diagonal `Tensor3x3` electric anisotropy is supported by FDTD, including supported CPML, dispersion, conduction, and adjoint compositions; use `DiagonalTensor3` when the material is naturally diagonal because it is cheaper.
- `LossyMetalMedium` is a narrowband, normal-incidence planar SIBC model. Curved, oblique, laterally finite, and adjoint SIBC workflows require a volumetric material model or a future generalized surface operator.
- The public differentiable path covers supported trainable material, geometry, and source inputs. Runtime capability checks reject combinations without a physically implemented reverse channel.
- `ModeSource`, `ModeMonitor`, and `ModePort` are available, but they are still marked experimental.

## License

Witwin Maxwell is available under a dual-license model for academic and
non-commercial research use or commercial and enterprise use. See the
[Witwin licensing page](https://witwin.ai/license) for the applicable terms.

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
