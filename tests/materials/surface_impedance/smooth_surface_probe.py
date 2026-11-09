"""Dump the compiled smooth-conductor surface paths for an exact cross-revision comparison.

Run as a script: ``python smooth_surface_probe.py OUT.pt``. It uses only public APIs that
predate the opt-in conductor-roughness feature, so the same probe runs unchanged on an
older revision and on the current tree; ``test_roughness_opt_in_bit_identity.py``
compares both dumps with exact tensor equality. Every configuration here carries no
roughness, so the dumps must agree bit for bit.
"""

from __future__ import annotations

import math
import sys

import torch

import witwin.maxwell as mw
from witwin.core import Box, Sphere
from witwin.maxwell.compiler.copper_foil import compile_copper_response
from witwin.maxwell.compiler.surface_impedance import fit_surface_impedance
from witwin.maxwell.fdtd.surface_impedance_reference import good_conductor_surface_impedance
from witwin.maxwell.scene import prepare_scene

_BAND = (0.5e9, 5.0e9)


def _put(out, key, value):
    if torch.is_tensor(value):
        out[key] = value.detach().cpu().clone()
    elif isinstance(value, bool):
        out[key] = torch.tensor(value)
    elif isinstance(value, int):
        out[key] = torch.tensor(value, dtype=torch.int64)
    elif isinstance(value, float):
        out[key] = torch.tensor(value, dtype=torch.float64)
    elif isinstance(value, complex):
        out[key] = torch.tensor(value, dtype=torch.complex128)


def _tree(out, prefix, value, depth=0):
    if depth > 6:
        return
    if torch.is_tensor(value) or isinstance(value, (bool, int, float, complex)):
        _put(out, prefix, value)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _tree(out, f"{prefix}.{key}", value[key], depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _tree(out, f"{prefix}[{index}]", item, depth + 1)
    elif isinstance(value, slice):
        _put(out, f"{prefix}.start", -1 if value.start is None else int(value.start))
        _put(out, f"{prefix}.stop", -1 if value.stop is None else int(value.stop))


def _rational_medium(order=6):
    freqs = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 120, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(50.0, freqs)).to(torch.complex128)
    model = mw.RationalSurfaceImpedance.fit(freqs, admittance, order=order, band=_BAND)
    return model, mw.SurfaceImpedanceMedium(impedance=model, name="coating")


def _metal_scene(material, *, device, geometry=None):
    dx = 0.005
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=8, strength=1.0, x=("pml", "pml")),
        device=device,
        sources=[mw.PointDipole(position=(-0.01, 0.0, 0.0), polarization=(0.0, 0.0, 1.0), width=2.0 * dx,
                                source_time=mw.CW(frequency=2.0e9, amplitude=40.0), name="pw")],
        structures=[mw.Structure(
            geometry=geometry if geometry is not None else Box(position=(0.012, 0.0, 0.0), size=(0.016, 0.04, 0.04)),
            material=material, name="metal")],
    )


def _foil_simulation(steps):
    port = mw.LumpedPort("feed", positive=(0., 0., .005), negative=(0., 0., -.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)))
    foil = mw.Structure(geometry=mw.Box(position=torch.tensor([.01, 0., 0.], dtype=torch.float64),
        size=torch.tensor([0., .02, .02], dtype=torch.float64)),
        material=mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9)), name="foil")
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3),
        grid=mw.GridSpec.uniform(.005), boundary=mw.BoundarySpec.none(),
        ports=[port], structures=[foil], device="cuda")
    return mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=steps),
        excitations=mw.PortExcitation("feed"), full_field_dft=True,
        # The pre-roughness baseline ran CPML; absorber="auto" would pick the graded
        # absorber (>= 40 layers) for this closed-transverse PML probe.
        absorber="cpml", spectral_sampler=mw.SpectralSampler(window="none"))


def _solver_tensors(out, prefix, solver):
    for name, value in sorted(vars(solver).items()):
        if torch.is_tensor(value) and value.numel() < 5_000_000:
            _put(out, f"{prefix}.{name}", value)
    _tree(out, f"{prefix}._surface_impedance", getattr(solver, "_surface_impedance", None))
    _tree(out, f"{prefix}._copper_foils", getattr(solver, "_copper_foils", None))


def collect() -> dict:
    out: dict[str, torch.Tensor] = {}
    freqs = torch.logspace(8.0, 11.0, 31, dtype=torch.float64)

    # Analytic smooth LossyMetalMedium helpers.
    metal = mw.LossyMetalMedium(conductivity=5.8e7, name="copper")
    _put(out, "lossy.z", torch.tensor([metal.surface_impedance_at_freq(float(f)) for f in freqs]))
    _put(out, "lossy.delta", torch.tensor([metal.skin_depth(float(f)) for f in freqs]))

    # Rational fits (public model fit and the compiler-layer Z-form fitter). The first
    # host fit in a fresh process can differ in the last ulp (allocator alignment of the
    # vectorized reductions), so a discarded warm-up fit precedes the recorded ones.
    _warm, _ = _rational_medium()
    fit_surface_impedance((_warm.sample_frequencies, _warm.surface_impedance(_warm.sample_frequencies)),
                          band=_BAND, order=6, dt=2.0e-12, representation="Z")
    model, rational = _rational_medium()
    _tree(out, "rational.model", {"poles": model.model.poles, "residues": model.model.residues,
                                  "direct": model.model.direct, "z": model.surface_impedance(freqs)})
    fitted = fit_surface_impedance((model.sample_frequencies, model.surface_impedance(model.sample_frequencies)),
                                   band=_BAND, order=6, dt=2.0e-12, representation="Z")
    _tree(out, "rational.zfit", {"A": fitted.discrete.A, "B": fitted.discrete.B, "C": fitted.discrete.C,
                                 "D": fitted.discrete.D, "poles": fitted.model.poles})

    # CopperFoil passive diffusion realization.
    foil = mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9))
    response = compile_copper_response(foil)
    _tree(out, "foil.response", {"rates": response.rates, "A": response.state_space.A,
                                 "B": response.state_space.B, "C": response.state_space.C,
                                 "D": response.state_space.D})
    _put(out, "foil.z", foil.surface_impedance(torch.linspace(.5e9, 2e9, 11, dtype=torch.float64)))

    # Compile-layer surface layouts (CPU scenes).
    for label, material, geometry in (
        ("box_lossy", mw.LossyMetalMedium(conductivity=50.0), None),
        ("box_rational", rational, None),
        ("sphere_lossy", mw.LossyMetalMedium(conductivity=50.0), Sphere(position=(0.008, 0.0, 0.0), radius=0.008)),
    ):
        scene = prepare_scene(_metal_scene(material, device="cpu", geometry=geometry))
        compiled = scene.compile_materials()
        layout = compiled["surface_impedance"]
        for index, face in enumerate(layout.faces):
            _tree(out, f"layout.{label}.face[{index}]", {
                "axis": face.axis, "side": face.metal_side == "high", "node": face.surface_node,
                "magnetic": face.magnetic_index, "slices": face.transverse_slices,
                "full": face.full_plane, "area": face.area, "mask": face.transverse_mask})
        for index, item in enumerate(layout.metals):
            _tree(out, f"layout.{label}.metal[{index}]", {
                "conductivity": item.conductivity, "slices": item.cell_slices,
                "mask": item.interior_node_mask})
        _tree(out, f"compiled.{label}", {key: value for key, value in compiled.items() if key != "surface_impedance"})

    # Prepared FDTD solvers and a short eager run (CUDA only).
    if torch.cuda.is_available():
        for label, material, geometry in (
            ("box_lossy", mw.LossyMetalMedium(conductivity=50.0), None),
            ("box_rational", _rational_medium()[1], None),
            ("sphere_lossy", mw.LossyMetalMedium(conductivity=50.0), Sphere(position=(0.008, 0.0, 0.0), radius=0.008)),
        ):
            def simulation():
                return mw.Simulation.fdtd(_metal_scene(material, device="cuda", geometry=geometry),
                                          frequencies=[2.0e9], run_time=mw.TimeConfig(time_steps=60),
                                          full_field_dft=True, absorber="cpml",
                                          spectral_sampler=mw.SpectralSampler(window="none"))
            _solver_tensors(out, f"prepared.{label}", simulation().prepare().solver)
            solver = simulation().run().solver
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                _put(out, f"run.{label}.{name}", getattr(solver, name))
        _solver_tensors(out, "prepared.foil", _foil_simulation(60).prepare().solver)
        result = _foil_simulation(60).run()
        solver = result.solver
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            _put(out, f"run.foil.{name}", getattr(solver, name))
        _tree(out, "run.foil.dft", result.fields)
        for name in ("voltage", "current"):
            _put(out, f"run.foil.port.{name}", getattr(result.port("feed"), name))
    out["__module__"] = torch.tensor(list(mw.__file__.encode()), dtype=torch.uint8)
    return out


if __name__ == "__main__":
    # Single-threaded host linear algebra keeps the rational fits reproducible run to run.
    torch.set_num_threads(1)
    torch.save(collect(), sys.argv[1])
