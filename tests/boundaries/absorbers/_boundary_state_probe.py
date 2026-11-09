"""Dump prepared boundary tensors and a short field run for one configuration.

Executed in a fresh interpreter by ``test_boundary_bit_identity.py`` against both
the current tree and an archived baseline tree. It touches only public API that
predates the absorbing-boundary extensions, so it runs unchanged on either tree.

Usage:
  python _boundary_state_probe.py <config> <output.pt>
  python _boundary_state_probe.py <config> <output.pt> save <checkpoint.pt>
  python _boundary_state_probe.py <config> <output.pt> resume <checkpoint.pt>

``save`` writes a disk resume checkpoint after ``_CHECKPOINT_STEP`` steps;
``resume`` continues such a checkpoint (possibly written by another tree) to the
full run length and records the final fields.
"""
from __future__ import annotations

import math
import sys

import torch

import witwin.maxwell as mw

# legacy subpixel pin; default coverage lives in tests/materials/compiler/test_subpixel_default_guard.py and tests/materials/compiler/test_kottke_edge.py
_LEGACY_SUBPIXEL = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase")

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_STATE_PREFIXES = ("cpml_", "psi", "boundary_", "inv_", "cex", "cey", "cez", "chx", "chy", "chz", "eps_", "mu_")


def _dipole_scene(boundary):
    scene = mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.35, 0.35), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=boundary,
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(geometry=mw.Box(position=(0.1, 0.0, 0.0), size=(0.2, 0.2, 0.2)), material=mw.Material(eps_r=3.0))
    )
    scene.add_source(
        mw.PointDipole(
            position=(-0.1, 0.05, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.5e9),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.2, 0.1, 0.05), fields=("Ez",)))
    return scene


def _bloch_scene():
    scene = mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(math.pi, 0.5 * math.pi, 0.0),
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            # Fixed-k Bloch runs accept CW sources only; see the broadband injection note.
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


CONFIGS = {
    "cpml_all": lambda: _dipole_scene(mw.BoundarySpec.pml(num_layers=6)),
    "cpml_pec": lambda: _dipole_scene(
        mw.BoundarySpec.faces(default="pml", num_layers=6, z_low="pec", y="pec")
    ),
    "bloch_xy_cpml_z": _bloch_scene,
    "mur_first_order": lambda: _dipole_scene(mw.BoundarySpec.mur()),
    "mur_cpml_mixed": lambda: _dipole_scene(mw.BoundarySpec.faces(default="pml", num_layers=6, x_high="mur")),
    "pmc_periodic": lambda: _dipole_scene(mw.BoundarySpec.faces(default="pml", num_layers=6, x="periodic", z_high="pmc")),
}


_RUN_STEPS = 80
_CHECKPOINT_STEP = 37


def _simulation(config):
    return mw.Simulation.fdtd(
        CONFIGS[config](),
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=_RUN_STEPS),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        # Frozen baselines ran CPML; the absorber='auto' default would pick the graded
        # absorber (>= 40 layers) for closed-transverse PML. Pin the baseline absorber.
        absorber="cpml",
        # Pin the legacy non-CFS CPML: complex fixed-k runs otherwise default to CFS.
        # A legacy pin for these short frozen-baseline runs, not a recommendation:
        # alpha_max = 0 is non-passive for long complex fixed-k runs (fdtd-cpml.md).
        cpml_config={"alpha_max": 0.0} if config == "bloch_xy_cpml_z" else None,
    )


def _fields(solver):
    return {
        name: _snapshot(getattr(solver, name))
        for name in _FIELDS + tuple(name + "_imag" for name in _FIELDS)
        if isinstance(getattr(solver, name, None), torch.Tensor)
    }


def save_checkpoint(config, output, checkpoint_path):
    checkpoint = _simulation(config).prepare().run_until(_CHECKPOINT_STEP)
    checkpoint.save(checkpoint_path)
    torch.save({"module_file": mw.__file__}, output)


def resume_checkpoint(config, output, checkpoint_path):
    restored = mw.FDTDResumeCheckpoint.load(checkpoint_path, map_location="cpu")
    prepared = _simulation(config).prepare()
    prepared.simulation._execute_fdtd_solve(prepared.solver, resume_from=restored)
    torch.save({"module_file": mw.__file__, "fields": _fields(prepared.solver)}, output)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    return value


def main(config, output):
    from witwin.maxwell.fdtd.resume import _configuration_fingerprint

    prepared = _simulation(config).prepare()
    solver = prepared.solver
    fingerprint = _configuration_fingerprint(solver)
    prepared_state = {
        name: _snapshot(value)
        for name, value in sorted(vars(solver).items())
        if isinstance(value, torch.Tensor) and name.startswith(_STATE_PREFIXES)
    }
    scalars = {
        name: value
        for name, value in sorted(vars(solver).items())
        if name.startswith("boundary_") and isinstance(value, (int, float, str, tuple))
    }
    scalars.update(
        {
            "dt": float(solver.dt),
            "has_pec_faces": bool(solver.has_pec_faces),
            "has_pml_faces": bool(solver.has_pml_faces),
            "has_mur_faces": bool(solver.has_mur_faces),
            "mur_faces": tuple(solver.mur_faces),
            "uses_cpml": bool(solver.uses_cpml),
            "complex_fields_enabled": bool(solver.complex_fields_enabled),
            "resolved_bloch_wavevector": tuple(solver.resolved_bloch_wavevector),
        }
    )
    mur = [
        {key: _snapshot(value) for key, value in entry.items()}
        for entry in getattr(solver, "_mur_state", ())
    ]
    prepared.run()
    fields = _fields(solver)
    mur_after = [
        {key: _snapshot(value) for key, value in entry.items()}
        for entry in getattr(solver, "_mur_state", ())
    ]
    torch.save(
        {
            "module_file": mw.__file__,
            "fingerprint": fingerprint,
            "prepared_state": prepared_state,
            "scalars": scalars,
            "mur": mur,
            "fields": fields,
            "mur_after": mur_after,
        },
        output,
    )


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[3] == "save":
        save_checkpoint(sys.argv[1], sys.argv[2], sys.argv[4])
    elif len(sys.argv) == 5 and sys.argv[3] == "resume":
        resume_checkpoint(sys.argv[1], sys.argv[2], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
