"""Resume fingerprints and short source-driven field runs for bit-identity checks.

Executed in a fresh interpreter by ``test_bit_identity_baseline.py`` against both
the current tree and an archived baseline tree. It uses only public API that the
baseline already had, so it runs unchanged on either tree.

Usage:
  python _bit_identity_probe.py fingerprints <output.pt>
  python _bit_identity_probe.py fields <output.pt>
"""
from __future__ import annotations

import math
import sys

import torch

import witwin.maxwell as mw

# legacy subpixel pin; default coverage lives in tests/materials/compiler/test_subpixel_default_guard.py and tests/materials/compiler/test_kottke_edge.py
_LEGACY_SUBPIXEL = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase")

_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_FREQUENCY = 1.0e9


def _pulse():
    return mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.5 * _FREQUENCY)


def _field_pulse():
    # A non-power-of-two amplitude makes every float32 regrouping of the
    # amplitude product observable in the injected samples.
    return mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.5 * _FREQUENCY, amplitude=5.3)


def _ricker():
    return mw.RickerWavelet(frequency=_FREQUENCY, amplitude=5.3)


def _box_scene(boundary, *, subpixel=1, material=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.35, 0.35), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=boundary,
        subpixel_samples=mw.SubpixelSpec(samples=subpixel, averaging="arithmetic", pec="staircase"),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.1, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
            material=material or mw.Material(eps_r=3.0),
        )
    )
    return scene


def _dipole(scene, source_time=None, polarization="Ez"):
    scene.add_source(
        mw.PointDipole(position=(-0.1, 0.05, 0.0), polarization=polarization, width=0.05,
                       source_time=source_time or _pulse())
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.2, 0.1, 0.05), fields=("Ez",)))
    return scene


def _bloch_scene():
    scene = mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=mw.BoundarySpec.faces(
            default="pml", num_layers=4, strength=1.0, x="bloch", y="bloch", z="pml",
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


def _open_scene():
    return mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        device="cuda",
    )


def _soft_plane_wave(source_time):
    scene = _open_scene()
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                  source_time=source_time))
    scene.add_monitor(mw.PointMonitor("probe", (0.05, 0.0, 0.1), fields=("Ex",)))
    return scene


def _tfsf_plane_wave(source_time):
    scene = _open_scene()
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                  source_time=source_time,
                                  injection=mw.TFSF(bounds=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)))))
    scene.add_monitor(mw.PointMonitor("probe", (0.05, 0.0, 0.1), fields=("Ex",)))
    return scene


def _gaussian_beam(source_time):
    scene = _open_scene()
    scene.add_source(mw.GaussianBeam(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                     beam_waist=0.2, focus=(0.0, 0.0, 0.0), source_time=source_time))
    scene.add_monitor(mw.PointMonitor("probe", (0.05, 0.0, 0.1), fields=("Ex",)))
    return scene


def _mode_source(source_time):
    scene = _open_scene()
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(2.0, 0.2, 0.3)),
                                     material=mw.Material(eps_r=4.0)))
    scene.add_source(mw.ModeSource(position=(-0.25, 0, 0), size=(0, 0.6, 0.6), polarization="Ez",
                                   source_time=source_time))
    scene.add_monitor(mw.PointMonitor("probe", (0.1, 0.0, 0.0), fields=("Ez",)))
    return scene


def _with_monitors(scene):
    scene.add_monitor(mw.PlaneMonitor("plane", axis="z", position=0.0, fields=("Ex", "Ez")))
    scene.add_monitor(mw.FluxMonitor("flux", axis="x", position=0.25))
    scene.add_monitor(mw.FieldTimeMonitor("trace", components=("Ez",), position=(0.0, 0.0, 0.1)))
    return scene


FINGERPRINT_CONFIGS = {
    # The six boundary-probe configurations.
    "cpml_all": lambda: _dipole(_box_scene(mw.BoundarySpec.pml(num_layers=6))),
    "cpml_pec": lambda: _dipole(_box_scene(mw.BoundarySpec.faces(default="pml", num_layers=6, z_low="pec", y="pec"))),
    "bloch_xy_cpml_z": _bloch_scene,
    "mur_first_order": lambda: _dipole(_box_scene(mw.BoundarySpec.mur())),
    "mur_cpml_mixed": lambda: _dipole(_box_scene(mw.BoundarySpec.faces(default="pml", num_layers=6, x_high="mur"))),
    "pmc_periodic": lambda: _dipole(_box_scene(mw.BoundarySpec.faces(default="pml", num_layers=6, x="periodic", z_high="pmc"))),
    # Broader scene features: material channels, subpixel path, sources, monitors.
    "subpixel_dispersive": lambda: _dipole(_box_scene(
        mw.BoundarySpec.pml(num_layers=6), subpixel=3,
        material=mw.Material(eps_r=2.0, sigma_e=0.01,
                             lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=2e9, gamma=1e8),)),
    )),
    "anisotropic": lambda: _dipole(_box_scene(
        mw.BoundarySpec.pml(num_layers=6),
        material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0)),
    )),
    "pec_structure": lambda: _dipole(_box_scene(mw.BoundarySpec.pml(num_layers=6), material=mw.Material(pec=True))),
    "cw_dipole_monitors": lambda: _with_monitors(_dipole(_box_scene(mw.BoundarySpec.pml(num_layers=6)),
                                                         source_time=mw.CW(frequency=_FREQUENCY, amplitude=2.0))),
    "soft_plane_wave": lambda: _soft_plane_wave(_pulse()),
    "tfsf_plane_wave": lambda: _tfsf_plane_wave(_ricker()),
    "gaussian_beam": lambda: _gaussian_beam(_pulse()),
    "mode_source": lambda: _mode_source(_pulse()),
}

FIELD_CONFIGS = {
    "bloch_tfsf_pulse": _bloch_scene,
    "soft_plane_wave_gaussian": lambda: _soft_plane_wave(_field_pulse()),
    "soft_plane_wave_ricker": lambda: _soft_plane_wave(_ricker()),
    "gaussian_beam_gaussian": lambda: _gaussian_beam(_field_pulse()),
    "gaussian_beam_ricker": lambda: _gaussian_beam(_ricker()),
    "mode_source_gaussian": lambda: _mode_source(_field_pulse()),
    "mode_source_ricker": lambda: _mode_source(_ricker()),
    "tfsf_plane_wave_gaussian": lambda: _tfsf_plane_wave(_field_pulse()),
    "tfsf_plane_wave_ricker": lambda: _tfsf_plane_wave(_ricker()),
    "dipole_gaussian": lambda: _dipole(_box_scene(mw.BoundarySpec.pml(num_layers=6)), source_time=_field_pulse()),
    "dipole_cw": lambda: _dipole(_box_scene(mw.BoundarySpec.pml(num_layers=6)),
                                 source_time=mw.CW(frequency=_FREQUENCY, amplitude=2.3)),
}


def _prepare(factory, steps=80):
    return mw.Simulation.fdtd(
        factory(),
        frequencies=[_FREQUENCY],
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        # Frozen baselines ran CPML; the absorber='auto' default would pick the graded
        # absorber (>= 40 layers) for closed-transverse PML. Pin the baseline absorber.
        absorber="cpml",
        # Pin the legacy non-CFS CPML: complex fixed-k runs otherwise default to CFS.
        # A legacy pin for these short frozen-baseline runs, not a recommendation:
        # alpha_max = 0 is non-passive for long complex fixed-k runs (fdtd-cpml.md).
        cpml_config={"alpha_max": 0.0} if factory is _bloch_scene else None,
    ).prepare()


def fingerprints(output):
    from witwin.maxwell.fdtd.resume import _configuration_fingerprint

    torch.save(
        {
            "module_file": mw.__file__,
            "fingerprints": {name: _configuration_fingerprint(_prepare(factory).solver)
                             for name, factory in FINGERPRINT_CONFIGS.items()},
        },
        output,
    )


def fields(output):
    payload = {}
    for name, factory in FIELD_CONFIGS.items():
        prepared = _prepare(factory, steps=160)
        prepared.run()
        solver = prepared.solver
        payload[name] = {
            field: getattr(solver, field).detach().to("cpu", copy=True)
            for field in _FIELDS + tuple(field + "_imag" for field in _FIELDS)
            if isinstance(getattr(solver, field, None), torch.Tensor)
        }
    torch.save({"module_file": mw.__file__, "fields": payload}, output)


if __name__ == "__main__":
    {"fingerprints": fingerprints, "fields": fields}[sys.argv[1]](sys.argv[2])
