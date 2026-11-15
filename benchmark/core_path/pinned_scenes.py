"""Gate scenes frozen at the definition their G1/G4 frozen-store payloads were captured on.

Most gate scenes that come from the benchmark registry are built from it live
(``build_core_case``), guarded by ``gate_scene_fingerprints.json``. A scene whose registry
definition was re-specified for benchmark reference accuracy is frozen here instead, as a
self-contained builder that does not read the registry or its helpers, so that later re-specs
cannot reach it. validation/bench-references (1182f3c19) moved four gate scenes:

- ``full_tensor_slab``: dx/4 grid with 32 PML layers (69^3 -> 270^3 nodes; the G1 run no
  longer fits the 8 GiB slot cap).
- ``mode_source_higher_order``: dx/2 grid (69^3 -> 120^3 nodes); validation/bench-respec
  re-specifies it again, to an anisotropic grid with 16 PML layers.
- ``dipole_vacuum`` and ``dipole_ey``: zero-DC sine-carrier pulses (``phase=pi/2``).

Each builder below is the registry definition at 12afa2626 (the parent of that merge), written
with the immutable Scene constructor.
Each one returns ``(scene, frequencies, absorber)``.
"""

from __future__ import annotations

import witwin.maxwell as mw

# benchmark/scenes/_common.py and planned.py constants at 12afa2626.
DX = 0.025
HALF_SPAN = 0.64
PML_LAYERS = 8
FREQUENCIES = (2.0e9,)
DIPOLE_FREQUENCIES = (1.5e9,)


def _scene(*, structures=(), sources=(), monitors=(), frequencies,
           subpixel=mw.SubpixelSpec(samples=3, averaging="polarized", pec="staircase")) -> mw.Scene:
    """The benchmark base scene (25 mm grid, 8 PML layers) with the plot monitors appended last."""
    plots = tuple(mw.PlaneMonitor(name=f"plot_field_{axis}", axis=axis, position=0.0, fields=("Ex", "Ey", "Ez"),
                                  frequencies=frequencies) for axis in ("x", "y", "z"))
    return mw.Scene(
        domain=mw.Domain(bounds=((-HALF_SPAN, HALF_SPAN),) * 3),
        grid=mw.GridSpec.uniform(DX),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        structures=structures, sources=sources, monitors=(*monitors, *plots),
        subpixel=subpixel,
    )


def _pulse() -> mw.GaussianPulse:
    return mw.GaussianPulse(frequency=FREQUENCIES[0], fwidth=0.5e9)


def _full_tensor_slab():
    tensor = mw.media.Tensor3x3(((3.0, 0.2, 0.0), (0.2, 2.5, 0.0), (0.0, 0.0, 2.0)))
    scene = _scene(
        # The slab runs through the transverse PML (4 x HALF_SPAN), as in the registry's _plane_scene.
        structures=(mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(4 * HALF_SPAN, 4 * HALF_SPAN, 0.1)),
                                 material=mw.Material(epsilon_tensor=tensor), name="sample"),),
        sources=(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(0.9436283192, 0.3310069414, 0.0),
                              source_time=_pulse()),),
        monitors=(
            mw.PlaneMonitor("field", axis="y", position=0.0, fields=("Ex",), frequencies=FREQUENCIES),
            mw.FluxMonitor("reflected", axis="z", position=-0.3, frequencies=FREQUENCIES, normal_direction="-"),
            mw.FluxMonitor("transmitted", axis="z", position=0.3, frequencies=FREQUENCIES),
        ),
        frequencies=FREQUENCIES,
        subpixel=mw.SubpixelSpec(samples=3, averaging="arithmetic"),
    )
    return scene, FREQUENCIES, None


def _mode_source_higher_order():
    plane = dict(size=(0, 0.525, 0.375))
    scene = _scene(
        structures=(mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(2 * HALF_SPAN, 0.30, 0.15)),
                                 material=mw.Material(eps_r=4.0), name="waveguide"),),
        sources=(mw.ModeSource(position=(-0.35, 0, 0), **plane, mode_index=1, polarization="Ez",
                               source_time=_pulse(), name="mode"),),
        monitors=(
            mw.FinitePlaneMonitor("field", position=(0.20, 0, 0), **plane, fields=("Ex", "Ey", "Ez"),
                                  frequencies=FREQUENCIES),
            mw.ModeMonitor("mode_out", position=(0.35, 0, 0), **plane, mode_index=1, polarization="Ez",
                           frequencies=FREQUENCIES),
            mw.FluxMonitor("reflected", axis="x", position=-0.5, frequencies=FREQUENCIES, normal_direction="-"),
            mw.FluxMonitor("transmitted", axis="x", position=0.5, frequencies=FREQUENCIES),
        ),
        frequencies=FREQUENCIES,
    )
    return scene, FREQUENCIES, None


def _dipole(polarization: str, flux_monitors):
    # A cosine-carrier pulse (phase 0); the registry rows moved to a sine carrier.
    dipole = mw.PointDipole(position=(0.0, 0.0, 0.0), polarization=polarization, profile="ideal",
                            source_time=mw.GaussianPulse(frequency=1.5e9, fwidth=0.5e9))
    field = mw.PlaneMonitor(name="field_xy", axis="z", position=0.0, fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                            frequencies=DIPOLE_FREQUENCIES)
    scene = _scene(sources=(dipole,), monitors=(field, *flux_monitors), frequencies=DIPOLE_FREQUENCIES)
    return scene, DIPOLE_FREQUENCIES, None


def _dipole_vacuum():
    return _dipole("Ez", (
        mw.FluxMonitor(name="flux_z_pos", axis="z", position=0.3, frequencies=DIPOLE_FREQUENCIES),
        mw.FluxMonitor(name="flux_z_neg", axis="z", position=-0.3, frequencies=DIPOLE_FREQUENCIES,
                       normal_direction="-"),
    ))


def _dipole_ey():
    return _dipole("Ey", (mw.FluxMonitor(name="flux_z", axis="z", position=0.3, frequencies=DIPOLE_FREQUENCIES),))


PINNED_SCENES = {
    "dipole_vacuum": _dipole_vacuum,
    "dipole_ey": _dipole_ey,
    "full_tensor_slab": _full_tensor_slab,
    "mode_source_higher_order": _mode_source_higher_order,
}
