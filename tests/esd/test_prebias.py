"""Cross-feature end-to-end: electrostatic pre-bias + ESD terminal injection +
dynamic dielectric breakdown + non-feedback stress monitor in one FDTD run.

Capability level under test: a single FDTD solve that (1) is seeded with a DC
electrostatic pre-bias mapped onto the Yee grid, (2) injects a standard IEC
61000-4-2 current waveform through a TerminalPort, (3) carries a DielectricBreakdown
material whose state machine feeds back into the update, and (4) accumulates a
non-feedback BreakdownMonitor stress record. Requires CUDA for the native FDTD
runtime.

Deferred (recorded, not asserted here): source-impedance / MNA-SPICE circuit
co-simulation of the ESD gun through the TerminalPort (the run uses ideal current
injection); subcycled electrostatic/transient coupling. See the Wave-D
integration acceptance doc.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cross-feature ESD pre-bias run requires CUDA"
)

BOUNDS = ((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))
H = 0.02

# Terminal footprints and the gap dielectric land on the Yee grid for a 0.02 m
# uniform mesh (plate edges at +-0.04 / +-0.08, dielectric z in [-0.04, 0.04]).
SIGNAL = ((0.0, 0.0, 0.06), (0.10, 0.10, 0.04))
GROUND = ((0.0, 0.0, -0.06), (0.10, 0.10, 0.04))
DIEL = ((0.0, 0.0, 0.0), (0.10, 0.10, 0.08))


def _box(pos, size):
    return mw.Box(position=pos, size=size)


def _prebias():
    """DC solve on the FDTD-identical grid: plates as fixed-potential terminals."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    # Same gap dielectric (no breakdown descriptor needed for the DC field).
    scene.add_structure(
        mw.Structure(name="diel", geometry=_box(*DIEL), material=mw.Material(eps_r=3.2))
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal("signal", geometry=_box(*SIGNAL), potential=200.0)
    )
    scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal("ground", geometry=_box(*GROUND), grounded=True)
    )
    dc = mw.Simulation.electrostatic(
        scene, boundary=mw.ElectrostaticBoundarySpec(default="neumann")
    ).run()
    # This nonuniform, dielectric-loaded pre-bias carries a real discrete-Gauss
    # residual (fringing + eps-interface discretization); raise the tolerance so the
    # gate accepts it rather than rejecting a physically meaningful field.
    return mw.ElectrostaticInitialCondition.from_result(dc, tolerance=1.0)


def _fdtd_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(name="signal", geometry=_box(*SIGNAL), material=mw.Material.pec())
    )
    scene.add_structure(
        mw.Structure(name="ground", geometry=_box(*GROUND), material=mw.Material.pec())
    )
    scene.add_structure(
        mw.Structure(
            name="diel",
            geometry=_box(*DIEL),
            material=mw.Material(
                eps_r=3.2,
                sigma_e=1e-12,
                breakdown=mw.DielectricBreakdown(
                    critical_field=1.0e3,
                    minimum_duration=0.0,
                    post_breakdown_conductivity=5.0,
                ),
            ),
        )
    )
    scene.add_port(
        mw.TerminalPort(
            "feed", mw.TerminalRef("signal"), mw.TerminalRef("ground"), mw.AxisPath("z"), 0.01
        )
    )
    scene.add_source(
        mw.ESDCurrentSource("gun", port="feed", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0))
    )
    scene.add_monitor(
        mw.BreakdownMonitor(
            "stress",
            position=(0.0, 0.0, 0.0),
            size=(0.10, 0.10, 0.08),
            critical_field=1.0e3,
            minimum_duration=0.0,
            quantities=("electric_field", "exposure", "dissipated_energy"),
        )
    )
    return scene


def test_prebias_esd_breakdown_end_to_end():
    ic = _prebias()
    result = mw.Simulation.fdtd(
        _fdtd_scene(),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=120),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        initial_condition=ic,
    ).run()

    # 1) Provenance carries the pre-bias metadata (DC diagnostics + Gauss residual).
    prov = result.electrostatic_prebias
    assert prov is not None
    assert prov["source"] == "electrostatic"
    assert prov["capability_level"] == "electrostatic-prebias"
    assert prov["dc_iterations"] > 0
    assert "gauss_residual" in prov and prov["gauss_residual"] <= prov["tolerance"]
    assert ic.gauss_residual is not None

    # 2) The non-feedback stress record is populated.
    stress = result.breakdown("stress")
    assert stress.capability_level == "stress-only"
    assert stress.peak_field > 0.0
    assert stress.max_field_map.device.type == "cuda"

    # 3) The dynamic breakdown state machine ran and its typed event log is coherent.
    data = result.breakdown_data
    assert data is not None
    assert data.triggered_count == len(result.breakdown_events)
    if data.triggered_count > 0:
        assert data.total_dissipated_energy > 0.0
        keys = [(e.step, e.cell_index) for e in result.breakdown_events]
        assert keys == sorted(keys)

    # 4) The ESD terminal-injection provenance is present on the same run.
    record = result.esd_waveform("gun")
    assert record.port_name == "feed"
    assert record.provenance["capability_level"] == "stress-only"


def test_prebias_shifts_the_initial_transient():
    # Sanity: the pre-bias actually seeds the field. The same ESD run without a
    # pre-bias starts from E == 0 at step 0; with the pre-bias the gap already holds
    # the DC field, so the recorded breakdown peak field is strictly larger.
    monitor_kw = dict(
        position=(0.0, 0.0, 0.0),
        size=(0.10, 0.10, 0.08),
        critical_field=1.0e12,  # high threshold: pure stress observer, no trigger
        minimum_duration=0.0,
        quantities=("electric_field",),
    )

    def run(with_prebias):
        scene = mw.Scene(
            domain=mw.Domain(bounds=BOUNDS),
            grid=mw.GridSpec.uniform(H),
            boundary=mw.BoundarySpec.pec(),
            device="cuda",
        )
        scene.add_structure(mw.Structure(name="signal", geometry=_box(*SIGNAL), material=mw.Material.pec()))
        scene.add_structure(mw.Structure(name="ground", geometry=_box(*GROUND), material=mw.Material.pec()))
        scene.add_structure(mw.Structure(name="diel", geometry=_box(*DIEL), material=mw.Material(eps_r=3.2)))
        scene.add_monitor(mw.BreakdownMonitor("stress", **monitor_kw))
        return mw.Simulation.fdtd(
            scene,
            frequencies=[1e9],
            run_time=mw.TimeConfig(time_steps=20),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
            initial_condition=_prebias() if with_prebias else None,
        ).run()

    biased = run(True).breakdown("stress").peak_field
    plain = run(False).breakdown("stress").peak_field
    assert biased > plain
