"""Circuit-driven end-to-end ESD (gate d): the Wave-D e2e upgraded from ideal
current injection to the standard-network MNA co-simulation.

One FDTD solve that (1) is seeded with a DC electrostatic pre-bias mapped onto
the Yee grid, (2) drives a standard IEC 61000-4-2 current waveform through the
330 ohm / 150 pF ESD generator network (:class:`ESDVoltageSource`) whose output
node is bound to a scene ``TerminalPort`` via the strong FDTD+MNA coupling, and
(3) accumulates a non-feedback ``BreakdownMonitor`` stress record. The port is
driven through a real source-impedance network, not the ideal current injection
of the companion ``test_prebias`` e2e. Requires CUDA.

Dynamic dielectric breakdown changes the field response after preparation.
Static conductive field/circuit elimination is supported, but a latched change
requires rebuilding that response and its circuit factors together. The dynamic
combination fails before execution; the circuit-driven control retains its
non-feedback stress monitor.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="circuit-driven ESD e2e requires CUDA"
)

BOUNDS = ((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))
H = 0.02
RUN_STEPS = 160

SIGNAL = ((0.0, 0.0, 0.06), (0.10, 0.10, 0.04))
GROUND = ((0.0, 0.0, -0.06), (0.10, 0.10, 0.04))
DIEL = ((0.0, 0.0, 0.0), (0.10, 0.10, 0.08))


def _box(pos, size):
    return mw.Box(position=pos, size=size)


def _prebias():
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
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
    return mw.ElectrostaticInitialCondition.from_result(dc, tolerance=1.0)


def _generator():
    return mw.ESDVoltageSource(
        "gun", port="feed", waveform=mw.ESDWaveform.iec_61000_4_2(8000.0)
    )


def _fdtd_scene(circuit, *, dielectric_material=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.pec(),
        circuits=(circuit,),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(name="signal", geometry=_box(*SIGNAL), material=mw.Material.pec())
    )
    scene.add_structure(
        mw.Structure(name="ground", geometry=_box(*GROUND), material=mw.Material.pec())
    )
    # This control uses a fixed dielectric response with a stress-only monitor.
    material = mw.Material(eps_r=3.2) if dielectric_material is None else dielectric_material
    scene.add_structure(mw.Structure(name="diel", geometry=_box(*DIEL), material=material))
    scene.add_port(
        mw.TerminalPort(
            "feed", mw.TerminalRef("signal"), mw.TerminalRef("ground"), mw.AxisPath("z"), 0.01
        )
    )
    scene.add_monitor(
        mw.BreakdownMonitor(
            "stress",
            position=(0.0, 0.0, 0.0),
            size=(0.10, 0.10, 0.08),
            critical_field=1.0e3,
            minimum_duration=0.0,
            quantities=("electric_field", "exposure"),
        )
    )
    return scene


def _breakdown_material():
    return mw.Material(
        eps_r=3.2,
        sigma_e=1e-12,
        breakdown=mw.DielectricBreakdown(
            critical_field=1.0e3,
            minimum_duration=0.0,
            post_breakdown_conductivity=5.0,
        ),
    )


def _dt() -> float:
    solver = mw.Simulation.fdtd(
        _fdtd_scene(_generator().build_circuit(t_end=1.0e-9)), frequency=1.0e9
    ).prepare().solver
    return float(solver.dt)


def test_circuit_driven_prebias_esd_stress_end_to_end():
    circuit = _generator().build_circuit(t_end=RUN_STEPS * _dt())
    result = mw.Simulation.fdtd(
        _fdtd_scene(circuit),
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=RUN_STEPS),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        initial_condition=_prebias(),
    ).run()

    # 1) Pre-bias provenance carries the DC diagnostics + Gauss residual.
    prov = result.electrostatic_prebias
    assert prov is not None
    assert prov["source"] == "electrostatic"
    assert prov["dc_iterations"] > 0
    assert "gauss_residual" in prov and prov["gauss_residual"] <= prov["tolerance"]

    # 2) The circuit-driven ESD generator ran through the strong MNA coupling and
    #    its port voltage is a genuine (non-zero) two-way-coupled trace.
    data = result.circuit(circuit.name)
    assert data.node_voltages.device.type == "cuda"
    assert torch.all(torch.isfinite(data.node_voltages))
    assert float(torch.max(torch.abs(data.node_voltage("tip")))) > 0.0

    # 3) The generator provenance (revision, level voltage, source network) rides
    #    through to the result (gate c on the e2e path).
    generator = result.esd_generator(circuit.name)
    assert generator["injection"] == "source_impedance_network"
    assert generator["waveform"]["standard_revision"] == "ed2-contact"
    assert generator["waveform"]["level_voltage"] == pytest.approx(8000.0)

    # 4) The non-feedback stress record is populated on device.
    stress = result.breakdown("stress")
    assert stress.capability_level == "stress-only"
    assert stress.peak_field > 0.0
    assert stress.max_field_map.device.type == "cuda"


def test_circuit_port_coupling_fails_closed_in_breakdown_media():
    """A changing conductivity must not retain the static MNA response."""

    circuit = _generator().build_circuit(t_end=RUN_STEPS * _dt())
    scene = _fdtd_scene(circuit, dielectric_material=_breakdown_material())
    with pytest.raises(NotImplementedError, match="dynamic dielectric breakdown"):
        mw.Simulation.fdtd(
            scene,
            frequencies=[1e9],
            run_time=mw.TimeConfig(time_steps=RUN_STEPS),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        ).prepare()
