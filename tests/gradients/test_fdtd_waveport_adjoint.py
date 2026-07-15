from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.waveport_sweep import resolve_waveport_run_manifest


_FREQUENCY = 8.0e8


def _wave_port(*, modes=None):
    return mw.WavePort(
        "input",
        position=(-0.30, 0.0, 0.0),
        size=(0.0, 0.60, 0.30),
        direction="+",
        reference_plane=-0.30,
        modes=(mw.WaveModeSpec("te", polarization="Ez"),) if modes is None else modes,
    )


def _guide_scene(*, device, density=None, region_position=(0.05, 0.0, 0.0), modes=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.5, 0.5), (-0.35, 0.35))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.none(),
        ports=(_wave_port(modes=modes),),
        device=device,
    )
    wall_specs = (
        ((0.0, 0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_high"),
        ((0.0, -0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_low"),
        ((0.0, 0.0, 0.25), (1.0, 0.60, 0.20), "wall_z_high"),
        ((0.0, 0.0, -0.25), (1.0, 0.60, 0.20), "wall_z_low"),
    )
    for position, size, name in wall_specs:
        scene.add_structure(
            mw.Box(position=position, size=size).with_material(mw.Material.pec(), name=name)
        )
    if density is not None:
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=region_position, size=(0.10, 0.30, 0.20)),
                density=density,
                eps_bounds=(1.0, 5.0),
            )
        )
    return scene


class _WaveguideDesign(mw.SceneModule):
    def __init__(self, value=0.0):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(float(value), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logit).expand(2, 6, 4)
        return _guide_scene(device="cuda", density=density)


def _simulation(scene, excitations):
    return mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        excitations=excitations,
        run_time=mw.TimeConfig(time_steps=240),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )


def _reflection_objective(model):
    result = _simulation(model, mw.PortSweep()).run()
    return torch.abs(result.network.s[0, 0, 0]).square(), result


def test_trainable_waveport_rejects_multimode_and_cross_section_designs():
    density = torch.full((2, 6, 4), 0.5, requires_grad=True)
    modes = (
        mw.WaveModeSpec("te", polarization="Ez"),
        mw.WaveModeSpec("te", mode_index=1, polarization="Ez"),
    )
    with pytest.raises(NotImplementedError, match="exactly one fixed mode"):
        _simulation(
            _guide_scene(device="cpu", density=density, modes=modes),
            mw.PortSweep(),
        )

    cross_section_density = torch.full((2, 6, 4), 0.5, requires_grad=True)
    cross_section_scene = _guide_scene(
        device="cpu",
        density=cross_section_density,
        region_position=(-0.30, 0.0, 0.0),
    )
    with pytest.raises(NotImplementedError, match="fixed port cross-sections"):
        resolve_waveport_run_manifest(
            cross_section_scene,
            mw.PortSweep(),
            (_FREQUENCY,),
        )

    for excitation in (
        mw.PortSweep(amplitude=torch.tensor(1.0, requires_grad=True)),
        mw.PortExcitation(
            "input",
            amplitude=torch.tensor(1.0, requires_grad=True),
            mode_name="TE0",
        ),
        ):
        with pytest.raises(NotImplementedError, match="trainable"):
            _simulation(
                _guide_scene(
                    device="cpu",
                    density=torch.full((2, 6, 4), 0.5, requires_grad=True),
                ),
                excitation,
            ).run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_direct_waveport_port_data_preserves_single_device_adjoint_graph():
    model = _WaveguideDesign().cuda()
    result = _simulation(
        model,
        mw.PortExcitation("input", mode_name="TE0"),
    ).run()

    port = result.port("input")
    assert port.a.grad_fn is not None
    assert port.b.grad_fn is not None
    assert port.voltage.grad_fn is not None
    assert port.voltage.device == model.logit.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device")
def test_waveport_network_adjoint_matches_three_step_central_difference():
    model = _WaveguideDesign().cuda()
    objective, result = _reflection_objective(model)

    network = result.network
    port = result.port("input")
    assert network.s.grad_fn is not None
    assert port.a.grad_fn is not None
    assert port.voltage.grad_fn is not None
    assert network.s.device == model.logit.device

    objective.backward()
    adjoint = float(model.logit.grad)
    assert abs(adjoint) > 1.0e-9

    center = float(model.logit.detach())
    finite_differences = []
    for step in (0.08, 0.04, 0.02):
        with torch.no_grad():
            model.logit.fill_(center + step)
        plus = float(_reflection_objective(model)[0].detach())
        with torch.no_grad():
            model.logit.fill_(center - step)
        minus = float(_reflection_objective(model)[0].detach())
        finite_differences.append((plus - minus) / (2.0 * step))
    with torch.no_grad():
        model.logit.fill_(center)

    relative_errors = [
        abs(value - adjoint) / max(abs(value), abs(adjoint), 1.0e-12)
        for value in finite_differences
    ]
    assert min(relative_errors) < 0.02, (
        f"adjoint={adjoint}, finite_differences={finite_differences}, "
        f"relative_errors={relative_errors}"
    )
