import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import PlanarEquivalentCurrents


_C0 = 299792458.0
_MU0 = 4.0 * math.pi * 1e-7
_EPS0 = 1.0 / (_MU0 * _C0**2)


def _result_with_differentiable_surface(monkeypatch, *, device="cpu"):
    frequencies = torch.tensor([1.0e9], device=device, dtype=torch.float64)
    a = torch.tensor([2.0 + 0.1j], device=device, dtype=torch.complex128, requires_grad=True)
    b = torch.tensor([0.4 - 0.2j], device=device, dtype=torch.complex128, requires_grad=True)
    port = mw.PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=a,
        b=b,
        z0=50.0,
    )
    u = torch.linspace(-0.08, 0.08, 7, device=device, dtype=torch.float64)
    v = torch.linspace(-0.06, 0.06, 5, device=device, dtype=torch.float64)
    j = torch.zeros((u.numel(), v.numel(), 3), device=device, dtype=torch.complex128)
    m = torch.zeros_like(j)
    j[..., 0] = 1.0 + 0.1j
    m[..., 1] = 0.4 - 0.2j
    j.requires_grad_()
    m.requires_grad_()
    currents = PlanarEquivalentCurrents(
        axis="z",
        position=0.0,
        frequency=1.0e9,
        u=u,
        v=v,
        J=j,
        M=m,
    )

    monkeypatch.setattr(
        "witwin.maxwell.postprocess.stratton_chu.equivalent_surface_currents_from_monitor",
        lambda result, name, **kwargs: currents,
    )
    solver = SimpleNamespace(c=_C0, eps0=_EPS0, mu0=_MU0)
    result = mw.Result(
        method="fdtd",
        scene=SimpleNamespace(monitors=()),
        frequency=1.0e9,
        solver=solver,
        ports={"feed": port},
    )
    return result, port, currents, a, b, j, m


def _closed_surface_result():
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    ).add_monitor(surface)
    coords = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)
    monitors = {}
    for face_index, face in enumerate(surface.faces):
        coord_names = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[face.axis]
        payload = {
            "kind": "plane",
            "fields": face.fields,
            "components": {},
            "samples": 8,
            "frequency": 1.0e9,
            "frequencies": (1.0e9,),
            "axis": face.axis,
            "position": face.plane_position,
            "compute_flux": False,
            "normal_direction": face.normal_direction,
            coord_names[0]: coords,
            coord_names[1]: coords,
            "coords": (coords, coords),
        }
        for component_index, component in enumerate(face.fields):
            amplitude = (face_index + 1) * (component_index + 1)
            data = torch.full(
                (coords.numel(), coords.numel()),
                complex(amplitude, 0.1 * amplitude),
                dtype=torch.complex128,
            )
            payload[component] = data
            payload["components"][component] = {
                "data": data,
                "coords": (coords, coords),
                "plane_index": 0,
                "plane_indices": (0,),
                "plane_weights": (1.0,),
                "plane_positions": (face.plane_position,),
            }
        monitors[face.name] = payload
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    port = mw.PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=torch.tensor([2.0 + 0.0j], dtype=torch.complex128),
        b=torch.tensor([0.2 + 0.0j], dtype=torch.complex128),
        z0=50.0,
    )
    return mw.Result(
        method="fdtd",
        scene=scene,
        frequency=1.0e9,
        solver=SimpleNamespace(c=_C0, eps0=_EPS0, mu0=_MU0),
        monitors=monitors,
        ports={"feed": port},
    )


def test_result_antenna_connects_port_power_far_field_and_surface_current(monkeypatch):
    result, port, currents, a, b, j, m = _result_with_differentiable_surface(
        monkeypatch
    )

    data = result.antenna(
        surface="nf2ff",
        driven_port="feed",
        theta_points=19,
        phi_points=25,
        radius=8.0,
        phase_center=torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64),
    )

    assert isinstance(data, mw.AntennaData)
    assert data.e_theta.shape == (1, 19, 25)
    assert data.surface_currents == (currents,)
    torch.testing.assert_close(data.p_accepted, port.accepted_power)
    torch.testing.assert_close(data.p_incident, port.incident_power)
    torch.testing.assert_close(
        data.realized_gain,
        data.mismatch_efficiency[:, None, None] * data.gain,
    )
    loss = data.realized_gain.square().mean() + data.p_rad.sum()
    loss.backward()
    for tensor in (a, b, j, m):
        assert tensor.grad is not None
        assert torch.all(torch.isfinite(tensor.grad))


def test_result_antenna_consumes_a_first_class_closed_surface_monitor():
    result = _closed_surface_result()

    data = result.antenna(
        surface="nf2ff",
        driven_port="feed",
        theta_points=11,
        phi_points=17,
        radius=4.0,
    )

    assert data.surface_currents is not None
    assert len(data.surface_currents[0].surfaces) == 6
    assert data.radiation_intensity.shape == (1, 11, 17)
    assert torch.all(torch.isfinite(data.realized_gain))


def test_result_antenna_frame_rotates_pattern_and_phase_center_changes_only_phase(
    monkeypatch,
):
    result, _, _, _, _, _, _ = _result_with_differentiable_surface(monkeypatch)
    theta = torch.linspace(0.0, math.pi, 13, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 17, dtype=torch.float64)
    identity = result.antenna(
        surface="nf2ff",
        driven_port="feed",
        theta=theta,
        phi=phi,
        radius=6.0,
    )
    shifted = result.antenna(
        surface="nf2ff",
        driven_port="feed",
        theta=theta,
        phi=phi,
        radius=6.0,
        phase_center=torch.tensor([0.02, 0.01, -0.03], dtype=torch.float64),
    )
    frame = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated = result.antenna(
        surface="nf2ff",
        driven_port="feed",
        theta=theta,
        phi=phi,
        radius=6.0,
        frame=frame,
    )

    torch.testing.assert_close(shifted.radiation_intensity, identity.radiation_intensity)
    assert not torch.allclose(shifted.e_theta, identity.e_theta)
    assert not torch.allclose(rotated.radiation_intensity, identity.radiation_intensity)


def test_result_antenna_rejects_frequency_and_grid_contract_violations(monkeypatch):
    result, port, _, _, _, _, _ = _result_with_differentiable_surface(monkeypatch)
    mismatched = mw.PortData.from_power_waves(
        port_name="other",
        frequencies=torch.tensor([1.1e9], dtype=torch.float64),
        a=torch.ones(1, dtype=torch.complex128),
        b=torch.zeros(1, dtype=torch.complex128),
        z0=50.0,
    )

    with pytest.raises(ValueError, match="must match"):
        result.antenna(surface="nf2ff", driven_port=mismatched)
    with pytest.raises(ValueError, match="theta_points"):
        result.antenna(surface="nf2ff", driven_port=port, theta_points=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_result_antenna_keeps_the_complete_workflow_on_one_cuda_device(monkeypatch):
    result, _, _, _, _, _, _ = _result_with_differentiable_surface(
        monkeypatch, device="cuda"
    )

    data = result.antenna(
        surface="nf2ff",
        driven_port="feed",
        theta_points=11,
        phi_points=13,
        radius=5.0,
    )

    assert data.device.type == "cuda"
    assert data.surface_currents[0].device.type == "cuda"
    for value in data.__dict__.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cuda"
