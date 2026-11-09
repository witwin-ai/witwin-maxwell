"""Physical-area and counterpropagating-wave checks for graded monitor planes."""
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.excitation.modes import sample_mode_source_component
from witwin.maxwell.fdtd.excitation.modes.tem import _solve_pec_tem_mode_torch
from witwin.maxwell.postprocess import modal
from witwin.maxwell.postprocess.stratton_chu import build_plane_points


def _project(monkeypatch, axis, nodes_u, nodes_v, electric, magnetic, forward=1., backward=0.):
    tangents = [label for label in "xyz" if label != axis]
    centers = [0.5 * (nodes[:-1] + nodes[1:]) for nodes in (nodes_u, nodes_v)]
    monitor = dict(axis=axis, position=0., normal_direction="+", frequency=1e9,
                   cell_widths=dict(zip(tangents, (torch.diff(nodes_u), torch.diff(nodes_v)))))
    monitor.update(zip(tangents, centers))
    for index, label in enumerate("xyz"):
        monitor["E" + label] = (forward + backward) * electric[..., index]
        monitor["H" + label] = (forward - backward) * magnetic[..., index]
    source = SimpleNamespace(direction="+", mode_index=0, polarization_axis=tangents[0])
    monkeypatch.setattr(modal, "_mode_overlap_context", lambda result: None)
    monkeypatch.setattr(modal, "_resolve_mode_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(modal, "_reference_mode_fields", lambda *args, **kwargs:
        (*centers, electric, magnetic, {"beta": 20., "effective_index": 1.}))
    return modal._compute_mode_overlap_from_payload(SimpleNamespace(scene=None), "port", monitor), monitor


@pytest.mark.parametrize("axis", "xyz")
def test_graded_constant_power_density_integrates_full_physical_plane(monkeypatch, axis):
    u = torch.tensor([-.03, -.012, -.01, -.001, .004, .03], dtype=torch.float64)
    v = torch.tensor([-.02, -.019, -.008, .001, .003, .02], dtype=torch.float64)
    electric = torch.zeros((len(u)-1, len(v)-1, 3), dtype=torch.complex128)
    magnetic = torch.zeros_like(electric)
    tangent = [index for index in range(3) if index != "xyz".index(axis)]
    electric[..., tangent[0]] = 6.
    magnetic[..., tangent[1]] = -2. if axis == "y" else 2.
    overlap, _ = _project(monkeypatch, axis, u, v, electric, magnetic)
    expected = 6. * float((u[-1]-u[0]) * (v[-1]-v[0]))
    assert float(overlap["total_power"]) == pytest.approx(expected, rel=1e-14)
    assert float(overlap["mode_power"]) == pytest.approx(expected, rel=1e-14)
    assert complex(overlap["amplitude_forward"]) == pytest.approx(1.)


@pytest.mark.parametrize("forward,backward", [(1.+0j, 0j), (0j, 1.+0j), (.4+.7j, -.2+.1j)])
def test_actual_graded_complex_tem_profiles_separate_counterpropagating_waves(monkeypatch, forward, backward):
    u = torch.tensor([-.02, -.012, -.008, -.003, -.001, .001, .003, .008, .012, .02], dtype=torch.float64)
    v = torch.tensor([-.02, -.015, -.007, -.002, 0., .002, .007, .015, .02], dtype=torch.float64)
    occupancy = torch.zeros((len(u), len(v)), dtype=torch.float64)
    occupancy[0, :] = occupancy[-1, :] = occupancy[:, 0] = occupancy[:, -1] = 1.
    occupancy[4:6, 3:6] = 1.
    epsilon = tuple(torch.full_like(occupancy, 2.25+.2j, dtype=torch.complex128) for _ in range(3))
    mu = tuple(torch.ones_like(value) for value in epsilon)
    _, profiles, _ = _solve_pec_tem_mode_torch(epsilon, mu, occupancy, k0=20., du=None, dv=None,
        coords_u=u, coords_v=v, mode_index=0, field_names=("Ey", "Ez", "Hy", "Hz"))
    for name in ("Ex", "Hx"):
        profiles[name] = torch.zeros_like(occupancy, dtype=torch.complex128)
    data = dict(component_profiles=profiles, tangential_axes=("y", "z"), coords_u=u, coords_v=v,
                staggered_profiles=True)
    points = build_plane_points("x", 0., .5*(u[:-1]+u[1:]), .5*(v[:-1]+v[1:]))
    electric = torch.stack([sample_mode_source_component(data, points, "E"+a) for a in "xyz"], dim=-1)
    magnetic = torch.stack([sample_mode_source_component(data, points, "H"+a) for a in "xyz"], dim=-1)
    overlap, _ = _project(monkeypatch, "x", u, v, electric, magnetic, forward, backward)
    assert complex(overlap["amplitude_forward"]) == pytest.approx(forward, abs=2e-14)
    assert complex(overlap["amplitude_backward"]) == pytest.approx(backward, abs=2e-14)
    # Independently sum each represented rectangle using its actual edge widths.
    expected = 0.
    for i, width_u in enumerate(torch.diff(u)):
        for j, width_v in enumerate(torch.diff(v)):
            e = (forward+backward)*electric[i, j]
            h = (forward-backward)*magnetic[i, j]
            expected += float(.5*torch.cross(e, h.conj(), dim=-1)[0].real*width_u*width_v)
    assert float(overlap["total_power"]) == pytest.approx(expected, rel=1e-13, abs=1e-16)


@pytest.mark.parametrize("bad_width", [[.01], [-.01, .02], [float("nan"), .02], [.01, float("inf")]])
def test_modal_quadrature_rejects_invalid_physical_widths(monkeypatch, bad_width):
    nodes = torch.tensor([0., .01, .03], dtype=torch.float64)
    electric = torch.zeros((2, 2, 3), dtype=torch.complex128)
    magnetic = torch.zeros_like(electric)
    electric[..., 1] = magnetic[..., 2] = 1.
    _, monitor = _project(monkeypatch, "x", nodes, nodes, electric, magnetic)
    monitor["cell_widths"]["y"] = bad_width
    with pytest.raises(ValueError, match="cell widths must be finite, positive and match"):
        modal._compute_mode_overlap_from_payload(SimpleNamespace(scene=None), "port", monitor)
