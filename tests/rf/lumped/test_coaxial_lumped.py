"""CPU contracts for annular (coaxial) lumped ports and resistors."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.coaxial import compile_coaxial_geometry
from witwin.maxwell.fdtd.coaxial import apply_coaxial_runtime, prepare_coaxial_runtime
from witwin.maxwell.fdtd.lumped import apply_lumped_runtime, prepare_lumped_runtime
from witwin.maxwell.scene import prepare_scene

H = 0.25e-3
A = 1.0e-3
B = 2.3e-3


def _scene(*items, half=3.0e-3, length=4.0e-3):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (0.0, length))),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.pec(),
        device="cpu",
    )
    for item in items:
        if isinstance(item, mw.CoaxialLumpedPort):
            scene.add_port(item)
        else:
            scene.add_lumped_element(item)
    return scene


def _port(**overrides):
    values = dict(center=(0.0, 0.0, 1.0e-3), inner_diameter=2 * A, outer_diameter=2 * B, normal_axis="z")
    values.update(overrides)
    return mw.CoaxialLumpedPort("coax", **values)


def test_coaxial_port_and_resistor_public_contract():
    port = _port(direction="-", reference_impedance=75.0, termination=mw.SeriesRLC(r=75.0))
    assert port.kind == "coaxial_lumped_port"
    assert port.normal_axis == "z" and port.direction_sign == -1
    assert port.reference_impedance == 75.0
    assert port.power_convention == "0.5*Re(V*conj(I))"
    resistor = mw.CoaxialLumpedResistor("load", center=(0, 0, 1e-3), inner_diameter=2 * A,
                                        outer_diameter=2 * B, normal_axis=2, resistance=50.0)
    assert resistor.normal_axis == "z"
    assert resistor.sheet_conductance == pytest.approx(math.log(B / A) / (2 * math.pi * 50.0))
    scene = _scene(port, resistor)
    assert scene.ports[0] is port and scene.lumped_elements[0] is resistor


@pytest.mark.parametrize("overrides, error", [
    (dict(inner_diameter=2 * B, outer_diameter=2 * A), ValueError),
    (dict(normal_axis="w"), ValueError),
    (dict(normal_axis=True), TypeError),
    (dict(direction="up"), ValueError),
    (dict(termination=mw.ParallelRLC(r=50.0)), TypeError),
    (dict(reference_impedance=-5.0), ValueError),
])
def test_coaxial_port_rejects_invalid_declarations(overrides, error):
    with pytest.raises(error):
        _port(**overrides)


def test_coaxial_resistor_rejects_nonpositive_or_trainable_resistance():
    arguments = dict(center=(0, 0, 0), inner_diameter=2 * A, outer_diameter=2 * B, normal_axis="z")
    with pytest.raises(ValueError):
        mw.CoaxialLumpedResistor("r", resistance=0.0, **arguments)
    with pytest.raises(NotImplementedError, match="Trainable CoaxialLumpedResistor.resistance"):
        mw.CoaxialLumpedResistor("r", resistance=torch.tensor(50.0, requires_grad=True), **arguments)


def test_scene_rejects_annulus_outside_domain_and_duplicate_names():
    with pytest.raises(ValueError, match="leaves the Scene domain"):
        _scene(_port(center=(1.0e-3, 0.0, 1.0e-3)))
    with pytest.raises(ValueError, match="already present"):
        _scene(_port(), _port())


def _tem_fields(scene):
    x, y = scene.x_nodes64, scene.y_nodes64
    xh, yh = scene.x_half64, scene.y_half64
    nz, nzh = len(scene.z_nodes64), len(scene.z_half64)

    def radial(px, py):
        r = np.hypot(px, py)
        return np.where((r > A) & (r < B), 1.0 / (r * r * math.log(B / A)), 0.0)

    def azimuthal(px, py):
        return 1.0 / (2.0 * math.pi * (px * px + py * py))

    xx, yy = np.meshgrid(xh, y, indexing="ij")
    ex = (radial(xx, yy) * xx)[:, :, None] * np.ones(nz)
    xx, yy = np.meshgrid(x, yh, indexing="ij")
    ey = (radial(xx, yy) * yy)[:, :, None] * np.ones(nz)
    hx = (-azimuthal(xx, yy) * yy)[:, :, None] * np.ones(nzh)
    xx, yy = np.meshgrid(xh, y, indexing="ij")
    hy = (azimuthal(xx, yy) * xx)[:, :, None] * np.ones(nzh)
    return {name: torch.as_tensor(value) for name, value in (("Ex", ex), ("Ey", ey), ("Hx", hx), ("Hy", hy))}


@pytest.mark.parametrize("direction", ["+", "-"])
def test_annular_row_and_loop_integrate_a_unit_tem_field(direction):
    scene = prepare_scene(_scene(_port(direction=direction)))
    geometry = compile_coaxial_geometry(scene, scene.ports[0], device="cpu")
    fields = _tem_fields(scene)
    assert geometry.voltage_components == ("Ex", "Ey")
    # Unit-voltage TEM field: the row is the azimuthal mean of the radial voltage.
    assert float(geometry.integrate_voltage(fields)) == pytest.approx(1.0, abs=0.05)
    # Unit inner-conductor current along +z, signed along the port direction.
    sign = 1.0 if direction == "+" else -1.0
    assert float(geometry.integrate_current(fields)) == pytest.approx(sign, rel=1.0e-3)
    assert geometry.metadata["current_half_index"] == (4 if direction == "+" else 3)


def _masked_solver(scene):
    """Coefficient masks of a staircased coax: edges with a midpoint inside either conductor are PEC."""
    x, y, z = scene.x_nodes64, scene.y_nodes64, scene.z_nodes64
    xh, yh, zh = scene.x_half64, scene.y_half64, scene.z_half64

    def free(px, py):
        r = np.hypot(px, py)
        return (r > A) & (r < B)

    ex = free(*np.meshgrid(xh, y, indexing="ij"))[:, :, None] * np.ones(len(z))
    ey = free(*np.meshgrid(x, yh, indexing="ij"))[:, :, None] * np.ones(len(z))
    ez = free(*np.meshgrid(x, y, indexing="ij"))[:, :, None] * np.ones(len(zh))
    return SimpleNamespace(scene=scene, device=torch.device("cpu"), cex_curl=torch.as_tensor(ex),
                           cey_curl=torch.as_tensor(ey), cez_curl=torch.as_tensor(ez))


def test_discrete_harmonic_row_obeys_kirchhoff_and_measures_the_conductor_voltage():
    from witwin.maxwell.fdtd.coaxial import harmonic_coaxial_geometry

    scene = prepare_scene(_scene(_port()))
    solver = _masked_solver(scene)
    geometry = harmonic_coaxial_geometry(solver, compile_coaxial_geometry(scene, scene.ports[0], device="cpu"))
    assert geometry.metadata["row"] == "discrete_harmonic"
    nx, ny = len(scene.x_nodes64), len(scene.y_nodes64)
    net = np.zeros((nx, ny))
    for component, indices, weights in zip(geometry.voltage_components, geometry.voltage_indices,
                                           geometry.voltage_weights):
        step = (1, 0) if component == "Ex" else (0, 1)
        for (i, j, _), weight in zip(indices.numpy(), weights.numpy()):
            current = weight / H
            net[i, j] -= current
            net[i + step[0], j + step[1]] += current
    xx, yy = np.meshgrid(scene.x_nodes64, scene.y_nodes64, indexing="ij")
    r = np.hypot(xx, yy)
    vacuum = (r > A + 0.5 * H) & (r < B - 0.5 * H)
    # Exact KCL at every vacuum node; unit current leaves the inner conductor.
    assert np.max(np.abs(net[vacuum])) < 1.0e-9
    assert net[r < A + 0.5 * H].sum() == pytest.approx(-1.0, abs=1.0e-9)
    assert net[r > B - 0.5 * H].sum() == pytest.approx(1.0, abs=1.0e-9)
    # Any in-plane gradient field with unit conductor voltage reads exactly 1:
    # arbitrary vacuum potentials, 1 on the discrete inner and 0 on the outer conductor.
    k = geometry.plane_index
    free_x = solver.cex_curl[:, :, k].numpy() > 0
    free_y = solver.cey_curl[:, :, k].numpy() > 0
    touches_pec = np.zeros((nx, ny), dtype=bool)
    touches_pec[:-1, :] |= ~free_x
    touches_pec[1:, :] |= ~free_x
    touches_pec[:, :-1] |= ~free_y
    touches_pec[:, 1:] |= ~free_y
    phi = np.random.default_rng(3).uniform(-2.0, 2.0, size=(nx, ny))
    phi[touches_pec] = np.where(r[touches_pec] < 0.5 * (A + B), 1.0, 0.0)
    fields = {
        "Ex": torch.as_tensor(-(phi[1:, :] - phi[:-1, :]) / H)[:, :, None].expand(-1, -1, len(scene.z_nodes64)),
        "Ey": torch.as_tensor(-(phi[:, 1:] - phi[:, :-1]) / H)[:, :, None].expand(-1, -1, len(scene.z_nodes64)),
    }
    assert float(geometry.integrate_voltage(fields)) == pytest.approx(1.0, abs=1.0e-9)


def test_discrete_harmonic_row_rejects_a_missing_outer_conductor():
    from witwin.maxwell.fdtd.coaxial import harmonic_coaxial_geometry

    scene = prepare_scene(_scene(_port()))
    solver = _masked_solver(scene)
    solver.cex_curl = torch.ones_like(solver.cex_curl)
    solver.cey_curl = torch.ones_like(solver.cey_curl)
    solver.cez_curl = torch.ones_like(solver.cez_curl)
    with pytest.raises(ValueError, match="not enclosed by PEC conductors"):
        harmonic_coaxial_geometry(solver, compile_coaxial_geometry(scene, scene.ports[0], device="cpu"))


def test_annular_compile_rejects_off_grid_plane_and_coarse_annulus():
    with pytest.raises(ValueError, match="node plane"):
        compile_coaxial_geometry(_scene(_port(center=(0.0, 0.0, 1.1e-3))), _port(center=(0.0, 0.0, 1.1e-3)))
    coarse = _port(inner_diameter=2 * A, outer_diameter=2 * (A + 0.3e-3))
    with pytest.raises(ValueError, match="fewer than two Yee cells"):
        compile_coaxial_geometry(_scene(coarse), coarse)


def _fake_solver(seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (9, 8, 7)
    eps0 = 8.8541878128e-12
    eps = torch.full(shape, eps0, dtype=torch.float64) * (1.0 + torch.rand(shape, generator=generator, dtype=torch.float64))
    dt = 1.0e-13
    return SimpleNamespace(
        device=torch.device("cpu"), dt=dt,
        Ex=torch.randn(shape, generator=generator, dtype=torch.float64),
        eps_Ex=eps, cex_curl=dt / eps,
    )


@pytest.mark.parametrize("termination", [mw.SeriesRLC(r=37.0), mw.SeriesRLC(r=20.0, l=2.0e-9, c=3.0e-12), None])
def test_one_component_annular_branch_reproduces_the_lumped_runtime_bitwise(termination):
    solver = _fake_solver()
    indices = torch.tensor([[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 5, 3]])
    weights = torch.tensor([0.3e-3, 0.25e-3, -0.1e-3, 0.4e-3], dtype=torch.float64)
    geometry = SimpleNamespace(port_name="row", voltage_components=("Ex",), voltage_indices=(indices,),
                               voltage_weights=(weights,))
    volume = torch.full(solver.Ex.shape, 1.0e-11, dtype=torch.float64)
    kwargs = dict(termination=termination) if termination is not None else dict(resistance=50.0)
    annular = prepare_coaxial_runtime(
        solver, geometry, edge_control_volume=lambda _s, _c: volume,
        history_currents=lambda _s, _row: (), validate_coefficient=lambda *_: None, **kwargs)
    reference = prepare_lumped_runtime(
        SimpleNamespace(port_name="row", voltage_indices=indices, voltage_weights=weights),
        dt=solver.dt, eps_edge=solver.eps_Ex, electric_update_coefficient=solver.cex_curl,
        yee_control_volume=volume, **kwargs)
    field_a = solver.Ex.clone()
    field_b = solver.Ex.clone()
    for step in range(25):
        drive = torch.tensor(math.sin(0.3 * step), dtype=torch.float64)
        field_a.add_(0.01 * torch.cos(field_a))
        field_b.add_(0.01 * torch.cos(field_b))
        apply_coaxial_runtime(annular, {"Ex": field_a}, thevenin_voltage=drive)
        apply_lumped_runtime(reference, field_b, thevenin_voltage=drive)
        assert torch.equal(field_a, field_b)
        assert torch.equal(annular.last_voltage_midpoint, reference.last_voltage_midpoint)
        assert torch.equal(annular.last_branch_current, reference.last_branch_current)


def test_annular_branch_drops_pec_suppressed_edges_and_rejects_an_all_pec_row():
    solver = _fake_solver()
    solver.cex_curl[1, 2, 3] = 0.0
    indices = torch.tensor([[1, 2, 3], [2, 2, 3]])
    weights = torch.tensor([1.0e-3, 1.0e-3], dtype=torch.float64)
    geometry = SimpleNamespace(port_name="row", voltage_components=("Ex",), voltage_indices=(indices,),
                               voltage_weights=(weights,))
    shared = dict(edge_control_volume=lambda *_: 1.0e-11, history_currents=lambda *_: (),
                  validate_coefficient=lambda *_: None)
    runtime = prepare_coaxial_runtime(solver, geometry, resistance=50.0, **shared)
    assert runtime.parts[0].linear_indices.numel() == 1
    solver.cex_curl[2, 2, 3] = 0.0
    with pytest.raises(ValueError, match="no unsuppressed tangential edge"):
        prepare_coaxial_runtime(solver, geometry, resistance=50.0, **shared)
    solver.cex_curl[2, 2, 3] = solver.dt / solver.eps_Ex[2, 2, 3]
    with pytest.raises(NotImplementedError, match="overlaps dispersive media"):
        prepare_coaxial_runtime(solver, geometry, resistance=50.0, edge_control_volume=lambda *_: 1.0e-11,
                                history_currents=lambda *_: (torch.zeros(1),), validate_coefficient=lambda *_: None)


def test_trainable_scene_with_coaxial_objects_is_rejected_by_the_adjoint_gate():
    scene = _scene(_port())
    scene.add_structure(mw.Structure(mw.Box(position=(0, 0, 3e-3), size=(1e-3, 1e-3, 0.5e-3)),
                                     mw.Material(eps_r=torch.tensor(2.0, requires_grad=True))))
    with pytest.raises(NotImplementedError, match="forward-only"):
        mw.Simulation.fdtd(scene, frequencies=(1e9,), run_time=mw.TimeConfig(time_steps=10))


@pytest.mark.parametrize("bound, network", [(True, None), (False, "net")])
def test_coaxial_port_rejects_circuit_and_embedded_network_coupling(bound, network):
    from witwin.maxwell.fdtd.ports import _prepare_coaxial_port_runtime

    with pytest.raises(NotImplementedError, match="cannot bind to a circuit or embedded network"):
        _prepare_coaxial_port_runtime(None, None, _port(), None, None, None, circuit_bound=bound, network_name=network)
