import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import (
    compute_bistatic_rcs,
    compute_directivity,
    EquivalentCurrentsSurface,
    infer_incident_plane_wave_amplitude,
    NearFieldFarFieldTransformer,
    PlanarEquivalentCurrents,
    StrattonChuPropagator,
    SurfaceEquivalentCurrents,
    equivalent_surface_currents_from_fields,
    equivalent_surface_currents_from_monitor,
    equivalent_surface_currents_from_surface_samples,
    transform_to_bistatic_rcs,
)


_C0 = 299792458.0
_MU0 = 4.0 * np.pi * 1e-7
_EPS0 = 1.0 / (_MU0 * _C0**2)
_ETA0 = math.sqrt(_MU0 / _EPS0)


def _trapz_weights(points: torch.Tensor) -> torch.Tensor:
    weights = torch.empty_like(points)
    diffs = points[1:] - points[:-1]
    weights[0] = diffs[0] / 2.0
    weights[-1] = diffs[-1] / 2.0
    weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    return weights


def _hertzian_dipole_sphere_samples(frequency, radius, n_theta, n_phi, moment=1.0):
    """Exact ``e^{-iωt}`` fields of a z-directed Hertzian dipole sampled on a
    sphere of the given radius, returned as ``(points, normals, areas, E, H)``
    torch tensors. The dipole is enclosed by the sphere, so its surface currents
    reproduce the dipole far field for any radius (surface equivalence). The
    outward normals are radial, so every quadrature point has a different,
    non-axis-aligned normal -- a genuinely curved Huygens surface."""
    omega = 2.0 * math.pi * frequency
    k = omega / _C0
    theta = torch.linspace(0.0, math.pi, n_theta, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi, dtype=torch.float64)
    weight_theta = _trapz_weights(theta)
    weight_phi = _trapz_weights(phi)
    th_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    wt_grid, wp_grid = torch.meshgrid(weight_theta, weight_phi, indexing="ij")
    sin_t, cos_t = torch.sin(th_grid), torch.cos(th_grid)
    sin_p, cos_p = torch.sin(phi_grid), torch.cos(phi_grid)
    r_hat = torch.stack([sin_t * cos_p, sin_t * sin_p, cos_t], dim=-1)
    theta_hat = torch.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], dim=-1)
    phi_hat = torch.stack([-sin_p, cos_p, torch.zeros_like(phi_grid)], dim=-1)

    kr = k * radius
    e_ikr = torch.exp(torch.tensor(1j * kr, dtype=torch.complex128))
    e_r = (_ETA0 * moment * cos_t) / (2.0 * math.pi * radius**2) * (1.0 + 1j / kr) * e_ikr
    e_t = (_ETA0 * k * moment * sin_t) / (4.0 * math.pi * radius) * (1.0 / kr + 1j * (1.0 / kr**2 - 1.0)) * e_ikr
    h_p = (k * moment * sin_t) / (4.0 * math.pi * radius) * (1.0 / kr - 1j) * e_ikr
    e_field = e_r[..., None] * r_hat + e_t[..., None] * theta_hat
    h_field = h_p[..., None] * phi_hat
    area = (radius**2) * sin_t * wt_grid * wp_grid
    return (
        (radius * r_hat).reshape(-1, 3),
        r_hat.reshape(-1, 3),
        area.reshape(-1),
        e_field.reshape(-1, 3),
        h_field.reshape(-1, 3),
    )


def _hertzian_far_e_theta(frequency, radius, theta, moment=1.0):
    """Analytic far-field ``E_theta`` (``e^{-iωt}``) of the same dipole."""
    k = 2.0 * math.pi * frequency / _C0
    return (
        (_ETA0 * k * moment * torch.sin(theta))
        / (4.0 * math.pi * radius)
        * (-1j)
        * torch.exp(torch.tensor(1j * k * radius, dtype=torch.complex128))
    )


def _synthetic_closed_surface_result(*, layered_exterior: bool = False) -> mw.Result:
    surface = mw.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        device="cpu",
    ).add_monitor(surface)
    if layered_exterior:
        scene.add_structure(
            mw.Structure(
                name="layer",
                geometry=mw.Box(position=(0.0, 0.0, 0.35), size=(1.0, 1.0, 0.3)),
                material=mw.Material(eps_r=2.5),
            )
        )

    coords = np.linspace(-0.3, 0.3, 7)
    monitors = {}
    for face in surface.faces:
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
        }
        for component in face.fields:
            data = np.ones((coords.size, coords.size), dtype=np.complex64)
            payload["components"][component] = {
                "data": data,
                "coords": (coords, coords),
                "plane_index": 0,
                "plane_indices": (0,),
                "plane_weights": (1.0,),
                "plane_positions": (face.plane_position,),
            }
            payload[component] = data
        payload[coord_names[0]] = coords
        payload[coord_names[1]] = coords
        payload["coords"] = (coords, coords)
        monitors[face.name] = payload

    return mw.Result(
        method="fdtd",
        scene=scene,
        frequency=1.0e9,
        monitors=monitors,
    )


def _build_reference_currents():
    frequency = 10.0e9
    u = np.linspace(-0.06, 0.06, 21)
    v = np.linspace(-0.04, 0.04, 17)
    j_field = np.zeros((u.size, v.size, 3), dtype=np.complex128)
    m_field = np.zeros((u.size, v.size, 3), dtype=np.complex128)
    m_field[..., 1] = 1.0 + 0.0j
    return PlanarEquivalentCurrents(
        axis="z",
        position=0.0,
        frequency=frequency,
        u=u,
        v=v,
        J=j_field,
        M=m_field,
    )


def _build_shifted_reference_currents():
    base = _build_reference_currents()
    shifted = torch.zeros_like(base.M)
    shifted[..., 0] = 0.5 + 0.0j
    return PlanarEquivalentCurrents(
        axis="x",
        position=0.03,
        frequency=base.frequency,
        u=base.u,
        v=base.v,
        J=torch.zeros_like(base.J),
        M=shifted,
    )


def test_equivalent_surface_currents_match_z_normal_convention():
    u = np.linspace(-0.5, 0.5, 4)
    v = np.linspace(-0.25, 0.25, 3)
    ex = np.full((u.size, v.size), 3.0 + 1.0j)
    ey = np.full((u.size, v.size), -2.0 + 0.5j)
    hx = np.full((u.size, v.size), 5.0 - 2.0j)
    hy = np.full((u.size, v.size), -7.0 + 4.0j)

    currents = equivalent_surface_currents_from_fields(
        axis="z",
        position=0.1,
        frequency=1.0e9,
        u=u,
        v=v,
        fields={
            "Ex": ex,
            "Ey": ey,
            "Hx": hx,
            "Hy": hy,
        },
        normal_direction="+",
    )

    np.testing.assert_allclose(currents.J[..., 0], -hy)
    np.testing.assert_allclose(currents.J[..., 1], hx)
    np.testing.assert_allclose(currents.J[..., 2], 0.0)
    np.testing.assert_allclose(currents.M[..., 0], ey)
    np.testing.assert_allclose(currents.M[..., 1], -ex)
    np.testing.assert_allclose(currents.M[..., 2], 0.0)


def test_equivalent_surface_currents_from_monitor_defaults_to_monitor_normal_direction():
    coords_u = np.linspace(-0.5, 0.5, 4)
    coords_v = np.linspace(-0.25, 0.25, 3)
    ex = np.full((coords_u.size, coords_v.size), 3.0 + 1.0j)
    ey = np.full((coords_u.size, coords_v.size), -2.0 + 0.5j)
    hx = np.full((coords_u.size, coords_v.size), 5.0 - 2.0j)
    hy = np.full((coords_u.size, coords_v.size), -7.0 + 4.0j)

    class DummyResult:
        def monitor(self, name, **kwargs):
            assert name == "nf"
            return {
                "kind": "plane",
                "axis": "z",
                "position": 0.1,
                "frequency": 1.0e9,
                "frequencies": (1.0e9,),
                "normal_direction": "-",
                "x": coords_u,
                "y": coords_v,
                "fields": ("Ex", "Ey", "Hx", "Hy"),
                "Ex": ex,
                "Ey": ey,
                "Hx": hx,
                "Hy": hy,
                "components": {
                    "Ex": {"data": ex, "coords": (coords_u, coords_v)},
                    "Ey": {"data": ey, "coords": (coords_u, coords_v)},
                    "Hx": {"data": hx, "coords": (coords_u, coords_v)},
                    "Hy": {"data": hy, "coords": (coords_u, coords_v)},
                },
            }

    currents = equivalent_surface_currents_from_monitor(DummyResult(), "nf")

    np.testing.assert_allclose(currents.J[..., 0], hy)
    np.testing.assert_allclose(currents.J[..., 1], -hx)
    np.testing.assert_allclose(currents.M[..., 0], -ey)
    np.testing.assert_allclose(currents.M[..., 1], ex)


def test_equivalent_surface_currents_from_monitor_crops_tangential_bounds():
    coords_u = np.linspace(-0.6, 0.6, 7)
    coords_v = np.linspace(-0.4, 0.4, 5)
    ex = np.add.outer(coords_u, coords_v) + 0.0j
    ey = np.add.outer(coords_u**2, coords_v) + 0.0j
    hx = np.add.outer(coords_u, -coords_v) + 0.0j
    hy = np.add.outer(coords_u * 0.5, coords_v * 0.25) + 0.0j

    class DummyResult:
        def monitor(self, name, **kwargs):
            assert name == "nf"
            return {
                "kind": "plane",
                "axis": "z",
                "position": 0.1,
                "frequency": 1.0e9,
                "frequencies": (1.0e9,),
                "normal_direction": "+",
                "x": coords_u,
                "y": coords_v,
                "fields": ("Ex", "Ey", "Hx", "Hy"),
                "Ex": ex,
                "Ey": ey,
                "Hx": hx,
                "Hy": hy,
            }

    currents = equivalent_surface_currents_from_monitor(
        DummyResult(),
        "nf",
        tangential_bounds={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    )

    np.testing.assert_allclose(currents.u.detach().cpu().numpy(), np.array([-0.2, 0.0, 0.2]))
    np.testing.assert_allclose(currents.v.detach().cpu().numpy(), np.array([-0.2, 0.0, 0.2]))
    np.testing.assert_allclose(currents.J[..., 0].detach().cpu().numpy(), -hy[2:5, 1:4])
    np.testing.assert_allclose(currents.J[..., 1].detach().cpu().numpy(), hx[2:5, 1:4])


def test_equivalent_surface_currents_from_monitor_accepts_first_class_closed_surface_workflow():
    result = _synthetic_closed_surface_result()

    currents = equivalent_surface_currents_from_monitor(result, "huygens")

    assert isinstance(currents, EquivalentCurrentsSurface)
    assert len(currents.surfaces) == 6
    for surface in currents.surfaces:
        np.testing.assert_allclose(surface.u.detach().cpu().numpy(), np.linspace(-0.2, 0.2, 5))
        np.testing.assert_allclose(surface.v.detach().cpu().numpy(), np.linspace(-0.2, 0.2, 5))


def test_equivalent_surface_currents_from_monitor_rejects_layered_exterior_for_closed_surface():
    result = _synthetic_closed_surface_result(layered_exterior=True)

    with pytest.raises(NotImplementedError, match="homogeneous exterior"):
        equivalent_surface_currents_from_monitor(result, "huygens")


def test_stratton_chu_parallel_plane_propagation_returns_field_grids():
    currents = _build_reference_currents()
    propagator = StrattonChuPropagator(
        currents,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )

    result = propagator.propagate_plane(position=0.25, batch_size=64)

    assert result["Ex"].shape == currents.shape
    assert result["Ey"].shape == currents.shape
    assert result["Ez"].shape == currents.shape
    assert result["Hx"].shape == currents.shape
    assert result["Hy"].shape == currents.shape
    assert result["Hz"].shape == currents.shape
    assert result["x"].shape == (currents.u.numel(),)
    assert result["y"].shape == (currents.v.numel(),)
    assert result["z"] == 0.25
    assert torch.max(torch.abs(result["E"])) > 0.0


def test_multi_plane_stratton_chu_matches_sum_of_individual_surfaces():
    first = _build_reference_currents()
    second = _build_shifted_reference_currents()
    combined = EquivalentCurrentsSurface((first, second))

    combined_propagator = StrattonChuPropagator(
        combined,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )
    first_propagator = StrattonChuPropagator(
        first,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )
    second_propagator = StrattonChuPropagator(
        second,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )

    points = np.asarray(
        [
            [0.0, 0.0, 0.25],
            [0.05, -0.02, 0.22],
        ]
    )
    combined_result = combined_propagator.propagate_points(points, batch_size=2)
    first_result = first_propagator.propagate_points(points, batch_size=2)
    second_result = second_propagator.propagate_points(points, batch_size=2)

    np.testing.assert_allclose(
        combined_result["E"].detach().cpu().numpy(),
        (first_result["E"] + second_result["E"]).detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        combined_result["H"].detach().cpu().numpy(),
        (first_result["H"] + second_result["H"]).detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_far_field_matches_exact_propagation_in_fraunhofer_limit():
    currents = _build_reference_currents()
    radius = 12.0

    propagator = StrattonChuPropagator(
        currents,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )
    transformer = NearFieldFarFieldTransformer(
        currents,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )

    exact = propagator.propagate_points([[0.0, 0.0, radius]], batch_size=1)
    far = transformer.transform(theta=np.array([0.0]), phi=np.array([0.0]), radius=radius, batch_size=1)

    np.testing.assert_allclose(far["E"][0], exact["E"][0], rtol=2.5e-2, atol=1e-5)
    np.testing.assert_allclose(far["H"][0], exact["H"][0], rtol=2.5e-2, atol=1e-5)
    assert far["power_density"][0] > 0.0


def test_multi_plane_nf2ff_matches_sum_of_individual_surfaces():
    first = _build_reference_currents()
    second = _build_shifted_reference_currents()
    combined = EquivalentCurrentsSurface((first, second))

    combined_transformer = NearFieldFarFieldTransformer(
        combined,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )
    first_transformer = NearFieldFarFieldTransformer(
        first,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )
    second_transformer = NearFieldFarFieldTransformer(
        second,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )

    directions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.1, np.sqrt(1.0 - 0.2**2 - 0.1**2)],
        ]
    )
    combined_far = combined_transformer.transform_directions(directions, radius=12.0, batch_size=2)
    first_far = first_transformer.transform_directions(directions, radius=12.0, batch_size=2)
    second_far = second_transformer.transform_directions(directions, radius=12.0, batch_size=2)

    np.testing.assert_allclose(
        combined_far["E"].detach().cpu().numpy(),
        (first_far["E"] + second_far["E"]).detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        combined_far["H"].detach().cpu().numpy(),
        (first_far["H"] + second_far["H"]).detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_infer_incident_plane_wave_amplitude_from_source_and_scene():
    source = mw.PlaneWave(
        direction=(0.0, 0.0, 1.0),
        polarization="Ex",
        source_time=mw.CW(frequency=3.0e9, amplitude=12.5),
    )
    assert infer_incident_plane_wave_amplitude(source=source) == 12.5

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_source(source)
    assert infer_incident_plane_wave_amplitude(scene=scene) == 12.5


def test_compute_bistatic_rcs_matches_closed_form_expression():
    frequency = 3.0e9
    radius = np.array([10.0])
    incident_amplitude = 4.0
    far_field = {
        "frequency": frequency,
        "radius": radius,
        "theta": np.array([0.0]),
        "phi": np.array([0.0]),
        "E_theta": np.array([2.0 + 0.0j]),
        "E_phi": np.array([0.0 + 0.0j]),
        "E": np.array([[2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]]),
    }

    rcs = compute_bistatic_rcs(
        far_field,
        incident_amplitude=incident_amplitude,
        c=_C0,
    )

    expected_sigma = 4.0 * np.pi * radius**2 * (2.0**2) / (incident_amplitude**2)
    expected_normalized = expected_sigma / ((_C0 / frequency) ** 2)
    expected_db = 10.0 * np.log10(expected_normalized)

    np.testing.assert_allclose(rcs["rcs"], expected_sigma)
    np.testing.assert_allclose(rcs["normalized_rcs"], expected_normalized)
    np.testing.assert_allclose(rcs["rcs_db"], expected_db)


def test_transform_to_bistatic_rcs_uses_transformer_and_plane_wave_amplitude():
    currents = _build_reference_currents()
    transformer = NearFieldFarFieldTransformer(
        currents,
        c=_C0,
        eps0=_EPS0,
        mu0=_MU0,
        device="cpu",
    )
    source = mw.PlaneWave(
        direction=(0.0, 0.0, 1.0),
        polarization="Ex",
        source_time=mw.CW(frequency=currents.frequency, amplitude=3.0),
    )

    rcs = transform_to_bistatic_rcs(
        transformer,
        theta=np.array([0.0, 0.1]),
        phi=np.array([0.0, 0.0]),
        radius=6.0,
        source=source,
    )

    assert rcs["rcs"].shape == (2,)
    assert rcs["normalized_rcs"].shape == (2,)
    assert rcs["rcs_db"].shape == (2,)
    assert torch.max(rcs["rcs"]) > 0.0


def test_stratton_chu_default_device_no_longer_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(mw.postprocess.stratton_chu.torch.cuda, "is_available", lambda: False)
    currents = _build_reference_currents()

    with pytest.raises(RuntimeError, match="defaults to CUDA"):
        StrattonChuPropagator(
            currents,
            c=_C0,
            eps0=_EPS0,
            mu0=_MU0,
        )


def test_bistatic_rcs_keeps_torch_gradients():
    e_theta = torch.tensor([2.0 + 0.0j, 1.5 + 0.0j], dtype=torch.complex128, requires_grad=True)
    radius = torch.tensor([10.0, 10.0], dtype=torch.float64)
    far_field = {
        "frequency": 3.0e9,
        "radius": radius,
        "theta": torch.tensor([0.0, 0.1], dtype=torch.float64),
        "phi": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "E_theta": e_theta,
        "E_phi": torch.zeros_like(e_theta),
        "E": torch.stack((e_theta, torch.zeros_like(e_theta), torch.zeros_like(e_theta)), dim=-1),
    }

    rcs = compute_bistatic_rcs(
        far_field,
        incident_amplitude=torch.tensor(4.0, dtype=torch.float64),
        c=_C0,
    )
    loss = rcs["rcs"].sum() + rcs["normalized_rcs"].sum()
    loss.backward()

    assert isinstance(rcs["rcs"], torch.Tensor)
    assert e_theta.grad is not None
    assert torch.all(torch.isfinite(e_theta.grad))


def test_surface_equivalent_currents_from_samples_uses_cross_product_convention():
    """``J = n x H`` and ``M = -n x E`` at each quadrature point, with the
    outward normals normalized to unit length and the per-point surface-area
    weights folded into the currents by ``quadrature()``. This is the curved
    generalization of the axis-aligned ``n x H`` convention: the two sample
    points below carry tilted, non-unit normals."""
    points = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.3]], dtype=torch.float64)
    normals = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]], dtype=torch.float64)
    areas = torch.tensor([0.5, 0.25], dtype=torch.float64)
    e_field = torch.tensor(
        [[1.0 + 1.0j, 2.0 - 1.0j, 0.5j], [0.0, 1.0 + 0.0j, -2.0 + 1.0j]], dtype=torch.complex128
    )
    h_field = torch.tensor(
        [[0.0, -1.0 + 2.0j, 3.0 + 0.0j], [1.0j, 2.0 + 0.0j, -1.0 + 0.0j]], dtype=torch.complex128
    )

    currents = equivalent_surface_currents_from_surface_samples(
        frequency=1.0e9,
        points=points,
        normals=normals,
        areas=areas,
        E=e_field,
        H=h_field,
    )

    assert isinstance(currents, SurfaceEquivalentCurrents)
    unit_normal = (normals / torch.linalg.norm(normals, dim=-1, keepdim=True)).to(torch.complex128)
    expected_j = torch.cross(unit_normal, h_field, dim=-1)
    expected_m = -torch.cross(unit_normal, e_field, dim=-1)
    torch.testing.assert_close(currents.J, expected_j)
    torch.testing.assert_close(currents.M, expected_m)

    quad_points, quad_j, quad_m = currents.quadrature()
    torch.testing.assert_close(quad_points, points)
    torch.testing.assert_close(quad_j, expected_j * areas[:, None].to(torch.complex128))
    torch.testing.assert_close(quad_m, expected_m * areas[:, None].to(torch.complex128))


def test_surface_equivalent_currents_reject_shape_and_normal_errors():
    good_points = torch.zeros((4, 3), dtype=torch.float64)
    good_vec = torch.ones((4, 3), dtype=torch.complex128)
    with pytest.raises(ValueError, match="normals must have shape"):
        equivalent_surface_currents_from_surface_samples(
            frequency=1.0e9,
            points=good_points,
            normals=torch.ones((3, 3), dtype=torch.float64),
            areas=torch.ones(4, dtype=torch.float64),
            E=good_vec,
            H=good_vec,
        )
    with pytest.raises(ValueError, match="non-zero"):
        equivalent_surface_currents_from_surface_samples(
            frequency=1.0e9,
            points=good_points,
            normals=torch.zeros((4, 3), dtype=torch.float64),
            areas=torch.ones(4, dtype=torch.float64),
            E=good_vec,
            H=good_vec,
        )


def test_stratton_chu_propagator_accepts_general_surface_currents():
    frequency = 1.0e9
    points, normals, areas, e_field, h_field = _hertzian_dipole_sphere_samples(frequency, 0.10, 48, 96)
    currents = equivalent_surface_currents_from_surface_samples(
        frequency=frequency,
        points=points,
        normals=normals,
        areas=areas,
        E=e_field,
        H=h_field,
    )
    propagator = StrattonChuPropagator(currents, c=_C0, eps0=_EPS0, mu0=_MU0, device="cpu")
    out = propagator.propagate_points([[0.0, 0.0, 12.0]], batch_size=4096)
    assert tuple(out["E"].shape) == (1, 3)
    assert float(torch.max(torch.abs(out["E"]))) > 0.0


def test_curved_sphere_huygens_reproduces_hertzian_dipole_far_field():
    """A spherical (curved, tilted-normal) Huygens surface enclosing a Hertzian
    dipole radiates the dipole's exact far field: E_theta matches the analytic
    dipole to quadrature error, the cross-polar E_phi is machine zero, and the
    reconstructed directivity is the 1.5 sin^2(theta) Hertzian pattern peaking
    broadside. This is the surface-equivalence far field on a non-box surface."""
    frequency = 1.0e9
    sphere_radius = 0.10
    obs_radius = 12.0
    points, normals, areas, e_field, h_field = _hertzian_dipole_sphere_samples(
        frequency, sphere_radius, 64, 128
    )
    currents = equivalent_surface_currents_from_surface_samples(
        frequency=frequency,
        points=points,
        normals=normals,
        areas=areas,
        E=e_field,
        H=h_field,
    )
    transformer = NearFieldFarFieldTransformer(currents, c=_C0, eps0=_EPS0, mu0=_MU0, device="cpu")

    theta = torch.linspace(0.0, math.pi, 91, dtype=torch.float64)
    phi = torch.zeros_like(theta)
    far = transformer.transform(theta, phi, radius=obs_radius, batch_size=4096)
    analytic = _hertzian_far_e_theta(frequency, obs_radius, theta)
    rel = torch.linalg.norm(far["E_theta"] - analytic) / torch.linalg.norm(analytic)
    assert float(rel) < 1e-3
    cross_pol = torch.max(torch.abs(far["E_phi"])) / torch.max(torch.abs(far["E_theta"]))
    assert float(cross_pol) < 1e-6

    theta_grid, phi_grid = torch.broadcast_tensors(
        torch.linspace(0.0, math.pi, 61, dtype=torch.float64)[:, None],
        torch.linspace(0.0, 2.0 * math.pi, 121, dtype=torch.float64)[None, :],
    )
    far_sphere = transformer.transform(theta_grid, phi_grid, radius=obs_radius, batch_size=8192)
    directivity = compute_directivity(far_sphere)
    assert abs(float(directivity["D_max"]) - 1.5) < 0.02
    assert abs(math.degrees(float(directivity["D_max_theta"])) - 90.0) < 1.0


def test_curved_huygens_surface_far_field_is_radius_independent():
    """Surface equivalence: the far field radiated from the equivalent currents
    on two different enclosing spheres must agree, because both enclose the same
    interior source. A convention or weighting bug would break this."""
    frequency = 1.0e9
    obs_radius = 12.0
    theta = torch.linspace(0.0, math.pi, 91, dtype=torch.float64)
    phi = torch.zeros_like(theta)

    def _far(sphere_radius):
        points, normals, areas, e_field, h_field = _hertzian_dipole_sphere_samples(
            frequency, sphere_radius, 64, 128
        )
        currents = equivalent_surface_currents_from_surface_samples(
            frequency=frequency,
            points=points,
            normals=normals,
            areas=areas,
            E=e_field,
            H=h_field,
        )
        transformer = NearFieldFarFieldTransformer(currents, c=_C0, eps0=_EPS0, mu0=_MU0, device="cpu")
        return transformer.transform(theta, phi, radius=obs_radius, batch_size=4096)["E_theta"]

    inner = _far(0.10)
    outer = _far(0.16)
    rel = torch.linalg.norm(outer - inner) / torch.linalg.norm(inner)
    assert float(rel) < 1e-3
