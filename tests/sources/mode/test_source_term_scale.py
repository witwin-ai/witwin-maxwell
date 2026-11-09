"""Do not discard finite modal or incident fields because their SI amplitudes are small."""

import math
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.excitation.tfsf_common import (
    _component_roundoff_floor,
    build_term_from_profile,
    build_terms_from_specs,
)
from witwin.maxwell.fdtd.excitation.tfsf_specs import build_discrete_tfsf_specs


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("tensor", [False, True])
def test_incident_surface_term_preserves_small_magnetic_field(length, tensor):
    value = 1e-10*length
    component = torch.tensor(value, dtype=torch.float64) if tensor else value
    specs = [{"incident_name":"Hz", "sign":1, "field_name":"Ey", "offsets":(0,0,0), "shape":(1,1,1), "delta_axis":"x"}]
    solver = SimpleNamespace(coefficient=torch.ones((1,1,1), dtype=torch.float64), scene=SimpleNamespace(**{f"d{axis}_dual64": [length] for axis in "xyz"}))
    terms = build_terms_from_specs(solver, specs, {"Hz":component}, {"Ey":"coefficient"}, lambda spec, patch, scale: patch*scale)
    assert len(terms) == 1
    assert terms[0].item() == pytest.approx(value/length, rel=1e-12, abs=0)


@pytest.mark.parametrize("length", [1., 1e-6])
def test_modal_surface_patch_keeps_finite_small_field(length):
    scale = torch.full((1,1,1), 1e-10*length, dtype=torch.float64)
    solver = SimpleNamespace(Ex=scale, dt=length/299792458)
    term = build_term_from_profile(solver, field_name="Ey", offsets=(0,0,0), scale=scale,
                                   delay_patch=torch.zeros_like(scale), activation_delay_patch=None,
                                   source_time={"kind":"cw", "amplitude":1.}, omega=1.)
    assert term is not None


@pytest.mark.parametrize("length", [1., 1e-6])
def test_interpolated_current_retains_finite_small_si_density(length):
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.excitation.injection import _append_interpolated_current
    from witwin.maxwell.scene import prepare_scene

    scene = prepare_scene(mw.Scene(domain=mw.Domain(bounds=((-length,length),)*3),
                                   grid=mw.GridSpec.uniform(.5*length), boundary=mw.BoundarySpec.none(), device="cpu"))
    solver = SimpleNamespace(scene=scene, device=torch.device("cpu"), Ex=torch.empty(1, dtype=torch.float64),
                              dt=length/299792458, _COMPONENT_HALF_OFFSET_AXES={"Ex":(True,False,False)})
    axes = tuple(torch.as_tensor(axis) for axis in (scene.x_half64, scene.y_nodes64, scene.z_nodes64))
    values = torch.full(tuple(len(axis) for axis in axes), 1e-28*length, dtype=torch.float64)
    terms = []
    _append_interpolated_current(solver, field_name="Ex", values=values, axes=axes,
                                 denom_tensor=torch.ones_like(values), region_lo=(-length,)*3,
                                 region_hi=(length,)*3, source_time={"kind":"cw", "amplitude":1.},
                                 source_omega=1., source_index=0, term_list=terms)
    assert len(terms) == 1
    assert terms[0]["patch"].abs().max().item() > 0


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
@pytest.mark.parametrize("construction", ["angles", "cross"])
def test_roundoff_vectors_keep_axis_aligned_term_count(length, dtype, construction):
    residual = math.cos(math.pi / 2)
    if construction == "angles":
        noisy = (residual, math.sin(math.pi / 2), 0.)
        exact = (0., 1., 0.)
    else:
        noisy = torch.linalg.cross(
            torch.tensor((1., residual, residual), dtype=torch.float64),
            torch.tensor((residual, 1., residual), dtype=torch.float64),
        ).tolist()
        exact = (0., 0., 1.)
    assert any(a != b for a, b in zip(noisy, exact))
    electric, magnetic = build_discrete_tfsf_specs((1, 1, 1), (3, 3, 3))
    solver = SimpleNamespace(
        coefficient=torch.ones((5, 5, 5), dtype=torch.float64),
        scene=SimpleNamespace(**{f"d{axis}_{kind}64": [length] * 5
                                 for axis in "xyz" for kind in ("primal", "dual")}),
    )
    for specs, family in ((electric, "H"), (magnetic, "E")):
        counts = []
        for vector in (exact, noisy, (0., 0., 0.)):
            values = {family + axis: value * 1e-10 * length for axis, value in zip("xyz", vector)}
            if dtype is not None:
                values = {name: torch.tensor(value, dtype=dtype) for name, value in values.items()}
            terms = build_terms_from_specs(
                solver, specs, values, {family + axis: "coefficient" for family in ("E", "H") for axis in "xyz"},
                lambda spec, patch, component: patch * component,
            )
            counts.append(len(terms))
        # One incident component contributes on four faces of a closed box.
        assert counts == [4, 4, 0]


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
@pytest.mark.parametrize("ratio", [1e-5, 1e-6])
def test_analytic_vectors_preserve_weak_real_components(length, dtype, ratio):
    electric, magnetic = build_discrete_tfsf_specs((1, 1, 1), (3, 3, 3))
    solver = SimpleNamespace(
        coefficient=torch.ones((5, 5, 5), dtype=torch.float32),
        scene=SimpleNamespace(**{f"d{axis}_{kind}64": [length] * 5
                                 for axis in "xyz" for kind in ("primal", "dual")}),
    )
    for specs, family in ((electric, "H"), (magnetic, "E")):
        vector = {family + axis: value for axis, value in zip("xyz", (0., ratio, 1.))}
        if dtype is not None:
            vector = {name: torch.tensor(value, dtype=dtype) for name, value in vector.items()}
        # The original 1e-12 component floor retains both nonzero components.
        baseline = [spec for spec in specs if abs(float(vector[spec["incident_name"]])) > 1e-12]
        terms = build_terms_from_specs(
            solver, specs, vector, {spec["field_name"]: "coefficient" for spec in specs},
            lambda spec, patch, component: patch * component,
        )
        assert len(terms) == len(baseline) == 8
        for term, spec in zip(terms, baseline):
            expected = solver.coefficient.new_full(term.shape, float(vector[spec["incident_name"]]) / length)
            torch.testing.assert_close(term, expected, rtol=2e-7, atol=0)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_component_profile_floor_uses_owner_and_dtype(length, dtype):
    peak = 1e-10 * length
    eps = torch.finfo(dtype).eps
    profiles = [torch.full((1, 1, 1), peak * factor, dtype=dtype) for factor in (1., 64 * eps, 128 * eps)]
    floor = _component_roundoff_floor(profiles, dtype=dtype)
    solver = SimpleNamespace(Ex=profiles[0].real, dt=length / 299792458)
    terms = [build_term_from_profile(
        solver, field_name="Ey", offsets=(0, 0, 0), scale=profile,
        delay_patch=torch.zeros_like(profile.real), activation_delay_patch=None,
        source_time={"kind": "cw", "amplitude": 1.}, omega=1., profile_floor=floor,
    ) for profile in profiles]
    assert [term is not None for term in terms] == [True, False, True]


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ratio", [0., math.cos(math.pi / 2), 1e-6])
def test_current_profiles_skip_only_exact_zero(length, dtype, ratio):
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.excitation.injection import _prepare_custom_current_source
    from witwin.maxwell.scene import prepare_scene

    scene = prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((-length, length),) * 3), grid=mw.GridSpec.uniform(.5 * length),
        boundary=mw.BoundarySpec.none(), device="cpu",
    ))
    shape = (5, 5, 5)
    peak = 1.
    arrays = {"Jx": torch.full(shape, peak, dtype=torch.float64).numpy(),
              "Jy": torch.full(shape, peak * ratio, dtype=torch.float64).numpy(),
              "Mz": torch.full(shape, peak * 1e-10, dtype=torch.float64).numpy()}
    dataset = mw.CurrentDataset(([-length, -.5 * length, 0., .5 * length, length],) * 3, arrays)
    for zero in (False, True):
        if zero:
            dataset = mw.CurrentDataset(dataset.coords, {key: array * 0 for key, array in arrays.items()})
        solver = SimpleNamespace(
            scene=scene, device=torch.device("cpu"), Ex=torch.empty(1, dtype=dtype), dt=length / 299792458,
            _COMPONENT_HALF_OFFSET_AXES={"Ex": (True, False, False), "Ey": (False, True, False), "Hz": (True, True, False)},
            eps_Ex=torch.ones(shape, dtype=dtype), eps_Ey=torch.ones(shape, dtype=dtype), mu_Hz=torch.ones(shape, dtype=dtype),
            _source_terms=[], _magnetic_source_terms=[],
        )
        _prepare_custom_current_source(solver, {"dataset": dataset, "source_time": {"kind": "cw", "frequency": 1.}}, source_index=0)
        expected_electric = [] if zero else (["Ex", "Ey"] if ratio else ["Ex"])
        assert [term["field_name"] for term in solver._source_terms] == expected_electric
        assert [term["field_name"] for term in solver._magnetic_source_terms] == ([] if zero else ["Hz"])
        if ratio and not zero:
            weak = solver._source_terms[1]
            # A constant dataset samples to its value times dual-cell coverage.
            from witwin.maxwell.fdtd.excitation.injection import _yee_box_overlap
            _, _, weights = _yee_box_overlap(solver, "Ey", (-length,) * 3, (length,) * 3, current_dataset=True)
            expected = -solver.dt * (torch.as_tensor(peak * ratio, dtype=dtype) * weights.to(dtype))
            torch.testing.assert_close(weak["patch"], expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("profile_dtype", [torch.float32, torch.float64, torch.complex128])
def test_mode_surface_profile_counts_ignore_roundoff(length, dtype, profile_dtype, monkeypatch):
    from witwin.maxwell.fdtd.excitation import injection

    source = {"normal_axis": "x", "direction_sign": 1, "source_time": {"kind": "cw", "frequency": 1. / length}}
    scene = SimpleNamespace(x_half=torch.tensor([-.5, .5, 1.5, 2.5]) * length,
                            **{f"d{axis}_{kind}64": [length] * 5
                               for axis in "xyz" for kind in ("primal", "dual")})
    solver = SimpleNamespace(scene=scene, Ex=torch.empty(1, dtype=dtype), dt=length / 299792458,
                             _plane_coordinate=lambda axis, index: index * length,
                             _electric_source_terms=[], _magnetic_source_terms=[],
                             coefficient=torch.ones((5, 5, 5), dtype=dtype))
    monkeypatch.setattr(injection, "_surface_plane_spec_positions",
                        lambda solver, spec: torch.zeros((*spec["shape"], 3), dtype=dtype))
    monkeypatch.setattr(injection, "E_CURL_ATTR", {name: "coefficient" for name in ("Ex", "Ey", "Ez")})
    monkeypatch.setattr(injection, "H_CURL_ATTR", {name: "coefficient" for name in ("Hx", "Hy", "Hz")})
    for residual, expected in ((0., (1, 1)), (math.cos(math.pi / 2), (1, 1)),
                               (128 * torch.finfo(torch.float64).eps, (2, 2)), (1e-5, (2, 2)), (1e-6, (2, 2))):
        profiles = {name: torch.full((3, 3), value * 1e-10 * length, dtype=profile_dtype)
                    for name, value in {"Ex": 0., "Ey": 1., "Ez": residual,
                                        "Hx": 0., "Hy": residual * 1e-10, "Hz": 1e-10}.items()}
        data = {"component_profiles": profiles, "normal_axis": "x", "tangential_axes": ("y", "z"),
                "coords_u": torch.linspace(-length, length, 3, dtype=dtype),
                "coords_v": torch.linspace(-length, length, 3, dtype=dtype),
                "box_lower": (2, 1, 1), "box_upper": (3, 3, 3), "plane_index": 2,
                "effective_index": 1., "beta": 1. / length}
        monkeypatch.setattr(injection, "solve_mode_source_profile", lambda solver, source: data)
        solver._electric_source_terms.clear()
        solver._magnetic_source_terms.clear()
        injection._prepare_mode_surface_source(solver, source, source_index=0)
        assert (len(solver._electric_source_terms), len(solver._magnetic_source_terms)) == expected


@pytest.mark.parametrize("length", [1., 1e-6])
def test_field_sheet_profiles_skip_only_exact_zero(length, monkeypatch):
    import witwin.maxwell as mw
    from witwin.maxwell.constants import C_0
    from witwin.maxwell.fdtd.excitation import injection
    from witwin.maxwell.scene import prepare_scene

    scene = prepare_scene(mw.Scene(
        domain=mw.Domain(bounds=((-length, length),) * 3), grid=mw.GridSpec.uniform(.5 * length),
        boundary=mw.BoundarySpec.none(), device="cpu",
    ))
    solver = SimpleNamespace(scene=scene, device=torch.device("cpu"), Ex=torch.empty(1, dtype=torch.float64),
                             c=C_0, dt=length / (100 * C_0), coefficient=torch.ones((5, 5, 5), dtype=torch.float64),
                             _electric_source_terms=[], _magnetic_source_terms=[])
    monkeypatch.setattr(injection, "E_CURL_ATTR", {name: "coefficient" for name in ("Ex", "Ey", "Ez")})
    monkeypatch.setattr(injection, "H_CURL_ATTR", {name: "coefficient" for name in ("Hx", "Hy", "Hz")})
    for residual in (0., math.cos(math.pi / 2), 1e-6):
        profiles = {name: torch.full((1, 5, 5), value * 1e-10 * length, dtype=torch.float64).numpy()
                    for name, value in {"Ey": 1., "Ez": residual, "Hz": 1e-10, "Hy": residual * 1e-10}.items()}
        dataset = mw.FieldDataset(([0.], [-length, -.5 * length, 0., .5 * length, length],
                                   [-length, -.5 * length, 0., .5 * length, length]), profiles)
        solver._electric_source_terms.clear()
        solver._magnetic_source_terms.clear()
        injection._prepare_custom_field_source(solver, {
            "dataset": dataset, "normal_axis": "x",
            "source_time": {"kind": "cw", "frequency": C_0 / (.5 * length)},
        }, source_index=0)
        # The sheet is on an E_y node; H_z uses both bracketing half nodes.
        # Keep every supplied nonzero component, including numerical residuals.
        assert {term["field_name"] for term in solver._electric_source_terms} == ({"Ey", "Ez"} if residual else {"Ey"})
        assert {term["field_name"] for term in solver._magnetic_source_terms} == ({"Hz", "Hy"} if residual else {"Hz"})
