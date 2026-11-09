"""Scale covariance of Yee modal power, including zero and cancellation controls."""

import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.constants import C_0, EPSILON_0
from witwin.maxwell.fdtd.excitation.modes import solve_mode_source_profile
from witwin.maxwell.fdtd.excitation.modes.common import (
    _discrete_mode_profile_power,
    _normalize_mode_profiles_to_unit_power,
)
from witwin.maxwell.scene import prepare_scene


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))]


def _profiles(dtype, device, staggered, *, cancellation=0.0):
    # Nonuniform coordinates and both transverse terms distinguish the reference
    # quadrature from a node sum, an area estimate, or abs(integrated power).
    u = torch.tensor([0., .1, .4, .7, 1.], dtype=dtype, device=device)
    v = torch.tensor([0., .2, .6, 1.], dtype=dtype, device=device)
    shape = (len(u), len(v))
    p = {name: torch.zeros(shape, dtype=dtype, device=device) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    eu = torch.ones((len(u)-1, len(v)) if staggered else shape, dtype=dtype, device=device)
    ev = torch.ones((len(u), len(v)-1) if staggered else shape, dtype=dtype, device=device)
    p.update(Ey=eu, Hz=eu / 200, Ez=ev, Hy=ev * cancellation / 200)
    return p, u, v


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("direction", [1, -1])
def test_power_normalization_scales_and_preserves_valid_arithmetic(device, dtype, staggered, direction):
    p, u, v = _profiles(dtype, device, staggered, cancellation=-.3)
    for name in ("Hx", "Hy", "Hz"):
        p[name] = direction * p[name]
    normalized = []
    for length in (1., 1e-6):
        args = dict(coords_u=u*length, coords_v=v*length, normal_axis="x", staggered=staggered)
        power = _discrete_mode_profile_power(p, **args)
        result = _normalize_mode_profiles_to_unit_power(p, **args)
        # The valid-mode normalization arithmetic must remain bitwise identical.
        for name in p:
            assert torch.equal(result[name], p[name] * torch.rsqrt(power.abs()))
        assert power.sign().item() == direction
        assert _discrete_mode_profile_power(result, **args).item() == pytest.approx(direction, rel=2e-6)
        normalized.append(result)
    for name in p:
        torch.testing.assert_close(normalized[0][name], normalized[1][name]*1e-6, rtol=2e-6, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("case", ["zero", "cancellation", "reactive", "resolved"])
def test_zero_power_guard_uses_the_yee_reference(device, dtype, staggered, case):
    p, u, v = _profiles(dtype, device, staggered)
    args = dict(coords_u=u, coords_v=v, normal_axis="x", staggered=staggered)
    pos = _discrete_mode_profile_power(p, **args)
    negative = {name: value.clone() for name, value in p.items()}
    negative["Ey"].zero_()
    negative["Hy"] = negative["Ez"] / 200
    neg = _discrete_mode_profile_power(negative, **args)
    if case == "zero":
        p = {name: value * 0 for name, value in p.items()}
    elif case == "reactive":
        p = {name: value.to(torch.complex64 if dtype == torch.float32 else torch.complex128) for name, value in p.items()}
        p["Hz"] *= 1j
    else:
        residual = (16 if case == "cancellation" else 512) * torch.finfo(dtype).eps
        p["Hy"] = p["Ez"] * (pos / -neg) * (1-residual) / 200
    for length in (1., 1e-6):
        scaled = dict(args, coords_u=u*length, coords_v=v*length)
        reference = _discrete_mode_profile_power(p, magnitude_reference=True, **scaled)
        if case == "resolved":
            assert reference > 0
            _normalize_mode_profiles_to_unit_power(p, **scaled)
        else:
            with pytest.raises(RuntimeError, match="zero integrated Poynting"):
                _normalize_mode_profiles_to_unit_power(p, **scaled)


def _waveguide(length, direction, path, device, *, roundoff=False, trainable=False,
               weak_component=0., dtype=torch.float64):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-length, length),)*3),
        grid=mw.GridSpec.uniform(.1*length), boundary=mw.BoundarySpec.none(), device=device,
        subpixel_samples=mw.SubpixelSpec(samples=1, averaging="kottke_edge"),
    )
    scene.add_structure(mw.Structure(name="guide", geometry=mw.Box(position=(0.,0.,0.), size=(2*length,.45*length,.25*length)), material=mw.Material(eps_r=12.11)))
    source = mw.ModeSource(position=(0.,0.,0.), size=(0.,1.2*length,.8*length), polarization="Ey", direction=direction, source_time=mw.CW(frequency=C_0/(1.55*length)))
    scene.add_source(source)
    prepared = prepare_scene(scene)
    materials = dict(prepared.compile_materials())
    # Fix the sampled waveguide cross-section at both scales to isolate the
    # mode-power scale contract from geometry sampling.
    y, z = torch.meshgrid(prepared.y / length, prepared.z / length, indexing="ij")
    eps = torch.where((y.abs() < .225) & (z.abs() < .125), 12.11, 1.)
    eps = eps[None].expand(len(prepared.x), -1, -1).contiguous()
    eps.requires_grad_(trainable)
    materials["eps_components"] = {axis: eps for axis in "xyz"}
    context = SimpleNamespace(scene=prepared, Ex=torch.empty(1, dtype=dtype, device=device), c=C_0, _compiled_material_model=materials)
    if path == "scalar":
        from witwin.maxwell.compiler.materials import evaluate_material_components
        from witwin.maxwell.fdtd.excitation.modes.common import _average_node_tensor_to_component
        eps, _ = evaluate_material_components(materials, source.source_time.frequency)
        context._mode_source_rebuild_from_fields = True
        context.eps_Ey = EPSILON_0 * _average_node_tensor_to_component(eps["y"], "Ey")
    compiled = _compile_mode_source(source, default_frequency=source.source_time.frequency)
    if roundoff:
        residual = math.cos(math.pi / 2)
        compiled.update(direction_vector=(1., residual, residual), polarization=(residual, 1., residual))
    if weak_component:
        compiled.update(direction_vector=(1., 0., 0.), polarization=(0., 1., weak_component))
    return solve_mode_source_profile(context, compiled)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("path", ["scalar", "vector"])
def test_waveguide_sign_and_profiles_are_length_covariant(device, direction, path):
    modes = [_waveguide(length, direction, path, device) for length in (1.,1e-6)]
    # Document the existing scalar sign discrepancy; do not silently correct it.
    expected_sign = (1 if direction == "+" else -1) * (-1 if path == "scalar" else 1)
    for mode in modes:
        power = _discrete_mode_profile_power(mode["component_profiles"], coords_u=mode["coords_u"], coords_v=mode["coords_v"], normal_axis="x")
        assert power.item() == pytest.approx(expected_sign, rel=2e-6)
    assert modes[0]["effective_index"] == pytest.approx(modes[1]["effective_index"], rel=1e-6)
    assert modes[0]["beta"] == pytest.approx(modes[1]["beta"] * 1e-6, rel=1e-6)
    for name in modes[0]["component_profiles"]:
        torch.testing.assert_close(modes[0]["component_profiles"][name], modes[1]["component_profiles"][name]*1e-6, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("trainable", [False, True])
def test_scalar_mode_cross_product_discards_roundoff_components(length, trainable):
    exact = _waveguide(length, "+", "scalar", "cpu", trainable=trainable)
    noisy = _waveguide(length, "+", "scalar", "cpu", roundoff=True, trainable=trainable)
    for name in exact["component_profiles"]:
        assert torch.equal(exact["component_profiles"][name], noisy["component_profiles"][name])
    assert [name for name, profile in noisy["component_profiles"].items() if bool(profile.abs().max() > 0)] == ["Ey", "Hz"]
    if trainable:
        assert noisy["component_profiles"]["Hz"].requires_grad


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ratio", [1e-5, 1e-6])
def test_scalar_mode_preserves_weak_real_vector_components(length, dtype, ratio):
    mode = _waveguide(length, "+", "scalar", "cpu", trainable=True, weak_component=ratio, dtype=dtype)
    profiles = mode["component_profiles"]
    assert [name for name, profile in profiles.items() if bool(profile.abs().max() > 0)] == ["Ey", "Ez", "Hy", "Hz"]
    torch.testing.assert_close(profiles["Ez"], ratio * profiles["Ey"], rtol=2e-6, atol=0)
    torch.testing.assert_close(profiles["Hy"], -ratio * profiles["Hz"], rtol=2e-6, atol=0)
    assert profiles["Hy"].requires_grad
