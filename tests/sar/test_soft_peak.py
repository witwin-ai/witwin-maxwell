"""Phase 4 slice: differentiable soft_peak surrogate over mass-averaged SAR."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.scene import prepare_scene


def _ramp_field_result(*, dx=0.1, sigma=0.5, rho=1000.0, freq=1.0e9, requires_grad=False):
    """Uniform-cube result whose field amplitude ramps along x, so SAR is non-uniform."""
    material = mw.Material(sigma_e=sigma, mass_density=rho, name="tissue")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 2.0),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(position=(1.0, 1.0, 1.0), size=(8.0, 8.0, 8.0)),
                material=material,
            ),
        ),
        device="cpu",
    )
    monitor = mw.PowerLossMonitor(
        "loss", position=(1.0, 1.0, 1.0), size=(1.0, 1.0, 1.0),
        frequencies=(freq,), channels=("conduction",),
    )
    scene.add_monitor(monitor)
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    fields = {}
    for component in ("Ex", "Ey", "Ez"):
        shape = compiled.full_component_shapes[component]
        ramp = torch.linspace(1.0, 3.0, shape[0], dtype=torch.float64)
        amp = ramp[:, None, None].expand(shape).clone().to(torch.complex128)
        amp.requires_grad_(requires_grad)
        fields[component.upper()] = amp
    result = mw.Result(
        method="fdtd", scene=scene, prepared_scene=prepared,
        frequencies=(freq,), fields=fields,
    )
    return result, dict(rho=rho, m0=rho * (3 * dx) ** 3)


def test_soft_peak_below_hard_peak_and_above_mean():
    result, params = _ramp_field_result()
    averaging = mw.SARAveraging(mass=(params["m0"],))
    sar = result.sar("loss", averaging=averaging)

    hard = float(sar.peak(params["m0"]).sar[0])
    field = sar.averaged_sar(params["m0"])[0]
    valid = torch.isfinite(field)
    mean = float(field[valid].mean())

    soft = float(sar.soft_peak(temperature=1e-3, mass=params["m0"])[0])
    assert mean <= soft <= hard + 1e-9
    assert soft > mean  # temperature-weighted, strictly above the plain mean


def test_soft_peak_approaches_hard_peak_as_temperature_drops():
    result, params = _ramp_field_result()
    averaging = mw.SARAveraging(mass=(params["m0"],))
    sar = result.sar("loss", averaging=averaging)
    hard = float(sar.peak(params["m0"]).sar[0])

    warm = float(sar.soft_peak(temperature=1.0, mass=params["m0"])[0])
    cold = float(sar.soft_peak(temperature=1e-6, mass=params["m0"])[0])
    assert abs(cold - hard) < abs(warm - hard)
    assert cold == pytest.approx(hard, rel=1e-3)


def test_soft_peak_single_mass_defaults_mass():
    result, params = _ramp_field_result()
    averaging = mw.SARAveraging(mass=(params["m0"],))
    sar = result.sar("loss", averaging=averaging)
    explicit = sar.soft_peak(temperature=0.5, mass=params["m0"])
    implicit = sar.soft_peak(temperature=0.5)
    torch.testing.assert_close(explicit, implicit)


def test_soft_peak_is_differentiable():
    result, params = _ramp_field_result(requires_grad=True)
    averaging = mw.SARAveraging(mass=(params["m0"],))
    sar = result.sar("loss", averaging=averaging)
    soft = sar.soft_peak(temperature=0.2, mass=params["m0"])
    soft[0].backward()
    grad = result.fields["EX"].grad
    assert grad is not None
    assert bool((grad.abs() > 0).any())


def test_soft_peak_requires_averaging():
    result, params = _ramp_field_result()
    sar = result.sar("loss")  # no averaging
    with pytest.raises(ValueError, match="mass-averaged"):
        sar.soft_peak(temperature=0.1, mass=params["m0"])


def test_soft_peak_rejects_nonpositive_temperature():
    result, params = _ramp_field_result()
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(params["m0"],)))
    with pytest.raises(ValueError, match="temperature"):
        sar.soft_peak(temperature=0.0, mass=params["m0"])
