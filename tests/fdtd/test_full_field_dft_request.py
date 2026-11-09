"""Explicit volume-DFT requests preserve spectral observations and their VJP."""
import dataclasses
import re

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.execution import estimate_simulation_footprint_bytes
from tests.gradients.test_fdtd_adjoint_bridge import _DensityFluxScene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
FREQUENCIES = (0.8e9, 1.0e9)
MISSING_FIELDS = (
    "This FDTD run did not record frequency-domain volume fields "
    "(full_field_dft=False); pass full_field_dft=True."
)


def _scene(*, gap_port=False):
    scene = _DensityFluxScene().to_scene()
    scene.material_regions.clear()
    scene.add_monitor(mw.PointMonitor("point", (0., 0., 0.), fields=("Ez", "Hy")))
    scene.add_monitor(mw.PlaneMonitor("plane", axis="z", position=0.06, fields=("Ex", "Ey", "Hx", "Hy")))
    if gap_port:
        scene.add_port(mw.LumpedPort(
            "feed", positive=(0., 0., 0.12), negative=(0., 0., 0.),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(position=(0., 0., 0.06), size=(0.12, 0.12, 0.)),
            termination=mw.SeriesRLC(r=50.),
        ))
    return scene


def _simulation(scene, **kwargs):
    return mw.Simulation.fdtd(
        scene, frequencies=FREQUENCIES, cuda_graph=False,
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"), **kwargs,
    )


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


def test_default_multifrequency_footprint_includes_volume_dft():
    scene = _scene()
    estimates = [
        estimate_simulation_footprint_bytes(_simulation(scene, full_field_dft=requested))
        for requested in (None, True, False)
    ]
    assert estimates[0] == estimates[1] > estimates[2]


def test_default_multifrequency_records_volume_fields():
    simulation = _simulation(_scene())
    assert simulation.config.full_field_dft is None
    result = simulation.run()
    assert result.fields and result.E.z.is_complex()
    assert result.solver_stats["full_field_dft"] is True
    assert result.solver_stats["full_field_dft_source"] == "auto"


@pytest.mark.parametrize("gap_port", [False, True])
def test_explicit_false_preserves_every_monitor_and_port_tensor(gap_port):
    scene = _scene(gap_port=gap_port)
    result = _simulation(scene, full_field_dft=False).run()
    reference = _simulation(scene, full_field_dft=True).run()
    assert result.solver_stats["full_field_dft"] is False
    assert result.solver_stats["full_field_dft_source"] == "explicit"
    assert not result.solver.dft_enabled and not result.solver._dft_entries
    assert not result._fields
    for actual, expected in ((result.monitors, reference.monitors), (result.ports, reference.ports)):
        actual_tensors, expected_tensors = list(_tensors(actual)), list(_tensors(expected))
        assert len(actual_tensors) == len(expected_tensors)
        assert all(torch.equal(a, b) for a, b in zip(actual_tensors, expected_tensors))
    if gap_port:
        assert list(_tensors(result.ports))
    for access in (
        lambda: result.E, lambda: result.H, lambda: result.field(),
        lambda: result.fields, lambda: result.at(freq_index=0).E.z,
        lambda: result._native_tensor("Ez"), lambda: result.tensor("Ez"),
    ):
        with pytest.raises(RuntimeError, match=re.escape(MISSING_FIELDS)):
            access()


@pytest.mark.parametrize("channel", ["conduction", "electric_dispersion"])
def test_explicit_false_defers_automatic_power_loss_to_evaluation(channel):
    scene = _scene()
    scene.add_monitor(mw.PowerLossMonitor("loss", position=(0.,)*3, size=(0.12,)*3, channels=(channel,)))
    simulation = _simulation(scene, full_field_dft=False)
    simulation.prepare()
    result = simulation.run()

    with pytest.raises(RuntimeError, match="full_field_dft=True or pass electric_fields"):
        result.power_loss("loss")

    if channel == "conduction":
        reference = _simulation(scene, full_field_dft=True).run()
        electric_fields = {
            component: torch.stack(
                [reference._native_tensor(component, frequency=frequency) for frequency in FREQUENCIES],
                dim=0,
            )
            for component in ("Ex", "Ey", "Ez")
        }
        actual = result.power_loss("loss", electric_fields=electric_fields)
        expected = reference.power_loss("loss")
        actual_tensors, expected_tensors = list(_tensors(actual)), list(_tensors(expected))
        assert len(actual_tensors) == len(expected_tensors)
        for index, (actual, expected) in enumerate(zip(actual_tensors, expected_tensors)):
            assert torch.equal(actual, expected), (
                f"Power-loss tensor {index} differs: max_abs={(actual - expected).abs().max().item()}"
            )


def _transmission_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.36, 0.36), (-0.36, 0.36), (-0.48, 0.48))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_source(mw.PointDipole(
        position=(0., 0., -0.18), polarization="Ex", width=0.04,
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.8e9, amplitude=50.),
    ))
    scene.add_monitor(mw.FluxMonitor("flux_probe", axis="z", position=0.18, normal_direction="+"))
    return scene


class _TransmissionPermittivityScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.eps_r = torch.nn.Parameter(torch.tensor(3.5, device="cuda"))

    def to_scene(self):
        scene = _transmission_scene()
        scene.add_structure(mw.Structure(
            name="design", geometry=mw.Box(position=(0., 0., 0.), size=(0.36, 0.36, 0.12)),
            material=mw.Material(eps_r=self.eps_r),
        ))
        return scene


class _TransmissionDensityScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros((1, 1, 1), device="cuda"))

    def to_scene(self):
        scene = _transmission_scene()
        scene.subpixel = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")
        scene.add_material_region(mw.MaterialRegion(
            name="design", geometry=mw.Box(position=(0., 0., 0.), size=(0.36, 0.36, 0.12)),
            density=self.logits.sigmoid(), eps_bounds=(1., 6.), mu_bounds=(1., 1.),
        ))
        return scene


def _comparison(actual, expected):
    difference = (actual - expected).abs()
    scale = torch.maximum(actual.abs(), expected.abs()).clamp_min(torch.finfo(actual.dtype).tiny)
    return {
        "equal": torch.equal(actual, expected),
        "max_abs": difference.max().item(),
        "max_rel": (difference / scale).max().item(),
    }


def _flux_gradient(model, parameter, requested):
    simulation = _simulation(model, full_field_dft=requested)
    simulation.config.run_time = mw.TimeConfig(time_steps=96)
    simulation.config.adjoint_checkpoint_stride = 16
    result = simulation.run()
    flux = result.monitor("flux_probe")["flux"]
    loss = flux.abs().square().sum()
    loss.backward()
    gradient = parameter.grad.detach().clone()
    print(f"full_field_dft={requested}: flux={flux.detach().tolist()}, "
          f"loss={loss.item():.17g}, gradient={gradient.flatten().tolist()}, "
          f"abs_gradient/loss={gradient.abs().max().item() / loss.item():.17g}")
    assert result.solver_stats["full_field_dft"] is requested
    if not requested:
        assert not result.solver._dft_entries and not result._fields
    return gradient, flux.detach().clone()


def test_explicit_false_flux_gradient_is_bitwise_equal():
    gradients, fluxes = [], []
    for requested in (True, True, False):
        model = _TransmissionPermittivityScene()
        gradient, flux = _flux_gradient(model, model.eps_r, requested)
        gradients.append(gradient)
        fluxes.append(flux)
        if len(gradients) == 2:
            print(f"gradient True/True: {_comparison(gradients[0], gradients[1])}")
            assert torch.equal(gradients[0], gradients[1]), "The True adjoint is not bitwise repeatable"
    print(f"gradient False/True: {_comparison(gradients[2], gradients[0])}")
    print(f"forward flux False/True: {_comparison(fluxes[2], fluxes[0])}")
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0].abs().max() > 1e-3 * fluxes[0].abs().square().sum()
    assert torch.equal(fluxes[2], fluxes[0])
    assert torch.equal(gradients[2], gradients[0])


def test_density_sampling_gradient_is_run_to_run_nondeterministic_within_roundoff():
    """CUDA 3-D grid_sample backward uses atomic adds for density sampling.

    MaterialRegion gradients can vary within roundoff between independent runs;
    this property is separate from the volume-DFT request. Equality is permitted.

    Deterministic mode raises: "grid_sampler_3d_backward_cuda does not have a
    deterministic implementation, but you set
    'torch.use_deterministic_algorithms(True)'. You can turn off determinism just
    for this operation, or you can use the 'warn_only=True' option, if that's
    acceptable for your application. You can also file an issue at
    https://github.com/pytorch/pytorch/issues to help us prioritize adding
    deterministic support for this operation."
    """
    gradients = []
    for _ in range(2):
        model = _TransmissionDensityScene()
        gradient, _ = _flux_gradient(model, model.logits, True)
        gradients.append(gradient)
    comparison = _comparison(*gradients)
    print(f"density gradient True/True: {comparison}")
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0].abs().max() > 0
    assert comparison["max_rel"] <= 1e-6

    texture = torch.ones((1, 1, 1, 1, 1), device="cuda", requires_grad=True)
    queries = torch.zeros((1, 1, 1, 2, 3), device="cuda")
    sampled = torch.nn.functional.grid_sample(texture, queries, align_corners=False)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="grid_sampler_3d_backward_cuda") as error:
            sampled.sum().backward()
        print(f"deterministic density sampling: {error.value}")
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
