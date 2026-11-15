"""FDTD never drops a gradient silently: every trainable input is differentiated or refused by name.

CPU tests stop at admission: the refusals fire before any CUDA work, and an
admitted CPU scene reaches the bridge's CUDA requirement instead. GPU tests prove
the newly routed inputs against central differences.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.execution.adjoint import validate_material_channels
from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge
from witwin.maxwell.fdtd.time_traces import guard_nondifferentiable_traces

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9
_ADMITTED = "Differentiable FDTD RF and material inputs must be CUDA tensors"


# --------------------------------------------------------------------------- CPU


def _scene(*structures, regions=()):
    # Geometry gradients need a differentiable averaging.
    scene = mw.Scene(
        subpixel=mw.SubpixelSpec(samples=2, averaging="polarized"),
        domain=mw.Domain(bounds=((-0.48, 0.48),) * 3),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
    )
    scene = scene.updated(structures=(*scene.structures, *structures))
    scene = scene.updated(material_regions=(*scene.material_regions, *regions))
    scene = scene.updated(sources=(
        *scene.sources,
        mw.PointDipole(position=(0.0, 0.0, -0.3), polarization="Ez", width=0.04,
                       source_time=mw.GaussianPulse(frequency=FREQ, fwidth=0.25 * FREQ)),
    ), monitors=(*scene.monitors, mw.PointMonitor("probe", (0.0, 0.0, 0.3), fields=("Ez",))))
    return scene


def _box(material, position=(0.12, 0.0, 0.0), name=None, **kwargs):
    return mw.Structure(name=name, geometry=mw.Box(position=position, size=(0.24, 0.24, 0.24)),
                        material=material, **kwargs)


def _moving(x=0.12):
    return torch.stack((torch.tensor(x, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0)))


def _trainable(value):
    return torch.tensor(value, requires_grad=True)


def _region(*, density_grad=True, geometry=(0.0, 0.0, 0.0), eps_bounds=(1.0, 4.0), mu_bounds=(1.0, 1.0)):
    return mw.MaterialRegion(name="design", geometry=mw.Box(position=geometry, size=(0.36, 0.36, 0.36)),
                             density=torch.full((3, 3, 3), 0.5, requires_grad=density_grad),
                             eps_bounds=eps_bounds, mu_bounds=mu_bounds)


def _refusal(run, feature, *fragments):
    with pytest.raises(mw.UnsupportedFeatureError) as caught:
        run()
    error = caught.value
    assert (error.feature, error.method) == (feature, "fdtd")
    for fragment in fragments:
        assert any(fragment in item for item in error.items), (fragment, error.items)
        assert fragment in str(error)
    return error


def _run(scene_like):
    return mw.Simulation.fdtd(scene_like, frequencies=[FREQ], config=mw.FDTDConfig(device="cpu")).run()


class _PecPositionScene(mw.SceneModule):
    """The partial-admission probe: a trainable dielectric next to a trainable PEC position."""

    def __init__(self):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor(3.0))
        self.pec_x = torch.nn.Parameter(torch.tensor(0.06))

    def to_scene(self):
        position = torch.stack((self.pec_x, self.pec_x.new_tensor(0.0), self.pec_x.new_tensor(0.12)))
        return _scene(_box(mw.Material(eps_r=self.eps), position=(0.0, 0.0, -0.12), name="dielectric"),
                      _box(mw.Material(pec=True), position=position, name="plate"))


def test_trainable_pec_position_next_to_trainable_dielectric_is_refused_by_name():
    # Before p2f the bridge differentiated eps only and pec_x.grad stayed None.
    _refusal(lambda: _run(_PecPositionScene()), "gradient.material_channel",
             "scene.structures[1] ('plate'): trainable geometry of a PEC structure")


def test_direct_pec_geometry_next_to_trainable_dielectric_is_refused_by_name():
    scene = _scene(_box(mw.Material(eps_r=_trainable(3.0)), position=(0.0, 0.0, -0.12)),
                   _box(mw.Material(pec=True), position=_moving(), name="plate"))
    _refusal(lambda: _run(scene), "gradient.material_channel", "trainable geometry of a PEC structure")


@pytest.mark.parametrize("material", [mw.Material(eps_r=2.0, mu_r=2.0), mw.Material(eps_r=2.0, sigma_m=10.0)],
                         ids=["permeability", "magnetic_conductivity"])
def test_moving_boundary_in_a_magnetic_scene_is_refused(material):
    scene = _scene(_box(mw.Material(eps_r=3.0), position=_moving(), name="mover"),
                   _box(material, position=(-0.24, 0.0, 0.0), name="magnet"))
    _refusal(lambda: _run(scene), "gradient.material_channel",
             "trainable geometry of scene.structures[0]", "scene.structures[1] ('magnet')")


def test_trainable_dielectric_geometry_next_to_fixed_pec_stays_admitted():
    scene = _scene(_box(mw.Material(eps_r=3.0), position=_moving()),
                   _box(mw.Material(pec=True), position=(-0.24, 0.0, 0.0)))
    with pytest.raises(RuntimeError, match=_ADMITTED):
        _run(scene)


def test_material_region_permeability_contrast_is_refused_on_single_device_and_batch():
    scene = _scene(regions=[_region(eps_bounds=(1.0, 3.0), mu_bounds=(1.0, 2.0))])
    _refusal(lambda: _run(scene), "gradient.material_channel", "mu_bounds=(1.0, 2.0)")
    _refusal(lambda: validate_material_channels(scene), "gradient.material_channel", "mu_bounds=(1.0, 2.0)")


@pytest.mark.parametrize("material", [mw.Material(pec=True), mw.Material(eps_r=2.0, sigma_e=0.1)], ids=["pec", "conductive"])
def test_batch_geometry_adjoint_keeps_refusing_pec_and_conductive_structures(material):
    # The single-device bridge admits a fixed PEC or a conductor next to a moving
    # dielectric; the batch material replay still has no channel for them.
    scene = _scene(_box(mw.Material(eps_r=3.0), position=_moving()), _box(material, position=(-0.24, 0.0, 0.0)))
    with pytest.raises(NotImplementedError, match="Batch geometry adjoint requires nonconductive structures without PEC"):
        validate_material_channels(scene)


def test_batch_moving_boundary_in_a_magnetic_scene_is_refused_by_the_shared_check():
    scene = _scene(_box(mw.Material(eps_r=3.0), position=_moving()),
                   _box(mw.Material(mu_r=2.0), position=(-0.24, 0.0, 0.0), name="magnet"))
    _refusal(lambda: validate_material_channels(scene), "gradient.material_channel", "scene.structures[1] ('magnet')")


def test_distributed_region_permeability_contrast_is_refused_before_allocation():
    scene = _scene(regions=[_region(eps_bounds=(1.0, 1.0), mu_bounds=(1.0, 4.0))])
    scene = scene.updated(boundary=mw.BoundarySpec.none())
    simulation = mw.Simulation.fdtd(scene, frequencies=[FREQ], config=mw.FDTDConfig(parallel=mw.fdtd.FDTDParallelConfig(
        devices=("cuda:0", "cuda:1"), transport="cuda_p2p", gather_fields=False, result_device="cuda:0")))
    _refusal(simulation.prepare, "gradient.material_channel", "mu_bounds=(1.0, 4.0)")


def test_trainable_input_of_a_disabled_structure_is_refused_by_name():
    scene = _scene(_box(mw.Material(eps_r=_trainable(3.0))),
                   _box(mw.Material(eps_r=_trainable(2.0)), position=(-0.24, 0.0, 0.0), enabled=False))
    error = _refusal(lambda: _run(scene), "gradient.unclaimed_tensor", "scene.structures[1].material.eps_r")
    assert error.items == ("scene.structures[1].material.eps_r",)


def _modulated(amplitude):
    return mw.Material(eps_r=2.0, modulation=mw.media.ModulationSpec(frequency=1.0e8, amplitude=amplitude))


class _PermeabilityScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor(3.0))

    def to_scene(self):
        # One parameter drives both eps_r (claimed) and mu_r (no channel).
        return _scene(_box(mw.Material(eps_r=self.eps, mu_r=self.eps)))


@pytest.mark.parametrize(
    ("build", "path"),
    [
        (lambda: _scene(_box(mw.Material(eps_r=2.0, mu_r=_trainable(2.0)))), "scene.structures[0].material.mu_r"),
        (lambda: _scene(_box(_modulated(torch.full((3, 3, 3), 0.1, requires_grad=True)))),
         "scene.structures[0].material.modulation.amplitude"),
        (lambda: _scene(_box(mw.Material(eps_r=_trainable(3.0), mu_r=_trainable(2.0)))),
         "scene.structures[0].material.mu_r"),
    ],
    ids=["permeability", "modulation_amplitude", "permeability_next_to_permittivity"],
)
def test_uncensused_trainable_tensors_are_refused_at_construction(build, path):
    scene_like = build()
    error = _refusal(lambda: mw.Simulation.fdtd(scene_like, frequencies=[FREQ], config=mw.FDTDConfig(device="cpu")), "gradient.unclaimed_tensor", path)
    assert error.items == (path,)


def test_permeability_sharing_a_permittivity_input_is_refused_before_the_run():
    # The construction census sees the shared tensor through eps_r; the channel
    # check still names the permeability it also drives.
    _refusal(lambda: _run(_PermeabilityScene()), "gradient.material_channel",
             "scene.structures[0] (None): trainable permeability mu_r")


def test_uncensused_trainable_tensors_run_forward_only_under_no_grad():
    scene = _scene(_box(mw.Material(eps_r=2.0, mu_r=_trainable(2.0))))
    with torch.no_grad():
        simulation = mw.Simulation.fdtd(scene, frequencies=[FREQ], config=mw.FDTDConfig(device="cpu"))
    assert not simulation.has_trainable_parameters
    with pytest.raises(mw.UnsupportedFeatureError, match="mu_r"):
        simulation.run()


def test_mass_density_stays_a_postprocessing_input():
    density = torch.full((3, 3, 3), 1000.0, requires_grad=True)
    scene = _scene(_box(mw.Material(eps_r=2.0, sigma_e=0.1, mass_density=density)))
    assert not mw.Simulation.fdtd(scene, frequencies=[FREQ], config=mw.FDTDConfig(device="cpu")).has_trainable_parameters


def test_material_region_geometry_is_routed_through_the_bridge():
    from witwin.maxwell.adjoint_inputs import material_dependent_inputs, scene_trainable_material_tensors

    region = _region(density_grad=False, geometry=_moving(0.0))
    scene = _scene(regions=[region])
    simulation = mw.Simulation.fdtd(scene, frequencies=[FREQ], config=mw.FDTDConfig(device="cpu"))
    assert simulation.has_trainable_parameters
    position = region.geometry.position
    assert any(tensor is position for tensor in material_dependent_inputs(scene, scene_trainable_material_tensors(scene), device="cpu"))
    with pytest.raises(RuntimeError, match=_ADMITTED):
        _FDTDGradientBridge(simulation)


def _payloads():
    t = torch.arange(3.0)
    trace = torch.ones(3)
    return {
        "field": {"kind": "field_time", "t": t, "components": {"Ez": trace}},
        "flux": {"kind": "flux_time", "t": t, "flux": trace},
        "spectral": {"kind": "point", "components": {"Ez": trace}},
    }


def test_field_and_flux_time_traces_refuse_backward():
    monitors = guard_nondifferentiable_traces(_payloads())
    field = monitors["field"]
    assert "data" not in field and "field" not in field
    assert not monitors["spectral"]["components"]["Ez"].requires_grad
    for name, value in (("field", field["components"]["Ez"]), ("flux", monitors["flux"]["flux"])):
        assert torch.equal(value.detach(), torch.ones(3))
        with pytest.raises(RuntimeError, match=f"Monitor '{name}'.*detached time-domain traces"):
            value.sum().backward()


# --------------------------------------------------------------------------- GPU


def _region_scene(x):
    position = torch.stack((x, x.new_tensor(0.0), x.new_tensor(0.06)))
    density = torch.linspace(0.2, 0.9, 27, device="cuda").reshape(3, 3, 3)
    scene = mw.Scene(
                subpixel=mw.SubpixelSpec(samples=5, averaging="polarized"),
                domain=mw.Domain(bounds=((-0.24, 0.24),) * 3),
                grid=mw.GridSpec.uniform(0.06),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                material_regions=(
            mw.MaterialRegion(name="design", geometry=mw.Box(position=position, size=(0.18,) * 3),
                              density=density, eps_bounds=(1.0, 6.0)),
            ),
                sources=(
            mw.PointDipole(position=(0.0, 0.0, -0.12), polarization="Ez", width=0.04,
                           source_time=mw.GaussianPulse(frequency=FREQ, fwidth=0.25 * FREQ, amplitude=5000.0)),
            ),
                monitors=(
            mw.PointMonitor("probe", (0.0, 0.0, 0.12), fields=("Ez",)),
            mw.monitors.FieldTimeMonitor("trace", components=("Ez",), position=(0.0, 0.0, 0.12), interval=1),
            ),
            )
    return scene


def _region_simulation(x):
    return mw.Simulation.fdtd(_region_scene(x), frequencies=[FREQ], config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=60),
                              spectral_sampler=mw.fdtd.SpectralSampler(window="none"), full_field_dft=False))


def _probe_power(result):
    return result.monitor("probe", freq_index=0).sel(component="Ez").values.abs().square().sum()


@requires_cuda
def test_material_region_geometry_gradient_matches_central_difference():
    x = torch.tensor(0.0137, device="cuda", requires_grad=True)
    result = _region_simulation(x).run()
    _probe_power(result).backward()
    assert x.grad is not None and bool(x.grad != 0)
    # The sub-sample fill is piecewise smooth in the position: central
    # differences converge to the adjoint as the step shrinks (relative error
    # 3.1e-2, 6.2e-3, 1.5e-3, 2.2e-5 at h = 2e-3, 1e-3, 5e-4, 2.5e-4).
    delta = 2.5e-4
    with torch.no_grad():
        plus, minus = (_probe_power(_region_simulation(x.detach() + step).run()) for step in (delta, -delta))
    finite_difference = (plus - minus) / (2.0 * delta)
    torch.testing.assert_close(x.grad, finite_difference.to(x.grad), rtol=1e-3, atol=0.0)


@requires_cuda
def test_field_time_trace_of_a_differentiable_run_refuses_backward():
    x = torch.tensor(0.0137, device="cuda", requires_grad=True)
    trace = _region_simulation(x).run().monitor("trace").sel(component="Ez").values
    with pytest.raises(RuntimeError, match="Monitor 'trace' \\(field_time\\) records detached time-domain traces"):
        trace.square().sum().backward()
