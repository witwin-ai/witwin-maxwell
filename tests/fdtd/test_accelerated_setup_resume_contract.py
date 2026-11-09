"""Cross-path material identity and disk checkpoint resume contracts."""

from dataclasses import fields, is_dataclass

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.resume import _configuration_fingerprint


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _scene(kind):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.32, 0.32),) * 3),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pec(),
        subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_edge"),
        device="cuda",
    )
    if kind == "lorentz_sphere":
        geometry = mw.Sphere(position=(0.04, 0.0, 0.0), radius=0.13)
        material = mw.Material.lorentz(
            eps_inf=2.0, delta_eps=1.0, resonance_frequency=1.8e9, gamma=1e8,
        )
    else:
        geometry = mw.Box(position=(0.04, 0.0, 0.0), size=(0.22, 0.26, 0.24))
        material = mw.Material(eps_r=3.0, sigma_e=0.02 if kind == "lossy_pec" else 0.0)
    scene.add_structure(mw.Structure(geometry=geometry, material=material))
    if kind == "lossy_pec":
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0.16, 0.0, 0.0), size=(0.08,) * 3),
            material=mw.Material(pec=True),
        ))
    scene.add_source(mw.PointDipole(
        position=(-0.14, 0.0, 0.0), polarization="Ez",
        source_time=mw.CW(frequency=1e9),
    ))
    scene.add_monitor(mw.PointMonitor(
        "probe", position=(-0.10, 0.02, 0.0), fields=_COMPONENTS,
    ))
    scene.add_monitor(mw.FieldTimeMonitor(
        "trace", position=(-0.10, 0.02, 0.0), components=_COMPONENTS, interval=1,
    ))
    return scene


def _assert_equal_by_key(actual, expected, path):
    if isinstance(expected, torch.Tensor):
        assert (actual.dtype, actual.shape, actual.device) == (
            expected.dtype, expected.shape, expected.device,
        ), path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_equal_by_key(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, value in enumerate(expected):
            _assert_equal_by_key(actual[index], value, f"{path}[{index}]")
    elif is_dataclass(expected):
        assert type(actual) is type(expected), path
        for field in fields(expected):
            _assert_equal_by_key(
                getattr(actual, field.name), getattr(expected, field.name),
                f"{path}.{field.name}",
            )
    else:
        assert actual == expected, path


@pytest.mark.parametrize("kind", ["lorentz_sphere", "dielectric_box", "lossy_pec"])
def test_accelerated_setup_resume_contract(kind, tmp_path):
    """_hash_value sorts mapping keys, so fingerprints ignore dict insertion order."""
    scene = _scene(kind)

    def prepare(accelerated_setup):
        return mw.Simulation.fdtd(
            scene, frequency=1e9, accelerated_setup=accelerated_setup, fdtd_core_path=False,
            run_time=mw.TimeConfig(time_steps=40),
            cuda_graph=False, full_field_dft=False,
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        ).prepare()

    executions = {enabled: prepare(enabled) for enabled in (True, False)}
    accelerated = executions[True].solver
    reference = executions[False].solver
    _assert_equal_by_key(
        accelerated._compiled_material_model, reference._compiled_material_model,
        "_compiled_material_model",
    )
    assert _configuration_fingerprint(accelerated) == _configuration_fingerprint(reference)

    for source_enabled, target_enabled in ((True, False), (False, True)):
        checkpoint = executions[source_enabled].run_until(17)
        assert checkpoint.step == 17
        path = tmp_path / f"setup-{source_enabled}.pt"
        checkpoint.save(path)
        loaded = mw.FDTDResumeCheckpoint.load(path)
        resumed = prepare(target_enabled).run(resume_from=loaded)
        direct = prepare(target_enabled).run()
        assert resumed.stats()["steps_run"] == direct.stats()["steps_run"] == 40
        assert torch.count_nonzero(direct.solver.Ez) > 0
        for name in _COMPONENTS:
            assert torch.equal(
                getattr(resumed.solver, name), getattr(direct.solver, name),
            ), (source_enabled, target_enabled, name)
        for name in ("probe", "trace"):
            _assert_equal_by_key(
                resumed.monitor(name), direct.monitor(name),
                f"resume({source_enabled}->{target_enabled}).{name}",
            )
