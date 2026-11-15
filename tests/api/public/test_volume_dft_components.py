"""FDTDConfig.volume_dft_components: validation, refusals and the H mirror parity (CPU)."""
import types

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.errors import UnsupportedFeatureError
from witwin.maxwell.result import _expand_tensor_with_symmetry

EH = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def test_default_records_e_only():
    config = mw.FDTDConfig()
    assert config.volume_dft_components == ("Ex", "Ey", "Ez") and not config.records_magnetic_volume_dft


def test_components_are_canonical_and_accept_field_component_values():
    config = mw.FDTDConfig(full_field_dft=True,
                           volume_dft_components=("Hz", "Ez", mw.monitors.FieldComponent.EX, "Ey"))
    assert config.volume_dft_components == ("Ex", "Ey", "Ez", "Hz") and config.records_magnetic_volume_dft
    assert mw.FDTDConfig(full_field_dft=True, volume_dft_components=EH).volume_dft_components == EH


@pytest.mark.parametrize("components, error, match", [
    ("Ex", TypeError, "tuple of Yee component names"),
    (("Ex", "Ey"), ValueError, "must contain Ex, Ey and Ez"),
    (("Ex", "Ey", "Ez", "Ez"), ValueError, "unique Yee component names"),
    (("Ex", "Ey", "Ez", "Jx"), ValueError, "unique Yee component names"),
])
def test_invalid_components_raise(components, error, match):
    with pytest.raises(error, match=match):
        mw.FDTDConfig(full_field_dft=True, volume_dft_components=components)


@pytest.mark.parametrize("full_field_dft", [None, False])
def test_magnetic_components_require_an_explicit_volume_dft(full_field_dft):
    with pytest.raises(ValueError, match="require full_field_dft=True"):
        mw.FDTDConfig(full_field_dft=full_field_dft, volume_dft_components=EH)


def test_multi_gpu_refuses_the_magnetic_volume_dft():
    parallel = mw.fdtd.FDTDParallelConfig(devices=("cuda:0", "cuda:1"))
    with pytest.raises(UnsupportedFeatureError, match="Multi-GPU FDTD records no magnetic volume DFT"):
        mw.FDTDConfig(full_field_dft=True, volume_dft_components=EH, parallel=parallel)
    assert not mw.FDTDConfig(parallel=parallel).records_magnetic_volume_dft


def test_trainable_run_refuses_the_magnetic_volume_dft():
    eps = torch.tensor(2.0, requires_grad=True)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.02),
                     boundary=mw.BoundarySpec.pml(num_layers=4),
                     structures=(mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(0.04, 0.04, 0.04)),
                                              material=mw.Material(eps_r=eps), name="box"),),
                     sources=(mw.PointDipole(position=(-0.05, 0, 0), polarization="Ez",
                                             source_time=mw.GaussianPulse(1e9, fwidth=0.5e9)),))
    simulation = mw.Simulation.fdtd(scene, frequencies=(1e9,), config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=10), full_field_dft=True, volume_dft_components=EH, device="cpu"))
    assert simulation.has_trainable_parameters
    with pytest.raises(UnsupportedFeatureError, match="adjoint does not differentiate the magnetic volume DFT"):
        simulation.run()


def _expand_1d(component, entry):
    scene = types.SimpleNamespace(Nx=3, Ny=1, Nz=1, scene=types.SimpleNamespace(symmetry=(entry, None, None)))
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).reshape(-1, 1, 1)
    return _expand_tensor_with_symmetry(tensor, scene, component=component).reshape(-1).numpy()


@pytest.mark.parametrize("electric, magnetic", [("EX", "HX"), ("EY", "HY")])
@pytest.mark.parametrize("mode", ["PEC", "PMC"])
def test_h_mirrors_with_the_opposite_parity_of_e(electric, magnetic, mode):
    # H is a pseudovector: at a PEC plane H normal is odd and H tangential even (E the reverse).
    e = _expand_1d(electric, (mode, "low"))
    h = _expand_1d(magnetic, (mode, "low"))
    np.testing.assert_allclose(h[2:], e[2:])
    np.testing.assert_allclose(h[:2], -e[:2])


def test_batch_takes_volume_components_from_its_output_selection():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.02),
                     boundary=mw.BoundarySpec.pml(num_layers=4),
                     sources=(mw.PointDipole(position=(0, 0, 0), polarization="Ez",
                                             source_time=mw.GaussianPulse(1e9, fwidth=0.5e9)),))
    simulation = mw.Simulation.fdtd(scene, frequencies=(1e9,), config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=10), full_field_dft=True, volume_dft_components=EH, device="cpu"))
    with pytest.raises(ValueError, match="through BatchOutput.fields"):
        mw.execution.BatchSimulation([simulation], execution=mw.execution.BatchExecution(devices=("cuda:0",)),
                                     outputs=mw.execution.BatchOutput(fields=("Ex", "Ey", "Ez")))
