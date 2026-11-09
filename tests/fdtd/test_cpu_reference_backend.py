"""CPU reference FDTD backend: kernel parity and explicit profile admission."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary.common import BOUNDARY_NONE, BOUNDARY_PEC, BOUNDARY_PMC, BOUNDARY_PML
from witwin.maxwell.fdtd.cpu_reference import get_cpu_reference_module


@pytest.fixture
def frozen_reference(monkeypatch):
    from tests.fdtd.cuda import torch_reference
    monkeypatch.setattr(torch_reference, "_require_cuda_tensors", lambda *values: None)
    return torch_reference


def _random(shape, generator):
    return torch.randn(shape, generator=generator, dtype=torch.float32)


_H_FRAME = {"x": ("Hx", ("Ey", "Ez"), ("y", "z")), "y": ("Hy", ("Ex", "Ez"), ("x", "z")),
            "z": ("Hz", ("Ex", "Ey"), ("x", "y"))}
_E_FRAME = {"x": ("Ex", ("Hy", "Hz"), ("y", "z")), "y": ("Ey", ("Hx", "Hz"), ("x", "z")),
            "z": ("Ez", ("Hx", "Hy"), ("x", "y"))}
_SHAPES = {"Ex": (5, 7, 6), "Ey": (6, 6, 6), "Ez": (6, 7, 5), "Hx": (6, 6, 5), "Hy": (5, 7, 5), "Hz": (5, 6, 6)}
_NODES = {"x": 6, "y": 7, "z": 6}


@pytest.mark.parametrize("component", "xyz")
@pytest.mark.parametrize("cpml", [False, True])
def test_magnetic_kernels_match_frozen_reference(frozen_reference, component, cpml):
    generator = torch.Generator().manual_seed(1)
    name, sources, axes = _H_FRAME[component]
    kwargs = {name: _random(_SHAPES[name], generator)}
    for source in sources:
        kwargs[source] = _random(_SHAPES[source], generator)
    kwargs[name + "Decay"] = 1 + 0.01 * _random(_SHAPES[name], generator)
    kwargs[name + "Curl"] = _random(_SHAPES[name], generator)
    for axis in axes:
        kwargs["invD" + axis] = 1 + _random((_NODES[axis] - 1,), generator).abs()
    kernel_name = f"updateMagneticFieldH{component}3D" if cpml else f"updateMagneticFieldH{component}Standard3D"
    if cpml:
        for axis in axes:
            upper = axis.upper()
            kwargs[f"Psi{name}{upper}"] = _random(_SHAPES[name], generator)
            kwargs[f"InvKappa{name}{upper}"] = 1 + _random((_NODES[axis] - 1,), generator).abs()
            kwargs[f"By{name}{upper}"] = _random((_NODES[axis] - 1,), generator)
            kwargs[f"Cy{name}{upper}"] = _random((_NODES[axis] - 1,), generator)
    expected = {key: value.clone() if torch.is_tensor(value) else value for key, value in kwargs.items()}
    getattr(get_cpu_reference_module(), kernel_name)(**kwargs).launchRaw()
    frozen_reference._KERNELS[kernel_name](**expected)
    for key in kwargs:
        if torch.is_tensor(kwargs[key]):
            torch.testing.assert_close(kwargs[key], expected[key], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("component", "xyz")
@pytest.mark.parametrize("cpml", [False, True])
@pytest.mark.parametrize("modes", [(BOUNDARY_PEC, BOUNDARY_PMC), (BOUNDARY_PML, BOUNDARY_NONE), (BOUNDARY_PMC, BOUNDARY_PEC)])
def test_electric_kernels_match_frozen_reference(frozen_reference, component, cpml, modes):
    generator = torch.Generator().manual_seed(2)
    name, sources, axes = _E_FRAME[component]
    kwargs = {name: _random(_SHAPES[name], generator)}
    for source in sources:
        kwargs[source] = _random(_SHAPES[source], generator)
    kwargs[name + "Decay"] = 1 + 0.01 * _random(_SHAPES[name], generator)
    kwargs[name + "Curl"] = _random(_SHAPES[name], generator)
    for axis in axes:
        kwargs["invD" + axis] = 1 + _random((_NODES[axis],), generator).abs()
        kwargs[f"{axis}LowBoundaryMode"], kwargs[f"{axis}HighBoundaryMode"] = modes
    kernel_name = f"updateElectricFieldE{component}Cpml3D" if cpml else f"updateElectricFieldE{component}Standard3D"
    if cpml:
        for axis in axes:
            upper = axis.upper()
            kwargs[f"Psi{name}{upper}"] = _random(_SHAPES[name], generator)
            kwargs[f"InvKappa{name}{upper}"] = 1 + _random((_NODES[axis],), generator).abs()
            kwargs[f"B{name}{upper}"] = _random((_NODES[axis],), generator)
            kwargs[f"C{name}{upper}"] = _random((_NODES[axis],), generator)
    expected = {key: value.clone() if torch.is_tensor(value) else value for key, value in kwargs.items()}
    getattr(get_cpu_reference_module(), kernel_name)(**kwargs).launchRaw()
    frozen_reference._KERNELS[kernel_name](**expected)
    for key in kwargs:
        if torch.is_tensor(kwargs[key]):
            torch.testing.assert_close(kwargs[key], expected[key], rtol=1e-6, atol=1e-6)


def test_unsupported_kernel_raises_clear_error():
    with pytest.raises(NotImplementedError, match="outside the CPU reference profile"):
        get_cpu_reference_module().updateDebyeCurrent3D


def _scene(**kwargs):
    defaults = dict(domain=mw.Domain(bounds=((-.05, .05), (-.05, .05), (-.05, .05))), grid=mw.GridSpec.uniform(.01),
                    boundary=mw.BoundarySpec.faces(default="pml", num_layers=4), device="cpu",
                    sources=[mw.PointDipole(position=(0., 0., 0.), polarization="Ez", profile="ideal",
                                            source_time=mw.GaussianPulse(frequency=1e9, fwidth=5e8))],
                    monitors=[mw.PointMonitor("p", (.02, 0., 0.), fields=("Ez",))])
    defaults.update(kwargs)
    return mw.Scene(**defaults)


def test_cpu_scene_still_requires_explicit_backend():
    with pytest.raises(ValueError, match="requires scene.device to be CUDA"):
        mw.Simulation.fdtd(_scene(), frequency=1e9).prepare()


def test_cpu_reference_runs_the_public_pipeline():
    result = mw.Simulation.fdtd(_scene(), frequencies=[1e9], run_time=mw.TimeConfig(time_steps=40),
                                backend="cpu_reference").run()
    assert isinstance(result, mw.Result)
    value = result.monitors["p"]["Ez"]
    assert torch.isfinite(torch.as_tensor(value)).all() and abs(complex(value)) > 0


def test_cpu_reference_rejects_unqualified_profiles():
    with pytest.raises(ValueError, match="backend must be"):
        mw.Simulation.fdtd(_scene(), frequency=1e9, backend="cpu")
    # An open transverse face keeps auto on CPML, isolating the periodicity refusal.
    periodic = _scene(boundary=mw.BoundarySpec.faces(default="pml", num_layers=40, x="periodic", z="none"))
    with pytest.raises(NotImplementedError, match="non-periodic"):
        mw.Simulation.fdtd(periodic, frequency=1e9, backend="cpu_reference").prepare()
    # Distributed execution keeps its CUDA-only preparation guard.
    with pytest.raises(ValueError, match="requires scene.device to be CUDA"):
        mw.Simulation.fdtd(_scene(), frequency=1e9, backend="cpu_reference",
                           parallel=mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1"))).prepare()
