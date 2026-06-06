import os

import pytest
import torch

from witwin.maxwell.fdtd.cuda import backend


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native CUDA parity tests."),
    pytest.mark.skipif(
        os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
        reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA dispersive kernels.",
    ),
]


def _rand(shape, generator):
    return torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator).contiguous()


def test_compiled_cuda_extension_linear_dispersive_updates_match_torch_dispatcher(monkeypatch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(505)
    shape = (3, 4, 5)

    electric = _rand(shape, generator)
    polarization = _rand(shape, generator)
    current = _rand(shape, generator)
    drive = torch.rand(shape, device="cuda", dtype=torch.float32, generator=generator).contiguous()
    inv_permittivity = torch.rand(shape, device="cuda", dtype=torch.float32, generator=generator).contiguous() + 0.25

    expected_polarization = polarization.clone()
    expected_current = current.clone()
    actual_polarization = polarization.clone()
    actual_current = current.clone()
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._update_debye_current(
        ElectricField=electric,
        Polarization=expected_polarization,
        PolarizationCurrent=expected_current,
        DebyeDrive=drive,
        decay=0.83,
        dt=2.0e-16,
    )
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._update_debye_current(
        ElectricField=electric,
        Polarization=actual_polarization,
        PolarizationCurrent=actual_current,
        DebyeDrive=drive,
        decay=0.83,
        dt=2.0e-16,
    )
    torch.testing.assert_close(actual_polarization, expected_polarization, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(actual_current, expected_current, rtol=2.0e-6, atol=1.0e-6)

    expected_current = current.clone()
    actual_current = current.clone()
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._update_drude_current(
        ElectricField=electric,
        PolarizationCurrent=expected_current,
        DrudeDrive=drive,
        decay=0.72,
    )
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._update_drude_current(
        ElectricField=electric,
        PolarizationCurrent=actual_current,
        DrudeDrive=drive,
        decay=0.72,
    )
    torch.testing.assert_close(actual_current, expected_current, rtol=1.0e-6, atol=1.0e-6)

    expected_polarization = polarization.clone()
    expected_current = current.clone()
    actual_polarization = polarization.clone()
    actual_current = current.clone()
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._update_lorentz_current(
        ElectricField=electric,
        Polarization=expected_polarization,
        PolarizationCurrent=expected_current,
        LorentzDrive=drive,
        decay=0.66,
        restoring=0.12,
        dt=1.5e-16,
    )
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._update_lorentz_current(
        ElectricField=electric,
        Polarization=actual_polarization,
        PolarizationCurrent=actual_current,
        LorentzDrive=drive,
        decay=0.66,
        restoring=0.12,
        dt=1.5e-16,
    )
    torch.testing.assert_close(actual_polarization, expected_polarization, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(actual_current, expected_current, rtol=1.0e-6, atol=1.0e-6)

    expected_electric = electric.clone()
    actual_electric = electric.clone()
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend._apply_polarization_current(
        ElectricField=expected_electric,
        PolarizationCurrent=current,
        InvPermittivity=inv_permittivity,
        dt=1.25e-16,
    )
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend._apply_polarization_current(
        ElectricField=actual_electric,
        PolarizationCurrent=current,
        InvPermittivity=inv_permittivity,
        dt=1.25e-16,
    )
    torch.testing.assert_close(actual_electric, expected_electric, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize(
    ("backend_func", "method_name", "decay_name"),
    [
        (backend._update_kerr_ex, "update_kerr_ex_curl", "ExDecay"),
        (backend._update_kerr_ey, "update_kerr_ey_curl", "EyDecay"),
        (backend._update_kerr_ez, "update_kerr_ez_curl", "EzDecay"),
    ],
)
def test_compiled_cuda_extension_kerr_curl_updates_match_torch_dispatcher(
    monkeypatch,
    backend_func,
    method_name,
    decay_name,
):
    extension = backend.get_compiled_extension()
    assert hasattr(extension, method_name)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(507)
    dynamic_curl = torch.empty((3, 4, 5), device="cuda", dtype=torch.float32)
    ex = _rand((3, 4, 5), generator)
    ey = _rand((3, 4, 5), generator)
    ez = _rand((3, 4, 5), generator)
    linear_permittivity = torch.rand((3, 4, 5), device="cuda", dtype=torch.float32, generator=generator) + 0.75
    decay = torch.rand((3, 4, 5), device="cuda", dtype=torch.float32, generator=generator) + 0.5
    kerr_chi3 = torch.rand((3, 4, 5), device="cuda", dtype=torch.float32, generator=generator) * 0.1

    expected = dynamic_curl.clone()
    actual = dynamic_curl.clone()
    kwargs = {
        "DynamicCurl": expected,
        "Ex": ex,
        "Ey": ey,
        "Ez": ez,
        "LinearPermittivity": linear_permittivity,
        decay_name: decay,
        "KerrChi3": kerr_chi3,
        "dt": 2.0e-16,
        "eps0": 8.8541878128e-12,
    }
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0")
    backend_func(**kwargs)
    kwargs["DynamicCurl"] = actual
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    backend_func(**kwargs)

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
