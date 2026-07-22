"""Phase 3 power-normalization coverage: port accepted-power scaling and fail-closed paths."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.network import PortData

from tests.sar.test_point_sar import _uniform_cube_result


def _result_with_port(*, frequencies, accepted, z0=50.0):
    """A uniform-cube SAR result carrying a 'feed' port with a known accepted power."""
    result, params = _uniform_cube_result(frequencies=frequencies)
    freq = torch.as_tensor(frequencies, dtype=torch.float64)
    a = torch.sqrt(torch.as_tensor(accepted, dtype=torch.float64)).to(torch.complex128)
    b = torch.zeros_like(a)
    port = PortData.from_power_waves(
        port_name="feed", frequencies=freq, a=a, b=b, z0=z0
    )
    result._ports = {"feed": port}
    return result, params


def _valid_ratio(scaled, base):
    valid = base.valid
    expand = valid[None].expand_as(base.point_sar("total"))
    return scaled.point_sar("total")[expand] / base.point_sar("total")[expand]


def test_accepted_power_scales_by_watts_over_measured():
    """accepted_power(port, watts) multiplies SAR by watts / measured accepted power."""
    result, _ = _result_with_port(frequencies=(1.0e9,), accepted=(0.5,))
    base = result.sar("loss")
    scaled = result.sar(
        "loss", normalization=mw.PowerNormalization.accepted_power(port="feed", watts=2.0)
    )
    ratio = _valid_ratio(scaled, base)
    torch.testing.assert_close(
        ratio, torch.full_like(ratio, 4.0), rtol=1e-5, atol=1e-6
    )


def test_accepted_power_is_per_frequency():
    """The scale is resolved per frequency from the port spectrum."""
    accepted = (0.5, 0.25)
    watts = 1.0
    result, _ = _result_with_port(frequencies=(1.0e9, 2.0e9), accepted=accepted)
    base = result.sar("loss")
    scaled = result.sar(
        "loss", normalization=mw.PowerNormalization.accepted_power(port="feed", watts=watts)
    )
    valid = base.valid
    for f, p in enumerate(accepted):
        expected = watts / p
        b = base.point_sar("total")[f][valid]
        s = scaled.point_sar("total")[f][valid]
        ratio = s / b
        torch.testing.assert_close(
            ratio, torch.full_like(ratio, expected), rtol=1e-5, atol=1e-6
        )


def test_accepted_power_missing_port_fails_closed():
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    with pytest.raises(KeyError, match="feed"):
        result.sar(
            "loss", normalization=mw.PowerNormalization.accepted_power(port="feed", watts=1.0)
        )


def test_accepted_power_frequency_mismatch_fails_closed():
    """A SAR frequency absent from the port spectrum must fail closed."""
    result, _ = _result_with_port(frequencies=(1.0e9,), accepted=(0.5,))
    # Rebuild the port at a different frequency so the SAR frequency has no match.
    freq = torch.tensor([3.0e9], dtype=torch.float64)
    a = torch.tensor([complex(0.7)], dtype=torch.complex128)
    result._ports = {
        "feed": PortData.from_power_waves(
            port_name="feed", frequencies=freq, a=a, b=torch.zeros_like(a), z0=50.0
        )
    }
    with pytest.raises(KeyError, match="not all present"):
        result.sar(
            "loss", normalization=mw.PowerNormalization.accepted_power(port="feed", watts=1.0)
        )


def test_accepted_power_nonpositive_measured_fails_closed():
    """A non-positive measured accepted power cannot normalize to a target watt level."""
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    freq = torch.tensor([1.0e9], dtype=torch.float64)
    # |b| > |a| => accepted power negative.
    a = torch.tensor([complex(0.3)], dtype=torch.complex128)
    b = torch.tensor([complex(0.9)], dtype=torch.complex128)
    result._ports = {
        "feed": PortData.from_power_waves(
            port_name="feed", frequencies=freq, a=a, b=b, z0=50.0
        )
    }
    with pytest.raises(ValueError, match="strictly positive"):
        result.sar(
            "loss", normalization=mw.PowerNormalization.accepted_power(port="feed", watts=1.0)
        )


def test_input_power_fails_closed():
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    with pytest.raises(NotImplementedError, match="input_power"):
        result.sar("loss", normalization=mw.PowerNormalization.input_power(watts=1.0))


def test_source_amplitude_square_law_holds_exactly():
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    base = result.sar("loss")
    for amplitude in (0.5, 2.0, 3.0):
        scaled = result.sar(
            "loss", normalization=mw.PowerNormalization.source(amplitude=amplitude)
        )
        ratio = _valid_ratio(scaled, base)
        torch.testing.assert_close(
            ratio, torch.full_like(ratio, amplitude ** 2), rtol=1e-6, atol=1e-7
        )
