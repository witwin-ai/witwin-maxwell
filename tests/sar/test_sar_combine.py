"""Phase 3 multi-source combination: coherent field sum and incoherent power sum."""

import pytest
import torch

import witwin.maxwell as mw

from tests.sar.test_point_sar import _uniform_cube_result


def _valid(sar):
    return sar.valid[None].expand_as(sar.point_sar("total"))


def test_coherent_in_phase_doubles_field_and_quadruples_sar():
    """Two equal in-phase sources sum to 2E, so SAR (quadratic in E) quadruples."""
    a, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    b, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    single = a.sar("loss")
    combined = mw.combine_coherent_sar([a, b], monitor="loss")

    mask = _valid(single)
    ratio = combined.point_sar("total")[mask] / single.point_sar("total")[mask]
    torch.testing.assert_close(ratio, torch.full_like(ratio, 4.0), rtol=1e-5, atol=1e-6)
    assert combined.provenance["combination"]["mode"] == "coherent"


def test_coherent_opposite_phase_cancels():
    """Equal, opposite-phase sources cancel to near-zero SAR (interference)."""
    a, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    b, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    combined = mw.combine_coherent_sar([a, b], monitor="loss", weights=[1.0, -1.0])
    total = combined.point_sar("total")
    finite = total[torch.isfinite(total)]
    assert float(finite.abs().max()) < 1e-10


def test_incoherent_sum_adds_sar_fields():
    """Incoherent combination sums the power-domain SAR fields."""
    a, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    b, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=1.5)
    sar_a = a.sar("loss")
    sar_b = b.sar("loss")
    combined = mw.combine_incoherent_sar([sar_a, sar_b])

    expected = sar_a.point_sar("total")[_valid(sar_a)] + sar_b.point_sar("total")[_valid(sar_b)]
    got = combined.point_sar("total")[_valid(combined)]
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-9)
    assert combined.provenance["combination"]["mode"] == "incoherent"


def test_incoherent_recomputes_peaks_on_combined_power():
    """Incoherent combination with averaging recomputes the mass-averaged peak."""
    a, params = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    b, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    m0 = params["rho"] * 0.3 ** 3
    averaging = mw.SARAveraging(mass=(m0,))
    sar_a = a.sar("loss", averaging=averaging)
    sar_b = b.sar("loss", averaging=averaging)
    combined = mw.combine_incoherent_sar([sar_a, sar_b], averaging=averaging)

    single_peak = float(sar_a.peak(m0).sar[0])
    combined_peak = float(combined.peak(m0).sar[0])
    assert combined_peak == pytest.approx(2.0 * single_peak, rel=1e-4)


def test_incoherent_normalization_mismatch_fails_closed():
    a, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0)
    sar_a = a.sar("loss")
    sar_scaled = a.sar("loss", normalization=mw.PowerNormalization.source(amplitude=2.0))
    with pytest.raises(ValueError, match="normalization"):
        mw.combine_incoherent_sar([sar_a, sar_scaled])


def test_incoherent_requires_two_operands():
    a, _ = _uniform_cube_result(frequencies=(1.0e9,))
    sar_a = a.sar("loss")
    with pytest.raises(ValueError, match="at least two"):
        mw.combine_incoherent_sar([sar_a])


def test_coherent_preserves_field_autograd():
    """The coherent combiner keeps the per-run field autograd graph."""
    a, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0, requires_grad=True)
    b, _ = _uniform_cube_result(frequencies=(1.0e9,), e0=2.0, requires_grad=True)
    combined = mw.combine_coherent_sar([a, b], monitor="loss")
    total = combined.point_sar("total")
    torch.nansum(total).backward()
    grad = a.fields["EX"].grad
    assert grad is not None and bool((grad.abs() > 0).any())
