from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from witwin.maxwell.postprocess import ModeTrackingError, track_modes


def _complex_overlap(
    frequency_count: int,
    mode_count: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.zeros(
        (frequency_count - 1, mode_count, mode_count),
        dtype=torch.complex128,
        device=device,
    )


def test_track_modes_uses_global_assignment_and_stable_beta_order() -> None:
    beta = torch.tensor(
        [
            [1.00, 2.00, 3.00],
            [3.02, 1.01, 2.01],
            [2.02, 3.03, 1.02],
        ],
        dtype=torch.float64,
    )
    overlaps = _complex_overlap(3, 3)
    overlaps[0] = torch.tensor(
        [
            [0.01, 0.97 * torch.exp(torch.tensor(0.2j)), 0.01],
            [0.01, 0.01, 0.96 * torch.exp(torch.tensor(-0.3j))],
            [0.98 * torch.exp(torch.tensor(0.4j)), 0.01, 0.01],
        ],
        dtype=torch.complex128,
    )
    overlaps[1] = torch.tensor(
        [
            [0.01, 0.98 * torch.exp(torch.tensor(-0.1j)), 0.01],
            [0.01, 0.01, 0.97 * torch.exp(torch.tensor(0.3j))],
            [0.96 * torch.exp(torch.tensor(-0.2j)), 0.01, 0.01],
        ],
        dtype=torch.complex128,
    )

    result = track_modes(beta, overlaps)

    torch.testing.assert_close(
        result.assignment,
        torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
    )
    torch.testing.assert_close(
        result.beta,
        torch.tensor(
            [[1.00, 2.00, 3.00], [1.01, 2.01, 3.02], [1.02, 2.02, 3.03]],
            dtype=torch.float64,
        ),
    )
    assert bool(torch.all(result.overlap.real > 0.95))
    assert bool(torch.all(torch.abs(result.overlap.imag) < 1.0e-12))
    assert bool(torch.all(result.margin > 0.8))
    assert not bool(torch.any(result.degenerate))


def test_track_modes_removes_frequency_dependent_phase_from_modal_basis() -> None:
    beta = torch.tensor([[2.0], [2.1], [2.2]], dtype=torch.float64)
    reference = torch.tensor([1.0 + 0.0j, 0.2 + 0.4j], dtype=torch.complex128)
    phases = torch.tensor([0.0, 0.7, -1.1], dtype=torch.float64)
    modal_basis = torch.stack(
        [reference * torch.exp(1j * phase) for phase in phases],
        dim=0,
    ).unsqueeze(1)

    result = track_modes(beta, modal_basis=modal_basis)

    assert result.aligned_basis is not None
    expected = reference.expand(3, 1, -1)
    torch.testing.assert_close(result.aligned_basis, expected, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        result.phase_factors[:, 0],
        torch.exp(-1j * phases),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        result.overlap,
        torch.ones_like(result.overlap),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_track_modes_orients_a_near_degenerate_subspace_with_svd() -> None:
    beta = torch.tensor(
        [[1.0, 1.0 + 1.0e-7, 3.0], [1.0 + 2.0e-7, 1.0, 3.1]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.complex128)
    angle = torch.tensor(0.61, dtype=torch.float64)
    cosine = torch.cos(angle).to(torch.complex128)
    sine = torch.sin(angle).to(torch.complex128)
    rotation = torch.stack(
        [
            torch.stack([cosine, sine, torch.zeros_like(cosine)]),
            torch.stack([-sine, cosine, torch.zeros_like(cosine)]),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.complex128),
        ]
    )
    modal_basis = torch.stack([identity, rotation])

    result = track_modes(beta, modal_basis=modal_basis)

    assert result.aligned_basis is not None
    torch.testing.assert_close(
        result.aligned_basis[1],
        identity,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        result.subspace_singular_values[0],
        torch.ones(3, dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert result.degenerate[0].tolist() == [True, True, False]
    assert bool(torch.all(result.confidence[0, :2] > 1.0 - 1.0e-12))
    assert bool(result.confidence[0, 2] > 0.9)


def test_track_modes_rejects_an_ambiguous_ordinary_assignment() -> None:
    beta = torch.tensor([[1.0, 3.0], [1.0, 3.0]], dtype=torch.float64)
    overlaps = torch.tensor(
        [[[0.70, 0.69], [0.69, 0.70]]],
        dtype=torch.complex128,
    )

    with pytest.raises(ModeTrackingError, match="Low-confidence mode match"):
        track_modes(beta, overlaps, beta_weight=0.0, min_margin=0.05)


def test_track_modes_rejects_a_low_confidence_degenerate_subspace() -> None:
    beta = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    overlaps = torch.tensor(
        [[[0.9, 0.0], [0.0, 0.2]]],
        dtype=torch.complex128,
    )

    with pytest.raises(ModeTrackingError, match="degenerate subspace"):
        track_modes(beta, overlaps, min_subspace_singular_value=0.5)


def test_track_modes_preserves_autograd_for_continuous_outputs() -> None:
    beta = torch.tensor([[1.0], [1.1], [1.2]], dtype=torch.float64, requires_grad=True)
    modal_basis = torch.tensor(
        [
            [[1.0 + 0.0j, 0.2 + 0.3j]],
            [[0.8 + 0.2j, 0.4 + 0.1j]],
            [[0.7 - 0.1j, 0.3 + 0.4j]],
        ],
        dtype=torch.complex128,
        requires_grad=True,
    )

    result = track_modes(beta, modal_basis=modal_basis, min_overlap=0.1)
    assert result.aligned_basis is not None
    loss = (
        result.beta.square().sum()
        + result.overlap.real.sum()
        + result.aligned_basis.real.sum()
    )
    loss.backward()

    assert beta.grad is not None
    assert modal_basis.grad is not None
    assert bool(torch.all(torch.isfinite(beta.grad)))
    assert bool(torch.all(torch.isfinite(modal_basis.grad)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_track_modes_keeps_every_tensor_on_one_cuda_device() -> None:
    device = torch.device("cuda")
    beta = torch.tensor([[1.0, 2.0], [1.1, 2.1]], dtype=torch.float64, device=device)
    overlaps = torch.eye(2, dtype=torch.complex128, device=device).unsqueeze(0)

    result = track_modes(beta, overlaps)

    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, torch.Tensor):
            assert value.device.type == device.type
            assert value.device.index == torch.cuda.current_device()


def test_track_modes_rejects_non_normalized_overlap() -> None:
    beta = torch.tensor([[1.0], [1.1]], dtype=torch.float64)
    overlaps = torch.tensor([[[1.01 + 0.0j]]], dtype=torch.complex128)

    with pytest.raises(ValueError, match="normalized"):
        track_modes(beta, overlaps)
