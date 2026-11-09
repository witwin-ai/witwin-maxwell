"""Owned arbitrary scalar Taylor polynomials using the shared native FFT algebra."""
import torch
from .fmm_fft import CartesianFFTTranslation, FFTElectromagneticM2L
from .fmm_expansion import multi_indices


class CartesianCoefficientFFTTranslation(CartesianFFTTranslation):
    """Fixed Taylor coefficients about an explicit center, with independent scale."""
    def __init__(self, coefficients, expansion_center, normalization_length, order, *,
                 rhs_batch=4, tensor_workspace_budget_bytes=128*1024**2,
                 retained_budget_bytes=1024**2):
        if type(order) is not int or not 0 <= order <= 12:
            raise ValueError('Coefficient FFT M2L order must lie between zero and twelve')
        if (not isinstance(coefficients, torch.Tensor) or coefficients.device.type != 'cuda'
                or coefficients.dtype != torch.complex128 or coefficients.requires_grad
                or coefficients.shape != (len(multi_indices(2*order+1)),)
                or not bool(torch.isfinite(coefficients).all())):
            raise ValueError('Coefficient FFT M2L requires fixed finite CUDA complex128 Taylor coefficients')
        if not isinstance(expansion_center, torch.Tensor) or expansion_center.device != coefficients.device:
            raise ValueError('Coefficient FFT center must match its coefficients')
        if normalization_length is None:
            raise ValueError('Coefficient FFT M2L requires an explicit independent normalization length')
        self._prepare(expansion_center, order, rhs_batch, tensor_workspace_budget_bytes,
                      retained_budget_bytes, normalization_length=normalization_length)
        # Bloch/symmetry cancellations may legitimately zero the constant term
        # or the whole polynomial. The original free-space amplitude gate stays.
        self._bind_coefficients(coefficients.clone(), None, require_nonzero_amplitude=False)

    def receipt(self):
        record = super().receipt()
        record.update(version='experimental_cartesian_coefficient_fft_v1',
            coefficient_scope='owned_fixed_total_degree_Taylor_polynomial',
            normalization='independent_positive_fixed_length', material_gradient_qualified=False)
        return record


class CoefficientFFTElectromagneticM2L(FFTElectromagneticM2L):
    """Shared PMCHWT current/curl composition of an owned Taylor polynomial."""
    def __init__(self, coefficients, expansion_center, normalization_length, medium, order, **kwargs):
        from .physics import Medium
        if not isinstance(medium, Medium) or medium.frequency <= 0 or medium.k == 0:
            raise ValueError('Coefficient FFT EM M2L requires a finite-frequency scalar medium')
        self._bind_scalar(CartesianCoefficientFFTTranslation(coefficients, expansion_center,
                          normalization_length, order, **kwargs), medium)
