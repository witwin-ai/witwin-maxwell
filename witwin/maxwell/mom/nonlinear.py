"""Instantaneous scalar and tensor nonlinear harmonic sources in peak-phasor convention.

This constitutive primitive is not a harmonic-balance field solver. The linear
background is deliberately absent: surface/volume coupling must account for it
once through the regional Green operator.
"""
from dataclasses import dataclass
import math
import torch
from ..constants import EPSILON_0


@dataclass(frozen=True)
class HarmonicSources:
    polarization: torch.Tensor
    dissipative_current: torch.Tensor
    average_dissipation: torch.Tensor
    collocation_count: int
    convention: str = 'real_dc_plus_real_positive_peak_phasors_exp_minus_i_n_omega_t_v1'

    def polarization_current(self, fundamental_frequency):
        if (isinstance(fundamental_frequency, bool) or not math.isfinite(fundamental_frequency)
                or fundamental_frequency <= 0):
            raise ValueError('Fundamental frequency must be finite and positive')
        orders = torch.arange(len(self.polarization), device=self.polarization.device,
                              dtype=self.polarization.real.dtype)
        shape = (len(orders),)+(1,)*(self.polarization.ndim-1)
        return (-2j*math.pi*fundamental_frequency*orders.reshape(shape))*self.polarization


def instantaneous_harmonic_sources(electric, *, chi2=0., chi3=0., tpa_scale=0.,
                                   chi2_tensor=None, chi3_tensor=None):
    """Evaluate all DC through 3H sources without polynomial aliasing.

    electric[0] is real DC; electric[n>0] is a peak phasor so the real field is
    E0 + Re(sum E_n exp(-i*n*omega*t)). Missing retained harmonics are explicit
    zeros. Outputs include generated harmonics beyond the input truncation.
    Coefficients may be real native CUDA tensors broadcast over spatial points.
    Optional lab-frame tensors add eps0*chi2_ijk*E_j*E_k and
    eps0*chi3_ijkl*E_j*E_k*E_l. Their final axes have size three; leading
    axes broadcast over space. Units are m/V and m^2/V^2 respectively.
    No factorial, symmetrization, or scalar-to-tensor reinterpretation is
    implicit. Arbitrary real tensors do not necessarily define conservative
    polarization; the power ledger must check that separately.
    """
    if (not isinstance(electric, torch.Tensor) or electric.device.type != 'cuda'
            or electric.dtype not in (torch.complex64, torch.complex128)
            or electric.ndim < 2 or electric.shape[-1] != 3 or not len(electric)
            or not bool(torch.isfinite(electric).all()) or bool((electric[0].imag != 0).any())):
        raise ValueError('Harmonic electric fields require finite CUDA complex [H+1,...,3] with real DC')
    spatial_shape = electric.shape[1:-1]
    def coefficient(value, name):
        if not isinstance(value, torch.Tensor):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} requires finite real coefficients')
            value = electric.real.new_tensor(value)
        if (value.device != electric.device or value.dtype != electric.real.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{name} requires matching finite real CUDA coefficients')
        try:
            shape = torch.broadcast_shapes(value.shape, spatial_shape)
        except RuntimeError as error:
            raise ValueError(f'{name} cannot broadcast to the spatial field') from error
        if shape != spatial_shape:
            raise ValueError(f'{name} cannot introduce a new spatial dimension')
        return value
    chi2, chi3, tpa_scale = (coefficient(v, n) for v, n in
                           ((chi2, 'chi2'), (chi3, 'chi3'), (tpa_scale, 'tpa_scale')))
    if bool((tpa_scale < 0).any()):
        raise ValueError('Two-photon conductivity must be nonnegative')
    def susceptibility(value, rank, name):
        if value is None:return None
        if (not isinstance(value,torch.Tensor) or value.device!=electric.device
                or value.dtype!=electric.real.dtype or value.ndim<rank
                or value.shape[-rank:]!=(3,)*rank or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{name} requires matching finite real CUDA spatial tensors with {rank} Cartesian axes')
        try:shape=torch.broadcast_shapes(value.shape[:-rank],spatial_shape)
        except RuntimeError as error:
            raise ValueError(f'{name} cannot broadcast to the spatial field') from error
        if shape!=spatial_shape:raise ValueError(f'{name} cannot introduce a new spatial dimension')
        return value
    chi2_tensor=susceptibility(chi2_tensor,3,'chi2_tensor')
    chi3_tensor=susceptibility(chi3_tensor,4,'chi3_tensor')
    highest = len(electric)-1
    count = 6*highest+1
    coefficients = electric.new_zeros((count,)+electric.shape[1:])
    coefficients[0] = electric[0]
    if highest:
        coefficients[1:highest+1] = electric[1:]/2
        coefficients[-highest:] = electric[1:].flip(0).conj()/2
    real_field = torch.fft.fft(coefficients, dim=0).real
    intensity = real_field.square().sum(-1)
    cubic = intensity[..., None]*real_field
    polarization_time = EPSILON_0*(chi2[..., None]*real_field.square()+chi3[..., None]*cubic)
    if chi2_tensor is not None:
        polarization_time=polarization_time+EPSILON_0*torch.einsum(
            '...ijk,t...j,t...k->t...i',chi2_tensor,real_field,real_field)
    if chi3_tensor is not None:
        polarization_time=polarization_time+EPSILON_0*torch.einsum(
            '...ijkl,t...j,t...k,t...l->t...i',chi3_tensor,real_field,real_field,real_field)
    current_time = tpa_scale[..., None]*cubic
    def peak_phasors(values):
        transformed = torch.fft.ifft(values, dim=0)[:3*highest+1]
        dc = transformed[:1].real.to(electric.dtype)
        return torch.cat((dc, 2*transformed[1:]), dim=0)
    return HarmonicSources(peak_phasors(polarization_time), peak_phasors(current_time),
                           (tpa_scale*intensity.square()).mean(0), count)


def harmonic_power_ledger(electric, sources, fundamental_frequency, cell_volumes):
    """Local nonlinear work by harmonic, with independent constitutive closure.

    Positive-frequency peak phasors carry 1/2; real DC carries 1. Positive
    work transfers energy from the field to matter. Conservative polarization
    can exchange energy between harmonics but its cycle-averaged sum is zero.
    Arbitrary tensor coefficients need not be conservative; their measured
    polarization work remains explicit and can fail the conservation check.
    This is not an exterior flux balance or a harmonic truncation certificate.
    """
    if (not isinstance(electric,torch.Tensor) or electric.device.type != 'cuda'
            or electric.dtype not in (torch.complex64,torch.complex128) or electric.ndim != 3
            or electric.shape[-1] != 3 or len(electric) < 1 or not bool(torch.isfinite(electric).all())
            or bool(electric[0].imag.any())):
        raise ValueError('Power ledger requires real-DC CUDA complex [H+1,cell,3] fields')
    if (not isinstance(cell_volumes,torch.Tensor) or cell_volumes.device != electric.device
            or cell_volumes.dtype != electric.real.dtype or cell_volumes.shape != electric.shape[1:2]
            or not bool(torch.isfinite(cell_volumes).all()) or not bool((cell_volumes > 0).all())):
        raise ValueError('Power ledger requires matching positive CUDA cell volumes')
    if not isinstance(sources,HarmonicSources):
        raise TypeError('Power ledger requires HarmonicSources')
    count = len(electric)
    for value in (sources.polarization,sources.dissipative_current):
        if (not isinstance(value,torch.Tensor) or value.device != electric.device or value.dtype != electric.dtype
                or value.ndim != 3 or value.shape[1:] != electric.shape[1:] or len(value) < count
                or not bool(torch.isfinite(value).all()) or bool(value[0].imag.any())):
            raise ValueError('Power ledger source axes must cover the retained field harmonics')
    average = sources.average_dissipation
    if (not isinstance(average,torch.Tensor) or average.device != electric.device or average.dtype != electric.real.dtype
            or average.shape != cell_volumes.shape or not bool(torch.isfinite(average).all())):
        raise ValueError('Power ledger requires matching real constitutive dissipation')
    polarization_current = sources.polarization_current(fundamental_frequency)[:count]
    dissipative_current = sources.dissipative_current[:count]
    weights = electric.real.new_full((count,1),.5)
    weights[0] = 1.
    def work(current):
        return (weights*(electric.conj()*current).sum(-1).real*cell_volumes).sum(-1)
    exchange,dissipation = work(polarization_current),work(dissipative_current)
    expected = (average*cell_volumes).sum()
    def bound(current):
        return (weights*(electric.abs()*current.abs()).sum(-1)*cell_volumes).sum()
    roundoff = 512*torch.finfo(electric.real.dtype).eps
    exchange_threshold = roundoff*bound(polarization_current)
    dissipation_threshold = roundoff*(bound(dissipative_current)+expected.abs())
    return {'polarization_exchange_W':exchange,'dissipation_W':dissipation,
        'total_nonlinear_work_W':exchange+dissipation,
        'polarization_closure_W':exchange.sum(),'polarization_threshold_W':exchange_threshold,
        'dissipation_parseval_error_W':dissipation.sum()-expected,
        'dissipation_threshold_W':dissipation_threshold,'constitutive_dissipation_W':expected,
        'polarization_conservative':bool(exchange.sum().abs() <= exchange_threshold),
        'dissipation_parseval_closed':bool((dissipation.sum()-expected).abs() <= dissipation_threshold),
        'passive_dissipation':bool(expected >= -dissipation_threshold),
        'harmonics':tuple(range(count)),'work_sign':'positive_field_to_matter',
        'physical_qualification':False,'global_power_balance_verified':False,
        'harmonic_truncation_converged':False}


def rotate_nonlinear_susceptibility(tensor, rotation):
    """Rotate a material-frame rank-3/4 tensor into the lab frame.

    rotation maps material vector components to lab components. It must be a
    proper real orthogonal 3x3 CUDA matrix. This helper takes one unbatched
    material tensor; source evaluation also supports spatial batches. Autograd remains
    native for tensor entries and differentiably generated rotations.
    """
    if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cuda'
            or tensor.dtype not in (torch.float32,torch.float64)
            or tensor.shape not in ((3,3,3),(3,3,3,3)) or not bool(torch.isfinite(tensor).all())):
        raise ValueError('Rotation requires a finite real CUDA rank-3 or rank-4 material tensor')
    if (not isinstance(rotation,torch.Tensor) or rotation.device!=tensor.device
            or rotation.dtype!=tensor.dtype or rotation.shape!=(3,3)
            or not bool(torch.isfinite(rotation).all())):
        raise ValueError('Material rotation requires a matching finite real CUDA 3x3 matrix')
    eye=torch.eye(3,device=tensor.device,dtype=tensor.dtype)
    tolerance=64*torch.finfo(tensor.dtype).eps
    if (not torch.allclose(rotation.T@rotation,eye,rtol=0,atol=tolerance)
            or not torch.allclose(torch.linalg.det(rotation),tensor.new_tensor(1.),rtol=0,atol=tolerance)):
        raise ValueError('Material rotation must be proper orthogonal')
    if tensor.ndim==3:
        return torch.einsum('ia,jb,kc,abc->ijk',rotation,rotation,rotation,tensor)
    return torch.einsum('ia,jb,kc,ld,abcd->ijkl',rotation,rotation,rotation,rotation,tensor)
