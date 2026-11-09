"""Public linear-background sampling for internal SIE-VIE harmonic balance."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .static_tensor_material import StaticTensorChannel
from dataclasses import dataclass, fields
import math
from witwin.core import PhysicalMaterial
from ..media import Material
from ..constants import EPSILON_0, MU_0
from .physics import compile_medium
from .nonlinear import instantaneous_harmonic_sources


@dataclass(frozen=True)
class StaticChannel:
    """Small-frequency constitutive data, preserving singular DC channels."""
    kind: str
    finite_relative_part: complex
    conductivity: float
    plasma_angular_frequency_squared: float

    def dielectric_sample(self, absolute_scale):
        if self.kind != 'finite_positive_dielectric':
            raise NotImplementedError(f'DC channel requires its {self.kind} formulation')
        return self.finite_relative_part.real*absolute_scale


def _static_channel(base, conductivity, debye, drude, lorentz, absolute_scale):
    finite = complex(base)+sum(p.delta_eps for p in (*debye,*lorentz))
    sigma, plasma = float(conductivity), 0.
    for pole in drude:
        strength = pole.plasma_angular_frequency**2
        damping = pole.gamma_angular_frequency
        if damping == 0:
            plasma += strength
        else:
            sigma += absolute_scale*strength/damping
            finite -= strength/damping**2
    if not all(math.isfinite(v) for v in (finite.real,finite.imag,sigma,plasma)):
        raise ValueError('DC material expansion must be finite')
    kind = ('lossless_plasma_limit' if plasma else 'conductive_limit' if sigma else
            'unresolved_complex_dc' if finite.imag else 'degenerate_or_nonpositive_dc' if finite.real <= 0
            else 'finite_positive_dielectric')
    return StaticChannel(kind,finite,sigma,plasma)


@dataclass(frozen=True)
class HarmonicMaterial:
    fundamental_frequency: float
    linear_samples: tuple
    chi2: float
    chi3: float
    tpa_scale: float
    electric_dc: StaticChannel | StaticTensorChannel
    magnetic_dc: StaticChannel | StaticTensorChannel
    chi2_tensor: object = None
    chi3_tensor: object = None

    @property
    def is_nonlinear(self):
        return self.chi2!=0 or self.chi3!=0 or self.tpa_scale!=0 or self.chi2_tensor is not None or self.chi3_tensor is not None

    def sources(self, electric):
        if len(electric) != len(self.linear_samples)+1:
            raise ValueError('Harmonic field count must match sampled positive frequencies plus DC')
        return instantaneous_harmonic_sources(electric,chi2=self.chi2,chi3=self.chi3,tpa_scale=self.tpa_scale,
            chi2_tensor=self.chi2_tensor,chi3_tensor=self.chi3_tensor)


def _linear_background(material, *, tensor_sources=False):
    """Separate instantaneous sources without changing constitutive declarations."""
    if getattr(material,'has_tensor_nonlinearity',False) and not tensor_sources:
        raise NotImplementedError('Tensor nonlinear susceptibility HB compilation is not integrated yet')
    if type(material) is Material:
        # Material inherits identity fields that its public constructor does
        # not accept, so dataclasses.replace is not a valid cloning operation.
        linear = Material(**{name: getattr(material,name) for name in (
            'eps_r','mu_r','sigma_e','name','sigma_m','debye_poles','drude_poles',
            'lorentz_poles','mu_debye_poles','mu_drude_poles','mu_lorentz_poles',
            'epsilon_tensor','mu_tensor','sigma_e_tensor','orientation',
            'modulation','pec','breakdown','mass_density')})
        chi2,chi3,tpa = material.nonlinear_chi2,material.nonlinear_chi3,material.tpa_sigma_scale
        electric = (material.debye_poles,material.drude_poles,material.lorentz_poles)
        magnetic = (material.mu_debye_poles,material.mu_drude_poles,material.mu_lorentz_poles)
    else:
        linear,chi2,chi3,tpa = material,0.,0.,0.
        electric = magnetic = ((),(),())
    return linear,chi2,chi3,tpa,electric,magnetic


def _tensor_source_coefficients(material, *, device, dtype):
    from ..media import TensorNonlinearSusceptibility
    coefficients={}
    for spec in getattr(material,'nonlinearity',()):
        if isinstance(spec,TensorNonlinearSusceptibility):
            for name,value in spec.compile_lab_tensors(device=device,dtype=dtype).items():
                coefficients[name]=coefficients.get(name,0)+value
    return coefficients


def compile_harmonic_material(material, fundamental_frequency, harmonics, *,
                              tensor_sources=False, device=None, dtype=None):
    """Separate public instantaneous nonlinearity from each linear background.

    Conductivity and dispersion occur only in compile_medium. The source law
    receives exactly the public chi2/chi3/TPA coefficients, with no linear
    epsilon contrast. Singular DC channels are classified, never replaced by
    a finite-frequency sample or silently treated as finite dielectric data.
    Explicit tensor_sources compiles full Cartesian source coefficients on the
    requested CUDA device without promoting the scalar linear background.
    """
    if (type(harmonics) is not int or harmonics < 1 or isinstance(fundamental_frequency,bool)
            or not math.isfinite(fundamental_frequency) or fundamental_frequency <= 0):
        raise ValueError('Harmonic sampling requires positive frequency and positive integer count')
    if type(material) not in (Material,PhysicalMaterial):
        raise TypeError('Harmonic sampling requires a supported public scalar material')
    if type(tensor_sources) is not bool:raise TypeError('tensor_sources must be an explicit boolean')
    linear,chi2,chi3,tpa,electric,magnetic = _linear_background(material,tensor_sources=tensor_sources)
    coefficients={}
    if tensor_sources:
        import torch
        dtype=torch.complex128 if dtype is None else dtype
        if device is None or torch.device(device).type!='cuda' or dtype not in (torch.complex64,torch.complex128):
            raise ValueError('Tensor harmonic sources require an explicit CUDA device and complex precision')
        coefficients=_tensor_source_coefficients(material,device=device,
            dtype=torch.float64 if dtype==torch.complex128 else torch.float32)
    samples = tuple(compile_medium(linear,n*fundamental_frequency) for n in range(1,harmonics+1))
    return HarmonicMaterial(float(fundamental_frequency),samples,chi2,chi3,tpa,
        _static_channel(linear.eps_r,linear.sigma_e,*electric,EPSILON_0),
        _static_channel(linear.mu_r,getattr(linear,'sigma_m',0.),*magnetic,MU_0),**coefficients)


def compile_tensor_harmonic_material(material, fundamental_frequency, harmonics, *, device, dtype=None, tensor_sources=False):
    """Sample tensor AC backgrounds and analytic DC Laurent channels separately.

    Scalar instantaneous source laws retain their public coefficients. This
    internal sampler does not admit unresolved DC channels to an SPD solver or
    enable the public mixed harmonic runtime by itself.
    """
    import torch
    from ..media import GyromagneticFerrite
    from .tensor_medium import compile_tensor_medium
    from .static_tensor_material import compile_static_tensor_material
    if (type(harmonics) is not int or harmonics < 1 or isinstance(fundamental_frequency,bool)
            or not math.isfinite(fundamental_frequency) or fundamental_frequency <= 0):
        raise ValueError('Harmonic sampling requires positive frequency and positive integer count')
    if type(material) not in (Material,PhysicalMaterial,GyromagneticFerrite):
        raise TypeError('Tensor harmonic sampling requires a supported public material')
    dtype = torch.complex128 if dtype is None else dtype
    if type(tensor_sources) is not bool:raise TypeError('tensor_sources must be an explicit boolean')
    linear,chi2,chi3,tpa,_,_ = _linear_background(material,tensor_sources=tensor_sources)
    samples = tuple(compile_tensor_medium(linear,n*fundamental_frequency,device=device,dtype=dtype)
                    for n in range(1,harmonics+1))
    static = compile_static_tensor_material(linear,device=device,dtype=samples[0].epsilon_r.real.dtype)
    coefficients=(_tensor_source_coefficients(material,device=device,dtype=samples[0].epsilon_r.real.dtype)
                  if tensor_sources else {})
    return HarmonicMaterial(float(fundamental_frequency),samples,chi2,chi3,tpa,
                            static.electric_dc,static.magnetic_dc,**coefficients)


def material_signature(material):
    """Describe material fields independently of allocator identity and revision."""
    def value_identity(name,value):
        if name!='nonlinearity' or not getattr(material,'has_tensor_nonlinearity',False):
            return repr(value)
        import hashlib
        import torch
        from ..media import TensorNonlinearSusceptibility
        def coefficient(value):
            if isinstance(value,torch.Tensor):
                # Declaration identity is exact value I/O, independent of print
                # precision, tensor storage strides, device and autograd graph.
                data=value.detach().contiguous().cpu()
                return ('tensor',str(data.dtype),tuple(data.shape),
                        hashlib.sha256(data.numpy().tobytes()).hexdigest())
            return ('literal',value)
        return ('tensor_nonlinearity_declaration_v1',tuple(
            ('TensorNonlinearSusceptibility',coefficient(spec.chi2),coefficient(spec.chi3),coefficient(spec.rotation))
            if isinstance(spec,TensorNonlinearSusceptibility) else (type(spec).__name__,repr(spec))
            for spec in value))
    return (type(material).__name__, tuple(
        (field.name, value_identity(field.name,getattr(material, field.name)))
        for field in fields(material) if field.name not in ('material_id', 'version')))


def cell_nonlinear_coefficients(materials, volumes, sample):
    """Expand body coefficients in the caller's explicit volume/cell order."""
    result={name: sample.real.new_tensor([
        getattr(materials[volume.region_id-1], name)
        for volume in volumes for _ in range(len(volume.cells))])
        for name in ('chi2', 'chi3', 'tpa_scale')}

    import torch
    for name,rank in (('chi2_tensor',3),('chi3_tensor',4)):
        if not any(getattr(m,name) is not None for m in materials):continue
        arrays=[]
        for volume in volumes:
            value=getattr(materials[volume.region_id-1],name)
            if value is None:value=sample.real.new_zeros((3,)*rank)
            if (not isinstance(value,torch.Tensor) or value.shape!=(3,)*rank
                    or value.device!=sample.device or value.dtype!=sample.real.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Tensor nonlinear body coefficients must match the field device and real precision')
            arrays.append(value.expand(len(volume.cells),*((3,)*rank)))
        result[name]=torch.cat(arrays)
    return result
