"""Analytic zero-frequency tensor constitutive channels for internal DC paths.

No finite-frequency sampling, scalarization or imaginary-part truncation is
used to admit an SPD static solve. Singular channels remain explicit.
"""
from dataclasses import dataclass
import torch
from witwin.core import PhysicalMaterial
from ..media import Material,DiagonalTensor3,Tensor3x3,ComplexTensor3x3,GyromagneticFerrite
from ..constants import EPSILON_0,MU_0
from .harmonic_material import _static_channel


class StaticTensorChannel:
    """Owned Laurent data: finite relative tensor, SI conductivity, plasma w^2."""
    def __init__(self,finite,conductivity,plasma,absolute_scale):
        self.finite_relative_part=finite.clone()
        self.conductivity=conductivity.clone()
        self.plasma_angular_frequency_squared=plasma.clone()
        self.absolute_scale=absolute_scale
        if not bool(torch.isfinite(finite).all() & torch.isfinite(conductivity).all() & torch.isfinite(plasma).all()):
            raise ValueError('Static tensor expansion must be finite')
        if bool((plasma!=0).any()):kind='lossless_plasma_limit'
        elif bool((conductivity!=0).any()):kind='conductive_limit'
        elif bool((finite.imag!=0).any()):kind='unresolved_complex_dc'
        elif not torch.equal(finite.real,finite.real.T):kind='nonreciprocal_real_dc'
        else:
            _,info=torch.linalg.cholesky_ex(finite.real)
            kind='finite_positive_dielectric' if bool(info==0) else 'degenerate_or_nonpositive_dc'
        self.kind=kind
        self._binding=self._state()

    def _state(self):
        return (self.kind,self.absolute_scale,tuple((id(t),t._version) for t in
            (self.finite_relative_part,self.conductivity,self.plasma_angular_frequency_squared)))

    def assert_current(self):
        if self._state()!=self._binding:raise ValueError('Static tensor material channel changed')

    def conductivity_sample(self):
        """Return the actual symmetric PSD SI DC current coefficient."""
        self.assert_current()
        if self.kind not in ('conductive_limit','finite_positive_dielectric'):
            raise NotImplementedError(f'DC tensor current requires its {self.kind} formulation')
        value=self.conductivity
        if (value.device.type!='cuda' or value.dtype not in (torch.float32,torch.float64)
                or value.shape!=(3,3) or not bool(torch.isfinite(value).all())
                or not torch.equal(value,value.T) or bool((torch.linalg.eigvalsh(value)<0).any())):
            raise ValueError('DC conductivity requires a finite real symmetric PSD CUDA tensor')
        return value

    def dielectric_sample(self):
        """Return a real SI SPD coefficient only for its actual finite DC channel."""
        self.assert_current()
        if self.kind!='finite_positive_dielectric':
            raise NotImplementedError(f'DC tensor channel requires its {self.kind} formulation')
        value=self.finite_relative_part.real*self.absolute_scale
        _,info=torch.linalg.cholesky_ex(value)
        if not bool(torch.isfinite(value).all()) or bool(info!=0):
            raise ValueError('Absolute static tensor coefficient is not representable as SPD')
        return value


@dataclass(frozen=True)
class StaticTensorMaterial:
    electric_dc: StaticTensorChannel
    magnetic_dc: StaticTensorChannel

    def assert_current(self):
        self.electric_dc.assert_current();self.magnetic_dc.assert_current()


def compile_static_tensor_material(material,*,device,dtype=torch.float64):
    """Compile public homogeneous constitutive declarations at the analytic DC limit.

    Instantaneous nonlinear coefficients do not alter the linear background.
    Tensor bases and electric conductivity replace their scalar declarations;
    homogeneous pole increments remain isotropic. Ferrite gyrotropy vanishes
    at zero frequency, leaving the real transverse susceptibility Ms/|H0|.
    """
    if type(material) not in (Material,PhysicalMaterial,GyromagneticFerrite):
        raise TypeError('Static tensor compilation requires a supported public material')
    device=torch.device(device)
    if device.type!='cuda' or dtype not in (torch.float32,torch.float64):
        raise ValueError('Static tensor compilation requires real CUDA precision')
    if isinstance(material,Material) and (material.pec or material.modulation is not None
            or material.breakdown is not None or material.has_custom_poles):
        raise ValueError('Static tensor compilation requires homogeneous stationary material data')
    eye=torch.eye(3,device=device,dtype=dtype)
    complex_dtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    def declaration(value,fallback,*,complex_value=True):
        target=complex_dtype if complex_value else dtype
        if value is None:return complex(fallback)*eye.to(target) if complex_value else float(fallback)*eye
        if isinstance(value,DiagonalTensor3):return torch.diag(torch.tensor(value.as_tuple(),device=device,dtype=target))
        if isinstance(value,(Tensor3x3,ComplexTensor3x3)):
            return torch.tensor(value.rows,device=device,dtype=target)
        raise TypeError('Unsupported static tensor declaration')
    channels=[]
    for magnetic,scale in ((False,EPSILON_0),(True,MU_0)):
        prefix='mu_' if magnetic else ''
        poles=tuple(tuple(getattr(material,prefix+family+'_poles',())) for family in ('debye','drude','lorentz'))
        if any(getattr(p,'allow_gain',False) for family in poles for p in family):
            raise ValueError('Static tensor compilation excludes gain-enabled poles')
        expansion=_static_channel(0.,0.,*poles,scale)
        base=declaration(getattr(material,'mu_tensor' if magnetic else 'epsilon_tensor',None),
                         material.mu_r if magnetic else material.eps_r)
        if magnetic and isinstance(material,GyromagneticFerrite):
            bias=eye.new_tensor(material.bias_field)
            magnitude=bias.norm()
            if not bool(torch.isfinite(magnitude)) or not bool(magnitude>0):
                raise ValueError('Ferrite DC bias magnitude is not representable')
            unit=bias/magnitude
            base=(material.mu_infinity*eye+(material.saturation_magnetization/magnitude)*
                  (eye-unit[:,None]*unit[None,:])).to(complex_dtype)
        finite=base+expansion.finite_relative_part*eye
        conductivity=declaration(None,getattr(material,'sigma_m',0.),complex_value=False) if magnetic else declaration(
            getattr(material,'sigma_e_tensor',None),material.sigma_e,complex_value=False)
        # Validate the declared background independently, as the AC compiler
        # does. Passive pole increments must not mask an active conductivity.
        if (not bool(torch.isfinite(conductivity).all()) or not torch.equal(conductivity,conductivity.T)
                or bool((torch.linalg.eigvalsh(conductivity)<0).any())):
            raise ValueError('Declared static conductivity must be finite symmetric positive semidefinite')
        conductivity=conductivity+expansion.conductivity*eye
        if (not bool(torch.isfinite(conductivity).all()) or not torch.equal(conductivity,conductivity.T)
                or bool((torch.linalg.eigvalsh(conductivity)<0).any())):
            raise ValueError('Static conductivity must be finite symmetric positive semidefinite')
        plasma=expansion.plasma_angular_frequency_squared*eye
        channels.append(StaticTensorChannel(finite,conductivity,plasma,scale))
    return StaticTensorMaterial(*channels)


def compile_static_tensor_interface(system,exterior,interiors,*,channel='electric',
                                    pair_batch=4096,max_matrix_bytes=256*1024**2):
    """Build an internal interface from actual finite public-material DC samples.

    Electric and magnetic potential channels are separate. Conductive, plasma,
    complex and non-SPD channels reject before any interface assembly.
    """
    from .static_tensor_interface import StaticTensorInterfaceOperator
    system.assert_current()
    if channel not in ('electric','magnetic'):
        raise ValueError('Static potential channel must be electric or magnetic')
    interiors=tuple(interiors)
    if len(interiors)!=len(system.bodies):
        raise ValueError('Each static body requires one public material')
    samples=[compile_static_tensor_material(material,device=system.device,dtype=system.dtype)
             for material in (exterior,*interiors)]
    coefficients=[getattr(sample,channel+'_dc').dielectric_sample() for sample in samples]
    return StaticTensorInterfaceOperator(system,coefficients[0],coefficients[1:],
        pair_batch=pair_batch,max_matrix_bytes=max_matrix_bytes)
