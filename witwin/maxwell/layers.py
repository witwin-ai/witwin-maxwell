"""Planar, nonmagnetic process-stack descriptors (SI units).

A layer permittivity ``eps_r`` is either a constant (complex) relative permittivity or a
causal frequency-dependent permittivity model: :class:`WidebandDebye` or
:class:`PolePermittivity`. Models use the exp(-i*omega*t) convention (loss is
Im(eps) > 0); ``sigma`` always adds the conduction term i*sigma/(omega*eps0).
"""

import cmath
from dataclasses import dataclass, fields
from itertools import accumulate
import math

from ._fixed_inputs import require_fixed_input
from ._stability import experimental


def _real_parameter(value, name):
    require_fixed_input(value, name=name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class WidebandDebye:
    """Djordjevic-Sarkar wideband Debye permittivity, causal by construction.

    eps(f) = eps_inf + delta_eps * ln((f2 - i f) / (f1 - i f)) / ln(f2 / f1)

    is a uniform distribution of passive Debye relaxations between the corner
    frequencies f1 < f2 [Hz], so Kramers-Kronig holds exactly. Between the corners the
    loss tangent is nearly constant. eps(0) = eps_inf + delta_eps, eps(inf) = eps_inf.
    Passivity requires delta_eps >= 0; a dielectric also requires eps_inf >= 1.
    """

    eps_inf: float
    delta_eps: float
    f1: float
    f2: float

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _real_parameter(getattr(self, field.name),
                                                                 f"WidebandDebye.{field.name}"))
        if self.delta_eps < 0:
            raise ValueError("WidebandDebye.delta_eps < 0 is active (gain); a passive dielectric needs delta_eps >= 0")
        if self.eps_inf < 1:
            raise ValueError("WidebandDebye.eps_inf < 1 is non-physical for a dielectric; require eps_inf >= 1")
        if not 0 < self.f1 < self.f2:
            raise ValueError("WidebandDebye requires corner frequencies 0 < f1 < f2")

    @classmethod
    def from_loss_tangent(cls, eps_r, tan_delta, frequency, *, f1=1e3, f2=1e12):
        """Match Re(eps)=eps_r and Im(eps)/Re(eps)=tan_delta at ``frequency`` [Hz].

        Raises when the match needs eps_inf < 1 (the loss tangent is too large for the
        corner window).
        """
        eps_r, tan_delta, frequency, f1, f2 = (
            _real_parameter(v, n) for v, n in ((eps_r, "eps_r"), (tan_delta, "tan_delta"),
                                               (frequency, "frequency"), (f1, "f1"), (f2, "f2")))
        if tan_delta < 0 or not 0 < f1 < f2 or frequency <= 0:
            raise ValueError("from_loss_tangent requires tan_delta >= 0, frequency > 0 and 0 < f1 < f2")
        width = math.log(f2 / f1)
        delta_eps = eps_r * tan_delta * width / (math.atan(frequency / f1) - math.atan(frequency / f2))
        shift = .5 * math.log((f2 * f2 + frequency * frequency) / (f1 * f1 + frequency * frequency)) / width
        return cls(eps_inf=eps_r - delta_eps * shift, delta_eps=delta_eps, f1=f1, f2=f2)

    def relative_permittivity(self, frequency) -> complex:
        f = float(frequency)
        return self.eps_inf + self.delta_eps * (cmath.log(complex(self.f2, -f)) - cmath.log(complex(self.f1, -f))) / (
            math.log(self.f2 / self.f1))

    def key(self):
        """Hashable exact parameter signature."""
        return (type(self).__name__, self.eps_inf, self.delta_eps, self.f1, self.f2)


@dataclass(frozen=True)
class PolePermittivity:
    """eps(f) = eps_inf + sum of the repo's DebyePole/DrudePole/LorentzPole susceptibilities.

    A causal pole-sum adapter: it evaluates the same pole objects a dispersive
    ``Material`` carries (``pole.susceptibility_at_freq``). Passive poles only: Lorentz
    gain (delta_eps < 0) and spatially varying custom poles are refused; eps_inf >= 1.
    """

    eps_inf: float
    poles: tuple

    def __post_init__(self):
        from .media import DebyePole, DrudePole, LorentzPole
        object.__setattr__(self, "eps_inf", _real_parameter(self.eps_inf, "PolePermittivity.eps_inf"))
        object.__setattr__(self, "poles", tuple(self.poles))
        if self.eps_inf < 1:
            raise ValueError("PolePermittivity.eps_inf < 1 is non-physical for a dielectric; require eps_inf >= 1")
        for pole in self.poles:
            if type(pole) not in (DebyePole, DrudePole, LorentzPole):
                raise TypeError("PolePermittivity poles must be DebyePole, DrudePole or LorentzPole "
                                f"(spatially uniform); got {type(pole).__name__}")
            for field in fields(pole):
                require_fixed_input(getattr(pole, field.name), name=f"{type(pole).__name__}.{field.name}")
            if type(pole) is LorentzPole and float(pole.delta_eps) < 0:
                raise ValueError("PolePermittivity refuses a gain LorentzPole (delta_eps < 0): not passive")

    @classmethod
    def from_material(cls, material):
        """Adapt an isotropic, nonmagnetic, electric-dispersive Material (e.g. a fit_dispersion result).

        Only eps_r and the electric poles are taken; static conduction belongs in Layer.sigma.
        """
        unapplied = [name for name, active in (
            ("sigma_e", float(material.sigma_e) != 0), ("mu_r", float(material.mu_r) != 1),
            ("magnetic poles", material.is_magnetic_dispersive), ("anisotropy", material.is_anisotropic),
            ("nonlinearity", material.is_nonlinear), ("modulation", material.is_modulated),
            ("sigma_m", material.sigma_m != 0), ("pec/pmc", material.pec or material.pmc)) if active]
        if unapplied:
            raise ValueError(f"PolePermittivity.from_material does not apply {', '.join(unapplied)}; "
                             "static conduction goes in Layer.sigma")
        return cls(material.eps_r, (*material.debye_poles, *material.drude_poles, *material.lorentz_poles))

    def relative_permittivity(self, frequency) -> complex:
        value = complex(self.eps_inf)
        for pole in self.poles:
            value += pole.susceptibility_at_freq(float(frequency))
        return value

    def key(self):
        """Hashable exact parameter signature."""
        return (type(self).__name__, self.eps_inf,
                tuple((type(p).__name__, *(float(getattr(p, f.name)) for f in fields(p))) for p in self.poles))


PERMITTIVITY_MODELS = (WidebandDebye, PolePermittivity)


def is_dispersive(medium) -> bool:
    """True for a Layer/HalfSpace whose eps_r is a frequency-dependent model."""
    return medium != "pec" and isinstance(medium.eps_r, PERMITTIVITY_MODELS)


def relative_permittivity(medium, frequency) -> complex:
    """Relative permittivity of a Layer/HalfSpace at ``frequency`` without the sigma term."""
    eps = medium.eps_r
    return eps.relative_permittivity(frequency) if isinstance(eps, PERMITTIVITY_MODELS) else complex(eps)


def permittivity_key(eps_r):
    """Exact comparison key: the constant itself, or the model's parameter signature."""
    return eps_r.key() if isinstance(eps_r, PERMITTIVITY_MODELS) else eps_r


def _require_fixed_fields(medium):
    # Layered Green functions read these values as host scalars; a trainable
    # tensor would lose its gradient there.
    for field in fields(medium):
        require_fixed_input(getattr(medium, field.name), name=f"{type(medium).__name__}.{field.name}")


@experimental("mom_layered")
@dataclass(frozen=True)
class Layer:
    name: str
    thickness: float
    eps_r: complex | WidebandDebye | PolePermittivity = 1.0
    sigma: float = 0.0
    mu_r: float = 1.0

    def __post_init__(self):
        _require_fixed_fields(self)
        if self.mu_r != 1:
            raise ValueError("Layered Green functions require mu_r=1")


@experimental("mom_layered")
@dataclass(frozen=True)
class HalfSpace:
    eps_r: complex | WidebandDebye | PolePermittivity = 1.0
    sigma: float = 0.0
    mu_r: float = 1.0

    def __post_init__(self):
        _require_fixed_fields(self)
        if self.mu_r != 1:
            raise ValueError("Layered Green functions require mu_r=1")


@experimental("mom_layered")
@dataclass(frozen=True)
class LayerStack:
    """Layers ordered bottom to top, with the first finite layer starting at z=0."""

    layers: tuple[Layer, ...]
    bottom: HalfSpace | str
    top: HalfSpace | str

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))

    def interfaces(self) -> tuple[float, ...]:
        """Interface heights, including the lower and upper exterior interfaces."""
        return (0.0, *accumulate(layer.thickness for layer in self.layers))

    @property
    def dispersive(self) -> bool:
        """True when any layer or half space has a frequency-dependent permittivity model."""
        return any(is_dispersive(m) for m in (self.bottom, *self.layers, self.top))
