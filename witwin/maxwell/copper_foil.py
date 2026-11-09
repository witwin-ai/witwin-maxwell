"""Passive finite-thickness conductor model for two-sided planar foil interfaces."""
from __future__ import annotations

import math

import torch

from .media import Material, _coerce_frequency_range


MU_0 = 1.25663706212e-6
EPSILON_0 = 8.8541878128e-12


class CopperFoil(Material):
    """A finite conductive thickness represented on a zero-thickness support.

    Attach this material to an axis-aligned planar ``Box``. ``thickness`` is the
    physical foil thickness in meters; it does not create a mesh interval. The
    two tangential electric traces are independent and coupled through conductor
    diffusion, including transmission through a thin film. ``frequency_range``
    declares the validated band. The good-conductor approximation requires
    displacement current / conduction current <= 1e-5 throughout that band.

    Port order is the low and high faces, with currents directed into the foil.
    The repository uses exp(-i omega t). This material carries no normal sheet
    conductivity. Bulk dielectric loss remains a separate material assignment.

    ``edge_model`` selects the treatment of free foil edges (a tangential trace
    whose in-plane neighbour across the edge carries neither foil nor a
    conductor contact). A zero-thickness sheet's edge current is singular, so
    the Yee rim trace alone cannot represent the physical edge loss; it would
    also grow logarithmically under refinement. The finite-thickness edge model
    multiplies the rim trace's two-face impedance by a geometric factor
    kappa >= 1 derived from the rectangular-edge conformal map of a foil with
    this thickness (Lewin stopping distance t/(4 pi e^pi)) matched to the
    discrete Yee half-plane edge solution. Scaling the full impedance keeps the
    internal inductance consistent with the loss (Wheeler's incremental-
    inductance rule). The model assumes surface current on every face, i.e. a
    good-conductor foil, t >> skin depth.

    The correction is weighted by ``edge_weight``, a C1 smoothstep of
    x = t / delta(f_max) at the band's upper edge: 0 for x <= 2, 1 for x >= 4.
    kappa is therefore continuous and differentiable in thickness and
    conductivity across the thin/thick transition (gradients see the blend).
    Gating on the upper edge keeps the correction for broadband sweeps whose
    low end is thin (35 um Cu is 3 skin depths only above ~45 MHz); the model is
    least certain (~10%) near t ~ 3 delta, and a warning is emitted whenever any
    part of the band has t < 3 delta while the correction is active.

    ``"auto"`` (``DEFAULT_EDGE_MODEL``) applies the weighted correction,
    ``"finite_thickness"`` applies it at full weight (it requires t >= 3
    delta at the upper band edge), and ``"none"`` keeps the bare
    zero-thickness rim.
    """

    is_copper_foil = True
    EDGE_MODELS = ("auto", "finite_thickness", "none")
    DEFAULT_EDGE_MODEL = "auto"
    # Smoothstep window of t / delta(f_max) over which "auto" turns kappa on.
    EDGE_WEIGHT_WINDOW = (2., 4.)

    def __init__(self, *, thickness, frequency_range, conductivity=5.8e7, mode_count=None, name=None,
                 edge_model=None):
        from ._fixed_inputs import require_fixed_input
        require_fixed_input(frequency_range, name="CopperFoil.frequency_range")
        super().__init__(eps_r=1.0, mu_r=1.0, sigma_e=0.0, name=name)
        for parameter, value in (("thickness", thickness), ("conductivity", conductivity)):
            if torch.is_tensor(value) and (value.ndim != 0 or not value.is_floating_point()):
                raise TypeError(f"CopperFoil {parameter} tensor must be a real floating scalar.")
            resolved = float(value.detach()) if torch.is_tensor(value) else float(value)
            if not math.isfinite(resolved) or resolved <= 0:
                raise ValueError(f"CopperFoil {parameter} must be finite and positive.")
            object.__setattr__(self, parameter, value if torch.is_tensor(value) else resolved)
        if mode_count is not None and (isinstance(mode_count, bool) or not isinstance(mode_count, int) or mode_count < 2):
            raise ValueError("CopperFoil mode_count must be an integer of at least 2.")
        if mode_count is None and any(torch.is_tensor(value) and value.requires_grad for value in (thickness, conductivity)):
            raise ValueError("Trainable CopperFoil parameters require explicit mode_count to fix the modal topology.")
        tensors = [value for value in (thickness, conductivity) if torch.is_tensor(value)]
        if len({value.device for value in tensors}) > 1:
            raise ValueError("CopperFoil parameter tensors must share one device.")
        object.__setattr__(self, "mode_count", mode_count)
        band = _coerce_frequency_range(frequency_range, name="CopperFoil frequency_range")
        if 2 * math.pi * band[1] * EPSILON_0 / self.conductivity > 1e-5:
            raise ValueError("CopperFoil diffusion requires omega*epsilon0/conductivity <= 1e-5.")
        object.__setattr__(self, "frequency_range", band)
        edge_model = self.DEFAULT_EDGE_MODEL if edge_model is None else edge_model
        if edge_model not in self.EDGE_MODELS:
            raise ValueError(f"CopperFoil edge_model must be one of {self.EDGE_MODELS}.")
        object.__setattr__(self, "edge_model", edge_model)
        if edge_model == "finite_thickness" and float(self._skin_ratio(band[1])) < 3:
            raise ValueError("CopperFoil edge_model='finite_thickness' requires thickness >= 3 skin depths "
                             "at the upper band edge; a resistive thin film has a different edge physics.")
        if self.edge_correction_active and float(self._skin_ratio(band[0])) < 3:
            import warnings
            warnings.warn("CopperFoil free-edge model is applied to a band whose lower part has thickness < 3 "
                          f"skin depths (t/delta = {float(self._skin_ratio(band[0])):.3g} at {band[0]:.4g} Hz); "
                          "its edge loss is uncertain there (~10% near t ~ 3 delta).", stacklevel=2)

    def _skin_ratio(self, frequency, *, thickness=None, conductivity=None):
        """t / delta at one frequency, differentiable in tensor parameters."""
        thickness = self.thickness if thickness is None else thickness
        conductivity = self.conductivity if conductivity is None else conductivity
        if not torch.is_tensor(thickness) and not torch.is_tensor(conductivity):
            return torch.tensor(thickness * math.sqrt(math.pi * frequency * MU_0 * conductivity), dtype=torch.float64)
        return torch.as_tensor(thickness, dtype=torch.float64) * torch.sqrt(
            torch.as_tensor(math.pi * frequency * MU_0, dtype=torch.float64) * conductivity)

    def edge_weight(self, *, thickness=None, conductivity=None):
        """Weight in [0, 1] of the free-edge correction (C1 in thickness/conductivity)."""
        if self.edge_model == "none":
            return torch.zeros((), dtype=torch.float64)
        if self.edge_model == "finite_thickness":
            return torch.ones((), dtype=torch.float64)
        low, high = self.EDGE_WEIGHT_WINDOW
        ratio = self._skin_ratio(self.frequency_range[1], thickness=thickness, conductivity=conductivity)
        step = ((ratio - low) / (high - low)).clamp(0, 1)
        return step.square() * (3 - 2 * step)

    @property
    def edge_correction_active(self):
        """Whether free-edge traces receive a nonzero finite-thickness edge impedance."""
        return bool(self.edge_weight().detach() > 0)

    def _frequencies(self, frequencies):
        tensors = [value for value in (self.thickness, self.conductivity) if torch.is_tensor(value)]
        device = tensors[0].device if tensors else None
        values = torch.as_tensor(frequencies, dtype=torch.float64, device=device)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or not values.numel() or not bool(torch.all(torch.isfinite(values))):
            raise ValueError("CopperFoil frequencies must be a nonempty finite vector.")
        if not bool(torch.all((values >= self.frequency_range[0]) & (values <= self.frequency_range[1]))):
            raise ValueError("CopperFoil frequencies must lie inside its declared frequency_range.")
        return values

    def skin_depth(self, frequency):
        values = self._frequencies(frequency)
        return torch.sqrt(2 / (2 * math.pi * values * MU_0 * self.conductivity))

    def surface_impedance(self, frequencies):
        """Exact diffusion two-port Z matrix, shape [F, 2, 2], in ohms."""
        omega = 2 * math.pi * self._frequencies(frequencies)
        s = torch.complex(torch.zeros_like(omega), -omega)
        gamma = torch.sqrt(s * MU_0 * self.conductivity)
        zc = s * MU_0 / gamma
        argument = gamma * self.thickness
        exponential = torch.exp(-argument)
        denominator = -torch.expm1(-2 * argument)
        diagonal = zc * (1 + exponential.square()) / denominator
        mutual = zc * 2 * exponential / denominator
        return torch.stack((torch.stack((diagonal, mutual), -1),
                            torch.stack((mutual, diagonal), -1)), -2)

    def modal_admittance(self, frequencies):
        """Even/odd tangential face admittances, shape [F, 2]."""
        omega = 2 * math.pi * self._frequencies(frequencies)
        s = torch.complex(torch.zeros_like(omega), -omega)
        gamma = torch.sqrt(s * MU_0 * self.conductivity)
        zc = s * MU_0 / gamma
        half_argument = 0.5 * gamma * self.thickness
        tangent = torch.tanh(half_argument)
        return torch.stack((tangent / zc, 1 / (tangent * zc)), -1)

    def surface_admittance(self, frequencies):
        """Exact diffusion two-port Y matrix, shape [F, 2, 2], in siemens."""
        modes = self.modal_admittance(frequencies)
        diagonal = 0.5 * (modes[:, 0] + modes[:, 1])
        mutual = 0.5 * (modes[:, 0] - modes[:, 1])
        return torch.stack((torch.stack((diagonal, mutual), -1),
                            torch.stack((mutual, diagonal), -1)), -2)
