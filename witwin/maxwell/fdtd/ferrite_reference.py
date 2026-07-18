"""Torch reference model for the gyromagnetic (Polder) ferrite discretization.

This module is intentionally independent of the production stepping runtime. It
defines the Phase-0 numerical contract used to validate the native ferrite
implementation, and it is deliberately not a CPU fallback: nothing in the solver
may import it.

The physics contract it encodes is frozen in
``docs/reference/ferrite-physics-contract.md``. In short (``exp(-i*omega*t)``
convention, bias unit vector ``b``, ``omega_0 = gamma*mu_0*|H0|``,
``omega_m = gamma*mu_0*Ms``):

* Linearized LLG magnetization ADE ``dm/dt = P m + Q h`` with the skew-precession
  matrix ``K``, Gilbert mass ``G``, and drive ``S`` given in the contract.
* Scalar Polder response ``mu = mu_inf + omega_m*W/D``, ``kappa = omega_m*omega/D``
  with ``W = omega_0 - i*alpha*omega`` and ``D = W^2 - omega^2``.
* Lab-frame tensor ``mu_r = mu*(I - b b^T) + mu_inf*(b b^T) + i*kappa*[b]_x``.
* Implicit-midpoint (trapezoidal) update; unconditionally stable and passive for
  ``alpha >= 0`` (orthogonal propagator at ``alpha = 0`` -> exact energy
  conservation).

The binding to the production path will live in the slice-1c forward test, which
steps the compiled CUDA kernel against this reference at
``ACCEPTANCE_BUDGET.reference_polder_rtol``. Until then the reference is bound by
``tests/materials/ferrite/test_ferrite_reference.py`` (self-consistency, bias
reversal, passivity, convergence, and the ``GyromagneticFerrite`` analytic
cross-check). Keep those bindings alive: without a consumer this reference
silently stops constraining anything.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
SPEED_OF_LIGHT = 1.0 / math.sqrt(EPSILON_0 * MU_0)
GAMMA_DEFAULT = 1.760859e11  # rad/(s*T)


@dataclass(frozen=True)
class AcceptanceBudget:
    """Pre-registered acceptance limits for the gyromagnetic ferrite implementation.

    Mirrors ``docs/reference/ferrite-physics-contract.md`` section 6 verbatim;
    the two must stay identical. Loosening any value is governed by the change
    rule in that document (pre-register a named near-resonance scene with its
    physical budget); tightening is always allowed.
    """

    reference_polder_rtol: float = 1.0e-5
    analytic_response_rel_err: float = 2.0e-2
    analytic_phase_err_deg: float = 3.0
    passive_energy_residual: float = 1.0e-2
    convergence_tiers: int = 3
    param_gradient_rel_err: float = 2.0e-2
    bias_reversal_symmetry_rtol: float = 1.0e-5
    ferrite_free_perf_regression: float = 1.0e-2


ACCEPTANCE_BUDGET = AcceptanceBudget()


def _as_real_tensor(value, *, dtype=torch.float64, like=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.is_complex():
            raise TypeError("ferrite reference parameters must be real")
    else:
        tensor = torch.as_tensor(value, dtype=dtype)
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    return tensor


@dataclass(frozen=True)
class FerriteReferenceParameters:
    """Single-cell gyromagnetic parameters in SI units.

    ``bias_unit_vector`` is the (normalized) direction of the static bias; the
    gyrotropy handedness is carried by this vector, while ``omega_0`` and
    ``omega_m`` use magnitudes only, so a bias reversal is purely a sign flip of
    the vector (see :meth:`with_reversed_bias`).
    """

    saturation_magnetization: float
    bias_magnitude: float
    bias_unit_vector: tuple[float, float, float]
    gilbert_damping: float = 0.0
    gyromagnetic_ratio: float = GAMMA_DEFAULT
    mu_infinity: float = 1.0
    eps_r: float = 1.0

    def __post_init__(self):
        vec = tuple(float(component) for component in self.bias_unit_vector)
        if len(vec) != 3:
            raise ValueError("bias_unit_vector must be a 3-vector.")
        norm = math.sqrt(sum(component * component for component in vec))
        if norm == 0.0:
            raise ValueError("bias_unit_vector must be non-zero.")
        object.__setattr__(self, "bias_unit_vector", tuple(component / norm for component in vec))
        if float(self.saturation_magnetization) <= 0.0:
            raise ValueError("saturation_magnetization must be > 0.")
        if float(self.bias_magnitude) <= 0.0:
            raise ValueError("bias_magnitude must be > 0.")
        if float(self.gilbert_damping) < 0.0:
            raise ValueError("gilbert_damping must be >= 0.")
        if float(self.gyromagnetic_ratio) <= 0.0:
            raise ValueError("gyromagnetic_ratio must be > 0.")
        if float(self.mu_infinity) <= 0.0:
            raise ValueError("mu_infinity must be > 0.")

    @property
    def omega_0(self) -> float:
        return float(self.gyromagnetic_ratio) * MU_0 * float(self.bias_magnitude)

    @property
    def omega_m(self) -> float:
        return float(self.gyromagnetic_ratio) * MU_0 * float(self.saturation_magnetization)

    def with_reversed_bias(self) -> "FerriteReferenceParameters":
        bx, by, bz = self.bias_unit_vector
        return FerriteReferenceParameters(
            saturation_magnetization=self.saturation_magnetization,
            bias_magnitude=self.bias_magnitude,
            bias_unit_vector=(-bx, -by, -bz),
            gilbert_damping=self.gilbert_damping,
            gyromagnetic_ratio=self.gyromagnetic_ratio,
            mu_infinity=self.mu_infinity,
            eps_r=self.eps_r,
        )


# --- Torch-native analytic core (differentiable in every argument) -----------


def state_space_matrices(omega_0, omega_m, gilbert_damping, *, dtype=torch.float64):
    """Local-frame magnetization ADE matrices ``P``, ``Q`` (2x2 real tensors).

    ``dm/dt = P m + Q h`` in the transverse local frame ``b = z_hat``. See
    contract section 2.2.
    """
    w0 = _as_real_tensor(omega_0, dtype=dtype)
    wm = _as_real_tensor(omega_m, dtype=dtype, like=w0)
    alpha = _as_real_tensor(gilbert_damping, dtype=dtype, like=w0)
    c = 1.0 / (1.0 + alpha * alpha)
    zero = torch.zeros_like(w0)
    one = torch.ones_like(w0)
    P = c * w0 * torch.stack(
        [
            torch.stack([-alpha, -one]),
            torch.stack([one, -alpha]),
        ]
    )
    Q = c * wm * torch.stack(
        [
            torch.stack([alpha, one]),
            torch.stack([-one, alpha]),
        ]
    )
    return P, Q


def _scalar_polder(omega, omega_0, omega_m, gilbert_damping, mu_infinity, *, dtype=torch.complex128):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    w = _as_real_tensor(omega, dtype=real_dtype).to(dtype)
    w0 = _as_real_tensor(omega_0, dtype=real_dtype).to(dtype)
    wm = _as_real_tensor(omega_m, dtype=real_dtype).to(dtype)
    alpha = _as_real_tensor(gilbert_damping, dtype=real_dtype).to(dtype)
    mu_inf = _as_real_tensor(mu_infinity, dtype=real_dtype).to(dtype)
    i = torch.tensor(1j, dtype=dtype)
    W = w0 - i * alpha * w
    D = W * W - w * w
    mu = mu_inf + wm * W / D
    kappa = wm * w / D
    return mu, kappa


def _cross_matrix(b: torch.Tensor) -> torch.Tensor:
    bx, by, bz = b[0], b[1], b[2]
    zero = torch.zeros_like(bx)
    return torch.stack(
        [
            torch.stack([zero, -bz, by]),
            torch.stack([bz, zero, -bx]),
            torch.stack([-by, bx, zero]),
        ]
    )


def polder_tensor(
    omega,
    *,
    omega_0,
    omega_m,
    gilbert_damping,
    mu_infinity,
    bias_unit_vector,
    dtype=torch.complex128,
) -> torch.Tensor:
    """Lab-frame 3x3 complex Polder permeability tensor (contract section 2.4).

    Differentiable in ``omega`` and every keyword argument that is a leaf tensor.
    """
    mu, kappa = _scalar_polder(omega, omega_0, omega_m, gilbert_damping, mu_infinity, dtype=dtype)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    b = _as_real_tensor(bias_unit_vector, dtype=real_dtype)
    b = (b / torch.linalg.vector_norm(b)).to(dtype)
    eye = torch.eye(3, dtype=dtype)
    bbT = torch.outer(b, b)
    i = torch.tensor(1j, dtype=dtype)
    mu_parallel = _as_real_tensor(mu_infinity, dtype=real_dtype).to(dtype)
    return mu * (eye - bbT) + mu_parallel * bbT + i * kappa * _cross_matrix(b)


def continuous_susceptibility(params: FerriteReferenceParameters, omega, *, dtype=torch.complex128) -> torch.Tensor:
    """Transverse 2x2 susceptibility ``chi = (-i*omega*I - P)^-1 Q`` (contract 2.3)."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    P, Q = state_space_matrices(params.omega_0, params.omega_m, params.gilbert_damping, dtype=real_dtype)
    P = P.to(dtype)
    Q = Q.to(dtype)
    w = _as_real_tensor(omega, dtype=real_dtype).to(dtype)
    i = torch.tensor(1j, dtype=dtype)
    eye = torch.eye(2, dtype=dtype)
    return torch.linalg.solve(-i * w * eye - P, Q)


def discrete_susceptibility(params: FerriteReferenceParameters, omega, dt, *, dtype=torch.complex128) -> torch.Tensor:
    """Exact discrete CW transfer function of the implicit-midpoint update (contract 3.2)."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    P, Q = state_space_matrices(params.omega_0, params.omega_m, params.gilbert_damping, dtype=real_dtype)
    P = P.to(dtype)
    Q = Q.to(dtype)
    w = _as_real_tensor(omega, dtype=real_dtype).to(dtype)
    dt_t = _as_real_tensor(dt, dtype=real_dtype).to(dtype)
    i = torch.tensor(1j, dtype=dtype)
    z = torch.exp(-i * w * dt_t)
    eye = torch.eye(2, dtype=dtype)
    lhs = (z - 1.0) * eye - (dt_t / 2.0) * (z + 1.0) * P
    return dt_t * torch.sqrt(z) * torch.linalg.solve(lhs, Q)


def continuous_polder(params: FerriteReferenceParameters, omega, *, dtype=torch.complex128):
    """Analytic scalar ``(mu, kappa)`` for the parameter set (contract 2.3)."""
    return _scalar_polder(
        omega, params.omega_0, params.omega_m, params.gilbert_damping, params.mu_infinity, dtype=dtype
    )


def discrete_polder(params: FerriteReferenceParameters, omega, dt, *, dtype=torch.complex128):
    """Scalar ``(mu, kappa)`` read from the discrete transfer function.

    ``chi_xx = mu - mu_inf`` and ``chi_yx = +i*kappa`` (contract 2.3/2.4).
    """
    chi = discrete_susceptibility(params, omega, dt, dtype=dtype)
    mu_inf = _as_real_tensor(params.mu_infinity, dtype=torch.float64).to(dtype)
    mu = chi[0, 0] + mu_inf
    kappa = chi[1, 0] / torch.tensor(1j, dtype=dtype)
    return mu, kappa


def reference_polder_tensor(params: FerriteReferenceParameters, omega, *, dtype=torch.complex128) -> torch.Tensor:
    """Lab-frame 3x3 tensor from the reference parameter set (continuous)."""
    return polder_tensor(
        omega,
        omega_0=params.omega_0,
        omega_m=params.omega_m,
        gilbert_damping=params.gilbert_damping,
        mu_infinity=params.mu_infinity,
        bias_unit_vector=params.bias_unit_vector,
        dtype=dtype,
    )


def circular_permeabilities(
    params: FerriteReferenceParameters, omega, *, propagation=(0.0, 0.0, 1.0), dtype=torch.complex128
):
    """Circular-polarization eigen-permeabilities ``mu_pm = mu +/- s*kappa`` (contract 2.5).

    The scalar ``kappa`` is magnitude-based, so the gyrotropy handedness is carried
    by ``s = sign(b . k_hat)``, the projection of the bias unit vector onto the
    propagation direction. This is what makes a bias reversal (``b -> -b``) swap
    ``mu_+`` and ``mu_-`` and flip the Faraday rotation.
    """
    mu, kappa = continuous_polder(params, omega, dtype=dtype)
    b = torch.as_tensor(params.bias_unit_vector, dtype=torch.float64)
    k = torch.as_tensor(propagation, dtype=torch.float64)
    projection = float(torch.dot(b, k / torch.linalg.vector_norm(k)))
    sign = 1.0 if projection >= 0.0 else -1.0
    return mu + sign * kappa, mu - sign * kappa


def faraday_rotation_angle(
    params: FerriteReferenceParameters, omega, length, *, propagation=(0.0, 0.0, 1.0), dtype=torch.complex128
) -> torch.Tensor:
    """1D Faraday rotation angle ``theta = (beta_+ - beta_-)*L/2`` [rad] (contract 2.5).

    ``beta_pm = Re((omega/c)*sqrt(eps_r * mu_pm))`` are the circular propagation
    constants for the two circular polarizations along ``propagation``. The sign
    flips under bias reversal.
    """
    mu_plus, mu_minus = circular_permeabilities(params, omega, propagation=propagation, dtype=dtype)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    eps_r = _as_real_tensor(params.eps_r, dtype=real_dtype).to(dtype)
    w = _as_real_tensor(omega, dtype=real_dtype).to(dtype)
    length_t = _as_real_tensor(length, dtype=real_dtype).to(dtype)
    k0 = w / SPEED_OF_LIGHT
    beta_plus = torch.real(k0 * torch.sqrt(eps_r * mu_plus))
    beta_minus = torch.real(k0 * torch.sqrt(eps_r * mu_minus))
    return (beta_plus - beta_minus) * torch.real(length_t) / 2.0


def recurrence_identity_residual(params: FerriteReferenceParameters, omega, dt, *, dtype=torch.complex128) -> float:
    """Relative residual of the closed-form phasor in the implicit-midpoint update.

    This is the pure-algebra identity gated by ``reference_polder_rtol``: the
    closed-form discrete phasor ``m^n = chi_d h_0 z^n`` must satisfy the update
    ``m^{n+1} = Phi m^n + Gamma h^{n+1/2}`` (contract section 3.2).
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    P, Q = state_space_matrices(params.omega_0, params.omega_m, params.gilbert_damping, dtype=real_dtype)
    P = P.to(dtype)
    Q = Q.to(dtype)
    eye = torch.eye(2, dtype=dtype)
    dt_t = _as_real_tensor(dt, dtype=real_dtype).to(dtype)
    Ainv = torch.linalg.inv(eye - (dt_t / 2.0) * P)
    Phi = Ainv @ (eye + (dt_t / 2.0) * P)
    Gamma = (Ainv * dt_t) @ Q
    chi = discrete_susceptibility(params, omega, dt, dtype=dtype)
    w = _as_real_tensor(omega, dtype=real_dtype).to(dtype)
    i = torch.tensor(1j, dtype=dtype)
    z = torch.exp(-i * w * dt_t)
    h0 = torch.tensor([1.0, 0.0], dtype=dtype)
    m_n = chi @ h0
    m_np1 = (chi @ h0) * z
    h_half = h0 * torch.sqrt(z)
    resid = m_np1 - (Phi @ m_n + Gamma @ h_half)
    return float(torch.linalg.vector_norm(resid) / torch.linalg.vector_norm(m_np1))


# --- Single-cell time-stepping reference -------------------------------------


class LLGReference:
    """Single-cell implicit-midpoint magnetization-ADE stepper (contract 3.1).

    A verification oracle only. ``run_cw`` drives a real CW field and DFT-extracts
    the steady-state susceptibility; ``energy_trajectory`` advances the free
    (undriven) precession to check passivity.
    """

    def __init__(self, params: FerriteReferenceParameters, dt: float, *, dtype=torch.float64):
        self.params = params
        self.dt = float(dt)
        self.dtype = dtype
        P, Q = state_space_matrices(params.omega_0, params.omega_m, params.gilbert_damping, dtype=dtype)
        eye = torch.eye(2, dtype=dtype)
        Ainv = torch.linalg.inv(eye - (self.dt / 2.0) * P)
        self.Phi = Ainv @ (eye + (self.dt / 2.0) * P)
        self.Gamma = (Ainv * self.dt) @ Q

    def energy_trajectory(self, m0, nsteps: int) -> torch.Tensor:
        """Free-precession energy ``E^n = 0.5 |m^n|^2`` over ``nsteps`` (no drive)."""
        m = torch.as_tensor(m0, dtype=self.dtype).clone()
        energies = torch.empty(nsteps + 1, dtype=self.dtype)
        energies[0] = 0.5 * (m @ m)
        for n in range(nsteps):
            m = self.Phi @ m
            energies[n + 1] = 0.5 * (m @ m)
        return energies

    def run_cw(self, omega: float, *, periods: int = 4000, settle: int = 2000):
        """Drive ``h = (cos(omega t), 0)`` and DFT-extract steady-state ``chi``.

        Returns ``(chi_xx, chi_yx)`` complex tensors extracted at the drive
        frequency over ``periods`` full periods after a ``settle`` transient.
        """
        omega = float(omega)
        steps_per_period = max(1, int(round(2.0 * math.pi / omega / self.dt)))
        n_settle = settle * steps_per_period
        n_window = periods * steps_per_period
        m = torch.zeros(2, dtype=self.dtype)
        acc = torch.zeros(2, dtype=torch.complex128)
        for n in range(n_settle + n_window):
            t_half = (n + 0.5) * self.dt
            h = torch.tensor([math.cos(omega * t_half), 0.0], dtype=self.dtype)
            m = self.Phi @ m + self.Gamma @ h
            if n >= n_settle:
                phase = torch.tensor(math.cos(omega * n * self.dt) + 1j * math.sin(omega * n * self.dt))
                acc = acc + m.to(torch.complex128) * phase
        chi = 2.0 * acc / n_window
        return chi[0], chi[1]
