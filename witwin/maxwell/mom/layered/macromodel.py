"""Wideband passive macromodel of a sampled layered network (plan C3).

Pipeline (``passive_macromodel``):

1. Shared-pole vector fit of the scattering matrix and its real state-space
   realization S(s) = C (sI - A)^-1 B + D, s = -i 2 pi f. The fit is the shared
   ``fit_rational`` algorithm (same starting poles, real basis and relocation,
   reused from rational.py) with one difference: every least-squares problem is
   solved by Householder QR (``_qr_lstsq``), which is bitwise repeatable on CPU
   and CUDA. The shared fit uses the default CPU least-squares driver, whose
   results vary between calls at the 1e-16 level.
2. Bounded-real Hamiltonian test. With R = D^T D - I and Q = D D^T - I,

       M = [[A - B R^-1 D^T C,  -B R^-1 B^T],
            [C^T Q^-1 C,        -A^T + C^T D R^-1 B^T]]

   has an eigenvalue i w exactly where a singular value of S(i w) equals 1. A
   stable model with |D|_2 < 1 and no imaginary eigenvalue of M therefore has
   |S(i w)|_2 < 1 at every frequency from DC to infinity (continuity from the
   passive limit at infinity). The crossings split the axis into bands; a band
   whose midpoint has |S|_2 > 1 is a violation band. Since A, B, C, D are real,
   S(-i w) = conj S(i w) and the test is independent of the phasor sign.
3. Passivity enforcement. Asymptotic first: singular values of D above
   1 - margin are clipped and C is refitted to the samples (fixed poles). Then residue perturbation: only C changes (poles and
   stability are kept). Each step minimizes the response change
   sum_f |dC X(s_f)|_F^2, X(s) = (sI - A)^-1 B, over the fit samples plus a
   uniform band grid (a sampled Gramian G = Re sum X X^H), subject to the
   first-order singular-value constraints sigma_i + Re(u_i^H dC X v_i) <= 1 - margin
   at sample points inside every violation band (points accumulate over steps).
   The quadratic program is solved exactly in whitened coordinates through its
   dual (Lawson-Hanson active set). Steps repeat until the Hamiltonian test passes.

Exports. ``admittance_system`` is the exact linear-fractional image of the
scattering realization in SI units (V in, I out) for real per-port reference
impedances; ``to_spice`` writes it as a subcircuit of grounded state
capacitors and voltage-controlled current sources (standard G/C cards).
SPICE AC analysis uses exp(+j w t), so its port admittances are the complex
conjugates of this package's exp(-i w t) values; the realization is real and
both describe the same circuit.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ...network import NetworkData
from ...rational import (RationalModel, StateSpaceNetwork, _canonicalize_relocated_poles, _conjugate_groups,
                         _initial_poles, _parameter_basis, _parameters_to_residues, bilinear_discretize)
from ..._stability import experimental

AXIS_TOLERANCE = 1e-6
CROSSING_TOLERANCE = 1e-6
BAND_SAMPLES = 17
GRAMIAN_GRID = 512
ACTIVE_SET_STEPS = 400


@experimental("mom_layered")
@dataclass(frozen=True)
class HamiltonianReport:
    """Bounded-real Hamiltonian passivity test of a scattering state-space model.

    ``passive`` is the global (DC to infinity) certificate: stable poles,
    ``direct_norm`` < 1 and no eigenvalue of the Hamiltonian within
    ``AXIS_TOLERANCE`` (relative) of the imaginary axis. ``crossings_hz`` are the
    frequencies where a singular value equals 1, ``violation_bands_hz`` the bands
    with |S|_2 > 1 and ``max_singular_value`` the largest sampled |S|_2 in them.
    ``axis_distance`` is min |Re l| / |l| over the Hamiltonian eigenvalues away
    from DC; ``dc_lossless`` marks a singular value of exactly 1 at f = 0.
    """

    passive: bool
    crossings_hz: tuple
    violation_bands_hz: tuple
    max_singular_value: float
    direct_norm: float
    axis_distance: float
    dc_lossless: bool
    state_count: int


@experimental("mom_layered")
@dataclass(frozen=True)
class MacromodelReport:
    """Accuracy and enforcement record; errors are max |dS| entries over the samples.

    ``fit_*`` describe the unconstrained vector fit, ``*_error`` the passive
    model. ``enforcement_change`` is max over the samples and the band grid of
    |S_passive - S_fit|_2. ``before`` and ``after`` are the Hamiltonian tests.
    """

    order: int
    fit_max_error: float
    fit_rms_error: float
    max_error: float
    rms_error: float
    enforcement_change: float
    enforcement_steps: int
    constraint_points: int
    margin: float
    band_hz: tuple
    before: HamiltonianReport
    after: HamiltonianReport


@experimental("mom_layered")
@dataclass(frozen=True)
class PassiveMacromodel:
    """Passive real state-space scattering model with reference impedances z0 [P] (ohm)."""

    system: StateSpaceNetwork
    z0: torch.Tensor
    port_names: tuple
    report: MacromodelReport

    def evaluate(self, frequencies):
        """NetworkData of the model at ``frequencies`` (Hz)."""
        frequencies = torch.as_tensor(frequencies, device=self.system.A.device, dtype=torch.float64)
        return NetworkData(frequencies=frequencies, s=self.system.evaluate(frequencies), z0=self.z0,
                           port_names=self.port_names)

    def hamiltonian(self):
        return hamiltonian_passivity(self.system)

    def admittance_system(self):
        return admittance_system(self.system, self.z0)

    def discretize(self, dt):
        """Bilinear (Tustin) discretization; it preserves bounded-realness exactly."""
        return bilinear_discretize(self.system, dt)

    def to_spice(self, path, *, name="macromodel", state_capacitance=1e-12):
        """Write the admittance realization as a SPICE subcircuit; returns the path."""
        path = Path(path)
        path.write_text(spice_subcircuit(self.admittance_system(), self.port_names, name=name,
                                         state_capacitance=state_capacitance))
        return path


def _response(system, s):
    """X(s) = (sI - A)^-1 B for complex s [F]: [F, n, P]."""
    n = system.state_count
    dtype = torch.complex128
    identity = torch.eye(n, device=system.A.device, dtype=dtype)
    matrix = s[:, None, None] * identity - system.A.to(dtype)
    return torch.linalg.solve(matrix, system.B.to(dtype).expand(len(s), -1, -1))


def _laplace(frequencies):
    return torch.complex(torch.zeros_like(frequencies), -2 * torch.pi * frequencies)


def _singular_max(system, frequencies):
    return torch.linalg.svdvals(system.evaluate(frequencies)).amax(dim=-1)


def hamiltonian_passivity(system):
    """HamiltonianReport of a real scattering StateSpaceNetwork (module docstring).

    Eigenvalues within AXIS_TOLERANCE of the largest eigenvalue magnitude are the
    DC point: a singular value of exactly 1 at f = 0 is the lossless DC limit of
    a network without a resistive DC path (an open even mode), which bounded
    realness allows; ``dc_lossless`` records it and it is not a violation, since
    with no crossing at f > 0 and |D|_2 < 1 every singular value is below 1 on
    (0, infinity).
    """
    A, B, C, D = system.A, system.B, system.C, system.D
    ports, n = D.shape[0], system.state_count
    identity = torch.eye(ports, device=D.device, dtype=D.dtype)
    direct_norm = float(torch.linalg.matrix_norm(D, ord=2))
    stable = bool(torch.all(torch.linalg.eigvals(A.clone()).real < 0))
    if not stable or direct_norm >= 1:
        return HamiltonianReport(False, (), (), float("inf"), direct_norm, 0.0, False, n)
    R = D.T @ D - identity
    Q = D @ D.T - identity
    Ri_Dt_C = torch.linalg.solve(R, D.T @ C)
    Ri_Bt = torch.linalg.solve(R, B.T)
    M = torch.cat((torch.cat((A - B @ Ri_Dt_C, -B @ Ri_Bt), 1),
                   torch.cat((C.T @ torch.linalg.solve(Q, C), -A.T + C.T @ D @ Ri_Bt), 1)), 0)
    eigenvalues = torch.linalg.eigvals(M)
    magnitude = eigenvalues.abs()
    dc = magnitude <= AXIS_TOLERANCE * float(magnitude.max())
    relative = eigenvalues.real.abs() / magnitude.clamp_min(torch.finfo(torch.float64).tiny)
    axis_distance = float(relative[~dc].min()) if bool((~dc).any()) else 1.0
    candidates = eigenvalues[(relative <= AXIS_TOLERANCE) & ~dc & (eigenvalues.imag >= 0)]
    omega = torch.sort(candidates.imag.abs()).values / (2 * torch.pi)
    crossings, confirmed = [], True
    if len(omega):
        # A candidate is a crossing when some singular value is 1 there.
        sigma = torch.linalg.svdvals(system.evaluate(omega))
        hit = ((sigma - 1).abs() <= CROSSING_TOLERANCE * sigma.amax(-1, keepdim=True)).any(-1)
        crossings = [float(f) for f, h in zip(omega, hit) if h]
        # An unconfirmed near-axis eigenvalue voids the certificate (conservative).
        confirmed = bool(hit.all())
    dc_lossless = bool(dc.any())
    if dc_lossless:
        sigma0 = float(torch.linalg.svdvals(system.evaluate(torch.zeros(1, device=D.device,
                                                                        dtype=torch.float64))).max())
        if abs(sigma0 - 1) > CROSSING_TOLERANCE:
            confirmed = False
    bands, worst = [], 0.0
    if crossings:
        edges = torch.tensor([0.0, *crossings], device=D.device, dtype=torch.float64)
        upper = torch.tensor([*crossings, 2 * crossings[-1]], device=D.device, dtype=torch.float64)
        intervals = torch.stack((edges, upper), 1)[:-1]
        intervals = intervals[intervals[:, 1] > intervals[:, 0]]
        t = torch.linspace(0, 1, BAND_SAMPLES, device=D.device, dtype=torch.float64)[1:-1]
        points = intervals[:, :1] + (intervals[:, 1:] - intervals[:, :1]) * t
        sigma = _singular_max(system, points.reshape(-1)).reshape(points.shape)
        peak = sigma.amax(1)
        bands = [tuple(map(float, band)) for band, p in zip(intervals, peak) if p > 1]
        worst = float(peak.max())
    passive = not crossings and confirmed
    return HamiltonianReport(passive, tuple(crossings), tuple(bands), worst, direct_norm, axis_distance,
                             dc_lossless, n)


def _active_set(constraints, bounds):
    """argmin |z|^2 s.t. constraints @ z <= bounds, through its dual NNLS-type problem.

    z = -K^T l with l = argmin_{l >= 0} l^T Q l / 2 + bounds^T l, Q = K K^T. Lawson-Hanson
    active set: add the most violated dual gradient, then solve on the passive set and step
    back to the feasible boundary until every passive multiplier is positive (finite).
    """
    gram = constraints @ constraints.T
    count = len(bounds)
    ridge = 1e-13 * float(gram.diagonal().max())
    tolerance = 1e-12 * max(1.0, float(bounds.abs().max()))
    multipliers = torch.zeros_like(bounds)
    passive = torch.zeros(count, dtype=torch.bool, device=bounds.device)
    for _ in range(ACTIVE_SET_STEPS):
        gradient = -(gram @ multipliers + bounds)
        gradient[passive] = -float("inf")
        candidate = int(torch.argmax(gradient))
        if float(gradient[candidate]) <= tolerance:
            return -constraints.T @ multipliers
        passive[candidate] = True
        while True:
            index = torch.nonzero(passive).flatten()
            block = gram[index][:, index] + ridge * torch.eye(len(index), device=gram.device, dtype=gram.dtype)
            trial = torch.zeros_like(multipliers)
            trial[index] = -torch.linalg.solve(block, bounds[index])
            if bool((trial[index] > 0).all()):
                multipliers = trial
                break
            blocked = index[trial[index] <= 0]
            step = float(torch.min(multipliers[blocked] / (multipliers[blocked] - trial[blocked])))
            multipliers = multipliers + step * (trial - multipliers)
            passive &= multipliers > 0
            multipliers[~passive] = 0
    raise RuntimeError(f"Passivity quadratic program did not settle in {ACTIVE_SET_STEPS} active-set steps")


def enforce_passivity(system, frequencies, samples, *, margin=1e-4, max_steps=24):
    """Residue (C) perturbation until the Hamiltonian test passes; returns (system, steps, points, before).

    ``samples`` [F, P, P] are the S values at ``frequencies``: when the direct term
    is clipped, C is refitted to them by least squares with A, B and the new D fixed.
    """
    device = system.A.device
    if not bool(torch.all(torch.linalg.eigvals(system.A.clone()).real < 0)):
        raise ValueError("The macromodel has unstable poles")
    before = hamiltonian_passivity(system)
    if before.direct_norm > 1 - margin:
        # Asymptotic passivity first: clip the singular values of the real direct term, then refit
        # C (fixed poles) so the in-band response absorbs the change: C X_f ~ S_f - D for all f.
        u, sigma, vh = torch.linalg.svd(system.D)
        direct = (u * sigma.clamp_max(1 - margin)) @ vh
        X = _response(system, _laplace(frequencies))                       # [F, n, P]
        residual = samples.to(torch.complex128) - direct                   # [F, P, P]
        # Row r of C: C_r X_f = residual_f[r]  ->  X_f^T C_r^T = residual_f[r]^T, stacked over f.
        matrix = X.transpose(1, 2).reshape(-1, system.state_count)          # [F P, n]
        target = residual.transpose(1, 2).reshape(-1, residual.shape[1])    # [F P, P]
        system = replace(system, D=direct, C=_qr_lstsq(matrix, target).T)
    grid = torch.linspace(float(frequencies.min()), float(frequencies.max()), GRAMIAN_GRID, device=device,
                          dtype=torch.float64)
    X = _response(system, _laplace(torch.cat((frequencies, grid))))
    gramian = torch.einsum("fip,fjp->ij", X, X.conj()).real
    gramian = gramian + 1e-12 * float(gramian.diagonal().max()) * torch.eye(len(gramian), device=device,
                                                                              dtype=torch.float64)
    L = torch.linalg.cholesky(gramian)
    points = torch.empty(0, device=device, dtype=torch.float64)
    report, steps = hamiltonian_passivity(system), 0
    while not report.passive:
        if steps == max_steps:
            raise RuntimeError(f"Passivity enforcement did not converge in {max_steps} steps "
                               f"(max |S|_2 {report.max_singular_value:.9g})")
        t = torch.linspace(0, 1, BAND_SAMPLES, device=device, dtype=torch.float64)
        bands = torch.tensor(report.violation_bands_hz, device=device, dtype=torch.float64).reshape(-1, 2)
        crossings = torch.tensor(report.crossings_hz, device=device, dtype=torch.float64)
        points = torch.unique(torch.cat((points, crossings,
                                         (bands[:, :1] + (bands[:, 1:] - bands[:, :1]) * t).reshape(-1))))
        u, sigma, vh = torch.linalg.svd(system.evaluate(points))
        Xp = _response(system, _laplace(points))
        select = sigma > 1 - 2 * margin
        f_index, mode = torch.nonzero(select, as_tuple=True)
        left = u[f_index, :, mode]                                    # [m, P]
        right = (Xp[f_index] @ vh[f_index, mode].conj()[..., None])[..., 0]   # [m, n]
        # Re(u^H dC X v) = sum_rc Re(conj(u_r) (X v)_c) dC_rc
        rows = (left.conj()[:, :, None] * right[:, None, :]).real    # [m, P, n]
        whitened = torch.linalg.solve_triangular(L, rows.transpose(1, 2), upper=False).transpose(1, 2)
        bounds = (1 - margin) - sigma[f_index, mode]
        z = _active_set(whitened.reshape(len(rows), -1), bounds)
        delta = torch.linalg.solve_triangular(L.T, z.reshape(system.C.shape).T, upper=True).T
        system = replace(system, C=system.C + delta)
        steps += 1
        report = hamiltonian_passivity(system)
    return system, steps, len(points), before


def _qr_lstsq(matrix, target):
    """Column-scaled real least squares of a complex system by Householder QR (fixed order, repeatable)."""
    real_matrix = torch.cat((matrix.real, matrix.imag), 0)
    real_target = torch.cat((target.real, target.imag), 0)
    scales = torch.linalg.vector_norm(real_matrix, dim=0)
    q, r = torch.linalg.qr(real_matrix / scales)
    return torch.linalg.solve_triangular(r, q.T @ real_target, upper=True) / scales[:, None]


def vector_fit(frequencies, values, order, iterations):
    """Real shared-pole fit of values [F, P, P] (no proportional term): RationalModel in S.

    Mirrors rational.fit_rational (initial poles, real basis, pole relocation by the
    sigma-function zeros, stable canonical poles) with QR least squares.
    """
    s = torch.complex(torch.zeros_like(frequencies), -2 * torch.pi * frequencies)
    flat = values.reshape(len(frequencies), -1).to(torch.complex128)
    count = flat.shape[1]
    poles = _initial_poles(order, frequencies, dtype=torch.complex128)
    for _ in range(iterations):
        basis, groups = _parameter_basis(s, poles)
        width = basis.shape[1] + 1
        matrix = torch.zeros((len(s) * count, count * width + basis.shape[1]), device=s.device, dtype=s.dtype)
        for k in range(count):
            rows = slice(k * len(s), (k + 1) * len(s))
            matrix[rows, k * width:(k + 1) * width] = torch.cat((basis, torch.ones_like(s[:, None])), 1)
            matrix[rows, count * width:] = -flat[:, k, None] * basis
        sigma = _qr_lstsq(matrix, flat.T.reshape(-1, 1))[count * width:, 0]
        residues = _parameters_to_residues(sigma, groups, order, dtype=poles.dtype)
        relocation = torch.diag(poles) - torch.ones((order, 1), device=s.device, dtype=s.dtype) @ residues[None, :]
        poles = _canonicalize_relocated_poles(torch.linalg.eigvals(relocation), enforce_stability=True)
    basis, groups = _parameter_basis(s, poles)
    parameters = _qr_lstsq(torch.cat((basis, torch.ones_like(s[:, None])), 1), flat).T
    residues = _parameters_to_residues(parameters[:, :-1], groups, order, dtype=poles.dtype)
    ports = values.shape[1]
    return RationalModel(poles=poles, residues=residues.reshape(ports, ports, order),
                         direct=parameters[:, -1].to(poles.dtype).reshape(ports, ports), representation="S")


def rational_from_realization(model, system):
    """RationalModel with model's poles whose residues/direct are read back from system's C and D.

    Inverse of RationalModel.to_state_space: input column j owns states
    [j*order, (j+1)*order); a real pole carries C = r, a conjugate pair C = [2 Re r, -2 Im r].
    """
    order, ports = model.order, model.input_count
    residues = torch.zeros_like(model.residues)
    for j in range(ports):
        cursor = j * order
        for first, second in _conjugate_groups(model.poles):
            if second is None:
                residues[:, j, first] = system.C[:, cursor].to(residues.dtype)
                cursor += 1
            else:
                value = torch.complex(system.C[:, cursor], -system.C[:, cursor + 1]) / 2
                residues[:, j, first], residues[:, j, second] = value, value.conj()
                cursor += 2
    return RationalModel(poles=model.poles, residues=residues, direct=system.D.to(residues.dtype),
                         representation=model.representation)


def enforce_rational_passivity(model, frequencies, samples, *, margin=1e-4, max_steps=24):
    """Hamiltonian-certified passive copy of a stable square S RationalModel (no proportional term).

    ``samples`` are the S values the model was fitted to at ``frequencies``.

    Returns (model, HamiltonianReport after, steps); the poles are unchanged.
    """
    system, steps, _, _ = enforce_passivity(model.to_state_space(), frequencies.to(torch.float64), samples,
                                            margin=margin, max_steps=max_steps)
    return rational_from_realization(model, system), hamiltonian_passivity(system), steps


def _macromodel(network, frequencies, order, iterations, margin, max_steps):
    model = vector_fit(frequencies, network.s, order, iterations)
    fitted = model.to_state_space(port_order=network.port_names)
    system, steps, points, before = enforce_passivity(fitted, frequencies, network.s, margin=margin,
                                                      max_steps=max_steps)
    samples = network.s.to(torch.complex128)

    def errors(candidate):
        error = (candidate.evaluate(frequencies) - samples).abs()
        return float(error.max()), float(error.square().mean().sqrt())

    grid = torch.linspace(float(frequencies.min()), float(frequencies.max()), GRAMIAN_GRID,
                          device=frequencies.device, dtype=torch.float64)
    check = torch.cat((frequencies, grid))
    change = float(torch.linalg.matrix_norm(system.evaluate(check) - fitted.evaluate(check), ord=2).max())
    report = MacromodelReport(order, *errors(fitted), *errors(system), change, steps, points, margin,
                              (float(frequencies.min()), float(frequencies.max())), before,
                              hamiltonian_passivity(system))
    return PassiveMacromodel(system, network.z0[0].real.to(torch.float64), tuple(network.port_names), report)


@experimental("mom_layered")
def passive_macromodel(network, *, tolerance=1e-3, order=None, max_order=24, iterations=8, margin=1e-4,
                       max_steps=24):
    """Fit, certify and (if needed) enforce a passive scattering macromodel of ``network``.

    ``network`` is a complete NetworkData (for example ``Result.network`` of an
    adaptive layered sweep) with real, positive, frequency-independent
    reference impedances. With ``order=None`` the even orders 2..max_order are
    tried in turn and the first passive model whose max |dS| against the samples
    is <= ``tolerance`` is returned; an explicit ``order`` tries only that order.
    Raises if no tried order meets the tolerance. Enforcement targets
    |S|_2 <= 1 - ``margin`` in every violation band (module docstring); the
    passive model's error obeys max_error <= fit_max_error + enforcement_change.
    """
    z0 = network.z0
    if bool((z0.imag != 0).any()) or not bool((z0 == z0[:1]).all()) or bool((z0.real <= 0).any()):
        raise ValueError("passive_macromodel needs real, positive, frequency-independent z0")
    frequencies = network.frequencies.to(torch.float64)
    orders = (order,) if order is not None else range(2, max_order + 1, 2)
    errors = []
    for candidate in orders:
        if len(frequencies) < candidate + 2:
            break
        model = _macromodel(network, frequencies, candidate, iterations, margin, max_steps)
        if model.report.max_error <= tolerance:
            return model
        errors.append(f"{candidate}: {model.report.max_error:.3g}")
    raise RuntimeError(f"No passive macromodel within max |dS| <= {tolerance:g} of the samples "
                       f"(order: error {', '.join(errors)}); add samples or raise max_order")


def admittance_system(system, z0):
    """Exact Y realization (V in, I out, SI) of a scattering realization with real reference z0 [P].

    With normalized waves a = (v + i)/2, b = (v - i)/2, v = V/sqrt(z0), i = sqrt(z0) I:
    A_y = A - B (I+D)^-1 C, B_n = B (I+D)^-1, C_n = -2 (I+D)^-1 C, D_n = (I+D)^-1 (I-D).
    """
    A, B, C, D = system.A, system.B, system.C, system.D
    identity = torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
    inverse = torch.linalg.inv(identity + D)
    root = z0.real.to(D.dtype).rsqrt()
    return StateSpaceNetwork(A=A - B @ inverse @ C, B=(B @ inverse) * root, C=root[:, None] * (-2 * inverse @ C),
                             D=root[:, None] * (inverse @ (identity - D)) * root, representation="Y",
                             port_order=system.port_order)


def _number(value):
    return f"{value:.17g}"


def spice_subcircuit(system, port_names, *, name, state_capacitance):
    """SPICE subcircuit text of a Y realization: ports (p_k, p_k_ref), states x_j on capacitors to ground.

    Node x_j carries state j: C_x_j = state_capacitance and VCCS cards inject
    state_capacitance * (A x + B V) into it. Each port pair draws I = C x + D V.
    Exact zero coefficients are omitted.
    """
    A, B, C, D = (value.detach().cpu().tolist() for value in (system.A, system.B, system.C, system.D))
    ports = [(f"p{k + 1}", f"p{k + 1}_ref") for k in range(len(port_names))]
    lines = [f"* Passive macromodel {name}: {len(A)} states, {len(ports)} ports "
             f"({', '.join(f'{p}/{r}={n}' for (p, r), n in zip(ports, port_names))})",
             "* Y realization dx/dt = A x + B V, I = C x + D V; port current enters p_k and leaves p_k_ref.",
             "* SPICE AC (exp(+jwt)) admittances are the complex conjugates of the exp(-iwt) source data.",
             f".SUBCKT {name} {' '.join(node for pair in ports for node in pair)}"]
    for j in range(len(A)):
        lines.append(f"C_x{j + 1} x{j + 1} 0 {_number(state_capacitance)}")
    for j, row in enumerate(A):
        lines += [f"G_A_{j + 1}_{k + 1} 0 x{j + 1} x{k + 1} 0 {_number(state_capacitance * v)}"
                  for k, v in enumerate(row) if v != 0]
    for j, row in enumerate(B):
        lines += [f"G_B_{j + 1}_{k + 1} 0 x{j + 1} {ports[k][0]} {ports[k][1]} {_number(state_capacitance * v)}"
                  for k, v in enumerate(row) if v != 0]
    for i, row in enumerate(C):
        lines += [f"G_C_{i + 1}_{j + 1} {ports[i][0]} {ports[i][1]} x{j + 1} 0 {_number(v)}"
                  for j, v in enumerate(row) if v != 0]
    for i, row in enumerate(D):
        lines += [f"G_D_{i + 1}_{k + 1} {ports[i][0]} {ports[i][1]} {ports[k][0]} {ports[k][1]} {_number(v)}"
                  for k, v in enumerate(row) if v != 0]
    lines.append(f".ENDS {name}")
    return "\n".join(lines) + "\n"
