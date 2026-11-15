"""Device-parameter extraction from layered MoM networks (PyTorch-native, differentiable).

Every function reads a public ``NetworkData`` (``Result.network``) and returns
complex128/float64 tensors on the network's device. They use only torch
operations, so when the network carries GR1 autograd graphs (trainable
conductivity, roughness or sheet thickness) every quantity is differentiable.

Conventions (``exp(-i*omega*t)``, the NetworkData phasor convention):

- A series R-L branch has impedance ``Z = R - i*omega*L``, so
  ``L = -Im(Z)/omega``, ``R = Re(Z)`` and ``Q = -Im(Z)/Re(Z)``.
- A shunt G-C branch has admittance ``Y = G - i*omega*C``, so
  ``C = -Im(Y)/omega``, ``G = Re(Y)``.
- Port currents enter the network at the positive terminal.

Terminals. ``circuit_impedance`` is the core: a *terminal* is either a port
(name or index), driven against that port's own reference, or a pair
``(positive_port, negative_port)``, driven between the two ports with the
current entering ``positive`` and leaving ``negative`` (the common-mode
current is zero, i.e. floating). With the selected ports S and the incidence
D (``I_S = D I_w``, ``V_w = D^T V_S``) the terminal impedance matrix is
``Z_w = D^T Z_S D``. Ports outside S are either shorted (``others="short"``:
``Z_S = (Y_SS)^-1``) or open (``others="open"``: ``Z_S = Z_SS``). When S holds
every port the two coincide and ``Z_S = network.to_z()``.

``others`` has no default because the two choices give different devices:
a single-ended inductor measured at port 1 with port 2 grounded is
``inductance(net, "p1", others="short")`` (``Z_in = 1/Y11``); a transformer
winding with the other winding open is
``inductance(net, ("p1", "p2"), others="open")``. The differential inductance
of a two-port inductor, ``inductance(net, ("p1", "p2"), others=...)``, uses
every port, so ``others`` does not change it (``Z_d = Z11 + Z22 - Z12 - Z21``).
"""

from dataclasses import dataclass
import math

import torch


SHORT, OPEN = "short", "open"


def _omega(network):
    return 2 * math.pi * network.frequencies.to(torch.float64)


def _port(network, port):
    if isinstance(port, str):
        if port not in network.port_names:
            raise ValueError(f"Unknown port {port!r}; ports are {network.port_names}")
        return network.port_names.index(port)
    if type(port) is not int or not 0 <= port < len(network.port_names):
        raise ValueError(f"Port index {port!r} is outside 0..{len(network.port_names) - 1}")
    return port


def _terminal(network, terminal):
    """(positive index, negative index or None) of one terminal."""
    if isinstance(terminal, tuple):
        if len(terminal) != 2:
            raise ValueError("A differential terminal is a (positive_port, negative_port) pair")
        positive, negative = (_port(network, p) for p in terminal)
        if positive == negative:
            raise ValueError("A differential terminal needs two distinct ports")
        return positive, negative
    return _port(network, terminal), None


def port_matrix(network, ports, *, others, kind="z"):
    """Z (``kind="z"``) or Y of the selected ports with all other ports shorted or open."""
    if others not in (SHORT, OPEN):
        raise ValueError("others must be 'short' or 'open'")
    if kind not in ("z", "y"):
        raise ValueError("kind must be 'z' or 'y'")
    index = [_port(network, p) for p in ports]
    if len(set(index)) != len(index):
        raise ValueError("port_matrix needs distinct ports")
    if sorted(index) == list(range(len(network.port_names))):
        matrix = network.to_z() if kind == "z" else network.to_y()
        return matrix[:, index][:, :, index]
    # Shorted ports: V=0 there, so the selected block of Y is exact; open ports: I=0, Z block.
    native = network.to_y() if others == SHORT else network.to_z()
    block = native[:, index][:, :, index]
    if (others == SHORT) == (kind == "y"):
        return block
    return 1 / block if len(index) == 1 else torch.linalg.inv(block)


def circuit_impedance(network, terminals, *, others):
    """``[F, W, W]`` impedance matrix ``Z_w = D^T Z_S D`` of the declared terminals.

    ``terminals`` is a sequence of ports and ``(positive, negative)`` port
    pairs (see the module docstring). A port may appear in several terminals
    (for example a centre tap).
    """
    pairs = [_terminal(network, t) for t in terminals]
    if not pairs:
        raise ValueError("circuit_impedance needs at least one terminal")
    selected = sorted({i for pair in pairs for i in pair if i is not None})
    z = port_matrix(network, selected, others=others, kind="z")
    incidence = torch.zeros((len(selected), len(pairs)), dtype=z.dtype, device=z.device)
    for column, (positive, negative) in enumerate(pairs):
        incidence[selected.index(positive), column] = 1
        if negative is not None:
            incidence[selected.index(negative), column] = -1
    if incidence.shape[0] == incidence.shape[1] and torch.equal(
            incidence, torch.eye(len(selected), dtype=z.dtype, device=z.device)):
        return z
    return incidence.transpose(0, 1) @ z @ incidence


def terminal_impedance(network, terminal, *, others):
    """``[F]`` driving-point impedance of one terminal (port or differential pair)."""
    return circuit_impedance(network, [terminal], others=others)[:, 0, 0]


def inductance(network, terminal, *, others):
    """``L = -Im(Z)/omega`` of one terminal (effective; negative above self-resonance)."""
    return -terminal_impedance(network, terminal, others=others).imag / _omega(network)


def resistance(network, terminal, *, others):
    """``R = Re(Z)`` of one terminal."""
    return terminal_impedance(network, terminal, others=others).real


def quality_factor(network, terminal, *, others):
    """``Q = -Im(Z)/Re(Z)`` of one terminal.

    One-port definition for a port, differential definition for a
    ``(positive, negative)`` pair (common mode floating).
    """
    z = terminal_impedance(network, terminal, others=others)
    return -z.imag / z.real


def mutual_inductance(network, first, second, *, others):
    """``M = -Im(Z_w[0, 1])/omega`` between two terminals.

    Positive M means a current entering both terminals' positive ends links
    flux in the same sense.
    """
    z = circuit_impedance(network, [first, second], others=others)
    return -z[:, 0, 1].imag / _omega(network)


def coupling_coefficient(network, first, second, *, others):
    """``k = M / sqrt(L1 L2)``; NaN where either effective self-inductance is nonpositive."""
    z = circuit_impedance(network, [first, second], others=others)
    inductances = -z.imag / _omega(network)[:, None, None]
    first_l, second_l, mutual = inductances[:, 0, 0], inductances[:, 1, 1], inductances[:, 0, 1]
    valid = (first_l > 0) & (second_l > 0)
    # The masked product keeps the backward pass finite where k is undefined.
    product = torch.where(valid, first_l * second_l, torch.ones_like(first_l))
    return torch.where(valid, mutual / torch.sqrt(product), torch.full_like(first_l, math.nan))


def self_resonance(network, terminal, *, others):
    """First self-resonance frequency (0-d tensor, NaN if none is sampled).

    The first sample interval where L changes from positive to nonpositive;
    the reactance ``X = Im(Z)`` is interpolated linearly to its zero. The value
    is differentiable with respect to the two bracketing samples.
    """
    frequencies = network.frequencies.to(torch.float64)
    if len(frequencies) < 2:
        raise ValueError("self_resonance needs at least two frequencies")
    if bool((frequencies.diff() <= 0).any()):
        raise ValueError("self_resonance needs strictly increasing frequencies")
    z = terminal_impedance(network, terminal, others=others)
    reactance = z.imag
    inductive = -reactance / _omega(network) > 0
    crossing = inductive[:-1] & ~inductive[1:]
    i = torch.argmax(crossing.to(torch.int64))
    x0, x1 = reactance[i], reactance[i + 1]
    span = torch.where(crossing[i], x1 - x0, torch.ones_like(x0))
    value = frequencies[i] + (frequencies[i + 1] - frequencies[i]) * (-x0) / span
    return torch.where(crossing.any(), value, torch.full_like(value, math.nan))


def peak_quality_factor(network, terminal, *, others):
    """``(Q_peak, frequency)`` over the sampled inductive frequencies (L > 0).

    The peak is the first maximum among samples; no interpolation. ``Q_peak``
    is differentiable with respect to that sample.
    """
    z = terminal_impedance(network, terminal, others=others)
    q = -z.imag / z.real
    inductive = -z.imag / _omega(network) > 0
    if not bool(inductive.any()):
        raise ValueError("peak_quality_factor needs at least one inductive sample (L > 0)")
    i = torch.argmax(torch.where(inductive, q.detach(), torch.full_like(q.detach(), -math.inf)))
    return q[i], network.frequencies[i]


def _band(network, band):
    frequencies = network.frequencies.to(torch.float64)
    if band is None:
        return torch.ones_like(frequencies, dtype=torch.bool)
    low, high = band
    select = (frequencies >= low) & (frequencies <= high)
    if not bool(select.any()):
        raise ValueError(f"No sampled frequency lies in the fit band {band}")
    return select


@dataclass(frozen=True)
class PiModel:
    """Per-frequency pi equivalent of two ports (other ports shorted), exp(-i omega t).

    ``series_impedance = -2/(Y12+Y21)``; ``shunt_admittance = (Y11+Y12, Y22+Y21)``.
    """

    frequencies: torch.Tensor
    series_impedance: torch.Tensor
    shunt_admittance: tuple

    @property
    def series_inductance(self):
        return -self.series_impedance.imag / (2 * math.pi * self.frequencies)

    @property
    def series_resistance(self):
        return self.series_impedance.real

    @property
    def shunt_capacitance(self):
        return tuple(-y.imag / (2 * math.pi * self.frequencies) for y in self.shunt_admittance)

    @property
    def shunt_conductance(self):
        return tuple(y.real for y in self.shunt_admittance)


def pi_model(network, first=0, second=1):
    """Exact per-frequency pi decomposition of the two-port ``(first, second)``."""
    y = port_matrix(network, [first, second], others=SHORT, kind="y")
    transfer = (y[:, 0, 1] + y[:, 1, 0]) / 2
    return PiModel(network.frequencies.to(torch.float64), -1 / transfer,
                   (y[:, 0, 0] + y[:, 0, 1], y[:, 1, 1] + y[:, 1, 0]))


@dataclass(frozen=True)
class TModel:
    """Per-frequency T equivalent of two ports (other ports open), exp(-i omega t).

    Arms ``Z11 - Zm`` and ``Z22 - Zm`` with shunt ``Zm = (Z12+Z21)/2``. For
    coupled inductors the arm inductances are ``L1 - M`` and ``L2 - M`` and the
    shunt inductance is ``M``.
    """

    frequencies: torch.Tensor
    arm_impedance: tuple
    shunt_impedance: torch.Tensor

    @property
    def arm_inductance(self):
        return tuple(-z.imag / (2 * math.pi * self.frequencies) for z in self.arm_impedance)

    @property
    def arm_resistance(self):
        return tuple(z.real for z in self.arm_impedance)

    @property
    def shunt_inductance(self):
        return -self.shunt_impedance.imag / (2 * math.pi * self.frequencies)

    @property
    def shunt_resistance(self):
        return self.shunt_impedance.real


def t_model(network, first=0, second=1, *, others):
    """Exact per-frequency T decomposition; ``first``/``second`` may be differential pairs."""
    z = circuit_impedance(network, [first, second], others=others)
    mutual = (z[:, 0, 1] + z[:, 1, 0]) / 2
    return TModel(network.frequencies.to(torch.float64),
                  (z[:, 0, 0] - mutual, z[:, 1, 1] - mutual), mutual)


def fit_t_model(network, first=0, second=1, *, others, band=None):
    """Constant ``R - i omega L`` per T branch by linear least squares over ``band``.

    For each branch impedance Z(omega): ``R = mean(Re Z)`` and
    ``L = -sum(omega Im Z) / sum(omega^2)`` over the band samples. Closed form,
    hence differentiable. Returns a dict of SI values (``L1_H, R1_ohm, L2_H,
    R2_ohm, M_H, Rm_ohm``; arm values are ``L - M``) and the relative RMS
    residual of each branch.
    """
    model = t_model(network, first, second, others=others)
    select = _band(network, band)
    omega = 2 * math.pi * model.frequencies[select]
    values, residuals = {}, {}
    for name, z in (("1", model.arm_impedance[0]), ("2", model.arm_impedance[1]), ("m", model.shunt_impedance)):
        z = z[select]
        r = z.real.mean()
        l = -(omega * z.imag).sum() / (omega * omega).sum()
        fitted = r - 1j * omega * l
        residuals[name] = torch.sqrt(((fitted - z).abs() ** 2).mean() / (z.abs() ** 2).mean())
        values[name] = (r, l)
    return dict(L1_H=values["1"][1] + values["m"][1], R1_ohm=values["1"][0],
                L2_H=values["2"][1] + values["m"][1], R2_ohm=values["2"][0],
                M_H=values["m"][1], Rm_ohm=values["m"][0],
                relative_branch_rms=torch.stack([residuals[k] for k in ("1", "2", "m")]))


# Nine-element pi of an on-chip inductor over a lossy substrate (F-A R6 model):
# series (Rs - i omega Ls) parallel Cs, each shunt arm Cox in series with
# (Rsub parallel Csub).
PI_PARAMETERS = ("Ls_H", "Rs_ohm", "Cs_F", "Cox1_F", "Csub1_F", "Rsub1_ohm", "Cox2_F", "Csub2_F", "Rsub2_ohm")
PI_INITIAL = (3.3e-9, 3., 5e-15, 80e-15, 30e-15, 1000., 80e-15, 30e-15, 1000.)
PI_LOWER = (.01e-9, .001, .001e-15, *((.001e-15, .001e-15, .001) * 2))
PI_UPPER = (100e-9, 1e4, 1e-10, *((1e-10, 1e-10, 1e10) * 2))


def substrate_pi_branches(frequencies, parameters):
    """``[F, 3]`` complex (series admittance, shunt 1, shunt 2) of the nine-element model."""
    omega = 2 * math.pi * frequencies
    ls, rs, cs, cox1, csub1, rsub1, cox2, csub2, rsub2 = parameters.unbind(-1)
    series = 1 / (rs - 1j * omega * ls) - 1j * omega * cs
    arms = []
    for cox, csub, rsub in ((cox1, csub1, rsub1), (cox2, csub2, rsub2)):
        substrate = 1 / rsub - 1j * omega * csub
        arms.append(1 / (1 / (-1j * omega * cox) + 1 / substrate))
    return torch.stack((series, *arms), -1)


def _pi_targets(network, first, second):
    y = port_matrix(network, [first, second], others=SHORT, kind="y")
    return torch.stack((-(y[:, 0, 1] + y[:, 1, 0]) / 2, y[:, 0, 0] + y[:, 0, 1], y[:, 1, 1] + y[:, 1, 0]), -1)


def _pi_residual(log_parameters, frequencies, target):
    scale = torch.sqrt((target.abs() ** 2).mean(0))
    error = (substrate_pi_branches(frequencies, torch.exp(log_parameters)) - target) / scale
    return torch.cat((error.real.reshape(-1), error.imag.reshape(-1)))


@dataclass(frozen=True)
class PiFit:
    """Fitted nine-element pi; ``parameters`` is ``[9]`` float64 in ``PI_PARAMETERS`` order."""

    parameters: torch.Tensor
    active_bounds: tuple
    relative_branch_rms: torch.Tensor
    cost: float
    iterations: int
    band: tuple

    def as_dict(self):
        return dict(zip(PI_PARAMETERS, self.parameters.unbind()))

    def admittance(self, frequencies):
        """``[F, 2, 2]`` model admittance at ``frequencies`` (Hz tensor)."""
        branches = substrate_pi_branches(frequencies.to(torch.float64), self.parameters)
        series, first, second = branches.unbind(-1)
        return torch.stack((torch.stack((series + first, -series), -1),
                            torch.stack((-series, series + second), -1)), -2)


def fit_pi_model(network, first=0, second=1, *, band=(.5e9, 8e9), initial=PI_INITIAL,
                 lower=PI_LOWER, upper=PI_UPPER, max_iterations=500):
    """Fit the nine-element substrate pi to the per-frequency pi branches in ``band``.

    Objective (identical to the F-A R6 analysis): log-parameter least squares
    of the three complex branch errors, each normalised by its RMS over the
    band, with box bounds. The minimiser is a deterministic projected
    Levenberg-Marquardt iteration in float64 on the network's device. It stops
    when a step changes the cost by less than 1e-15 relative or the projected
    gradient is below 1e-12 relative, and raises if neither happens.

    Gradients use the implicit-function theorem at the optimum: with free
    parameters x, loss L(x, y) = |r|^2/2 and exact Hessian H = d^2L/dx^2,
    ``dx/dy = -H^-1 d^2L/dx dy``. Parameters on an active bound have zero
    derivative. The forward value is exactly the iterate.
    """
    frequencies = network.frequencies.to(torch.float64)
    select = _band(network, band)
    if int(select.sum()) < 2:
        raise ValueError("fit_pi_model needs at least two band samples: each three-element branch has "
                         "three parameters and one complex sample gives two real equations")
    target = _pi_targets(network, first, second)[select]
    f = frequencies[select]
    device = target.device
    low = torch.log(torch.tensor(lower, dtype=torch.float64, device=device))
    high = torch.log(torch.tensor(upper, dtype=torch.float64, device=device))
    x = torch.log(torch.tensor(initial, dtype=torch.float64, device=device)).clamp(low, high)
    fixed_target = target.detach()

    def residual(values):
        return _pi_residual(values, f, fixed_target)

    with torch.no_grad():
        r = residual(x)
    cost = .5 * float(r @ r)
    damping, converged, iteration = 1e-3, False, 0
    for iteration in range(1, max_iterations + 1):
        jacobian = torch.func.jacfwd(residual)(x)
        with torch.no_grad():
            r = residual(x)
            gradient = jacobian.T @ r
            # Projected active set: bounds the descent direction would cross.
            active = ((x <= low) & (gradient > 0)) | ((x >= high) & (gradient < 0))
            free = ~active
            if float(gradient[free].abs().max()) <= 1e-12 * max(cost, 1e-300) ** .5 * float(
                    jacobian[:, free].abs().max()):
                converged = True
                break
            normal = jacobian[:, free].T @ jacobian[:, free]
            accepted = False
            while damping < 1e16:
                system = normal + damping * torch.diag(torch.diagonal(normal))
                step = torch.zeros_like(x)
                step[free] = torch.linalg.solve(system, -gradient[free])
                trial = (x + step).clamp(low, high)
                trial_r = residual(trial)
                trial_cost = .5 * float(trial_r @ trial_r)
                if trial_cost < cost:
                    accepted = True
                    break
                damping *= 10
            if not accepted:
                converged = True
                break
            change = (cost - trial_cost) / cost
            x, cost, damping = trial, trial_cost, max(damping / 10, 1e-12)
            if change < 1e-15:
                converged = True
                break
    if not converged:
        raise RuntimeError(f"fit_pi_model did not converge in {max_iterations} iterations")
    with torch.no_grad():
        gradient = torch.func.jacfwd(residual)(x).T @ residual(x)
    active = ((x <= low) & (gradient > 0)) | ((x >= high) & (gradient < 0))
    free = ~active
    if target.requires_grad:
        def loss(free_values, data):
            values = torch.where(free, torch.zeros_like(x), x)
            values = values.index_put((free.nonzero().squeeze(-1),), free_values)
            # Real view: functorch transforms require real inputs.
            r = _pi_residual(values, f, torch.view_as_complex(data))
            return .5 * (r @ r)

        hessian = torch.func.hessian(loss)(x[free], torch.view_as_real(fixed_target))
        g = torch.func.grad(loss)(x[free], torch.view_as_real(target))
        correction = -torch.linalg.solve(hessian, g)
        update = torch.zeros_like(x).index_put((free.nonzero().squeeze(-1),), correction - correction.detach())
        x = x + update
    parameters = torch.exp(x)
    branches = substrate_pi_branches(f, parameters)
    error = branches - target
    relative = torch.sqrt((error.abs() ** 2).mean(0) / (target.abs() ** 2).mean(0))
    return PiFit(parameters, tuple(name for name, a in zip(PI_PARAMETERS, active.tolist()) if a),
                 relative, cost, iteration, tuple(band) if band is not None else None)
