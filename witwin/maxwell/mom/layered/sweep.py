"""Deterministic greedy rational sampling for the layered public runtime."""

from dataclasses import dataclass

import torch

from ...network import NetworkData
from ...rational import PassivityReport, RationalFitConfig, RationalModel


@dataclass(frozen=True)
class AdaptiveSweepResult:
    """Sampled Result.network stays physical; this holds its in-band surrogate.

    dense_network is the 2000-point evaluated model, exportable through the
    existing NetworkData.to_touchstone method. Passivity is checked on that
    grid and with the shared pole-aware interval check. Violations are rejected;
    no residue or direct-term perturbation changes the physical response.
    """

    model: RationalModel
    passivity_report: PassivityReport
    dense_network: NetworkData
    estimated_error: float
    solved_frequencies: int
    enforcement_change: float = 0.0

    def evaluate_network(self, frequencies):
        frequencies = torch.as_tensor(frequencies, device=self.model.poles.device, dtype=torch.float64)
        return NetworkData(frequencies=frequencies, s=self.model.evaluate(frequencies),
                           z0=self.dense_network.z0[0], port_names=self.dense_network.port_names)


def _fit(network):
    order = min(8, len(network.frequencies) - 2)
    return network.fit_rational(RationalFitConfig(order=order, iterations=8,
                               diagnostic_only=True), representation="S")


def _join(networks):
    first = networks[0]
    frequencies = torch.cat([n.frequencies for n in networks])
    order = torch.argsort(frequencies, stable=True)
    metadata = dict(first.metadata)
    indices = order.tolist()
    for key in ("factor_dtype", "refinement_steps", "backward_error"):
        if key in metadata:
            values = [value for n in networks for value in n.metadata[key]]
            metadata[key] = tuple(values[i] for i in indices)
    return NetworkData(frequencies=frequencies[order], s=torch.cat([n.s for n in networks])[order],
                       z0=first.z0[0], port_names=first.port_names, metadata=metadata)


def adaptive_network(sweep, initial_network, initial_currents, solve):
    """Nested fits select interval midpoints; all validation solves count.

    Nine uniform seeds resolve both band ends. Compare each new fit with the
    previous nested fit on 2000 points. Bisect the interval with largest change.
    Once both fit change and training residual are <= tolerance/4, independently
    solve the midpoint with largest S curvature, then require its prediction
    error and the updated fit change to pass too. This is an estimator, not a
    uniform error proof; a budget exhaustion raises rather than returning a
    surrogate with an unqualified tolerance.
    """
    networks, current_batches = [initial_network], [initial_currents]
    subset = torch.arange(0, 9, 2, device=initial_network.s.device)
    previous = _fit(NetworkData(frequencies=initial_network.frequencies[subset],
        s=initial_network.s[subset], z0=initial_network.z0[0], port_names=initial_network.port_names))
    grid = torch.linspace(sweep.f_min, sweep.f_max, 2000, device=initial_network.s.device, dtype=torch.float64)
    validated = False
    validation_error = float("inf")
    while True:
        network = _join(networks)
        model = _fit(network)
        response = model.evaluate(grid)
        change = (response - previous.evaluate(grid)).abs().amax(dim=(-1, -2))
        error = float(change.max())
        residual = float((model.evaluate(network.frequencies) - network.s).abs().max())
        if validated and max(error, residual, validation_error) <= sweep.tolerance / 4:
            report = model.check_passivity(grid)
            if not report.passive:
                raise RuntimeError(f"Adaptive rational model violates in-band passivity by {report.max_violation:g}")
            dense = NetworkData(frequencies=grid, s=response, z0=network.z0[0], port_names=network.port_names)
            frequencies = torch.cat([n.frequencies for n in networks])
            order = torch.argsort(frequencies, stable=True)
            currents = torch.cat(current_batches)[order]
            network.metadata["adaptive_sweep"] = dict(
                solved_frequencies=len(network.frequencies), estimated_error=max(error,residual,validation_error),
                passivity_margin=report.margin, passivity_certified=report.certified,
                passivity_samples=report.sample_count, enforcement_change=0.0)
            return network, currents, AdaptiveSweepResult(model,report,dense,
                max(error,residual,validation_error),len(network.frequencies))
        if len(network.frequencies) >= sweep.max_points:
            raise RuntimeError(f"Adaptive sweep exhausted max_points={sweep.max_points} before tolerance {sweep.tolerance:g}")
        midpoints = (network.frequencies[:-1] + network.frequencies[1:]) / 2
        if max(error, residual) <= sweep.tolerance / 4:
            # Validate where linear interpolation is least consistent with the
            # current rational response, so resonant/high-curvature gaps win.
            midpoint_response = model.evaluate(midpoints)
            linear = (network.s[:-1] + network.s[1:]) / 2
            scores = (midpoint_response-linear).abs().amax(dim=(-1,-2))
            validated = True
        else:
            midpoint_response = model.evaluate(midpoints)
            scores = (midpoint_response-previous.evaluate(midpoints)).abs().amax(dim=(-1,-2))
            validated = False
        index = int(torch.argmax(scores))
        frequency = float(midpoints[index])
        batch, currents = solve((frequency,))
        validation_error = float((batch.s-model.evaluate(batch.frequencies)).abs().max())
        networks.append(batch)
        current_batches.append(currents)
        previous = model
