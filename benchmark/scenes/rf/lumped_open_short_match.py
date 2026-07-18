"""Lumped one-port open/short/match calibration scene.

A feed port is excited with a matched Thevenin source and a second, passive port
placed two cells away is terminated by a resistive load. The three canonical
calibration standards are realised by the load resistance:

    matched : R = Z0  -> Gamma -> 0
    short   : R -> 0   -> Gamma -> -1
    open    : R -> inf -> Gamma -> +1

The reflection coefficient is read from the feed-port power waves after a real
broadband FDTD run (full pulse window, real time stepping, DFT accumulation) --
not from the single implicit-update algebraic identity the audit flagged. The
lumped two-port bench has near-field coupling between the ports, so the measured
reflection magnitude and phase are recorded against the analytic Gamma with the
coupling gap stated; the binding wave-level matched-|S11| gate (a propagating
transmission structure) lives in ``tests/rf/wave_validation``.
"""

from __future__ import annotations

import witwin.maxwell as mw

# Practical stand-ins for the ideal open/short standards.
SHORT_RESISTANCE = 1.0e-2
OPEN_RESISTANCE = 1.0e6


def analytic_gamma(load_resistance: float, reference_impedance: float = 50.0) -> complex:
    return complex(
        (load_resistance - reference_impedance)
        / (load_resistance + reference_impedance)
    )


def lumped_one_port_scene(
    *,
    load_resistance: float = 50.0,
    reference_impedance: float = 50.0,
    dx: float = 0.005,
    half_span: float = 0.02,
    device: str = "cuda",
) -> mw.Scene:
    """Build the feed + resistively-terminated load two-port calibration bench."""

    def _port(name, x, term=None):
        return mw.LumpedPort(
            name=name,
            positive=(x, 0.0, dx),
            negative=(x, 0.0, -dx),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(
                position=(x, 0.0, -0.5 * dx), size=(dx, 3.0 * dx, 0.0)
            ),
            reference_impedance=reference_impedance,
            termination=term,
        )

    load_termination = mw.SeriesRLC(r=load_resistance)
    return mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-half_span, half_span),
                (-half_span, half_span),
                (-half_span, half_span),
            )
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(_port("feed", -dx), _port("load", dx, term=load_termination)),
        device=device,
    )
