"""Series/parallel RLC resonator scene (two-port lumped bench).

An excited feed port and a passive port terminated by a ``SeriesRLC`` (or
``ParallelRLC`` network) sit two cells apart in a small PML box. A real FDTD run
drives the feed with a broadband pulse; the load-port branch current shows the
circuit resonance directly in the FDTD fields (series RLC = minimum impedance,
maximum branch current at ``f0``). The measured resonance is compared against the
closed-form ``f0 = 1 / (2*pi*sqrt(L*C))`` and ``Q`` references.

This is *not* the discretised-impedance-formula sweep the audit flagged: the
solver is stepped through resonance and the peak is read from field-derived port
data. The near-field coupling of the lumped two-port bench shifts the measured
resonance relative to the ideal circuit value; the wave-level gate uses the
resonance-existence-and-location tolerance derived in
``tests/rf/wave_validation`` and the benchmark records the measured gap to the
plan-01 2% target honestly rather than asserting a tighter bound than the bench
supports.
"""

from __future__ import annotations

import witwin.maxwell as mw


def series_rlc_scene(
    *,
    r: float = 8.0,
    l: float = 0.5e-9,  # noqa: E741 - circuit notation
    c: float = 1.0e-12,
    reference_impedance: float = 50.0,
    dx: float = 0.005,
    half_span: float = 0.02,
    parallel: bool = False,
    device: str = "cuda",
) -> mw.Scene:
    """Build the two-port RLC resonator bench (feed + RLC-terminated load)."""

    termination = (
        mw.ParallelRLC(r=r, l=l, c=c) if parallel else mw.SeriesRLC(r=r, l=l, c=c)
    )

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
        ports=(_port("feed", -dx), _port("load", dx, term=termination)),
        device=device,
    )
