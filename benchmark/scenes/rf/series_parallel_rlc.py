"""Series / parallel RLC resonator on the coax bench (in-line lumped element).

Rebuild (audit S1 follow-up). The retired bench placed a feed and an
RLC-terminated port two cells apart in a tiny PML box: the load-port branch
current was dominated by the feed/load near-field parasitic and barely tracked
the circuit ``C`` (the peak did NOT move with C -- recorded as a strict xfail open
gap). This rebuild inserts the lumped RLC as an IN-LINE two-terminal element in
the inner conductor of the PROVEN air coax line, ahead of a matched (through-PML)
continuation, so the element carries the full axial line current and its
resonance genuinely controls the feed reflection.

Topology: a single ``WavePort`` TEM feed launches the coax mode; the inner
conductor is cut by a two-cell axial gap at :data:`LOAD_PLANE` and bridged by a
``LumpedPort`` (voltage path along the line axis, current contour encircling the
rod) carrying the ``SeriesRLC`` or ``ParallelRLC`` termination. The coax continues
on through the right PML (a matched Z0 continuation), so the input seen by the
feed is ``Z_RLC + Z0`` (series element) -- the reflection therefore images the RLC
impedance directly:

* **series RLC** -- ``Z_RLC = R + jwL + 1/jwC`` is minimum (``= R``) at
  ``f0 = 1/(2*pi*sqrt(L C))``; with a small ``R`` the feed sees a near-match there,
  so ``|S11|`` shows a deep NOTCH at the resonance;
* **parallel RLC** -- ``Z_RLC`` is maximum (``= R``) at the anti-resonance
  ``f0``; with a large ``R`` the feed sees a near-open there, so ``|S11|`` shows a
  PEAK.

Parasitic model (measured, documented -- NOT hidden): the axial rod gap adds a
parallel fringe capacitance and the current loop a small series inductance, so the
measured resonance sits BELOW the ideal ``1/(2*pi*sqrt(L C))`` (series: ~13% low
and highly consistent; parallel: a larger, dilution-type shift because the fringe
capacitance adds directly to the parallel ``C``). What the gate binds is that the
resonance TRACKS ``C`` -- ``f_res * sqrt(C)`` is constant (series) and the
resonance moves in the correct direction under a +/-20% change in ``C`` -- which
is precisely what the retired parasitic-dominated bench failed to do.
"""

from __future__ import annotations

import math

import witwin.maxwell as mw

from benchmark.scenes.rf.coax_thru import (
    DOMAIN_TRANSVERSE,
    INNER_RADIUS,
    OUTER_RADIUS,
    SHIELD_OUTER,
    _HollowCylinder,
    _coax_port,
)

DOMAIN_X = 0.12
PML_LAYERS = 12
FEED_X = -0.08
LOAD_PLANE = 0.06

# Validated component defaults (resonance ~0.5 GHz at the default L/C tier).
DEFAULT_L = 25.0e-9
DEFAULT_C = 4.0e-12
SERIES_R = 5.0       # small R -> deep notch at series resonance
PARALLEL_R = 500.0   # large R -> pronounced peak at parallel anti-resonance


def line_length(dx: float) -> float:
    """Conductor length that runs through the PML to the padded grid edges."""
    return 2.0 * (DOMAIN_X + PML_LAYERS * dx) + 8.0 * dx


def resonance_frequency(l: float = DEFAULT_L, c: float = DEFAULT_C) -> float:  # noqa: E741
    """Ideal LC resonance ``f0 = 1/(2*pi*sqrt(L C))``."""
    return 1.0 / (2.0 * math.pi * math.sqrt(l * c))


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def default_frequencies(*, parallel: bool = False) -> tuple[float, ...]:
    """Sweep band bracketing the (parasitic-shifted) resonance for each topology."""
    if parallel:
        return tuple(float(x) for x in _linspace(0.27e9, 0.43e9, 9))
    return tuple(float(x) for x in _linspace(0.38e9, 0.54e9, 9))


def _snapped_loop_half(dx: float, *, target: float = 0.10) -> float:
    """Half-extent of the current loop, snapped to the transverse Yee half-grid.

    The loop boundary at +-half must land on ``(k + 0.5)*dx`` (symmetric about 0),
    strictly between the inner rod (``INNER_RADIUS``) and the shield
    (``OUTER_RADIUS``) so the contour encircles the rod's axial current.
    """
    k = round(target / dx - 0.5)
    half = (k + 0.5) * dx
    if not (INNER_RADIUS < half < OUTER_RADIUS):
        raise ValueError(
            f"snapped current-loop half {half} must sit between the rod and shield "
            f"({INNER_RADIUS}, {OUTER_RADIUS}); adjust dx."
        )
    return half


def series_rlc_scene(
    *,
    r: float | None = None,
    l: float = DEFAULT_L,  # noqa: E741 - circuit notation
    c: float = DEFAULT_C,
    parallel: bool = False,
    dx: float = 0.01,
    device: str = "cuda",
) -> mw.Scene:
    """Build the coax in-line RLC resonator bench (series or parallel)."""

    resistance = (PARALLEL_R if parallel else SERIES_R) if r is None else r
    termination = (
        mw.ParallelRLC(r=resistance, l=l, c=c)
        if parallel
        else mw.SeriesRLC(r=resistance, l=l, c=c)
    )

    length = line_length(dx)
    left_edge = -0.5 * length
    right_edge = 0.5 * length
    loop_half = _snapped_loop_half(dx)

    load = mw.LumpedPort(
        name="load",
        positive=(LOAD_PLANE + dx, 0.0, 0.0),
        negative=(LOAD_PLANE - dx, 0.0, 0.0),
        voltage_path=mw.AxisPath("x"),
        current_surface=mw.Box(
            position=(LOAD_PLANE + 0.5 * dx, 0.0, 0.0),
            size=(0.0, 2.0 * loop_half, 2.0 * loop_half),
        ),
        reference_impedance=83.0,
        termination=termination,
    )

    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-DOMAIN_X, DOMAIN_X),
                (-DOMAIN_TRANSVERSE, DOMAIN_TRANSVERSE),
                (-DOMAIN_TRANSVERSE, DOMAIN_TRANSVERSE),
            )
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        ports=(_coax_port("feed", FEED_X, "+", dx=dx), load),
        device=device,
    )

    scene.add_structure(
        _HollowCylinder(
            position=(0.0, 0.0, 0.0),
            inner_radius=OUTER_RADIUS,
            outer_radius=SHIELD_OUTER,
            length=length,
        ).with_material(mw.Material.pec(), name="shield")
    )

    # Inner rod: two segments straddling a two-cell axial gap at the load plane.
    seg1_length = (LOAD_PLANE - dx) - left_edge
    seg1_center = 0.5 * (left_edge + LOAD_PLANE - dx)
    scene.add_structure(
        mw.Cylinder(
            position=(seg1_center, 0.0, 0.0),
            radius=INNER_RADIUS,
            height=seg1_length,
            axis="x",
        ).with_material(mw.Material.pec(), name="inner_feed_side")
    )
    seg2_length = right_edge - (LOAD_PLANE + dx)
    seg2_center = 0.5 * (LOAD_PLANE + dx + right_edge)
    scene.add_structure(
        mw.Cylinder(
            position=(seg2_center, 0.0, 0.0),
            radius=INNER_RADIUS,
            height=seg2_length,
            axis="x",
        ).with_material(mw.Material.pec(), name="inner_load_side")
    )

    return scene
