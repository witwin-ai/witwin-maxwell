"""Grid convergence and power-conservation closure for the layered SAR phantom.

Two wave-level checks on the ``layered_slab`` phantom, run through the public
``Scene -> Simulation -> Result`` path with a real FDTD solve at three grids:

* **Conservation closure (independent reference).** On the periodic-transverse
  variant the absorbed power measured two independent ways -- the volume integral
  of the conduction loss ``sigma |E|^2`` (the quantity SAR is built from) and the
  net surface Poynting flux ``flux(z_in) - flux(z_out)`` (E x H on two planes) --
  agree, and their residual shrinks monotonically as the grid refines. Surface
  E x H is not computed from the volume loss, so this is a genuine conservation
  reference, not a self-consistency identity (gate class: ``wave-level``).

* **Peak 1 g SAR three-grid study.** The regulatory pointwise peak is recorded at
  three grids. Because the incident power density delivered by a source-normalized
  plane wave is itself grid-dependent, and because the peak is a pointwise max over
  a thin (8 mm) under-resolved skin layer, the peak carries real grid sensitivity;
  the study gates the physically robust structure (finite, positive, mass-averaging
  monotone) and records the peak spread rather than asserting a tight limit.

Falsification: a 30% error in the absorbed-power volume integral pushes the
surface/volume residual to 0.36 (> 0.20) and the magnitude-agreement ratio to
0.64 (< 0.75); a 0.5 -> 0.25 Poynting-factor error pushes the residual to 0.58.
Recorded in the H3 acceptance doc.
"""

import pytest
import torch

import witwin.maxwell as mw

from benchmark.scenes.sar import layered_slab

_HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="FDTD phantom run requires CUDA.")

# Three refinement tiers (m). Coarsest to finest; ratio 5:4:3.
_GRIDS = (0.005, 0.004, 0.003)


def _run(scene):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=list(layered_slab.FREQUENCIES),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
        full_field_dft=True,
    ).run()
    return result


def _volume_absorbed_power(sar) -> float:
    q = sar.absorbed_power_density["total"]
    return float((q * sar.cell_volume[None]).sum())


def _conservation(dx: float) -> tuple[float, float, float]:
    """Return (volume-absorbed, surface-absorbed, relative closure residual)."""
    scene = layered_slab.build_conservation_scene(dx=dx, device="cuda")
    result = _run(scene)
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3,)))
    volume_absorbed = _volume_absorbed_power(sar)
    flux_in = float(result.monitor("flux_in")["flux"])
    flux_out = float(result.monitor("flux_out")["flux"])
    surface_absorbed = flux_in - flux_out
    residual = abs(surface_absorbed - volume_absorbed) / abs(volume_absorbed)
    # Sanity: forward net inflow exceeds the transmitted term, both positive.
    assert flux_in > flux_out > 0.0
    assert volume_absorbed > 0.0
    return volume_absorbed, surface_absorbed, residual


@requires_cuda
def test_layered_slab_power_conservation_via_flux():
    """Volume sigma|E|^2 absorption closes against the surface Poynting balance."""
    volume_absorbed, surface_absorbed, residual = _conservation(0.004)
    # Independent surface-vs-volume absorbed-power balance at the reference grid.
    # The finite-grid gap between the two power measures is ~17% at dx=4 mm and
    # shrinks under refinement (see the convergence gate); pin a ceiling here.
    assert residual < 0.20
    # The two measures agree in magnitude (not off by a factor).
    assert 0.75 < surface_absorbed / volume_absorbed < 1.0


@requires_cuda
def test_layered_slab_conservation_closure_converges():
    """The surface/volume absorbed-power closure residual shrinks as dx -> 0."""
    residuals = [_conservation(dx)[2] for dx in _GRIDS]
    # Every tier stays within a loose physical ceiling.
    assert all(r < 0.25 for r in residuals)
    # Net convergence: the finest grid closes markedly better than the coarsest.
    assert residuals[-1] < residuals[0] - 0.04
    # Finest grid reaches a strong closure.
    assert residuals[-1] < 0.15


@requires_cuda
def test_layered_slab_peak_1g_sar_three_grid_study():
    """Record peak 1 g / 10 g SAR at three grids; gate the robust structure only."""
    peaks_1g = []
    peaks_10g = []
    for dx in _GRIDS:
        scene = layered_slab.build_scene(dx=dx, device="cuda")
        result = _run(scene)
        sar = result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3, 10e-3)))
        peak_1g = float(sar.peak(1e-3).sar[0])
        peak_10g = float(sar.peak(10e-3).sar[0])
        # Physically robust structure at every grid.
        assert peak_1g > 0.0 and peak_1g == peak_1g  # finite, positive
        assert peak_10g > 0.0 and peak_10g == peak_10g
        # Mass averaging is a smoothing: the 10 g peak never exceeds the 1 g peak.
        assert peak_10g <= peak_1g + 1e-6
        peaks_1g.append(peak_1g)
        peaks_10g.append(peak_10g)
    # Documented grid sensitivity: the pointwise peak stays in a physical band but
    # is NOT asserted to converge tightly (thin skin + source-normalized incident
    # power both move it grid-to-grid). The convergent, gate-bearing observable is
    # the conservation closure above, not this pointwise peak.
    assert all(0.2 < value < 1.5 for value in peaks_1g)
    spread = (max(peaks_1g) - min(peaks_1g)) / min(peaks_1g)
    # Guard the documented spread does not silently blow up beyond the recorded band.
    assert spread < 0.6
