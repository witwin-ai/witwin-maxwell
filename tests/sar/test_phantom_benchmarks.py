"""Golden / analytic gates for the canonical phantom SAR benchmark family.

Scenes live under ``benchmark/scenes/sar/``. The synthetic ``one_gram_cube`` gate
is hand-computable (no solver); the plane-wave phantom gates run a compact
deterministic FDTD and check exact power-conservation closure, physical structure
(monotone mass-averaging, peak tissue), and a golden regression anchor. The
``antenna_near_phantom`` gate pins the recorded conductive-media port blocker as
fail-closed behaviour.
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.scene import prepare_scene

from benchmark.scenes.sar import (
    antenna_near_phantom,
    layered_slab,
    one_gram_cube,
    uniform_lossy_cube,
)

_HAS_CUDA = torch.cuda.is_available()
_DEVICE = "cuda" if _HAS_CUDA else "cpu"
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="FDTD phantom run requires CUDA.")


def _nanpeak(tensor):
    flat = tensor.reshape(tensor.shape[0], -1)
    return float(
        torch.where(torch.isnan(flat), torch.full_like(flat, -float("inf")), flat).max()
    )


def _closure_residual(sar):
    q = sar.absorbed_power_density["total"]
    region = (q * sar.cell_volume[None]).sum(dim=(1, 2, 3))
    reference = sar.provenance["electric_channel_power"].to(region.device)
    return float((region - reference).abs().max() / (reference.abs().max() + 1e-30))


def _run_planewave_sar(scene):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=list(scene.monitors[0].frequencies or ()),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
        full_field_dft=True,
    ).run()
    return result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3, 10e-3)))


# --------------------------------------------------------------------------- #
# one_gram_cube: synthetic, hand-computable 1 g average (no solver).           #
# --------------------------------------------------------------------------- #

def test_one_gram_cube_exact_hand_computed_average():
    e0 = 2.0
    scene = one_gram_cube.build_scene(device=_DEVICE)
    monitor = next(m for m in scene.monitors if m.name == "loss")
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    fields = {
        component.upper(): torch.full(
            compiled.full_component_shapes[component],
            complex(e0),
            dtype=torch.complex64,
            device=_DEVICE,
        )
        for component in ("Ex", "Ey", "Ez")
    }
    result = mw.Result(
        method="fdtd",
        scene=scene,
        prepared_scene=prepared,
        frequencies=one_gram_cube.FREQUENCIES,
        fields=fields,
    )
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3,)))
    peak = sar.peak(mass=1e-3)

    # 27 cells at rho=1000 on the one-gram grid weigh exactly 1 g.
    assert one_gram_cube.one_gram_cell_mass() == pytest.approx(1e-3 / 27.0, rel=1e-12)
    # The 1 g window is a symmetric 3x3x3 cube (half-width 1) enclosing exactly 1 g.
    assert int(peak.cube_half_width[0]) == 1
    assert float(peak.mass_kg[0]) == pytest.approx(1e-3, rel=1e-5)
    # Under a uniform field the 1 g average equals the point SAR = 0.5*sigma*3*e0^2/rho.
    analytic = 0.5 * one_gram_cube.SIGMA_E * (3.0 * e0**2) / one_gram_cube.MASS_DENSITY
    assert float(peak.sar[0]) == pytest.approx(analytic, rel=1e-5)


# --------------------------------------------------------------------------- #
# uniform_lossy_cube: exact power closure + monotone averaging + golden.        #
# --------------------------------------------------------------------------- #

@requires_cuda
def test_uniform_lossy_cube_power_closure_and_golden():
    scene = uniform_lossy_cube.build_scene(device="cuda")
    sar = _run_planewave_sar(scene)

    # Absorbed-power volume integral closes exactly against the channel total.
    assert _closure_residual(sar) < 1e-5

    peak_point = _nanpeak(sar.point_sar("total"))
    peak_1g = _nanpeak(sar.averaged_sar(1e-3))
    peak_10g = _nanpeak(sar.averaged_sar(10e-3))
    assert peak_point > 0.0
    # Mass averaging is a smoothing: 10 g peak <= 1 g peak <= point peak.
    assert peak_10g <= peak_1g <= peak_point + 1e-6
    # Golden regression anchor (deterministic run; loose tol for CUDA reductions).
    assert peak_point == pytest.approx(0.4386, rel=3e-2)
    assert peak_1g == pytest.approx(0.3763, rel=3e-2)
    assert peak_10g == pytest.approx(0.2841, rel=3e-2)


# --------------------------------------------------------------------------- #
# layered_slab: three tissues, peak in a lossy layer (not fat), golden 1 g.     #
# --------------------------------------------------------------------------- #

@requires_cuda
def test_layered_slab_tissue_peak_and_golden():
    scene = layered_slab.build_scene(device="cuda")
    sar = _run_planewave_sar(scene)

    assert _closure_residual(sar) < 1e-5

    # Three tissues are resolved with the published densities.
    stats = sar.statistics
    assert len(stats) == 3

    peak_1g = sar.peak(1e-3)
    peak_10g = sar.peak(10e-3)
    assert float(peak_1g.sar[0]) >= float(peak_10g.sar[0])
    # The 1 g peak sits in the front (skin) layer, not in low-loss fat.
    z_peak = float(peak_1g.position[0][2])
    skin_lo, skin_hi = layered_slab.layer_bounds()["skin_dry"]
    assert skin_lo <= z_peak <= skin_hi

    # Fat (lowest conductivity) has the lowest per-tissue peak SAR of the three.
    max_sars = sorted(float(s["max_sar"]) for s in stats.values())
    fat_max = min(float(s["max_sar"]) for s in stats.values())
    fat_sigma = 0.051
    # Identify fat as the lowest-max tissue and confirm it is well below the top.
    assert fat_max == max_sars[0]
    assert fat_max < max_sars[-1]

    # Golden regression anchors.
    assert float(peak_1g.sar[0]) == pytest.approx(0.5697, rel=3e-2)
    assert float(peak_10g.sar[0]) == pytest.approx(0.2959, rel=3e-2)


# --------------------------------------------------------------------------- #
# antenna_near_phantom: recorded conductive-media port blocker (fail-closed).   #
# --------------------------------------------------------------------------- #

@requires_cuda
def test_antenna_near_phantom_fails_closed_on_conductive_media():
    scene = antenna_near_phantom.build_scene(device="cuda")
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=list(antenna_near_phantom.FREQUENCIES),
        excitations=mw.PortExcitation(
            "feed",
            source_time=mw.GaussianPulse(
                frequency=antenna_near_phantom.DESIGN_FREQUENCY, fwidth=1.5e9
            ),
        ),
        full_field_dft=True,
    )
    with pytest.raises(NotImplementedError, match="conductive media|conductive background"):
        simulation.prepare()
