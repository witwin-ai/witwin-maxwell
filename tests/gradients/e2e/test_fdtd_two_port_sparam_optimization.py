"""End-to-end two-port S-parameter-style inverse-design acceptance test.

This is the flagship acceptance test for the multi-source FDTD adjoint landed in
phase P5.1. Before P5.1 the adjoint bridge rejected any scene with more than one
source (``FDTD adjoint currently supports at most one source per scene``), which
made multi-port / S-parameter inverse design impossible. This test drives a
*genuine two-port* waveguide (a ``ModeSource`` at each end) through the full
public ``Scene -> Simulation -> Result`` adjoint loop and optimizes a
multi-parameter ``MaterialRegion`` barrier so that the coupled field at both
ports drops monotonically under Adam.

The objective is S-parameter-style: two point monitors sit just outside the
design on the two port sides (``port2_thru`` on the transmission side of the
strong port-1 drive, ``port1_refl`` on its reflection side), and the loss is the
summed steady-state intensity at both ports. Reducing it is exactly the
"suppress the coupled port response" objective a multi-port matching/isolation
design solves. The two ports carry unequal drive amplitudes so both sources
genuinely shape the field the objective sees (a single-source fallback would
compute a different gradient and a different descent).

Acceptance (mirrors ``fdtd_gap_05_functional_completeness.md`` P5.1):

* two sources and two monitors,
* a >= 2-parameter design region,
* Adam for >= 20 iterations,
* the objective decreases monotonically (tolerance 1e-9 per step),
* and finishes materially below the start.
"""

import pytest
import torch

import witwin.maxwell as mw

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")


def _abs2(z):
    if z.is_complex():
        return (z * z.conj()).real
    return z * z


class _TwoPortBarrierScene(mw.SceneModule):
    """Inline two-port dielectric waveguide with a trainable barrier in the middle.

    ``port1`` (strong) drives from the left, ``port2`` (weaker) from the right;
    the design fills the full guide cross-section so its density has real control
    authority over how much field couples through to each port monitor.
    """

    def __init__(self, shape=(4, 3, 3), init=0.0):
        super().__init__()
        # (Nx, Ny, Nz) trainable logits -> per-voxel density -> per-voxel eps.
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.90, 0.90), (-0.60, 0.60), (-0.60, 0.60))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="waveguide",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.56, 0.36, 0.36)),
                material=mw.Material(eps_r=8.0),
            )
        )
        # Full-cross-section barrier spanning 4 cells along the guide; the wide
        # eps range lets the optimizer turn it into a matched section or a
        # reflector, giving a strong, smooth handle on the port response.
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.48, 0.36, 0.36)),
                density=density,
                eps_bounds=(1.0, 20.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.ModeSource(
                position=(-0.48, 0.0, 0.0),
                size=(0.0, 0.36, 0.36),
                polarization="Ez",
                direction="+",
                source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
                name="port1",
            )
        )
        scene.add_source(
            mw.ModeSource(
                position=(0.48, 0.0, 0.0),
                size=(0.0, 0.36, 0.36),
                polarization="Ez",
                direction="-",
                source_time=mw.CW(frequency=1.0e9, amplitude=15.0),
                name="port2",
            )
        )
        scene.add_monitor(mw.PointMonitor("port2_thru", (0.30, 0.0, 0.0), fields=("Ez",)))
        scene.add_monitor(mw.PointMonitor("port1_refl", (-0.30, 0.0, 0.0), fields=("Ez",)))
        return scene


def _sparam_objective(model, *, time_steps):
    """Steady-state coupled intensity summed over the two port monitors."""
    result = mw.Simulation.fdtd(
        model,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    thru = result.monitor("port2_thru")["data"]
    refl = result.monitor("port1_refl")["data"]
    return _abs2(thru) + _abs2(refl)


@_CUDA
def test_two_port_sparam_optimization_descends_monotonically():
    """A two-source S-parameter-style objective is driven down by Adam end to end.

    Exercises the full multi-source forward + adjoint + optimizer loop: both
    ``ModeSource`` ports contribute to the field, the adjoint pulls gradients
    back to every design voxel, and Adam reduces the summed port response.
    """
    model = _TwoPortBarrierScene(shape=(4, 3, 3), init=0.0).cuda()

    # Sanity-check the scene really is a two-port, two-monitor, multi-parameter
    # problem before spending the optimization budget on it.
    scene = model.to_scene()
    assert len(scene.sources) == 2, "acceptance requires a genuine two-source scene"
    assert len(scene.monitors) == 2, "acceptance requires two port monitors"
    assert model.logits.numel() >= 2, "design must have >= 2 trainable parameters"

    time_steps = 160
    iterations = 24
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    losses = []
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = _sparam_objective(model, time_steps=time_steps)
        losses.append(loss.item())
        loss.backward()
        assert model.logits.grad is not None, "adjoint did not reach the design logits"
        assert torch.isfinite(model.logits.grad).all(), "non-finite design gradient"
        optimizer.step()

    assert len(losses) == iterations

    # Monotonic descent: no step may increase the objective beyond a 1e-9 floor.
    max_increase = max(
        (losses[i] - losses[i - 1] for i in range(1, len(losses))),
        default=0.0,
    )
    assert max_increase <= 1.0e-9, (
        f"objective increased by {max_increase:.3e} on some step (not monotonic): "
        f"{[f'{value:.4e}' for value in losses]}"
    )

    # Materially below the start (observed ~34% reduction; 10% is a safe bar).
    assert losses[-1] < 0.90 * losses[0], (
        f"objective did not drop materially: {losses[0]:.4e} -> {losses[-1]:.4e}"
    )
