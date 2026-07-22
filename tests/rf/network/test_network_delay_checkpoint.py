"""E4b: checkpoint/resume of an embedded network with explicit port delay.

Before this slice the frozen checkpoint layout captured only the network
state-space vector and carried voltage, never the bidirectional-delay
reference-plane rings, the Thiran fractional-filter memory, or the shared ring
cursor. A delayed network resumed mid-run would restart its reference planes
from zero and silently desynchronize from the uninterrupted run. This gate:

* round-trip: ``run_until(k)`` + ``run(resume_from=...)`` reproduces the
  uninterrupted final fields and embedded-network diagnostics bit-for-bit, and
  the delay ring/filter/cursor state is actually captured into the checkpoint;
* drop-one falsification: zeroing the captured delay ring state before resume
  genuinely breaks the resumed field (proving the state is load-bearing).
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.delay import delay_phase_matrix


def _delayed_two_port(
    frequencies: torch.Tensor,
    delays: tuple[float, float],
) -> tuple[mw.NetworkData, mw.StateSpaceNetwork]:
    core = torch.tensor(
        ((0.1, 0.25), (0.25, 0.1)),
        dtype=torch.float64,
        device=frequencies.device,
    )
    scattering = core.to(torch.complex128)[None, ...] * delay_phase_matrix(
        frequencies, delays
    )
    data = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("left", "right"),
    )
    model = mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64),
        B=torch.zeros((0, 2), dtype=torch.float64),
        C=torch.zeros((2, 0), dtype=torch.float64),
        D=core,
        representation="S",
        port_order=data.port_names,
    )
    return data, model


def _port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(x, 0.0, -0.0025), size=(0.005, 0.005, 0.0)),
        reference_impedance=50.0,
    )


def _block() -> mw.NetworkBlock:
    frequencies = torch.linspace(0.0, 4.0e9, 81, dtype=torch.float64)
    delays = (0.2e-9, 0.3e-9)
    data, model = _delayed_two_port(frequencies, delays)
    return mw.NetworkBlock(
        name="delayed_link",
        network=data,
        connections={"left": "p_left", "right": "p_right"},
        fit=False,
        model=model,
        delay_seconds=delays,
        max_delay_steps=128,
    )


def _scene() -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.025, 0.025), (-0.02, 0.02), (-0.02, 0.02))),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.CW(frequency=2.0e9),
            ),
        ),
        ports=(_port("p_left", -0.01), _port("p_right", 0.01)),
        networks=(_block(),),
        device="cuda",
    )


def _simulation() -> mw.Simulation:
    return mw.Simulation.fdtd(
        _scene(),
        frequencies=(2.0e9,),
        run_time=mw.TimeConfig(time_steps=48),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=False,
    )


def _delay_state_keys(checkpoint) -> list[str]:
    return [name for name in checkpoint.physics.tensors if "_delay_" in name]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_delayed_network_resume_matches_uninterrupted_run() -> None:
    reference = _simulation().run()
    checkpoint = _simulation().prepare().run_until(24)

    keys = _delay_state_keys(checkpoint)
    assert keys, "bidirectional-delay state was not captured into the checkpoint"
    ring_keys = [name for name in keys if name.endswith("_ring")]
    assert ring_keys
    assert any(
        float(checkpoint.physics.tensors[name].abs().max()) > 0.0 for name in ring_keys
    ), "captured delay ring state is all-zero at the checkpoint step"

    resumed = _simulation().prepare().run(resume_from=checkpoint)

    assert torch.equal(resumed.field("Ez"), reference.field("Ez"))
    assert torch.equal(
        resumed.embedded_network("delayed_link").voltage,
        reference.embedded_network("delayed_link").voltage,
    )
    assert torch.equal(
        resumed.embedded_network("delayed_link").current,
        reference.embedded_network("delayed_link").current,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zeroing_captured_delay_state_breaks_resume() -> None:
    reference = _simulation().run()
    checkpoint = _simulation().prepare().run_until(24)
    keys = _delay_state_keys(checkpoint)
    assert keys
    for name in keys:
        if name.endswith("_cursor"):
            continue
        checkpoint.physics.tensors[name].zero_()

    corrupted = _simulation().prepare().run(resume_from=checkpoint)

    assert not torch.equal(corrupted.field("Ez"), reference.field("Ez")), (
        "zeroing the captured delay ring/filter state left the resumed field "
        "unchanged, so the state is not actually load-bearing in resume"
    )
