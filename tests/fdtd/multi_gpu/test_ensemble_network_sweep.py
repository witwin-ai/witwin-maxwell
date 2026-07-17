"""Phase 2 RF-aware ensemble: distribute N-port excitation columns over GPUs.

Plan 01's ``NetworkRunManifest`` expands an N-port ``PortSweep`` into one
independent single-active-port Simulation per column. These run through the same
ensemble executor and are reassembled by plan 01's ``aggregate_network_columns``
into an identical ``NetworkData`` matrix. The single-device ensemble leg asserts
exact equality with serial (isolating the executor path); the two-GPU leg asserts
a tight numerical match plus per-column device provenance.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.execution import MultiGPUExecution
from witwin.maxwell.lumped import PortSweep

_FREQUENCIES = (1.0e9, 2.0e9)


def _port(name, x):
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.004),
        negative=(x, 0.0, -0.004),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(x, 0.0, -0.002), size=(0.012, 0.012, 0.0)),
        reference_impedance=50.0,
    )


def _four_port_scene(*, device):
    xs = (-0.008, -0.004, 0.004, 0.008)
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        ports=tuple(_port(f"p{i + 1}", x) for i, x in enumerate(xs)),
        device=device,
    )


def _sweep_sim(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=_FREQUENCIES,
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        excitations=PortSweep(),
    )


def test_single_device_ensemble_sweep_matches_serial_exactly(
    cuda_p2p_devices, cuda_memory_cleanup
):
    device = str(cuda_p2p_devices[0])
    serial = _sweep_sim(_four_port_scene(device=device)).run()

    execution = MultiGPUExecution.ensemble(devices=(device,), max_concurrency=1)
    prepared = _sweep_sim(_four_port_scene(device=device)).prepare()
    ensemble = prepared.run(execution=execution)

    assert ensemble.network.port_names == serial.network.port_names == (
        "p1",
        "p2",
        "p3",
        "p4",
    )
    # Executor path (column expansion + ordered gather + assembly) reproduces the
    # serial matrix bit-for-bit on a single device.
    assert torch.equal(ensemble.network.s, serial.network.s)
    assert torch.equal(ensemble.network.z0, serial.network.z0)


def test_two_gpu_ensemble_sweep_matches_serial_with_provenance(
    cuda_p2p_devices, cuda_memory_cleanup
):
    devices = tuple(str(device) for device in cuda_p2p_devices)
    serial = _sweep_sim(_four_port_scene(device=devices[0])).run()

    execution = MultiGPUExecution.ensemble(devices=devices, max_concurrency=2)
    prepared = _sweep_sim(_four_port_scene(device=devices[0])).prepare()
    ensemble = prepared.run(execution=execution)

    assert ensemble.network.port_names == serial.network.port_names
    # Same ordered ResultSequence assembled into the same matrix across two GPUs.
    torch.testing.assert_close(
        ensemble.network.s, serial.network.s, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        ensemble.network.z0, serial.network.z0, rtol=1e-5, atol=1e-6
    )

    provenance = ensemble.solver_stats["ensemble"]
    column_devices = provenance["column_devices"]
    # Per-column provenance: one leased device per excitation column, all valid.
    assert len(column_devices) == 4
    assert set(column_devices).issubset(set(devices))
    # The assembled matrix and gathered ports live on the single result device.
    assert str(ensemble.network.s.device) == devices[0]
