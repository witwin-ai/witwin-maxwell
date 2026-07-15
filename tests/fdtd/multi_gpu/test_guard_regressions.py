from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed import DistributedFDTD
from witwin.maxwell.fdtd_parallel import FDTDParallelConfig


_FREQUENCY = 1.0e9


def _parallel(*, devices=("cuda:0", "cuda:1"), result_device="cuda:0"):
    return FDTDParallelConfig(
        devices=devices,
        transport="cuda_p2p",
        gather_fields=False,
        result_device=result_device,
    )


def _scene(*, device="cpu") -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )


@pytest.mark.parametrize(
    "material, message",
    (
        (
            mw.Material(
                eps_r=2.0,
                nonlinearity=mw.NonlinearSusceptibility(chi3=1.0e-10),
            ),
            "nonlinear media",
        ),
        (
            mw.Material(
                epsilon_tensor=mw.Tensor3x3(
                    ((2.0, 0.2, 0.1), (0.2, 2.5, 0.3), (0.1, 0.3, 3.0))
                )
            ),
            "off-diagonal anisotropy",
        ),
    ),
    ids=("nonlinear", "full_offdiagonal_anisotropy"),
)
def test_public_prepare_rejects_materials_without_complete_interface_halos(
    material,
    message,
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    scene = _scene(device="cuda:0")
    scene.add_structure(
        mw.Structure(
            name="cross_interface_material",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=material,
        )
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        parallel=_parallel(devices=cuda_p2p_devices),
    )

    with pytest.raises(ValueError, match=message):
        simulation.prepare()


def test_invalid_dipole_emission_source_is_rejected_before_hardware_prepare():
    scene = _scene()
    scene.add_monitor(
        mw.DipoleEmissionMonitor(
            "emission",
            source_name="missing_dipole",
            frequencies=(_FREQUENCY,),
        )
    )

    with pytest.raises(ValueError, match="missing_dipole"):
        DistributedFDTD(
            scene,
            frequency=_FREQUENCY,
            parallel=_parallel(),
        )


def test_thin_wire_is_explicitly_rejected_before_distributed_hardware_prepare():
    scene = _scene()
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)),
            radius=1.0e-3,
            conductor=mw.WireConductor.pec(),
        )
    )

    with pytest.raises(NotImplementedError, match="fragment/state ownership"):
        DistributedFDTD(
            scene,
            frequency=_FREQUENCY,
            parallel=_parallel(),
        )


@pytest.mark.parametrize(
    "monitor",
    (
        mw.PermittivityMonitor(
            "permittivity",
            position=(0.0, 0.0, 0.0),
            size=(0.2, 0.2, 0.2),
        ),
        mw.MediumMonitor(
            "medium",
            position=(0.0, 0.0, 0.0),
            size=(0.2, 0.2, 0.2),
        ),
    ),
    ids=("permittivity", "medium"),
)
def test_material_monitors_are_explicitly_rejected_before_hardware_prepare(monitor):
    scene = _scene()
    scene.add_monitor(monitor)

    with pytest.raises(ValueError, match="material monitors"):
        DistributedFDTD(
            scene,
            frequency=_FREQUENCY,
            parallel=_parallel(),
        )


def _count_shard(frequencies=(), counts=()):
    return SimpleNamespace(
        solver=SimpleNamespace(
            observer_frequencies=tuple(frequencies),
            observer_sample_counts=tuple(counts),
            dft_sample_counts=tuple(counts),
        )
    )


def test_observer_counts_merge_by_frequency_across_different_shard_subsets():
    solver = object.__new__(DistributedFDTD)
    solver.shards = (
        _count_shard((_FREQUENCY,), (7,)),
        _count_shard((1.5e9,), (11,)),
    )
    solver._observer_frequencies = (_FREQUENCY, 1.5e9)

    assert solver.observer_frequencies == (_FREQUENCY, 1.5e9)
    assert solver.observer_sample_counts == (7, 11)


def test_empty_distributed_sample_counts_use_empty_tuple_contract():
    solver = object.__new__(DistributedFDTD)
    solver.shards = ()
    solver._observer_frequencies = ()

    assert solver.dft_sample_counts == ()
    assert solver.observer_sample_counts == ()


def test_active_absorber_type_aggregates_x_high_only_pml_from_last_shard():
    solver = object.__new__(DistributedFDTD)
    solver.shards = (
        SimpleNamespace(solver=SimpleNamespace(active_absorber_type="none")),
        SimpleNamespace(solver=SimpleNamespace(active_absorber_type="cpml")),
    )

    assert solver.active_absorber_type == "cpml"


def _hardware_probe(monkeypatch, *, missing_pair=None):
    solver = object.__new__(DistributedFDTD)
    solver.devices = tuple(torch.device(f"cuda:{index}") for index in range(3))
    solver.device = torch.device("cuda:0")
    solver.transport = SimpleNamespace(preflight=Mock())
    calls = []

    def can_access(source, destination):
        pair = (int(source), int(destination))
        calls.append(pair)
        return pair != missing_pair

    properties = SimpleNamespace(name="mock-gpu", major=8, minor=6)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    return solver, calls


@pytest.mark.parametrize("missing_pair", ((0, 2), (2, 0)))
def test_result_device_preflight_rejects_missing_bidirectional_peer_link(
    monkeypatch,
    missing_pair,
):
    solver, _calls = _hardware_probe(monkeypatch, missing_pair=missing_pair)

    with pytest.raises(RuntimeError, match="result gathering/reduction"):
        solver._validate_hardware()


def test_result_device_preflight_checks_every_shard_in_both_directions(monkeypatch):
    solver, calls = _hardware_probe(monkeypatch)

    solver._validate_hardware()

    assert {(0, 1), (1, 0), (0, 2), (2, 0)} <= set(calls)
    solver.transport.preflight.assert_called_once_with()
