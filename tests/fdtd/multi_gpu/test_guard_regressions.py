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


def test_public_prepare_rejects_gyromagnetic_ferrite(
    cuda_p2p_devices,
    cuda_memory_cleanup,
):
    """Multi-GPU distributed FDTD must fail closed on a gyromagnetic ferrite.

    The shard phases never run the magnetization-ADE hooks and the shard-local
    layout has no rank-seam handling, so a joint solve would silently simulate a
    reciprocal medium (frozen contract boundary 8, rejected until Phase 4).
    """
    scene = _scene(device="cuda:0")
    scene.add_structure(
        mw.Structure(
            name="ferrite",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=mw.GyromagneticFerrite(
                eps_r=14.5,
                saturation_magnetization=1.40e5,
                bias_field=(0.0, 0.0, 1.75e5),
                gilbert_damping=1.0e-2,
            ),
        )
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        parallel=_parallel(devices=cuda_p2p_devices),
    )

    with pytest.raises(NotImplementedError, match="GyromagneticFerrite"):
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


def _trainable_circuit_scene() -> mw.Scene:
    """Two-shard circuit scene whose load resistance is a trainable tensor.

    The trainable resistance is a circuit-parameter channel the public
    ``_scene_trainable_circuit_parameters`` collector covers but that the
    solver-level guard's earlier hand-rolled list did not enumerate.
    """

    circuit = mw.Circuit("trainable_two_shard")
    left = circuit.node("left")
    right = circuit.node("right")
    circuit.add(
        mw.CurrentSource(
            "Iexcite",
            left,
            circuit.ground,
            0.0,
            waveform=mw.SineWaveform(0.0, 1.0e-3, _FREQUENCY),
        )
    )
    circuit.add(mw.Resistor("Rlink", left, right, 75.0))
    circuit.add(
        mw.Resistor(
            "Rload", right, circuit.ground, torch.tensor(50.0, requires_grad=True)
        )
    )
    circuit.bind_port("left_port", positive=left, negative=circuit.ground)
    circuit.bind_port("right_port", positive=right, negative=circuit.ground)

    def _port(name, x):
        return mw.LumpedPort(
            name=name,
            positive=(x, 0.0, 0.004),
            negative=(x, 0.0, -0.004),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(position=(x, 0.0, -0.002), size=(0.012, 0.012, 0.0)),
            reference_impedance=50.0,
        )

    return mw.Scene(
        domain=mw.Domain(bounds=((-0.024, 0.024),) * 3),
        grid=mw.GridSpec.uniform(0.004),
        boundary=mw.BoundarySpec.none(),
        ports=(_port("left_port", -0.008), _port("right_port", 0.008)),
        circuits=(circuit,),
        device="cpu",
    )


def _absorber_guard_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.4, 0.4), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda:0",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            profile="ideal",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=5.0e8),
            name="drive",
        )
    )
    return scene


@pytest.mark.parametrize("absorber", ("cpml", "stablepml"))
def test_require_distributed_adjoint_support_accepts_cpml_family(
    absorber, cuda_p2p_devices, cuda_memory_cleanup
):
    """The reverse-support guard now accepts the CPML absorbing update (S4).

    ``uses_cpml`` (absorber "cpml"/"stablepml") is a verified distributed adjoint
    capability: the guard accepts it and asserts the x-CPML pinning invariant. The
    legacy graded-sigma absorbers stay rejected (companion test below).
    """

    from witwin.maxwell.fdtd.distributed.adjoint import (
        require_distributed_adjoint_support,
    )

    distributed = DistributedFDTD(
        _absorber_guard_scene(),
        frequency=_FREQUENCY,
        parallel=_parallel(devices=cuda_p2p_devices),
        absorber_type=absorber,
    )
    distributed.init_field()

    assert distributed.active_absorber_type == absorber
    assert all(shard.solver.uses_cpml for shard in distributed.shards)
    require_distributed_adjoint_support(distributed)  # must not raise


@pytest.mark.parametrize("absorber", ("pml", "absorber"))
def test_require_distributed_adjoint_support_rejects_graded_sigma_absorber(
    absorber, cuda_p2p_devices, cuda_memory_cleanup
):
    """Defense in depth: the reverse-support guard fails closed on graded sigma.

    The legacy graded-sigma absorbers ("pml"/"absorber") have no verified
    distributed reverse core (they do not set ``uses_cpml``), so the guard must
    reject them independently of the public ``Simulation`` prepare guard.
    """

    from witwin.maxwell.fdtd.distributed.adjoint import (
        require_distributed_adjoint_support,
    )

    distributed = DistributedFDTD(
        _absorber_guard_scene(),
        frequency=_FREQUENCY,
        parallel=_parallel(devices=cuda_p2p_devices),
        absorber_type=absorber,
    )
    distributed.init_field()

    assert distributed.active_absorber_type == absorber
    with pytest.raises(ValueError, match="CPML absorbing update"):
        require_distributed_adjoint_support(distributed)


def test_solver_trainable_guard_covers_circuit_parameter_channel():
    """Defense in depth: the distributed solver's own trainable guard rejects a
    trainable circuit parameter even though the public ``Simulation`` guard would
    have caught it first.

    Regression for the solver-level guard previously enumerating a partial
    trainable list (density/geometry/perturbation only) that missed the circuit
    and RF/port channels the public collectors cover. Constructing
    ``DistributedFDTD`` directly bypasses the public boundary; the assertion that
    the public ``Simulation`` sees the same parameter pins that this is genuine
    defense in depth rather than the only guard.
    """

    scene = _trainable_circuit_scene()

    # The public boundary detects the trainable circuit parameter (and would
    # reject the parallel run in _reject_trainable_parallel_fdtd).
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        parallel=_parallel(),
    )
    assert simulation.has_trainable_parameters

    # Directly constructing the distributed solver bypasses that public guard; the
    # solver-level guard must still fail closed on the circuit-parameter channel.
    with pytest.raises(ValueError, match="trainable"):
        DistributedFDTD(
            scene,
            frequency=_FREQUENCY,
            parallel=_parallel(),
        )


def _wire(scene):
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)),
            radius=1.0e-3,
            conductor=mw.WireConductor.pec(),
        )
    )
    return scene


def test_supported_thin_wire_forward_passes_static_distributed_validation():
    # The distributed PEC thin-wire forward is supported (Phase 4). A plain wire on
    # a non-absorbing boundary must no longer be rejected at static validation; the
    # constructor runs the full static capability check without touching hardware.
    solver = DistributedFDTD(
        _wire(_scene()),
        frequency=_FREQUENCY,
        parallel=_parallel(),
    )
    assert getattr(solver.logical_scene, "thin_wires", ())


def test_thin_wire_with_distributed_cpml_is_rejected_before_hardware_prepare():
    scene = _wire(_scene())
    scene = scene.clone(boundary=mw.BoundarySpec.pml(num_layers=4))
    with pytest.raises(NotImplementedError, match="distributed CPML"):
        DistributedFDTD(scene, frequency=_FREQUENCY, parallel=_parallel())


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
    solver._nccl = False
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
