from __future__ import annotations

from dataclasses import replace
import io
import math

import pytest
import torch
import torch.nn.functional as F

import witwin.maxwell as mw
from witwin.maxwell.fdtd.wire import (
    _group_indices,
    _target_masses,
    _validate_compiled_topology,
    _wire_cfl_limit,
    deposit_wire_current,
    sample_and_update_wire,
)
from witwin.maxwell.fdtd.thin_wire_reference import (
    ACCEPTANCE_BUDGET,
    AxisAlignedWireReference,
    WireReferenceState,
)
from witwin.maxwell.fdtd.checkpoint import (
    capture_checkpoint_state,
    validate_checkpoint_state,
)
from witwin.maxwell.fdtd.adjoint.core import _replay_segment_states
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")

_FREQUENCY = 2.0e9


def _scene(
    *,
    points=None,
    radius=2.0e-3,
    source=True,
    monitor=True,
    quantities=None,
    endpoints=None,
    structures=(),
    boundary=None,
):
    if points is None:
        points = ((0.0, 0.0, -0.08), (0.0, 0.0, 0.08))
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none() if boundary is None else boundary,
        structures=structures,
        device="cuda",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=points,
            radius=radius,
            conductor=mw.WireConductor.pec(),
            endpoints=endpoints,
            snap="strict",
        )
    )
    if source:
        scene.add_source(
            mw.PointDipole(
                name="drive",
                position=(0.0, 0.0, 0.02),
                polarization="Ez",
                width=0.04,
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
            )
        )
    if monitor:
        monitor_kwargs = {}
        if quantities is not None:
            monitor_kwargs["quantities"] = quantities
        scene.add_monitor(
            mw.WireMonitor(
                name="wire_state",
                wire="wire",
                frequencies=(_FREQUENCY,),
                **monitor_kwargs,
            )
        )
    return scene


def _prepared_solver(scene, *, frequency=_FREQUENCY):
    return mw.Simulation.fdtd(
        scene,
        frequencies=(frequency,),
        run_time=mw.TimeConfig(time_steps=8),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver


def _target_values(solver, components, offsets):
    values = torch.empty(offsets.numel(), device=solver.device, dtype=solver.Ex.dtype)
    for component, field in enumerate((solver.Ex, solver.Ey, solver.Ez)):
        indices = torch.nonzero(components == component, as_tuple=False).reshape(-1)
        if indices.numel():
            values.index_copy_(
                0,
                indices,
                field.reshape(-1).index_select(0, offsets.index_select(0, indices)),
            )
    return values


def _set_target_values(solver, components, offsets, values):
    for component, field in enumerate((solver.Ex, solver.Ey, solver.Ez)):
        indices = torch.nonzero(components == component, as_tuple=False).reshape(-1)
        if indices.numel():
            field.reshape(-1).index_copy_(
                0,
                offsets.index_select(0, indices),
                values.index_select(0, indices),
            )


def _promote_wire_runtime_to_float64(solver):
    """Re-derive the compiled wire runtime in float64.

    The reference oracle is a float64 dense model. Promoting the native runtime
    keeps the comparison limited by the discrete contract rather than by float32
    rounding, so a real deviation cannot hide under the reference budget.
    """

    for name in ("Ex", "Ey", "Ez", "eps_Ex", "eps_Ey", "eps_Ez"):
        setattr(solver, name, getattr(solver, name).to(torch.float64))
    runtime = solver._wire_runtime
    runtime.current = runtime.current.to(torch.float64)
    runtime.charge = runtime.charge.to(torch.float64)
    runtime.emf = runtime.emf.to(torch.float64)
    for name, value in tuple(runtime.coefficients.items()):
        if value.is_floating_point():
            runtime.coefficients[name] = value.to(torch.float64)
    coeff = runtime.coefficients
    masses = _target_masses(solver, coeff["target_components"], coeff["target_offsets"])
    edge_groups = _group_indices(coeff["edge_group_offsets"])
    coeff["contribution_scales"] = (
        float(solver.dt)
        * coeff["contribution_weights"]
        / masses.index_select(0, edge_groups)
    ).contiguous()
    # Recompute the reverse-path scales from float64 masses too, so the promoted
    # fixture is internally consistent in precision rather than carrying
    # float32-rounded values inside float64 containers.
    sample_masses = _target_masses(solver, coeff["edge_components"], coeff["edge_offsets"])
    coeff["sample_masses"] = sample_masses
    coeff["sample_deposition_scales"] = (
        float(solver.dt) * coeff["weights"] / sample_masses
    ).contiguous()
    return solver, masses


def _dense_wire_operators(solver):
    """Rebuild dense ``G`` and ``B`` from the compiled sparse network.

    The transpose ``assert_close`` below only checks that the compiler's own
    ``searchsorted``/``edge_group_offsets`` bookkeeping is self-consistent: the
    deposition list is emitted as a permutation of the sampling weights
    (``compiler/thin_wire.py``), so ``deposition == sampling.mT`` holds for any
    weights and is NOT an independent reciprocity check. The real end-to-end
    ``G``/``G^T`` adjointness of the CUDA kernels is verified separately in
    ``test_wire_sample_and_deposit_kernels_are_energy_adjoint_on_shared_edges``.
    """

    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    device = solver.device
    dtype = solver.Ex.dtype
    segment_count = int(coeff["inductance"].numel())
    node_count = int(coeff["node_capacitance"].numel())
    target_count = int(coeff["target_offsets"].numel())

    stride = max(field.numel() for field in (solver.Ex, solver.Ey, solver.Ez))
    target_keys = coeff["target_components"].to(torch.int64) * stride + coeff["target_offsets"]
    sample_keys = coeff["edge_components"].to(torch.int64) * stride + coeff["edge_offsets"]
    sorted_targets, order = torch.sort(target_keys)
    positions = torch.searchsorted(sorted_targets, sample_keys)
    assert torch.equal(sorted_targets.index_select(0, positions), sample_keys), (
        "every sampled Yee edge must also be a deposition target"
    )
    sample_dofs = order.index_select(0, positions)

    sampling = torch.zeros((segment_count, target_count), device=device, dtype=dtype)
    sampling.index_put_(
        (_group_indices(coeff["segment_offsets"]), sample_dofs),
        coeff["weights"],
        accumulate=True,
    )
    deposition = torch.zeros((target_count, segment_count), device=device, dtype=dtype)
    deposition.index_put_(
        (_group_indices(coeff["edge_group_offsets"]), coeff["contribution_segments"]),
        coeff["contribution_weights"],
        accumulate=True,
    )
    torch.testing.assert_close(deposition, sampling.mT, rtol=0.0, atol=0.0)

    segments = torch.arange(segment_count, device=device, dtype=torch.int64)
    incidence = torch.zeros((node_count, segment_count), device=device, dtype=dtype)
    incidence.index_put_(
        (coeff["tail"], segments), torch.ones_like(coeff["inductance"]), accumulate=True
    )
    incidence.index_put_(
        (coeff["head"], segments), -torch.ones_like(coeff["inductance"]), accumulate=True
    )
    return sampling, incidence


def _wire_reference_for(solver, masses):
    coeff = solver._wire_runtime.coefficients
    sampling, incidence = _dense_wire_operators(solver)
    return AxisAlignedWireReference(
        incidence=incidence,
        sampling=sampling,
        segment_inductance=coeff["inductance"],
        node_capacitance=coeff["node_capacitance"],
        field_mass=masses,
        dt=float(solver.dt),
    )


@pytest.mark.parametrize(
    ("points", "expected_components"),
    (
        (((0.0, 0.0, -0.08), (0.0, 0.0, 0.08)), {2}),
        (((-0.08, 0.0, 0.0), (0.08, 0.0, 0.0)), {0}),
        (((-0.08, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.08, 0.0)), {0, 1}),
    ),
    ids=("straight_ez", "straight_ex", "l_bend_ex_ey"),
)
def test_native_wire_step_matches_the_torch_reference_oracle(points, expected_components):
    """Phase 0/1 gate: the CUDA recurrence must reproduce the reference contract.

    Parametrized over the plan's named Phase 1 topologies (straight and L-bend)
    and both transverse orientations so the ``Ex``/``Ey`` branches of the
    component plan are actually exercised, not only the degenerate straight-``Ez``
    single-segment case.
    """

    solver, masses = _promote_wire_runtime_to_float64(
        _prepared_solver(_scene(points=points, source=False, monitor=False))
    )
    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    reference = _wire_reference_for(solver, masses)
    assert not bool(coeff["grounded"].any()), "oracle parity assumes open endpoints"
    assert (
        set(coeff["target_components"].unique().tolist()) == expected_components
    ), "fixture must exercise the intended field components"

    electric = torch.linspace(
        -2.0, 3.0, masses.numel(), device=solver.device, dtype=torch.float64
    )
    _set_target_values(
        solver, coeff["target_components"], coeff["target_offsets"], electric
    )
    state = WireReferenceState(
        electric.clone(),
        torch.zeros_like(runtime.charge),
        torch.zeros_like(runtime.current),
    )

    errors = []
    for _ in range(64):
        sample_and_update_wire(solver)
        deposit_wire_current(solver)
        state = reference.step(state)
        native_electric = _target_values(
            solver, coeff["target_components"], coeff["target_offsets"]
        )
        for native, expected in (
            (runtime.current, state.current_half),
            (runtime.charge, state.charge),
            (native_electric, state.electric),
        ):
            scale = expected.abs().max()
            errors.append(
                float((native - expected).abs().max() / torch.clamp(scale, min=1.0e-300))
            )

    assert max(errors) < ACCEPTANCE_BUDGET.reference_rtol
    # The comparison is only meaningful if the recurrence actually moved.
    assert float(state.charge.abs().max()) > 0.0
    assert float(state.current_half.abs().max()) > 0.0


def _gershgorin_wire_cfl_limit(solver, masses):
    """Recompute the production Gershgorin bound from the promoted coefficients.

    The oracle eigenvalue limit and this bound are then derived from the exact
    same float64 inductance/capacitance/weights/masses, so the comparison is a
    pure Gershgorin-vs-eigenvalue statement rather than a float32 rounding
    artefact of the prepare-time value.
    """

    coeff = solver._wire_runtime.coefficients
    return _wire_cfl_limit(
        inductance=coeff["inductance"],
        node_capacitance=coeff["node_capacitance"],
        grounded=coeff["grounded"],
        node_offsets=coeff["node_offsets"],
        node_segments=coeff["node_segments"],
        edge_group_offsets=coeff["edge_group_offsets"],
        contribution_segments=coeff["contribution_segments"],
        contribution_weights=coeff["contribution_weights"],
        target_masses=masses,
    )


def test_native_wire_cfl_bound_is_tight_on_uniform_topology_and_never_exceeds_exact_limit():
    """Gershgorin equals the exact ``2/omega_max`` limit on a uniform wire.

    Gershgorin's disc theorem upper-bounds the largest eigenvalue by the largest
    absolute row sum, so it lower-bounds ``dt``: the production bound can never
    exceed the exact leapfrog limit. On a uniform-coefficient axis-aligned wire
    that inequality is an *equality* (the bound is TIGHT, not conservative),
    because the constant-sign coupling makes the maximal row sum coincide with
    the top eigenvalue. Both quantities are recomputed here from the identical
    float64 coefficients, so this is a mathematical statement, not the float32
    prepare-time rounding accident it replaces.
    """

    solver, masses = _promote_wire_runtime_to_float64(
        _prepared_solver(_scene(source=False, monitor=False))
    )
    runtime = solver._wire_runtime
    exact_limit = float(_wire_reference_for(solver, masses).maximum_stable_dt())
    gershgorin_limit = _gershgorin_wire_cfl_limit(solver, masses)

    assert math.isfinite(exact_limit)
    # Safety property: Gershgorin lower-bounds dt (never exceeds the exact limit).
    assert gershgorin_limit <= exact_limit * (1.0 + 1.0e-12)
    # Tightness: on a uniform wire the bound coincides with the exact eigenvalue
    # limit. Measured ratio is 1.0 in float64 (auditor and re-derived on-host).
    assert gershgorin_limit >= exact_limit * (1.0 - 1.0e-9)
    # The float64 prepare-time value the solver actually uses to gate dt must
    # also respect the exact limit (it is computed from float32-valued masses, so
    # allow a small margin) and the chosen step must stay strictly below it.
    assert runtime.wire_cfl_limit <= exact_limit * (1.0 + 1.0e-6)
    assert float(solver.dt) < runtime.wire_cfl_limit


def test_native_wire_cfl_bound_has_strict_gershgorin_slack_on_nonuniform_topology():
    """On strongly non-uniform coefficients Gershgorin is strictly conservative.

    Varying per-segment radius and segment length makes the self terms
    (``L'``/``C'``) differ across segments, so the maximal row sum overshoots the
    top eigenvalue and Gershgorin gains real slack. Measured ratio on this
    fixture is ~0.936 (a ~6.4% margin); the gate below only requires a >3% strict
    slack, which is comfortably inside the measured value and still non-trivial.
    """

    wire = mw.ThinWire(
        name="nonuniform",
        points=(
            (-0.08, 0.0, 0.0),
            (-0.04, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.08, 0.0, 0.0),
        ),
        radius=(2.0e-4, 5.0e-3, 8.0e-4),
        conductor=mw.WireConductor.pec(),
        snap="strict",
    )
    solver, masses = _promote_wire_runtime_to_float64(
        _prepared_solver(_network_scene((wire,)))
    )
    exact_limit = float(_wire_reference_for(solver, masses).maximum_stable_dt())
    gershgorin_limit = _gershgorin_wire_cfl_limit(solver, masses)

    assert math.isfinite(exact_limit)
    # Still a valid lower bound on dt.
    assert gershgorin_limit <= exact_limit
    # ...and here it is strictly, non-trivially below the exact limit.
    assert gershgorin_limit < exact_limit * (1.0 - 3.0e-2)


def _run_native_wire_recurrence_peak(solver, dt_fraction, *, steps=400):
    """Drive the real CUDA sample/update/deposit kernels at a scaled ``dt``.

    ``dt`` and the deposition scales are rebuilt consistently for the requested
    fraction of the production wire CFL limit, then the compressed recurrence is
    stepped with the actual production kernels (no Maxwell curl, so the only
    dynamics are the wire-field coupling the CFL bound governs).
    """

    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    masses = _target_masses(solver, coeff["target_components"], coeff["target_offsets"])
    solver.dt = dt_fraction * runtime.wire_cfl_limit
    edge_groups = _group_indices(coeff["edge_group_offsets"])
    coeff["contribution_scales"] = (
        float(solver.dt)
        * coeff["contribution_weights"]
        / masses.index_select(0, edge_groups)
    ).contiguous()
    node_count = masses.numel()
    seed = torch.where(
        torch.arange(node_count, device=solver.device) % 2 == 0, 1.0, -1.0
    ).to(solver.Ex.dtype)
    _set_target_values(
        solver, coeff["target_components"], coeff["target_offsets"], seed
    )
    runtime.current.zero_()
    runtime.charge.zero_()
    peak = 0.0
    for _ in range(steps):
        sample_and_update_wire(solver)
        deposit_wire_current(solver)
        electric = _target_values(
            solver, coeff["target_components"], coeff["target_offsets"]
        )
        peak = max(
            peak,
            float(runtime.charge.abs().max()),
            float(runtime.current.abs().max()),
            float(electric.abs().max()),
        )
    return peak


def _closed_loop_wire():
    return mw.ThinWire(
        name="loop",
        points=(
            (-0.08, -0.08, 0.0),
            (0.08, -0.08, 0.0),
            (0.08, 0.08, 0.0),
            (-0.08, 0.08, 0.0),
            (-0.08, -0.08, 0.0),
        ),
        radius=2.0e-3,
        conductor=mw.WireConductor.pec(),
        snap="strict",
    )


def test_native_wire_recurrence_diverges_when_dt_straddles_the_cfl_limit():
    """The native kernels are the real safety gate, not just the reference.

    The reference-side straddle test lives in ``test_thin_wire_reference.py``.
    This is the production-path analogue: the same closed-loop network stepped
    through the actual ``sampleWireEmf3D``/``updateWireState1D``/
    ``depositWireCurrent3D`` kernels stays bounded just below the wire CFL limit
    and blows up just above it. Measured on-host: 0.99x -> ~5.9 peak, 1.01x ->
    non-finite.
    """

    stable = _run_native_wire_recurrence_peak(
        _prepared_solver(_network_scene((_closed_loop_wire(),))), 0.99
    )
    unstable = _run_native_wire_recurrence_peak(
        _prepared_solver(_network_scene((_closed_loop_wire(),))), 1.01
    )

    assert math.isfinite(stable)
    assert stable < 1.0e2
    assert (not math.isfinite(unstable)) or unstable > 1.0e3


def test_wire_sample_and_deposit_kernels_are_energy_adjoint_on_shared_edges():
    """Real ``G``/``G^T`` adjointness through the production CUDA kernels.

    Draws random field and current tensors and runs the actual ``sampleWireEmf3D``
    and ``depositWireCurrent3D`` kernels. The energy-adjoint identity

        <sample(E), I>  ==  sum_e  mass_e * E_e * (-dE_e) / dt

    (where ``dE`` is the field increment the deposition kernel applies for current
    ``I``) must hold to float64 machine precision. Unlike the compiler-list
    transpose check in ``_dense_wire_operators``, this is NOT forced by
    construction: it fails if either kernel mis-maps an edge or segment, or if the
    per-edge segmented reduction is wrong. The fixture is a bent oblique polyline
    chosen to exercise BOTH failure modes empirically: it compiles to more than
    one segment (asserted below), so a segment-to-edge mis-map on the second arm
    breaks the identity; and several fragments deposit onto a shared Yee edge
    (also asserted below), exercising the per-edge segmented reduction rather than
    the axis-aligned one-contribution-per-edge case.
    """

    solver, masses = _promote_wire_runtime_to_float64(
        _prepared_solver(
            _scene(
                points=(
                    (-0.08, -0.04, 0.0),
                    (0.0, 0.04, 0.0),
                    (0.08, -0.04, 0.0),
                ),
                source=False,
                monitor=False,
            )
        )
    )
    runtime = solver._wire_runtime
    coeff = runtime.coefficients

    num_segments = int(coeff["segment_offsets"].numel()) - 1
    assert num_segments > 1, (
        "fixture must compile to multiple segments so the identity gates the "
        "segment-to-edge mapping, not just a single segment"
    )
    edge_group_offsets = coeff["edge_group_offsets"]
    contributions_per_edge = edge_group_offsets[1:] - edge_group_offsets[:-1]
    assert int(contributions_per_edge.max()) > 1, (
        "fixture must exercise multiple fragments depositing onto one Yee edge"
    )

    device = solver.device
    torch.manual_seed(0)
    electric = torch.randn(masses.numel(), device=device, dtype=torch.float64)
    current = torch.randn(runtime.current.numel(), device=device, dtype=torch.float64)

    # sample(E): zero the fields, imprint the random E on the sampled edges, and
    # run the real sampling kernel to obtain the segment EMF.
    for name in ("Ex", "Ey", "Ez"):
        getattr(solver, name).zero_()
    _set_target_values(
        solver, coeff["target_components"], coeff["target_offsets"], electric
    )
    runtime.emf.zero_()
    solver.fdtd_module.sampleWireEmf3D(
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        segmentOffsets=coeff["segment_offsets"],
        edgeComponents=coeff["edge_components"],
        edgeOffsets=coeff["edge_offsets"],
        weights=coeff["weights"],
        emf=runtime.emf,
    ).launchRaw()
    sampled_inner_product = float(torch.dot(runtime.emf, current))

    # deposit(I): zero the fields again, load the random current, and run the real
    # deposition kernel; the resulting field increment is exactly -deposited.
    for name in ("Ex", "Ey", "Ez"):
        getattr(solver, name).zero_()
    runtime.current.copy_(current)
    deposit_wire_current(solver)
    field_increment = _target_values(
        solver, coeff["target_components"], coeff["target_offsets"]
    )
    deposited_inner_product = float(
        torch.sum(masses * electric * (-field_increment) / float(solver.dt))
    )

    scale = max(abs(sampled_inner_product), abs(deposited_inner_product), 1.0e-300)
    assert (
        abs(sampled_inner_product - deposited_inner_product) / scale < 1.0e-12
    )
    # Guard against a silent zero-vs-zero pass.
    assert abs(sampled_inner_product) > 1.0e-8


def test_compressed_cuda_recurrence_conserves_charge_and_staggered_energy():
    solver = _prepared_solver(_scene(source=False, monitor=False))
    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    masses = _target_masses(
        solver,
        coeff["target_components"],
        coeff["target_offsets"],
    )
    # One deterministic field value per owned Yee edge. No 3D wire state is used.
    initial_electric = torch.linspace(
        -2.0, 3.0, masses.numel(), device=solver.device, dtype=solver.Ex.dtype
    )
    _set_target_values(
        solver,
        coeff["target_components"],
        coeff["target_offsets"],
        initial_electric,
    )

    energies = []
    max_continuity = torch.zeros((), device=solver.device, dtype=solver.Ex.dtype)
    for _ in range(512):
        electric_before = _target_values(
            solver, coeff["target_components"], coeff["target_offsets"]
        )
        charge_before = runtime.charge.clone()
        current_before = runtime.current.clone()
        sample_and_update_wire(solver)
        current_after = runtime.current.clone()
        incidence_current = torch.zeros_like(runtime.charge)
        incidence_current.index_add_(0, coeff["tail"], current_after)
        incidence_current.index_add_(0, coeff["head"], -current_after)
        continuity = runtime.charge - charge_before + float(solver.dt) * incidence_current
        max_continuity = torch.maximum(max_continuity, continuity.abs().max())
        energies.append(
            0.5 * torch.sum(masses * electric_before.square())
            + 0.5 * torch.sum(charge_before.square() / coeff["node_capacitance"])
            + 0.5
            * torch.sum(coeff["inductance"] * current_after * current_before)
        )
        deposit_wire_current(solver)

    energy = torch.stack(energies)
    relative_drift = (energy - energy[0]).abs().max() / energy[0].abs()
    assert float(max_continuity) < max(
        1.0e-6 * float(runtime.charge.abs().max() + 1.0e-30),
        1.0e-20,
    )
    assert float(relative_drift) < 1.0e-4
    assert runtime.current.numel() == runtime.network.segment_count
    assert runtime.charge.numel() == runtime.network.node_count
    assert runtime.state_bytes == (
        2 * runtime.network.segment_count + runtime.network.node_count
    ) * solver.Ex.element_size()
    assert float(solver.dt) < runtime.cfl_limit


def test_joint_maxwell_wire_cfl_tightens_low_frequency_step_and_stays_bounded():
    solver = _prepared_solver(
        _scene(source=False, monitor=False),
        frequency=1.0e6,
    )
    runtime = solver._wire_runtime
    assert runtime.dt_adjusted is True
    assert float(solver.dt) == pytest.approx(0.99 * runtime.cfl_limit)
    assert runtime.cfl_limit < min(
        runtime.maxwell_cfl_limit,
        runtime.wire_cfl_limit,
    )

    torch.manual_seed(710)
    fields = (solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    for field in fields:
        field.copy_(1.0e-3 * torch.randn_like(field))
    impedance = math.sqrt(float(solver.mu0) / float(solver.eps0))
    initial_max = max(
        max(float(field.abs().max()) for field in fields[:3]),
        impedance * max(float(field.abs().max()) for field in fields[3:]),
    )
    for _ in range(600):
        _field_update_block(solver, 0.0)
    torch.cuda.synchronize()
    final_max = max(
        max(float(field.abs().max()) for field in fields[:3]),
        impedance * max(float(field.abs().max()) for field in fields[3:]),
    )
    assert math.isfinite(final_max)
    assert final_max < 10.0 * initial_max


def test_preparation_validator_rejects_unsafe_native_topology_contents():
    network = _prepared_solver(_scene(source=False, monitor=False))._wire_runtime.network

    invalid_components = network.edge_components.clone()
    invalid_components[0] = 7
    malformed_offsets = network.segment_offsets.clone()
    malformed_offsets[-1] += 1
    invalid_segments = network.contribution_segments.clone()
    invalid_segments[0] = network.segment_count
    duplicate_tail_signs = network.node_signs.clone()
    head_entry = torch.nonzero(
        (network.node_segments == 0) & (network.node_signs == -1), as_tuple=False
    ).reshape(-1)[0]
    duplicate_tail_signs[head_entry] = 1
    duplicate_offsets = network.target_offsets.clone()
    duplicate_components = network.target_components.clone()
    duplicate_offsets[1] = duplicate_offsets[0]
    duplicate_components[1] = duplicate_components[0]
    duplicate_membership = torch.cat(
        (network.wire_node_indices, network.wire_node_indices[:1])
    )
    duplicate_membership_offsets = network.wire_node_offsets.clone()
    duplicate_membership_offsets[-1] += 1

    invalid_networks = (
        replace(network, edge_components=invalid_components),
        replace(network, segment_offsets=malformed_offsets),
        replace(network, contribution_segments=invalid_segments),
        replace(network, node_signs=duplicate_tail_signs),
        replace(
            network,
            target_offsets=duplicate_offsets,
            target_components=duplicate_components,
        ),
        replace(
            network,
            wire_node_indices=duplicate_membership,
            wire_node_offsets=duplicate_membership_offsets,
        ),
    )
    for invalid in invalid_networks:
        with pytest.raises(ValueError, match="Invalid compiled thin-wire topology"):
            _validate_compiled_topology(invalid)


def _network_scene(wires):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    for wire in wires:
        scene.add_thin_wire(wire)
    return scene


def _assert_network_continuity_and_energy(solver):
    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    masses = _target_masses(
        solver, coeff["target_components"], coeff["target_offsets"]
    )
    _set_target_values(
        solver,
        coeff["target_components"],
        coeff["target_offsets"],
        torch.linspace(-2.0, 3.0, masses.numel(), device=solver.device),
    )
    energies = []
    max_continuity = torch.zeros((), device=solver.device)
    for _ in range(512):
        electric_before = _target_values(
            solver, coeff["target_components"], coeff["target_offsets"]
        )
        charge_before = runtime.charge.clone()
        current_before = runtime.current.clone()
        sample_and_update_wire(solver)
        current_after = runtime.current.clone()
        incidence_current = torch.zeros_like(runtime.charge)
        incidence_current.index_add_(0, coeff["tail"], current_after)
        incidence_current.index_add_(0, coeff["head"], -current_after)
        continuity = runtime.charge - charge_before + float(solver.dt) * incidence_current
        max_continuity = torch.maximum(max_continuity, continuity.abs().max())
        energies.append(
            0.5 * torch.sum(masses * electric_before.square())
            + 0.5 * torch.sum(charge_before.square() / coeff["node_capacitance"])
            + 0.5 * torch.sum(coeff["inductance"] * current_after * current_before)
        )
        deposit_wire_current(solver)
    energy = torch.stack(energies)
    relative_drift = (energy - energy[0]).abs().max() / energy[0].abs()
    assert float(max_continuity) < 1.0e-6 * float(runtime.charge.abs().max() + 1.0e-30)
    assert float(relative_drift) < 1.0e-4


def test_branch_and_closed_loop_cuda_recurrence_meet_continuity_and_energy_gates():
    junction = mw.WireEnd.node("J")
    branch = (
        mw.ThinWire(
            name="a",
            points=((-0.08, 0.0, 0.0), (0.0, 0.0, 0.0)),
            radius=2.0e-3,
            conductor=mw.WireConductor.pec(),
            endpoints=(mw.WireEnd.open(), junction),
        ),
        mw.ThinWire(
            name="b",
            points=((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)),
            radius=2.0e-3,
            conductor=mw.WireConductor.pec(),
            endpoints=(junction, mw.WireEnd.open()),
        ),
        mw.ThinWire(
            name="c",
            points=((0.0, 0.0, 0.0), (0.0, 0.08, 0.0)),
            radius=2.0e-3,
            conductor=mw.WireConductor.pec(),
            endpoints=(junction, mw.WireEnd.open()),
        ),
    )
    loop = (
        mw.ThinWire(
            name="loop",
            points=(
                (-0.08, -0.08, 0.0),
                (0.08, -0.08, 0.0),
                (0.08, 0.08, 0.0),
                (-0.08, 0.08, 0.0),
                (-0.08, -0.08, 0.0),
            ),
            radius=2.0e-3,
            conductor=mw.WireConductor.pec(),
        ),
    )
    branch_solver = _prepared_solver(_network_scene(branch), frequency=2.0e9)
    wrong_owner = branch_solver._wire_runtime.network.node_wire_ids.clone()
    wrong_owner[0] = 1
    with pytest.raises(ValueError, match="Invalid compiled thin-wire topology"):
        _validate_compiled_topology(
            replace(branch_solver._wire_runtime.network, node_wire_ids=wrong_owner)
        )
    _assert_network_continuity_and_energy(branch_solver)
    _assert_network_continuity_and_energy(
        _prepared_solver(_network_scene(loop), frequency=2.0e9)
    )


def test_wire_checkpoint_v2_round_trip_and_segment_replay_match_native_forward():
    solver = _prepared_solver(_scene(source=False, monitor=False))
    runtime = solver._wire_runtime
    coeff = runtime.coefficients
    _set_target_values(
        solver,
        coeff["target_components"],
        coeff["target_offsets"],
        torch.linspace(-0.5, 0.75, coeff["target_offsets"].numel(), device=solver.device),
    )
    runtime.current.copy_(
        torch.linspace(-1.0e-6, 2.0e-6, runtime.current.numel(), device=solver.device)
    )
    runtime.charge.copy_(
        torch.linspace(5.0e-15, -4.0e-15, runtime.charge.numel(), device=solver.device)
    )
    checkpoint = capture_checkpoint_state(solver, step=0)
    assert checkpoint.schema.version == 2
    assert checkpoint.schema.wire_state_names == ("wire_current", "wire_charge")
    assert tuple(checkpoint.tensors)[-2:] == checkpoint.schema.wire_state_names
    assert checkpoint.tensors["wire_current"].data_ptr() != runtime.current.data_ptr()
    assert checkpoint.tensors["wire_charge"].data_ptr() != runtime.charge.data_ptr()

    serialized = io.BytesIO()
    torch.save(checkpoint, serialized)
    serialized.seek(0)
    loaded = torch.load(serialized, weights_only=False)
    assert loaded.schema == checkpoint.schema
    for name in checkpoint.schema.state_names:
        torch.testing.assert_close(loaded.tensors[name], checkpoint.tensors[name], rtol=0.0, atol=0.0)

    missing_wire_charge = dict(checkpoint.tensors)
    del missing_wire_charge["wire_charge"]
    with pytest.raises(RuntimeError, match="layout drifted"):
        validate_checkpoint_state(replace(checkpoint, tensors=missing_wire_charge))

    for step_index in range(4):
        _field_update_block(solver, step_index * solver.dt)
    expected = capture_checkpoint_state(solver, step=4)
    replayed = _replay_segment_states(solver, checkpoint, 0, 4)[-1]
    for name in checkpoint.schema.state_names:
        torch.testing.assert_close(
            replayed[name],
            expected.tensors[name],
            rtol=2.0e-5,
            atol=2.0e-6,
            msg=lambda message, state_name=name: f"{state_name}: {message}",
        )


def _run_forward(scene, *, cuda_graph):
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=180),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        cuda_graph=cuda_graph,
    ).run()
    return result.monitor("wire_state"), result.stats()


def test_straight_wire_forward_monitor_is_graph_exact_and_radius_sensitive():
    eager, eager_stats = _run_forward(_scene(), cuda_graph=False)
    captured, captured_stats = _run_forward(_scene(), cuda_graph=True)

    assert eager_stats["cuda_graph_active"] is False
    assert captured_stats["cuda_graph_active"] is True
    assert captured_stats["thin_wire_enabled"] is True
    assert captured_stats["thin_wire_segments"] == captured.current.shape[1]
    assert captured_stats["thin_wire_state_bytes"] > 0
    assert torch.count_nonzero(eager.current).item() > 0
    assert torch.count_nonzero(eager.charge).item() > 0
    torch.testing.assert_close(captured.current, eager.current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(captured.charge, eager.charge, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(captured.ohmic_loss).item() == 0

    wider, _ = _run_forward(_scene(radius=3.0e-3), cuda_graph=False)
    assert not torch.allclose(wider.current, eager.current, rtol=1.0e-3, atol=0.0)


def test_wire_monitor_quantities_control_accumulation_and_typed_output():
    current_only, _ = _run_forward(
        _scene(quantities=("current",)), cuda_graph=False
    )

    assert current_only.current is not None
    assert current_only.charge is None
    assert current_only.ohmic_loss is None
    assert current_only.metadata["quantities"] == ("current",)


def test_l_wire_forward_propagates_current_across_the_corner():
    points = ((-0.08, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.08, 0.0))
    scene = _scene(points=points, source=False)
    scene.add_source(
        mw.PointDipole(
            name="drive",
            position=(-0.02, 0.0, 0.0),
            polarization="Ex",
            width=0.04,
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
        )
    )
    data, stats = _run_forward(scene, cuda_graph=False)

    assert stats["thin_wire_segments"] == 2
    magnitude = data.current.abs()[0]
    assert torch.count_nonzero(magnitude).item() == 2
    assert torch.all(torch.isfinite(data.charge))


def test_named_pec_grounded_endpoint_remains_zero_during_forward():
    ground = mw.Structure(
        name="ground",
        geometry=mw.Box(position=(0.0, 0.0, -0.08), size=(0.02, 0.02, 0.02)),
        material=mw.Material.pec(),
    )
    scene = _scene(
        structures=(ground,),
        endpoints=(mw.WireEnd.grounded(structure="ground"), mw.WireEnd.open()),
    )
    data, _stats = _run_forward(scene, cuda_graph=False)

    assert torch.all(torch.isfinite(data.current))
    torch.testing.assert_close(data.charge[:, 0], torch.zeros_like(data.charge[:, 0]))


@pytest.mark.parametrize(
    ("material", "message"),
    (
        (mw.Material(eps_r=2.0, sigma_e=0.1), "conductive background"),
        (
            mw.Material(
                eps_r=2.0,
                debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1.0e-10),),
            ),
            "dispersive background",
        ),
        (
            mw.Material(
                eps_r=2.0,
                nonlinearity=mw.NonlinearSusceptibility(chi3=1.0e-10),
            ),
            "nonlinear or time-modulated",
        ),
        (
            mw.Material(
                eps_r=2.0,
                modulation=mw.ModulationSpec(frequency=1.0e8, amplitude=0.1),
            ),
            "nonlinear or time-modulated",
        ),
    ),
    ids=("conductive", "dispersive", "nonlinear", "modulated"),
)
def test_phase1_rejects_unsupported_host_compositions(material, message):
    host = mw.Structure(
        name="host",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.24, 0.24)),
        material=material,
    )
    with pytest.raises(NotImplementedError, match=message):
        _prepared_solver(
            _scene(structures=(host,), source=False, monitor=False)
        )


def test_phase1_rejects_surface_impedance_conductor_ownership():
    surface = mw.Structure(
        name="surface_metal",
        geometry=mw.Box(position=(0.08, 0.0, 0.0), size=(0.08, 0.24, 0.24)),
        material=mw.LossyMetalMedium(conductivity=5.8e7),
    )
    with pytest.raises(NotImplementedError, match="share conductor ownership"):
        _prepared_solver(
            _scene(structures=(surface,), source=False, monitor=False)
        )


def test_phase3_real_periodic_interior_wire_prepares_and_steps():
    simulation = mw.Simulation.fdtd(
        _scene(
            boundary=mw.BoundarySpec.periodic(),
            source=False,
        ),
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=2),
    )

    result = simulation.run()

    assert result.solver._wire_runtime is not None
    assert torch.all(torch.isfinite(result.solver._wire_runtime.current))
    assert torch.all(torch.isfinite(result.solver._wire_runtime.charge))


def test_phase3_bloch_wire_composition_remains_fail_closed():
    with pytest.raises(NotImplementedError, match="Bloch-periodic phase topology"):
        _prepared_solver(
            _scene(
                boundary=mw.BoundarySpec.bloch((0.1, 0.0, 0.0)),
                source=False,
                monitor=False,
            )
        )


def _half_wave_dipole_profile(segment_count: int) -> torch.Tensor:
    wavelength = 299792458.0 / _FREQUENCY
    half_length = 0.5 * wavelength
    spacing = half_length / segment_count
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half_length, half_length),) * 3),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        device="cuda",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="dipole",
            points=tuple(
                (0.0, 0.0, float(position))
                for position in torch.linspace(
                    -0.5 * half_length,
                    0.5 * half_length,
                    segment_count + 1,
                    dtype=torch.float64,
                )
            ),
            radius=0.02 * (half_length / 12.0),
            conductor=mw.WireConductor.pec(),
        )
    )
    # Two equal adjacent-edge drives preserve mirror symmetry about the center node.
    for index, z_position in enumerate((-0.5 * spacing, 0.5 * spacing)):
        scene.add_source(
            mw.PointDipole(
                name=f"drive_{index}",
                position=(0.0, 0.0, z_position),
                polarization="Ez",
                width=spacing,
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=5.0),
            )
        )
    scene.add_monitor(
        mw.WireMonitor(
            name="dipole_state",
            wire="dipole",
            frequencies=(_FREQUENCY,),
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=60 * segment_count),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    return result.monitor("dipole_state").current.abs()[0]


def test_center_driven_half_wave_dipole_forward_converges_to_sinusoidal_profile():
    profiles = {
        segment_count: _half_wave_dipole_profile(segment_count)
        for segment_count in (8, 10, 12)
    }
    errors = []
    resampled = []
    for segment_count, magnitude in profiles.items():
        centers = (
            torch.arange(segment_count, device=magnitude.device, dtype=magnitude.dtype)
            + 0.5
        ) / segment_count - 0.5
        oracle = torch.cos(math.pi * centers)
        normalized = magnitude / magnitude.max()
        errors.append(float(torch.linalg.vector_norm(normalized - oracle) / math.sqrt(segment_count)))
        resampled.append(
            F.interpolate(
                normalized.reshape(1, 1, -1),
                size=96,
                mode="linear",
                align_corners=True,
            ).reshape(-1)
        )

        assert magnitude.shape == (segment_count,)
        assert torch.all(torch.isfinite(magnitude))
        assert torch.count_nonzero(magnitude).item() == magnitude.numel()
        torch.testing.assert_close(magnitude, magnitude.flip(0), rtol=0.15, atol=0.0)
        assert magnitude[segment_count // 2 - 1 : segment_count // 2 + 1].mean() > magnitude[
            [0, -1]
        ].mean()
    coarse_change = torch.sqrt(torch.mean((resampled[1] - resampled[0]).square()))
    fine_change = torch.sqrt(torch.mean((resampled[2] - resampled[1]).square()))
    assert float(fine_change) < float(coarse_change)
    assert max(errors) < 0.15


def test_no_wire_scene_keeps_zero_state_and_no_wire_kernel_runtime():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.08, 0.08),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=2),
    ).prepare().solver

    assert solver._wire_runtime is None
