"""Electrostatic pre-bias initial condition: mapping, grid guard, Gauss gate, and
FDTD steady-state physics.

Capability level under test: an electrostatic (DC) solution mapped onto the Yee
grid as an FDTD initial field. The mapping interpolates the cell-centred DC
potential onto the primary nodes and takes Yee edge differences, so the injected
field is a discrete gradient -- its discrete curl is exactly zero, hence a
lossless interior cell starts in a discrete FDTD steady state (H stays zero, E
stays constant with no source).

The pure-mapping and guard tests run on CPU (the electrostatic solver is
device-agnostic); the steady-state injection tests need the native FDTD runtime
and are CUDA-gated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.electrostatic.initial_condition import DEFAULT_GAUSS_TOLERANCE
from witwin.maxwell.scene import prepare_scene

BOUNDS = ((0.0, 0.5), (0.0, 0.5), (0.0, 0.5))
H = 0.05


def _uniform_dc_result(*, device="cpu", bounds=BOUNDS, h=H):
    """A parallel-plate DC field: z-face electrodes (1 V / 0 V), insulating sides.

    The exact continuum solution is a uniform Ez = V / Lz, and the discrete
    finite-volume solution reproduces it exactly on a uniform grid."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.uniform(h),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    boundary = mw.ElectrostaticBoundarySpec(
        default="neumann", z_low=("dirichlet", 1.0), z_high=("dirichlet", 0.0)
    )
    return mw.Simulation.electrostatic(scene, boundary=boundary).run()


def _matching_fdtd_scene(*, device="cpu", boundary=None, bounds=BOUNDS, h=H):
    return prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=bounds),
            grid=mw.GridSpec.uniform(h),
            boundary=boundary if boundary is not None else mw.BoundarySpec.periodic(),
            device=device,
        )
    )


# ---------------------------------------------------------------------------
# Construction / validation (CPU).
# ---------------------------------------------------------------------------


def test_from_result_rejects_non_electrostatic_result():
    fake = mw.Scene(domain=mw.Domain(bounds=BOUNDS), grid=mw.GridSpec.uniform(H), device="cpu")
    from witwin.maxwell.result import Result

    bad = Result(method="fdtd", scene=fake, frequency=1e9)
    with pytest.raises(ValueError):
        mw.ElectrostaticInitialCondition.from_result(bad)


def test_from_result_rejects_negative_tolerance():
    dc = _uniform_dc_result()
    with pytest.raises(ValueError):
        mw.ElectrostaticInitialCondition.from_result(dc, tolerance=-1.0)


def test_from_result_captures_dc_provenance():
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    assert ic.tolerance == DEFAULT_GAUSS_TOLERANCE
    prov = ic.provenance
    assert prov["source"] == "electrostatic"
    assert prov["capability_level"] == "electrostatic-prebias"
    assert prov["grid_shape"] == tuple(dc.electrostatic.potential.shape)
    assert prov["dc_iterations"] == dc.electrostatic.iterations
    # Gauss residual is filled only at injection (needs FDTD eps).
    assert ic.gauss_residual is None


# ---------------------------------------------------------------------------
# Mapping: exact Yee shapes, curl-free, uniform-field recovery (CPU).
# ---------------------------------------------------------------------------


def test_mapping_yields_exact_yee_component_shapes():
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    prepared = _matching_fdtd_scene()
    Ex, Ey, Ez = ic.map_to_yee(prepared)
    Nx, Ny, Nz = prepared.Nx, prepared.Ny, prepared.Nz
    assert tuple(Ex.shape) == (Nx - 1, Ny, Nz)
    assert tuple(Ey.shape) == (Nx, Ny - 1, Nz)
    assert tuple(Ez.shape) == (Nx, Ny, Nz - 1)


def test_mapped_field_has_zero_discrete_yee_curl():
    # The injected E is a discrete gradient of a nodal scalar, so the discrete Yee
    # curl (the H-update source) is identically zero to roundoff -- this is exactly
    # what makes the pre-bias a discrete FDTD steady state.
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    prepared = _matching_fdtd_scene()
    Ex, Ey, Ez = (t.numpy() for t in ic.map_to_yee(prepared))
    dx = np.asarray(prepared.dx_primal64)
    dy = np.asarray(prepared.dy_primal64)
    dz = np.asarray(prepared.dz_primal64)
    scale = max(np.abs(Ex).max(), np.abs(Ey).max(), np.abs(Ez).max())

    curl_z = (Ey[1:, :, :] - Ey[:-1, :, :]) / dx[:, None, None] - (
        Ex[:, 1:, :] - Ex[:, :-1, :]
    ) / dy[None, :, None]
    curl_x = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy[None, :, None] - (
        Ey[:, :, 1:] - Ey[:, :, :-1]
    ) / dz[None, None, :]
    curl_y = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz[None, None, :] - (
        Ez[1:, :, :] - Ez[:-1, :, :]
    ) / dx[:, None, None]
    tol = 1e-9 * scale / min(dx.min(), dy.min(), dz.min())
    assert np.abs(curl_z).max() <= tol
    assert np.abs(curl_x).max() <= tol
    assert np.abs(curl_y).max() <= tol


def test_uniform_field_recovered_exactly():
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    prepared = _matching_fdtd_scene()
    _, _, Ez = ic.map_to_yee(prepared)
    # V = 1 across Lz = 0.5 -> uniform Ez = 2 V/m everywhere.
    assert torch.allclose(Ez, torch.full_like(Ez, 2.0), atol=1e-9)


# ---------------------------------------------------------------------------
# Grid-identity guard (CPU).
# ---------------------------------------------------------------------------


def test_grid_mismatch_is_rejected():
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    # A different resolution -> different node grid -> fail closed.
    mismatched = _matching_fdtd_scene(h=0.1)
    with pytest.raises(ValueError, match="does not match the FDTD scene grid"):
        ic.map_to_yee(mismatched)


def test_pml_boundary_grid_extension_is_rejected():
    # A PML boundary pads the FDTD grid, so its node grid no longer matches the
    # electrostatic (non-extending) grid: the identity guard must reject it.
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    pml = _matching_fdtd_scene(boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0))
    with pytest.raises(ValueError, match="does not match the FDTD scene grid"):
        ic.map_to_yee(pml)


# ---------------------------------------------------------------------------
# Fail-closed capability guards on the run configuration (CPU; direct validator).
# ---------------------------------------------------------------------------


def _sim_with_ic(boundary, *, parallel=None):
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=boundary,
        device="cpu",
    )
    return mw.Simulation.fdtd(scene, frequencies=[1e9], initial_condition=ic, parallel=parallel)


def test_run_rejects_non_initial_condition_object():
    scene = mw.Scene(domain=mw.Domain(bounds=BOUNDS), grid=mw.GridSpec.uniform(H), device="cpu")
    sim = mw.Simulation.fdtd(scene, frequencies=[1e9], initial_condition="not-an-ic")
    with pytest.raises(TypeError):
        sim._validate_initial_condition_support()


def test_run_rejects_bloch_prebias():
    sim = _sim_with_ic(mw.BoundarySpec.bloch(wavevector=(1.0, 0.0, 0.0)))
    with pytest.raises(NotImplementedError, match="Bloch"):
        sim._validate_initial_condition_support()


def test_run_rejects_distributed_prebias():
    sim = _sim_with_ic(
        mw.BoundarySpec.periodic(),
        parallel=mw.FDTDParallelConfig(devices=("cuda:0", "cuda:1")),
    )
    with pytest.raises(NotImplementedError, match="[Dd]istributed|multi-GPU"):
        sim._validate_initial_condition_support()


def test_run_rejects_trainable_prebias():
    dc = _uniform_dc_result()
    ic = mw.ElectrostaticInitialCondition.from_result(dc)
    density = torch.rand((4, 4, 4), dtype=torch.float32, requires_grad=True)
    scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.periodic(),
        device="cpu",
    )
    scene.add_material_region(
        mw.MaterialRegion(
            name="design",
            geometry=mw.Box(position=(0.25, 0.25, 0.25), size=(0.2, 0.2, 0.2)),
            density=density,
            eps_bounds=(1.0, 6.0),
            mu_bounds=(1.0, 1.0),
        )
    )
    sim = mw.Simulation.fdtd(scene, frequencies=[1e9], initial_condition=ic)
    assert sim.has_trainable_parameters
    with pytest.raises(NotImplementedError, match="[Tt]rainable|adjoint"):
        sim._validate_initial_condition_support()


# ---------------------------------------------------------------------------
# FDTD steady-state physics and Gauss gate (CUDA).
# ---------------------------------------------------------------------------


def _plate(zc):
    # A parallel plate spanning the full x,y cross-section, one z-cell thick.
    return mw.Box(position=(0.25, 0.25, zc), size=(1.0, 1.0, 0.05))


def _plate_prebias_and_solver(time_steps=300, *, device="cuda", tolerance=DEFAULT_GAUSS_TOLERANCE):
    # DC: parallel plates as fixed-potential terminals, insulating sides.
    dc_scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.none(),
        device=device,
    )
    dc_scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal("hot", geometry=_plate(0.075), potential=1.0)
    )
    dc_scene.add_electrostatic_terminal(
        mw.ElectrostaticTerminal("gnd", geometry=_plate(0.425), grounded=True)
    )
    dc = mw.Simulation.electrostatic(
        dc_scene, boundary=mw.ElectrostaticBoundarySpec(default="neumann")
    ).run()
    ic = mw.ElectrostaticInitialCondition.from_result(dc, tolerance=tolerance)

    # FDTD: the plates are PEC-material structures; periodic sides keep the uniform
    # interior field from being clipped, so it is an exact discrete steady state.
    fdtd_scene = mw.Scene(
        domain=mw.Domain(bounds=BOUNDS),
        grid=mw.GridSpec.uniform(H),
        boundary=mw.BoundarySpec.periodic(),
        device=device,
    )
    fdtd_scene.add_structure(
        mw.Structure(name="hot", geometry=_plate(0.075), material=mw.Material.pec())
    )
    fdtd_scene.add_structure(
        mw.Structure(name="gnd", geometry=_plate(0.425), material=mw.Material.pec())
    )
    sim = mw.Simulation.fdtd(
        fdtd_scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        initial_condition=ic,
    )
    return ic, sim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_prebias_charged_plates_are_an_fdtd_steady_state():
    ic, sim = _plate_prebias_and_solver(time_steps=300)
    solver = sim.prepare().solver
    # The discrete-Gauss gate accepts the well-resolved uniform pre-bias.
    assert ic.gauss_residual is not None and ic.gauss_residual < DEFAULT_GAUSS_TOLERANCE

    Ez0 = solver.Ez.detach().clone()
    # Between-plate interior region (plates occupy z-cells 1 and 8): uniform Ez.
    interior = (slice(None), slice(None), slice(2, 7))
    peak = float(Ez0.abs().max())
    assert peak > 0.0
    assert float(Ez0[interior].std()) < 1e-4 * peak  # spatially uniform

    solver.solve(300, full_field_dft=False)
    Ez1 = solver.Ez.detach()
    drift = float((Ez1[interior] - Ez0[interior]).abs().max())
    # No source, curl-free seed, no tangential clipping in the interior: the discrete
    # curl is exactly zero, so H stays identically zero and the interior E is frozen
    # bit-exactly (empirically drift == 0.0).
    assert drift < 1e-9 * peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_non_curlfree_seed_is_not_a_steady_state():
    # Discriminating (falsification) test: the curl-free mapped seed is frozen, but a
    # seed with a nonzero discrete curl -- here a checkerboard perturbation, the
    # farthest a field can be from a discrete gradient -- drifts strongly. This
    # proves the steady-state assertion is load-bearing (a half-cell-staggered or
    # otherwise curl-carrying mapping would fail it), not vacuously true.
    _, sim = _plate_prebias_and_solver(time_steps=300)
    solver = sim.prepare().solver
    Ez0 = solver.Ez.detach().clone()
    peak = float(Ez0.abs().max())
    nx, ny, nz = solver.Ez.shape
    ix = torch.arange(nx, device=solver.Ez.device).view(nx, 1, 1)
    iy = torch.arange(ny, device=solver.Ez.device).view(1, ny, 1)
    iz = torch.arange(nz, device=solver.Ez.device).view(1, 1, nz)
    sign = 1.0 - 2.0 * (((ix + iy + iz) % 2).to(Ez0.dtype))
    solver.Ez[...] = Ez0 * (1.0 + 0.3 * sign)
    seed = solver.Ez.detach().clone()
    solver.solve(300, full_field_dft=False)
    interior = (slice(None), slice(None), slice(2, 7))
    drift = float((solver.Ez.detach()[interior] - seed[interior]).abs().max())
    assert drift > 1e-2 * peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for native FDTD")
def test_gauss_residual_gate_fails_closed():
    # A tolerance below the mapped-field discrete-Gauss residual must reject the
    # injection rather than silently seeding an inconsistent field. The uniform
    # plate pre-bias has a small but non-zero residual; tolerance 0 rejects it.
    _, sim = _plate_prebias_and_solver(time_steps=1, tolerance=0.0)
    with pytest.raises(ValueError, match="discrete-Gauss consistency gate"):
        sim.prepare()
