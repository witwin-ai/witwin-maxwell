"""Full-row sheet flux recovery, including inconsistent and closed-loop states."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.thin_sheet_overhead import SCENARIOS, _scene
from tests.boundaries.cpml.test_guard_storage import _flux_plan, _mixed_copper_scene
from tests.materials.thin_sheet import test_thin_sheet_runtime as cavity
from tests.materials.thin_sheet.test_thin_sheet_chords import prism_scene
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, enforce_pec_boundaries
from witwin.maxwell.fdtd.thin_sheet import (
    _sheet_flux, _sheet_flux_factor, thin_sheet_energy, thin_sheet_energy_device,
)


def _incidence(runtime, solver):
    c = runtime.complex
    Kc = sp.coo_matrix((c.k_c_vals, (c.k_c_rows, c.k_c_cols)),
                       shape=(len(c.h_area)+len(c.sheet_face_area), runtime.chord_e.numel())).tocsr()
    Kc = Kc @ sp.diags(np.repeat(~c.chord_fixed, 2).astype(float))
    return (Kc[len(c.h_area):].T @ sp.diags(
        np.sqrt(c.sheet_face_dual/(solver.mu0*c.sheet_face_area_eff)))).tocsr()


def _check_flux(B, plan, forcing):
    reference = lsqr(B, forcing.numpy(), atol=1e-15, btol=1e-15, iter_lim=4*B.shape[1])[0]
    actual = _sheet_flux(plan, forcing)
    error = np.max(np.abs(actual.numpy()-reference))
    scale = np.max(np.abs(reference))
    assert error <= 1e-12*scale
    assert all(torch.equal(actual, _sheet_flux(plan, forcing)) for _ in range(3))
    residual = np.linalg.norm(B@reference-forcing.numpy())/np.linalg.norm(forcing.numpy())
    return residual, error/scale


@pytest.mark.parametrize("case", ["patch", "cavity", "mixed", "closed"])
def test_midrun_flux_and_energy_match_lsqr(case):
    if case == "patch":
        spec = {**SCENARIOS["copper_patch"], "cell": .004}
        scene = _scene(spec, True, device="cpu").clone(boundary=mw.BoundarySpec.pec())
    elif case == "cavity":
        scene = cavity._cavity(cavity._plate(mw.CopperFoil(
            thickness=2e-6, frequency_range=(.5e9, 4e9))))
    elif case == "closed":
        scene = prism_scene(mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 4e9)),
                            side_cells=6)
    else:
        scene = _mixed_copper_scene(mw.BoundarySpec.pec())
    solver = mw.Simulation.fdtd(scene, frequency=2e9, backend="cpu_reference",
                                run_time=mw.TimeConfig(time_steps=80)).prepare().solver
    runtime = solver._thin_sheets[0]
    runtime.ensure_energy(solver)
    p = runtime.energy
    B = _incidence(runtime, solver)
    factor = _sheet_flux_factor(B)
    assert factor["rank"] == np.linalg.matrix_rank(B.toarray())
    generator = torch.Generator().manual_seed(901)
    for name in ("Ex", "Ey", "Ez"):
        getattr(solver, name).normal_(generator=generator)
    runtime.scatter(solver, runtime.plan_all_e, runtime.e_extra,
                    torch.zeros(runtime.plan_all_e["size"], dtype=torch.float64))
    enforce_pec_boundaries(solver)
    for entry in solver._copper_foils:
        entry["right_e"].copy_(getattr(solver, entry["electric_name"])[entry["index"]])
    for step in range(1, 81):
        _field_update_block(solver, 0.)
        enforce_pec_boundaries(solver)
        if step not in (20, 40, 80):
            continue
        h = runtime.gather(solver, runtime.plan_all_h, runtime.h_extra)
        forcing = (runtime.chord_f.reshape(-1)-torch.mv(p["KcT"], p["dual"]*h)
                   + float(solver.dt)/4*torch.mv(p["KcT"], p["lh"]*torch.mv(
                       p["K_patch"], runtime.e_new-runtime.e_old)))
        residual, z_error = _check_flux(B, p["flux"], forcing)
        # Native fp32 fields make the forcing inconsistent; no row may be dropped.
        assert residual > 1e-13
        for modified in (False, True):
            reference = thin_sheet_energy(solver, modified=modified)
            actual = thin_sheet_energy_device(solver, modified=modified)
            energy_error = abs(float(actual)-reference)/abs(reference)
            assert energy_error <= 1e-12
            assert all(torch.equal(actual, thin_sheet_energy_device(solver, modified=modified))
                       for _ in range(3))
            print(dict(case=case, step=step, modified=modified, consistency_residual=residual,
                       B_shape=B.shape, row_deficiency=B.shape[0]-factor["rank"],
                       column_deficiency=B.shape[1]-factor["rank"],
                       factor_nnz=factor["L"].nnz, z_relative_error=z_error,
                       energy_relative_error=energy_error))


@pytest.mark.parametrize("inconsistent", [False, True])
def test_closed_loop_flux_keeps_minimum_norm_and_all_forcing_rows(inconsistent):
    # A closed weighted face loop has both a column nullspace and dependent
    # rows. Duplicate one chord and perturb it to falsify row selection.
    B = sp.csr_matrix([[1., -2., 0.], [0., 2., -3.], [-1., 0., 3.], [1., -2., 0.]])
    plan = _flux_plan(B)
    assert plan["rank"] == 2
    assert "null" in plan
    for state in (np.array([.3, .6, -.2]), np.array([-.8, .3, .7]), np.array([.1, -.9, .4])):
        forcing = B@state
        if inconsistent:
            forcing[-1] += .015
        residual, error = _check_flux(B, plan, torch.from_numpy(forcing))
        actual = _sheet_flux(plan, torch.from_numpy(forcing)).numpy()
        reference = np.linalg.pinv(B.toarray())@forcing
        assert np.max(np.abs(actual-reference)) <= 1e-12*np.max(np.abs(reference))
        assert abs(actual@np.array([1., .5, 1/3])) < 1e-14
        print(dict(closed_loop=True, inconsistent=inconsistent, consistency_residual=residual,
                   row_deficiency=2, column_deficiency=1, z_relative_error=error))
    assert torch.count_nonzero(_sheet_flux(plan, torch.zeros(4, dtype=torch.float64))) == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; reviewer-owned")
def test_device_flux_on_current_stream_is_bitwise_and_has_no_host_reads(monkeypatch):
    from witwin.maxwell.fdtd.thin_sheet import _SheetRuntime, _prepare_sheet_flux

    B = sp.csr_matrix([[1., -2., 0.], [0., 2., -3.], [-1., 0., 3.], [1., -2., 0.]])
    forcing = np.array([.4, -.8, .2, .415])
    reference = lsqr(B, forcing, atol=1e-15, btol=1e-15)[0]
    runtime = _SheetRuntime.__new__(_SheetRuntime)
    runtime.device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=runtime.device)
    with torch.cuda.stream(stream):
        plan = _prepare_sheet_flux(runtime, B)
        rhs = torch.as_tensor(forcing, device=runtime.device)
        expected = torch.as_tensor(reference, device=runtime.device)
        actual = _sheet_flux(plan, rhs)
        assert (actual-expected).abs().max() <= 1e-12*expected.abs().max()

        def host_read(*args, **kwargs):
            raise AssertionError("flux sampling read device data on the host")

        with monkeypatch.context() as patch:
            for name in ("cpu", "numpy", "item", "__float__", "__bool__"):
                patch.setattr(torch.Tensor, name, host_read)
            torch.cuda.set_sync_debug_mode("error")
            try:
                samples = [_sheet_flux(plan, rhs) for _ in range(3)]
            finally:
                torch.cuda.set_sync_debug_mode(0)
        assert all(torch.equal(samples[0], value) for value in samples[1:])
        assert torch.equal(samples[0], actual)
        print(dict(repeated_bitwise=True, stream=stream.cuda_stream))


@pytest.mark.parametrize("matrix", [
    [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    [[2., 0., 0.], [0., 0., 0.], [0., -3., 0.]],
    [[2., -3., 4.], [0., 0., 0.], [5., 0., 0.]],
])
def test_fixed_row_reduction_handles_empty_and_uneven_rows(matrix):
    from witwin.maxwell.fdtd.thin_sheet import _sheet_flux_row_sum

    sparse = torch.tensor(matrix, dtype=torch.float64).to_sparse_csr()
    width = int(torch.diff(sparse.crow_indices()).max())
    value = torch.tensor([.3, -.7, .8], dtype=torch.float64)
    expected = torch.mv(sparse, value)
    actual = _sheet_flux_row_sum(sparse, width, value)
    torch.testing.assert_close(actual, expected, rtol=4*torch.finfo(torch.float64).eps, atol=0)
    assert all(torch.equal(actual, _sheet_flux_row_sum(sparse, width, value)) for _ in range(3))
