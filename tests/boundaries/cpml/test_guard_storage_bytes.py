"""Deterministic guard storage including lazily retained sheet diagnostics."""
import faulthandler
import gc
import time
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from tests.boundaries.cpml.test_guard_storage import _mixed_copper_scene
from witwin.maxwell.fdtd.runtime import passivity


@pytest.fixture(autouse=True)
def _sample_deferred_storage(monkeypatch):
    # v0.6.0 skips guarded copper/sheet scenes; this module qualifies the
    # counted functional that dense sampling will use once it is affordable.
    monkeypatch.setattr(passivity, "DEFERRED_STORAGE", ())


def _simulation(case, device="cpu"):
    scene = _mixed_copper_scene(mw.BoundarySpec.faces(x="pec", y="pec", z="pml", num_layers=2))
    scene.device = torch.device(device)
    scene.structures = [s for s in scene.structures
                        if case == "mixed" or s.name == ("planar" if case == "copper" else "sheet")]
    scene.add_monitor(mw.PointMonitor("probe", position=(-.06, .04, .04), fields=("Ez",)))
    return mw.Simulation.fdtd(
        scene, frequency=2e9, absorber="cpml", cuda_graph=True, shutoff=0.,
        backend="cpu_reference" if device == "cpu" else "native",
        run_time=mw.TimeConfig(time_steps=2*passivity.SAMPLE_INTERVAL, transient_cycles=0))


def _storages(value):
    """Inventory complete underlying storage, including sparse CSR components."""
    found = {}

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.layout == torch.sparse_csr:
                for part in (item.crow_indices(), item.col_indices(), item.values()):
                    visit(part)
            else:
                storage = item.untyped_storage()
                if storage.nbytes():
                    found[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for part in item.values():
                visit(part)
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)

    visit(value)
    return found


def _guard_tensors(solver):
    return ({name: value for name, value in vars(solver).items() if name.startswith("_passivity_")},
            _diagnostics(solver))


def _model_sizes(solver):
    # N is the number of native nodes, including exterior PML nodes.
    widths = [8*n for N in (solver.Nx, solver.Ny, solver.Nz) for n in (N-1, N)]
    return widths + [8*passivity.FIT_WINDOWS*3, 8, 8] + _diagnostic_model_sizes(solver)


def _diagnostics(solver):
    return (getattr(solver, "_thin_sheet_energy_widths", {}),
            [getattr(rt, "energy", {}) for rt in getattr(solver, "_thin_sheets", ())])


def _diagnostic_model_sizes(solver):
    """Bytes from cut-complex geometry, independently of device tensor storage."""
    import numpy as np
    import scipy.sparse as sp
    from witwin.maxwell.fdtd.thin_sheet import _sheet_flux_factor

    def csr_sizes(matrix):
        matrix = matrix.tocsr()
        index_bytes = 4 if matrix.nnz < 2**31-1 else 8
        return [index_bytes*(matrix.shape[0]+1), index_bytes*matrix.nnz, 8*matrix.nnz]

    sheets = getattr(solver, "_thin_sheets", ())
    if not sheets:
        return []
    sizes = [8*n for N in (solver.Nx, solver.Ny, solver.Nz) for n in (N-1, N)]
    # Host primal arrays alias the native cache; CUDA copies have distinct storage.
    if torch.device(solver.device).type == "cpu":
        sizes = sizes[1::2]
    for rt in sheets:
        c = rt.complex
        ne, nh = len(c.e_volume), len(c.h_area)
        rows, cols, vals = ((c.k_rows, c.k_cols, c.k_vals) if c.is_pec and not c.feeds else
                            (c.k_x_rows, c.k_x_cols, c.k_x_vals))
        K = sp.coo_matrix((vals, (rows, cols)), shape=(nh, ne)).tocsr()
        sizes += [8*ne, 8*nh, 8*nh, 8*nh] + csr_sizes(K) + csr_sizes(K[:, rt.patch_e])
        if rt.copper is not None:
            ns = len(c.sheet_face_area)
            nc = 2*len(c.chord_face)
            Kc = sp.coo_matrix((c.k_c_vals, (c.k_c_rows, c.k_c_cols)),
                               shape=(nh+ns, nc)).tocsr()
            Kc = Kc @ sp.diags(np.repeat(~c.chord_fixed, 2).astype(float))
            bulk, sheet = Kc[:nh], Kc[nh:]
            B = sheet.T @ sp.diags(np.sqrt(c.sheet_face_dual/(solver.mu0*c.sheet_face_area_eff)))
            sizes += [8*ns]
            for matrix in (bulk, bulk.T, sheet):
                sizes += csr_sizes(matrix)
            factor = _sheet_flux_factor(B)
            sizes += [8*factor["rank"]]  # column index
            for key in ("L", "LT", "BT"):
                sizes += csr_sizes(factor[key])
            if factor["null"].shape[1]:
                sizes += csr_sizes(factor["null"]) + csr_sizes(factor["nullT"])
    return sizes


@pytest.mark.parametrize("case", ["copper", "sheet", "mixed"])
def test_guard_storage_bytes(case, monkeypatch):
    solver = _simulation(case).prepare().solver
    assert bool(solver._copper_foils) == (case != "sheet")
    assert bool(solver._thin_sheets) == (case != "copper")
    assert sum(_storages(_guard_tensors(solver)).values()) == 0
    solver._passivity_state = passivity.make_passivity_state(solver)
    assert solver._passivity_state is not None
    measured = sum(_storages(_guard_tensors(solver)).values())
    expected = sum(_model_sizes(solver))
    assert measured == expected
    diagnostics = _storages(_diagnostics(solver))
    retained = sum(diagnostics.values())
    # Views and repeated aliases must not change the inventory.
    assert sum(_storages((_guard_tensors(solver), solver._passivity_state["history"][:, 1])).values()) == expected
    energy = passivity.total_energy(solver)
    assert energy.dtype == torch.float64
    assert _storages(_diagnostics(solver)) == diagnostics
    assert sum(_storages(_guard_tensors(solver)).values()) == expected
    # Match the cost script's explicit guard-off toggle on a fresh preparation.
    monkeypatch.setattr(passivity, "make_passivity_state", lambda solver: None)
    off_prepared = _simulation(case).prepare()
    off = off_prepared.solver
    off._passivity_state = passivity.make_passivity_state(off)
    assert sum(_storages(_guard_tensors(off)).values()) == 0
    off_prepared.run()
    assert sum(_storages(_guard_tensors(off)).values()) == 0
    print(f"{case}: analytic={expected} measured={measured} guard_off=0 "
          f"sampling_cache={retained}", flush=True)


@pytest.mark.parametrize("name,closed_z,expected", [
    ("copper_patch", False, 7528), ("pec_board", False, 7528),
    ("copper_board", False, 11624), ("copper_board_60k", False, 9384),
    ("copper_patch", True, 7208), ("copper_board_60k", True, 9064),
    ("planar", True, 2856),
])
def test_pcb_grid_storage_totals(name, closed_z, expected):
    """Resolve the real grids without preparing the large copper eigensystems.

    All-PML rows are the conditional cost of an enabled guard at that grid
    size; their real topology is unguarded. Keep that distinction explicit.
    """
    from scripts.dev.fdtd.thin_sheet_overhead import SCENARIOS, _scene
    from witwin.maxwell.scene import _build_axis_grid64

    if name == "planar":
        # Exact grid/domain/boundary of test_copper_fused_kernel._simulation.
        scene = mw.Scene(domain=mw.Domain(((-.3, .3), (-.01, .01), (-.01, .01))),
                         grid=mw.GridSpec.uniform(.005), device="cpu",
                         boundary=mw.BoundarySpec.faces(default="pml", num_layers=12, y="pmc", z="pec"))
    else:
        scene = _scene(SCENARIOS[name], True, device="cpu")
        if closed_z:
            scene = scene.clone(boundary=mw.BoundarySpec.faces(default="pml", num_layers=10, z="pec"))
    widths = {}
    cells = []
    for axis, (lo, hi) in zip("xyz", scene.domain.bounds):
        _, primal, _, dual = _build_axis_grid64(
            scene.grid, axis, lo, hi, scene.boundary.axis_kind(axis),
            tuple(scene.pml_thickness_for_face(axis, side) for side in ("low", "high")))
        widths[f"d{axis}_primal64"], widths[f"d{axis}_dual64"] = primal, dual
        cells.append(len(primal))
    energy_scene = SimpleNamespace(domain=scene.domain, **widths,
                                  boundary=mw.BoundarySpec.faces(x="pec", y="pec", z="pml", num_layers=2))
    solver = SimpleNamespace(scene=energy_scene, device="cpu", c=299792458., dt=1e-12,
                             source_frequency=2.4e9, active_absorber_type="cpml")
    solver._passivity_state = passivity.make_passivity_state(solver)
    measured = sum(_storages(_guard_tensors(solver)).values())
    assert measured == expected == 400+8*sum(2*n+1 for n in cells)
    print(f"{name} closed_z={closed_z} cells={cells}: analytic={expected} measured={measured}", flush=True)


@pytest.fixture
def cuda_time_budget():
    started = time.monotonic()
    faulthandler.dump_traceback_later(120, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert time.monotonic() - started <= 120


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; reviewer-owned")
@pytest.mark.parametrize("case", ["copper", "sheet", "mixed"])
def test_guard_allocated_bytes_and_graph(case, monkeypatch, cuda_time_budget):
    started = time.monotonic()
    print(f"{case}: allocation/capture check start (120 s budget)", flush=True)
    original = passivity.make_passivity_state
    deltas = {}
    for enabled in (False, True):
        prepared = _simulation(case, "cuda").prepare()
        solver = prepared.solver
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(solver.device)
        state = original(solver) if enabled else None
        solver._passivity_state = state
        torch.cuda.synchronize()
        deltas[enabled] = torch.cuda.memory_allocated(solver.device) - before
        expected = sum(_model_sizes(solver)) if enabled else 0
        rounded = sum(((n+511)//512)*512 for n in _model_sizes(solver)) if enabled else 0
        # Each persistent allocation occupies a 512 B allocator block. Graph
        # private pools and per-sample scratch are outside this state-build delta.
        assert expected <= deltas[enabled] <= rounded
        monkeypatch.setattr(passivity, "make_passivity_state", lambda solver, state=state: state)
        prepared.run()
        torch.cuda.synchronize()
        assert solver._cuda_graph_active is True
        if enabled:
            assert state["sample_count"] >= 2
        else:
            assert solver._passivity_state is None
        assert sum(_storages(_guard_tensors(solver)).values()) == expected
        print(f"{case} guard={enabled}: analytic={expected} allocated={deltas[enabled]} "
              f"rounded={rounded} graph={solver._cuda_graph_active}", flush=True)
        del prepared, solver, state
        gc.collect()
    assert deltas[False] == 0
    assert time.monotonic() - started <= 120
