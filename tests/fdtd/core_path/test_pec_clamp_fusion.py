"""Host admission/index proofs and T0 gates against the historical clamp sites."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import core_path, core_tables, stepping


FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _solver(sides=((True, True),) * 3, **overrides):
    solver = SimpleNamespace(
        scene=SimpleNamespace(boundary=mw.BoundarySpec.pec()), has_pec_faces=True,
        tfsf_enabled=False, complex_fields_enabled=False, electric_dispersive_enabled=False,
        _source_terms=[], _electric_source_terms=[], observers_enabled=False,
        _advance_magnetic_dispersive_state=lambda: None,
        _apply_magnetic_dispersive_corrections=lambda: None,
        _advance_dispersive_state=lambda: None, _apply_dispersive_corrections=lambda: None,
    )
    for component, shape in zip(("Ex", "Ey", "Ez"), ((4, 7, 9), (5, 6, 9), (5, 7, 8))):
        setattr(solver, component, torch.zeros(shape))
    for axis, pair in zip("xyz", sides):
        for side, pec in zip(("low", "high"), pair):
            setattr(solver, f"boundary_{axis}_{side}_code", 4 if pec else 0)
    vars(solver).update(overrides)
    return solver


def _term(field="Ez", offsets=(2, 3, 3), shape=(1, 1, 1)):
    return dict(field_name=field, offsets=offsets, patch=torch.zeros(shape))


def _sources(ordinary=True):
    return SimpleNamespace(auxiliary=None, groups={
        name: [SimpleNamespace(launch=lambda: None)] if ordinary and name == "ordinary" else []
        for name in ("ordinary", "electric", "magnetic", "tfsf_electric", "tfsf_magnetic")})


@pytest.mark.parametrize("dispersive,expected", [(False, (True, False)), (True, (True, True))])
def test_p1_dominating_clamp_without_electric_correction(dispersive, expected):
    # A face source forces P2 to retain call 1, isolating the P1 decision.
    solver = _solver(electric_dispersive_enabled=dispersive, _source_terms=[_term(offsets=(0, 3, 3))])
    assert core_path.pec_clamp_plan(solver, _sources()) == expected


@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize("writer", ["interior", "adjacent", "face", "electric_face", "cw_face", "unknown"])
def test_p2_uses_actual_compiled_patch_destinations(ordinary, writer):
    solver = _solver()
    term = _term(offsets={"face": (0, 3, 3), "electric_face": (0, 3, 3),
                          "cw_face": (0, 3, 3), "adjacent": (1, 3, 3)}.get(writer, (2, 3, 3)))
    if writer == "unknown":
        term = {"field_name": "Ez"}
    if writer == "cw_face":
        term["cw_cos_patch"], term["patch"] = term["patch"], None
    solver._source_terms = [term] if ordinary else []
    if not ordinary or writer == "electric_face":
        solver._electric_source_terms = [term]
    touches = writer not in ("interior", "adjacent")
    assert core_path.pec_clamp_plan(solver, _sources(ordinary)) == (
        (True, False) if ordinary and touches else (False, touches))


@pytest.mark.parametrize("tfsf_writer", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_tfsf_without_ordinary_sources_never_adds_clamp(tfsf_writer, auxiliary):
    solver = _solver(tfsf_enabled=True, electric_dispersive_enabled=True,
                     _tfsf_state={"electric_terms": [_term(offsets=(0 if tfsf_writer else 2, 3, 3))]})
    sources = _sources(False)
    sources.auxiliary = object() if auxiliary else None
    assert core_path.pec_clamp_plan(solver, sources) == (False, False)


@pytest.mark.parametrize("tfsf_writer", [False, True])
def test_tfsf_terms_participate_in_first_clamp_decision(tfsf_writer):
    solver = _solver(tfsf_enabled=True,
                     _tfsf_state={"electric_terms": [_term(offsets=(0 if tfsf_writer else 2, 3, 3))]})
    assert core_path.pec_clamp_plan(solver, _sources()) == (tfsf_writer, False)


@pytest.mark.parametrize("ordinary,tfsf", list(product((False, True), repeat=2)))
def test_complex_fields_keep_historical_real_and_imaginary_path(ordinary, tfsf):
    assert core_path.pec_clamp_plan(_solver(complex_fields_enabled=True, tfsf_enabled=tfsf),
                                     _sources(ordinary)) == (ordinary, not tfsf)


@pytest.mark.parametrize("attribute", ["_port_runtimes", "_lumped_element_runtimes",
                                       "_coaxial_element_runtimes", "_network_runtimes"])
def test_separate_physics_between_clamps_prevents_p1(attribute):
    solver = _solver(**{attribute: (object(),)})
    assert core_path.pec_clamp_plan(solver, _sources()) == (True, True)


def test_sheet_keeps_post_e_clamp_but_does_not_prevent_p1():
    assert core_path.pec_clamp_plan(_solver(_thin_sheets=(object(),)), _sources()) == (True, False)


def test_missing_term_tables_and_periodic_images_keep_clamp():
    solver = _solver(_source_terms=None)
    assert core_path.pec_clamp_plan(solver, _sources()) == (True, False)
    solver._source_terms = [_term()]
    solver.scene.boundary = mw.BoundarySpec.faces(default="pec", y="periodic")
    assert core_path.pec_clamp_plan(solver, _sources()) == (True, False)


@pytest.mark.parametrize("offsets,shape,touches", [
    ((-2, 3, 3), (3, 1, 1), True), ((-2, 3, 3), (2, 1, 1), False),
    ((4, 3, 3), (4, 1, 1), True), ((5, 3, 3), (1, 1, 1), False),
    ((2, -3, 3), (1, 1, 1), False),
])
def test_patch_clipping_matches_source_kernel(offsets, shape, touches):
    assert core_path._term_may_write_pec(_solver(), _term(offsets=offsets, shape=shape)) is touches


@pytest.mark.parametrize("field", ["Ex", "Ey", "Ez"])
def test_component_normal_face_is_not_clamped(field):
    offsets = [2, 3, 3]
    offsets["xyz".index(field[-1].lower())] = 0
    assert not core_path._term_may_write_pec(_solver(), _term(field, tuple(offsets)))


def _boundary_face_offsets(shape, axis, side):
    """NumPy translation of boundary.cu boundary_face_offset (full face)."""
    nx, ny, nz = shape
    face_linear = np.arange((ny * nz, nx * nz, nx * ny)[axis])
    if axis == 0:
        i, (j, k) = (0 if side == 0 else nx - 1), np.divmod(face_linear, nz)
    elif axis == 1:
        (i, k), j = np.divmod(face_linear, nz), (0 if side == 0 else ny - 1)
    else:
        (i, j), k = np.divmod(face_linear, ny), (0 if side == 0 else nz - 1)
    return (i * ny + j) * nz + k


@pytest.mark.parametrize("sides", list(product(((False, False), (True, False),
                                               (False, True), (True, True)), repeat=3)))
@pytest.mark.parametrize("component", range(3))
def test_existing_core_zero_set_equals_clamp_union_with_edges_and_corners(sides, component):
    # Odd computational sizes expose component-specific high-face off-by-one errors.
    shape = tuple(size - (axis == component) for axis, size in enumerate((5, 7, 9)))
    coords = np.indices(shape)
    core_zero = np.zeros(shape, dtype=bool)
    clamp_zero = np.zeros(np.prod(shape), dtype=bool)
    for axis in range(3):
        if axis == component:
            continue
        low, high = sides[axis]
        # yee.cu pec(coord, component_size, half.axes[axis]).
        core_zero |= ((coords[axis] == 0) & low) | ((coords[axis] + 1 == shape[axis]) & high)
        for side, enabled in enumerate((low, high)):
            if enabled:
                clamp_zero[_boundary_face_offsets(shape, axis, side)] = True
    np.testing.assert_array_equal(core_zero.reshape(-1), clamp_zero)
    # The production admission predicate must recognize every member of this set.
    solver = _solver(sides)
    for index in product(*(range(size) for size in shape)):
        assert core_path._term_may_write_pec(solver, _term("E" + "xyz"[component], index)) == core_zero[index]


@pytest.mark.parametrize("face,dispersive,sides,expected", [
    (False, False, ((True, True),) * 3, 0),
    (True, False, ((True, True),) * 3, 3),
    (False, True, ((True, True),) * 3, 3),
    (True, True, ((True, True),) * 3, 6),
    (False, False, ((True, False),) * 3, 0),
    (True, False, ((True, False),) * 3, 6),
])
def test_graph_builder_issues_expected_native_clamp_launches(monkeypatch, face, dispersive, sides, expected):
    launches = []

    class NativeFDTDModule:
        def clampPecBoundary3D(self, **kwargs):
            return SimpleNamespace(launchRaw=lambda: launches.append("pair"))

        def clampFieldFace3D(self, **kwargs):
            return SimpleNamespace(launchRaw=lambda: launches.append("face"))

    solver = _solver(sides, electric_dispersive_enabled=dispersive, fdtd_module=NativeFDTDModule(),
                     _source_terms=[_term(offsets=(0 if face else 2, 3, 3))])
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.sources, runtime.step = _sources(), torch.zeros(1, dtype=torch.int64)
    runtime.magnetic = runtime.electric = lambda: None
    runtime.extension = SimpleNamespace(CoreStepLaunch=lambda *a: SimpleNamespace(launch=lambda: None))
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda s: None)
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: None)
    from witwin.maxwell.fdtd import cut_surface
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: None)
    step, _ = runtime.prepare_step(solver, use_gpu_dft=False, time_steps=32)
    # The closures retain the graph-build verdict even if host metadata changes.
    solver._source_terms = [_term(offsets=(0 if not face else 2, 3, 3))]
    for _ in range(32):
        step()
    assert len(launches) == 32 * expected


def _simulation(case, *, core=True, steps=65, device="cuda"):
    # Binary-exact nodes keep the adjacent dipole's interpolation support off
    # the wall, including the float32 coordinate conversion during compilation.
    nodes = tuple(np.linspace(-.5, .5, size) for size in (9, 17, 17))
    boundary = mw.BoundarySpec.pec()
    if case == "one_side":
        boundary = mw.BoundarySpec.faces(x_low="pec", y_high="pec", z_low="pec")
    elif case == "cpml":
        # PEC faces next to 4-layer CPML on every axis: an open PML topology, so absorber="auto"
        # resolves to cpml with the passivity guard inactive and the core path stays eligible
        # (closed-transverse CPML is guard-sampled and core-refused since the guard merge).
        boundary = mw.BoundarySpec.faces(default="pml", x_low="pec", y_low="pec", num_layers=4)
    elif case == "mur":
        boundary = mw.BoundarySpec.faces(default="pec", z="mur")
    y = nodes[1][1] if case == "adjacent" else (nodes[1][0] if case == "on_face" else 0.)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
                     grid=mw.GridSpec.custom(*nodes), boundary=boundary, device="cpu")
    if case == "dispersive":
        scene.add_structure(mw.Structure(
            geometry=mw.Box(position=(0., 0., 0.), size=(1.2, 1.2, 1.2), device="cpu"),
            material=mw.Material.debye(eps_inf=2., delta_eps=1., tau=8e-11), name="dispersive_host"))
    scene.add_source(mw.PointDipole(position=(0., y, 0.), polarization="Ez", profile="ideal", width=.01,
                                   source_time=mw.CW(frequency=1e9, phase=.3)))
    scene.add_monitor(mw.PointMonitor("point", position=(.1, .1, .1), fields=FIELDS))
    scene.add_monitor(mw.PlaneMonitor("plane", axis="y", position=0., fields=FIELDS, frequencies=(1e9,)))
    # A Hanning CW window opens after the >=15-period transient (hundreds of
    # steps here); window="none" samples from step 0 so a short run records DFT
    # and monitor spectra while keeping the CW injection route.
    return mw.Simulation.fdtd(scene.clone(device=device), frequency=1e9,
        run_time=mw.TimeConfig(time_steps=steps), full_field_dft=True, fdtd_core_path=core,
        spectral_sampler=mw.SpectralSampler(window="none"), cuda_graph=True, shutoff=0.)


@pytest.mark.parametrize("case", ["box", "one_side", "cpml", "adjacent", "on_face", "dispersive", "mur"])
def test_gate_scenes_build_on_host(monkeypatch, case):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU scene construction initialized CUDA"))
    simulation = _simulation(case, device="cpu")
    assert simulation.config.fdtd_core_path
    assert len(simulation.scene.monitors) == 2
    if case == "mur":
        assert "MUR boundaries" in core_path.core_path_blockers(SimpleNamespace(scene=simulation.scene))


@pytest.mark.parametrize("case", ["box", "adjacent", "on_face"])
def test_real_dipole_patch_metadata_selects_expected_route(case):
    from witwin.maxwell.fdtd.excitation.injection import _prepare_point_dipole_source
    from witwin.maxwell.scene import PreparedScene

    scene = PreparedScene(_simulation(case, device="cpu").scene, lazy_meshgrid=True)
    source = scene.sources[0]
    shape = tuple(len(nodes) for nodes in (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64))
    solver = _solver(scene=scene, Nx=shape[0], Ny=shape[1], Nz=shape[2], dt=1e-11, device="cpu",
                     Ez=torch.zeros(shape[0], shape[1], shape[2] - 1),
                     eps_Ez=torch.ones(shape[0], shape[1], shape[2] - 1),
                     _iter_source_images=lambda position, cutoff: [(position, 1., 0.)])
    _prepare_point_dipole_source(solver, dict(position=source.position, width=source.width,
        profile=source.profile, polarization=source.polarization, source_time={"frequency": 1e9}), source_index=0)
    assert solver._source_terms
    assert any(core_path._term_may_write_pec(solver, term)
               for term in solver._source_terms) == (case == "on_face")
    if case == "adjacent":
        assert all(term["offsets"][1] == 1 and term["patch"].shape[1] == 1 for term in solver._source_terms)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="T0 comparison requires CUDA.")
@pytest.mark.parametrize("case", ["box", "one_side", "cpml", "adjacent", "on_face", "dispersive"])
@pytest.mark.parametrize("chunk", [1, 16, 32])
def test_fields_and_monitors_equal_historical_clamps_three_runs(monkeypatch, case, chunk):
    from benchmark.core_path.gates import tensors

    plan = core_path.pec_clamp_plan

    def run(oracle):
        # Only the test can restore the old two sites. Both sites still call
        # stepping.enforce_pec_boundaries, including its original native kernels.
        monkeypatch.setattr(core_path, "pec_clamp_plan", (
            lambda s, sources: (bool(sources.groups["ordinary"]), not s.tfsf_enabled)) if oracle else plan)
        execution = _simulation(case).prepare()
        solver = execution.solver
        solver._core_runtime.chunk_size = chunk
        hits = any(core_path._term_may_write_pec(solver, term) for term in solver._source_terms)
        assert hits == (case == "on_face")
        generator = torch.Generator(device="cpu").manual_seed(812)
        for name in FIELDS:
            field = getattr(solver, name)
            field.copy_(torch.randn(field.shape, generator=generator) * 2.**-20)
        result = execution.run()
        assert solver._core_step_graph_active
        assert solver._core_step_graph_k == chunk
        assert solver._core_runtime.pec_clamp_plan == (
            (True, True) if oracle else (case == "on_face", case == "dispersive"))
        values = {name: getattr(solver, name).cpu().clone() for name in FIELDS}
        values.update({name: value.cpu().clone() for name, value in tensors(result.monitors, "monitor").items()})
        values.update({name: value.cpu().clone()
                       for name, value in tensors(solver._dft_batched_fields, "dft").items()})
        assert any(name.startswith("monitor/") for name in values)
        return values

    reference = run(True)
    for _ in range(3):
        actual = run(False)
        assert actual.keys() == reference.keys()
        for name in reference:
            assert torch.equal(actual[name], reference[name]), (case, chunk, name)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mur regression requires CUDA.")
def test_pec_mur_keeps_existing_admission_and_noncore_execution():
    from benchmark.core_path.gates import tensors

    # Mur is already excluded by core_path_blockers; do not broaden admission
    # just to exercise an optimization in a path that cannot execute it.
    with pytest.raises(ValueError, match="MUR boundaries"):
        _simulation("mur").prepare()
    reference = None
    for _ in range(3):
        result = _simulation("mur", core=False).run()
        fields = {name: getattr(result.solver, name).cpu().clone() for name in FIELDS}
        fields.update(tensors(result.monitors, "monitor"))
        if reference is not None:
            assert fields.keys() == reference.keys()
            for name in fields:
                assert torch.equal(fields[name], reference[name]), name
        reference = fields
