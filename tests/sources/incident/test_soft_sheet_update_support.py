"""Soft launch currents must lie on E nodes advanced by the Yee update."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary import BOUNDARY_KIND_TO_CODE
from witwin.maxwell.fdtd.runtime import stepping
from witwin.maxwell.fdtd.runtime.core_tables import compile_source_tables


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _prepare(*, polarization="Ez", kind="plane", boundary=None, sign=1, device="cpu",
             steps=1, core=False):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
        grid=mw.GridSpec.uniform(0.025),
        boundary=boundary or mw.BoundarySpec.pml(num_layers=4), device=device,
    )
    kwargs = dict(direction=(sign, 0, 0), polarization=polarization, injection_axis="x",
                  source_time=mw.CW(frequency=1e9))
    if kind == "plane":
        source = mw.PlaneWave(**kwargs)
    elif kind == "gaussian":
        source = mw.GaussianBeam(beam_waist=0.16, focus=(0.03, 0.02, -0.01), **kwargs)
    else:
        source = mw.AstigmaticGaussianBeam(beam_waist=(0.16, 0.22),
                                         focus=(0.03, 0.02, -0.01), **kwargs)
    scene.add_source(source)
    if steps > 1:
        scene.add_monitor(mw.PointMonitor("probe", position=(0, 0, 0), fields=("Ez", "Hy")))
    return mw.Simulation.fdtd(
        scene, frequency=1e9, backend="cpu_reference" if device == "cpu" else "native",
        # These support checks exercise the CPML electric kernels: pin CPML explicitly.
        absorber="cpml", cuda_graph=False, fdtd_core_path=core, full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=steps),
    ).prepare()


def _patch(term):
    return term["cw_cos_patch"]


def _bounds(term):
    return tuple((start, start + size) for start, size in zip(term["offsets"], _patch(term).shape))


def _assert_supported(solver):
    for term in solver._electric_source_terms:
        for (start, stop), (lo, hi) in zip(
                _bounds(term), stepping.electric_update_support(solver, term["field_name"])):
            assert lo <= start < stop <= hi


@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize("polarization,kind", (("Ez", "plane"), ("Ey", "plane"),
                                               ("Ez", "gaussian"), ("Ez", "astigmatic")))
def test_full_aperture_sheets_exclude_exactly_inactive_electric_rows(polarization, kind, sign):
    solver = _prepare(polarization=polarization, kind=kind, sign=sign).solver
    _assert_supported(solver)
    electric, = solver._electric_source_terms
    magnetic, = solver._magnetic_source_terms
    assert electric["field_name"] == polarization
    if polarization == "Ez":
        assert _bounds(electric)[1:] == ((1, solver.Ny - 1), (0, solver.Nz - 1))
        assert magnetic["field_name"] == "Hy"
        assert _bounds(magnetic)[1:] == ((0, solver.Ny), (0, solver.Nz - 1))
    else:
        assert _bounds(electric)[1:] == ((0, solver.Ny - 1), (1, solver.Nz - 1))
        assert magnetic["field_name"] == "Hz"
        assert _bounds(magnetic)[1:] == ((0, solver.Ny - 1), (0, solver.Nz))


@pytest.mark.parametrize("polarization", ("Ey", "Ez"))
def test_periodic_y_keeps_all_rows_and_pml_z_excludes_only_node_walls(polarization, monkeypatch):
    from witwin.maxwell.fdtd import cpu_profile

    # Compile only: the CPU stepping profile does not implement periodic curls.
    monkeypatch.setattr(cpu_profile, "admit_cpu_reference_scene", lambda simulation, scene:
                        setattr(scene, "_fdtd_cpu_reference", True))
    boundary = mw.BoundarySpec.pml(num_layers=4).with_faces(y_low="periodic", y_high="periodic")
    solver = _prepare(polarization=polarization, boundary=boundary).solver
    _assert_supported(solver)
    electric, = solver._electric_source_terms
    field = getattr(solver, polarization)
    assert _bounds(electric)[1] == (0, field.shape[1])
    assert _bounds(electric)[2] == ((1, field.shape[2] - 1) if polarization == "Ey"
                                  else (0, field.shape[2]))


@pytest.mark.parametrize("low,high", (("pec", "pec"), ("pec", "pmc"), ("pmc", "pec")))
def test_pec_exclusion_follows_tangential_clamp(low, high):
    boundary = mw.BoundarySpec.pml(num_layers=4).with_faces(y_low=low, y_high=high)
    solver = _prepare(boundary=boundary).solver
    _assert_supported(solver)
    electric, = solver._electric_source_terms
    assert _bounds(electric)[1] == (int(low == "pec"), solver.Ny - int(high == "pec"))
    # The PEC clamp acts only on E tangent to y, leaving the y-half axis intact.
    assert stepping.electric_update_support(solver, "Ey")[1] == (0, solver.Ny - 1)
    solver.Ez.fill_(1)
    stepping.enforce_pec_boundaries(solver)
    assert bool(solver.Ez[:, 0, :].eq(0).all()) == (low == "pec")
    assert bool(solver.Ez[:, -1, :].eq(0).all()) == (high == "pec")


@pytest.mark.parametrize("kind", ("none", "pml", "pec", "pmc", "periodic", "bloch", "mur"))
@pytest.mark.parametrize("component", ("Ex", "Ey", "Ez"))
def test_support_per_face_kind_and_component(kind, component):
    # Asymmetric faces catch accidental axis-wide rather than per-face cropping.
    solver = SimpleNamespace(**{component: torch.empty(7, 8, 9)})
    for axis in "xyz":
        setattr(solver, f"boundary_{axis}_low_code", BOUNDARY_KIND_TO_CODE[kind])
        setattr(solver, f"boundary_{axis}_high_code", BOUNDARY_KIND_TO_CODE["pmc"])
    trim = kind in ("none", "pml", "pec", "mur")
    expected = tuple((int(trim and axis != component[-1]), size)
                     for axis, size in zip("xyz", (7, 8, 9)))
    assert stepping.electric_update_support(solver, component) == expected
    for axis in "xyz":
        setattr(solver, f"boundary_{axis}_high_code", BOUNDARY_KIND_TO_CODE[kind])
        setattr(solver, f"boundary_{axis}_low_code", BOUNDARY_KIND_TO_CODE["pmc"])
    expected = tuple((0, size - int(trim and axis != component[-1]))
                     for axis, size in zip("xyz", (7, 8, 9)))
    assert stepping.electric_update_support(solver, component) == expected


@pytest.mark.parametrize("kind", ("none", "pml", "pec", "pmc", "mur"))
@pytest.mark.parametrize("cpml", (False, True), ids=("standard", "cpml"))
def test_support_matches_electric_kernel_evolution(kind, cpml):
    boundary = mw.BoundarySpec.pml(num_layers=4).with_faces(
        y_low=kind, y_high=kind, z_low=kind, z_high=kind,
    )
    solver = _prepare(boundary=boundary).solver
    for axis in "xyz":
        getattr(solver, "E" + axis).fill_(1)
        getattr(solver, f"ce{axis}_decay").fill_(0.5)
    update = (stepping.update_electric_fields_cpml_dense if cpml
              else stepping.update_electric_fields_standard)
    update(solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    for component in ("Ex", "Ey", "Ez"):
        field = getattr(solver, component)
        expected = torch.ones_like(field)
        expected[tuple(slice(lo, hi) for lo, hi in
                       stepping.electric_update_support(solver, component))] = 0.5
        if kind == "pec":
            for axis, label in enumerate("xyz"):
                if label != component[-1] and label in "yz":
                    expected.select(axis, 0).zero_()
                    expected.select(axis, field.shape[axis] - 1).zero_()
        assert torch.equal(field, expected)


@pytest.mark.parametrize("kind", ("plane", "gaussian", "astigmatic"))
def test_normalization_and_retained_profiles_are_bitwise_unchanged(kind, monkeypatch):
    solver = _prepare(kind=kind).solver
    # With full E support the sheet builder reproduces the original full-aperture
    # algorithm. Compare both CW quadratures, including off-axis beam phases.
    with monkeypatch.context() as patch:
        patch.setattr(stepping, "electric_update_support", lambda s, c:
                      tuple((0, n) for n in getattr(s, c).shape))
        original = _prepare(kind=kind).solver
    electric, = solver._electric_source_terms
    untrimmed, = original._electric_source_terms
    slices = tuple(slice(start - base, stop - base) for (start, stop), base
                   in zip(_bounds(electric), untrimmed["offsets"]))
    for key in ("cw_cos_patch", "cw_sin_patch"):
        assert torch.equal(electric[key], untrimmed[key][slices])
        magnetic, = solver._magnetic_source_terms
        original_magnetic, = original._magnetic_source_terms
        assert magnetic["offsets"] == original_magnetic["offsets"]
        assert torch.equal(magnetic[key], original_magnetic[key])


def test_core_source_tables_use_the_same_trimmed_terms():
    solver = _prepare().solver
    extension = SimpleNamespace(CoreStepLaunch=lambda *args: SimpleNamespace(args=args))
    tables = compile_source_tables(solver, extension, torch.zeros(1, dtype=torch.int64), 2)
    for family in ("electric", "magnetic"):
        term, = getattr(solver, f"_{family}_source_terms")
        launch, = tables.groups[family]
        _, tensors, integers, _ = launch.args
        assert tensors[1] is getattr(solver, term["field_name"])
        assert tensors[2] is term["cw_cos_patch"]
        assert tensors[3] is term["cw_sin_patch"]
        assert tuple(integers[:3]) == term["offsets"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for late-time stability")
@pytest.mark.parametrize("core", (False, True), ids=("standard", "core"))
def test_soft_plane_wave_cw_energy_and_corner_field_stay_bounded(core):
    energies = []
    for steps in (2048, 4096):
        prepared = _prepare(device="cuda", steps=steps, core=core)
        prepared.run()
        solver = prepared.solver
        energy = sum((getattr(solver, "E" + axis).double().square().sum() * solver.eps0
                      + getattr(solver, "H" + axis).double().square().sum() * solver.mu0)
                     for axis in "xyz")
        energies.append(float(energy))
        hy = solver.Hy
        corners = torch.stack([hy[:, j, k].abs().max() for j in (0, -1) for k in (0, -1)])
        centre_line = hy[:, hy.shape[1] // 2, hy.shape[2] // 2].abs().max()
        assert float(centre_line) > 0
        assert float(corners.max()) <= 10 * float(centre_line)
    assert abs(energies[1] / energies[0] - 1) <= 0.10
