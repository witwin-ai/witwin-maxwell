"""Conformal thin-sheet admission: unsupported requests fail before stepping."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.thin_sheet import owns_thin_sheet

from .harness import pec_material, rectangle_mesh

NORMAL = (1.0, 0.6, 0.35)


def _scene(structure, *, source=(-0.03, 0.02, 0.01)):
    return mw.Scene(
        domain=mw.Domain(bounds=((-.05, .05), (-.05, .05), (-.05, .05))), grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.faces(default="pec"), device="cpu",
        sources=[mw.PointDipole(position=source, polarization="Ez", profile="ideal",
                                source_time=mw.GaussianPulse(frequency=2e9, fwidth=1.5e9))],
        structures=[structure], monitors=[mw.PointMonitor("p", (0.03, -0.02, 0.01), fields=("Ez",))])


def _foil(**kwargs):
    return mw.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 4e9), **kwargs)


def test_ownership_routes_only_declared_sheets():
    plate = rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02)
    assert owns_thin_sheet(mw.Structure(geometry=plate, material=_foil()))
    assert not owns_thin_sheet(mw.Structure(geometry=plate, material=pec_material()))
    declared = rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02, sheet_model="conformal_two_sided")
    assert owns_thin_sheet(mw.Structure(geometry=declared, material=pec_material()))
    with pytest.raises(ValueError, match="sheet_model"):
        rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02, sheet_model="voxel")


def test_cpu_solver_without_the_reference_backend_is_rejected():
    from witwin.maxwell.fdtd.thin_sheet import _require_scope
    solver = SimpleNamespace(cpu_reference=False, device="cpu", scene=SimpleNamespace())
    with pytest.raises(NotImplementedError, match="native CUDA backend or the explicit CPU reference backend"):
        _require_scope(solver)


def test_faceted_copper_mesh_prepares_with_folds():
    mesh = mw.SurfaceMesh.bent_rectangle(width=0.03, length=0.03, radius=0.03, segments=3)
    scene = _scene(mw.Structure(geometry=mesh, material=_foil()), source=(-0.03, 0.02, -0.03))
    prepared = mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                                  backend="cpu_reference").prepare()
    statistics = prepared.solver._thin_sheets[0].complex.statistics
    assert statistics["facets"] == 3 and statistics["fold_faces"] > 0


def test_faceted_surface_mesh_copper_runs_with_two_sided_state():
    mesh = mw.SurfaceMesh.bent_rectangle(width=.03, length=.03, radius=.03, segments=3)
    scene = _scene(mw.Structure(geometry=mesh, material=_foil()), source=(-.03, .02, -.03))
    result = mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=40),
                                backend="cpu_reference").run()
    sheet = result.solver._thin_sheets[0]
    assert sheet.copper is not None and sheet.complex.statistics["fold_faces"] > 0
    assert float(sheet.stored_copper_energy()) > 0
    field = torch.as_tensor(result.monitors["p"]["Ez"])
    assert torch.isfinite(field).all() and field.abs().max() > 0


def test_facets_narrower_than_a_cell_are_rejected():
    mesh = mw.SurfaceMesh.bent_rectangle(width=0.03, length=0.03, radius=0.03, segments=6)
    scene = _scene(mw.Structure(geometry=mesh, material=_foil()), source=(-0.03, 0.02, -0.03))
    with pytest.raises(NotImplementedError, match="refine the grid or widen the facets"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                           backend="cpu_reference").prepare()


def _prepare_mesh(vertices, triangles):
    scene = _scene(mw.Structure(geometry=mw.SurfaceMesh(np.asarray(vertices), triangles), material=_foil()))
    mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                       backend="cpu_reference").prepare()


def test_fold_sharper_than_ninety_degrees_is_rejected():
    from .harness import extruded_polyline_mesh
    mesh = extruded_polyline_mesh([(-0.02, 0.0013), (0.0011, 0.0013), (-0.02, 0.0213)], (-0.02, 0.02))
    with pytest.raises(NotImplementedError, match="fold by more than 90 degrees"):
        _prepare_mesh(*mesh.to_mesh())


def test_inconsistently_oriented_mesh_is_rejected():
    vertices = [(-0.02, -0.02, 0.001), (0.02, -0.02, 0.001), (0.02, 0.02, 0.003), (-0.02, 0.02, 0.003)]
    with pytest.raises(ValueError, match="consistent orientation"):
        _prepare_mesh(vertices, [(0, 1, 2), (0, 3, 2)])


def test_trainable_copper_is_rejected_as_forward_only():
    thickness = torch.tensor(2e-6, dtype=torch.float64, requires_grad=True)
    foil = mw.CopperFoil(thickness=thickness, frequency_range=(0.5e9, 4e9), mode_count=4)
    scene = _scene(mw.Structure(geometry=rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02), material=foil))
    with pytest.raises(NotImplementedError, match="forward-only; trainable (copper|scene) parameters are not supported"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                           backend="cpu_reference").prepare()


def test_trainable_surface_mesh_vertices_are_rejected():
    mesh = rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02)
    vertices, triangles = mesh.to_mesh()
    with pytest.raises(NotImplementedError, match="SurfaceMesh requires fixed geometry and topology"):
        mw.SurfaceMesh(vertices.clone().double().requires_grad_(True), triangles)


def test_source_on_the_patch_is_rejected():
    structure = mw.Structure(geometry=rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02), material=_foil())
    with pytest.raises(NotImplementedError, match="source overlaps a conformal thin-sheet patch"):
        mw.Simulation.fdtd(_scene(structure, source=(0.0, 0.0, 0.0)), frequencies=[2e9],
                           run_time=mw.TimeConfig(time_steps=2), backend="cpu_reference").prepare()


def test_checkpoint_capture_is_rejected():
    from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
    structure = mw.Structure(geometry=rectangle_mesh((0, 0, 0), NORMAL, 0.02, 0.02), material=_foil())
    prepared = mw.Simulation.fdtd(_scene(structure), frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                                  backend="cpu_reference").prepare()
    with pytest.raises(NotImplementedError, match="checkpoint/resume"):
        capture_checkpoint_state(prepared.solver, 0)


def test_explicit_dt_above_the_cut_bound_is_rejected():
    structure = mw.Structure(geometry=rectangle_mesh((0.001, 0.0013, -0.0007), NORMAL, 0.03, 0.025),
                             material=_foil())
    scene = _scene(structure)
    probe = mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                               backend="cpu_reference").prepare()
    limit = probe.solver._thin_sheet_cfl_limit
    metadata = probe.solver._time_step_metadata
    maxwell_allowed = metadata["safety_factor"] * metadata["stability_limit"]
    # Stay below the generic FDTD guard so the cut-complex guard is exercised.
    dt = (limit + maxwell_allowed) / 2
    assert limit < dt < maxwell_allowed
    print(dict(limit=limit, maxwell_allowed=maxwell_allowed, requested_dt=dt, dt=probe.solver.dt))
    with pytest.raises(ValueError, match="thin-sheet leapfrog limit"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2, dt=dt),
                           backend="cpu_reference").prepare()
    assert probe.solver.dt <= 0.985 * limit * (1 + 1e-12)


def test_patch_reaching_the_cpml_is_rejected():
    # Shifted +0.02 m in y so the patch genuinely enters the 4-cell PML collar.
    structure = mw.Structure(geometry=rectangle_mesh((0, 0.02, 0), NORMAL, 0.045, 0.045), material=_foil())
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-.05, .05), (-.05, .05), (-.05, .05))), grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=4), device="cpu",
        sources=[mw.PointDipole(position=(-0.03, 0.03, 0.03), polarization="Ez", profile="ideal",
                                source_time=mw.GaussianPulse(frequency=2e9, fwidth=1.5e9))],
        structures=[structure], monitors=[mw.PointMonitor("p", (0.03, -0.02, 0.01), fields=("Ez",))])
    with pytest.raises(NotImplementedError, match="CPML absorber"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                           backend="cpu_reference").prepare()


def test_third_material_in_the_cut_cells_is_rejected():
    from witwin.core import Box
    plate = mw.Structure(geometry=rectangle_mesh((0, 0, 0), (1.0, 0.0, 0.0), 0.02, 0.02), material=_foil())
    # A dielectric face 0.6 cells beyond the sheet puts a material boundary inside the cut region.
    slab = mw.Structure(geometry=Box(position=(0.0278, 0.0, 0.0), size=(0.05, 0.2, 0.2)), material=mw.Material(eps_r=4.0))
    scene = _scene(plate)
    scene.structures = (slab, plate)
    with pytest.raises(NotImplementedError, match="homogeneous isotropic medium|third material"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                           backend="cpu_reference").prepare()


def test_monitor_on_split_unknowns_warns_about_the_side():
    structure = mw.Structure(geometry=rectangle_mesh((0.001, 0.0013, -0.0007), NORMAL, 0.03, 0.025), material=_foil())
    scene = _scene(structure)
    scene.monitors = (mw.PointMonitor("on_sheet", (0.001, 0.0013, -0.0007), fields=("Ex", "Ey", "Ez")),)
    with pytest.warns(RuntimeWarning, match="read only the sheet's -1 side"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                           backend="cpu_reference").prepare()


@pytest.mark.parametrize("feature", [
    "complex_fields_enabled", "full_aniso_enabled", "nonlinear_enabled", "modulation_enabled",
    "gyromagnetic_enabled", "breakdown_enabled", "dispersive_enabled", "magnetic_dispersive_enabled",
    "tfsf_enabled",
])
def test_runtime_rejects_unqualified_execution_features(feature):
    from witwin.maxwell.fdtd.thin_sheet import _require_scope
    solver = SimpleNamespace(device="cpu", cpu_reference=True,
                             scene=SimpleNamespace(boundary=mw.BoundarySpec.faces(default="pec")))
    setattr(solver, feature, True)
    with pytest.raises(NotImplementedError, match=feature.removesuffix("_enabled")):
        _require_scope(solver)


@pytest.mark.parametrize("boundary", [mw.BoundarySpec.periodic(), mw.BoundarySpec.bloch((.2, .3, .4))],
                         ids=["periodic", "bloch"])
def test_runtime_rejects_periodic_boundaries(boundary):
    from witwin.maxwell.fdtd.thin_sheet import _require_scope
    with pytest.raises(NotImplementedError, match="periodic or Bloch"):
        _require_scope(SimpleNamespace(device="cpu", cpu_reference=True,
                                       scene=SimpleNamespace(boundary=boundary)))


@pytest.mark.parametrize("owner, attribute, value, message", [
    ("scene", "coordinates", object(), "Cartesian"),
    ("scene", "_fdtd_shard_layout", object(), "single-device"),
    ("solver", "_wire_runtime", object(), "wires in the same scene"),
], ids=["coordinates", "shards", "wires"])
def test_runtime_rejects_unqualified_scene_compositions(owner, attribute, value, message):
    from witwin.maxwell.fdtd.thin_sheet import _require_scope
    scene = SimpleNamespace(boundary=mw.BoundarySpec.faces(default="pec"))
    solver = SimpleNamespace(device="cpu", cpu_reference=True, scene=scene)
    setattr(scene if owner == "scene" else solver, attribute, value)
    with pytest.raises(NotImplementedError, match=message):
        _require_scope(solver)


def test_time_step_preparation_rejects_trainable_copper():
    from witwin.maxwell.fdtd.thin_sheet import prepare_thin_sheet_time_step
    foil = _foil()
    foil.thickness = torch.tensor(2e-6, requires_grad=True)
    structure = mw.Structure(rectangle_mesh((0, 0, 0), NORMAL, .02, .02), foil)
    solver = SimpleNamespace(device="cpu", cpu_reference=True, scene=_scene(structure))
    with pytest.raises(NotImplementedError, match="trainable copper parameters"):
        prepare_thin_sheet_time_step(solver)


@pytest.mark.parametrize("case", ["inhomogeneous_ring", "one_sided_third_medium", "split_third_medium"])
def test_side_materials_reject_unrepresented_media(case):
    from witwin.maxwell.fdtd.thin_sheet import _side_materials
    # Two homogeneous sampling rings and two near-sheet unknowns, on opposite sides.
    c = SimpleNamespace(e_component=np.zeros(5, dtype=int),
                        e_index=np.column_stack((np.arange(5), np.zeros((5, 2), dtype=int))),
                        e_phi=np.array([-1.8, -1.5, -.75, .75, 1.5]),
                        e_side=np.array([-1, -1, -1, 1, 1]),
                        e_fixed=np.zeros(5, dtype=bool), e_split=np.zeros(5, dtype=bool),
                        e_normal=np.tile([1., 0., 0.], (5, 1)))
    eps = torch.ones(5, 1, 1, dtype=torch.float64)
    if case == "inhomogeneous_ring":
        eps[0] = 2
        message = "homogeneous isotropic medium"
    else:
        eps[2] = 3
        c.e_split[2] = case == "split_third_medium"
        message = "third material"
    scene = SimpleNamespace(**{f"d{a}_primal64": np.ones(6) for a in "xyz"})
    solver = SimpleNamespace(scene=scene, eps_Ex=eps, sigma_e_Ex=torch.zeros_like(eps))
    with pytest.raises(NotImplementedError, match=message):
        _side_materials(solver, c)


@pytest.fixture(scope="module")
def prepared_sheet_solver():
    structure = mw.Structure(rectangle_mesh((0, 0, 0), NORMAL, .02, .02), _foil())
    return mw.Simulation.fdtd(_scene(structure), frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2),
                              backend="cpu_reference").prepare().solver


@pytest.mark.parametrize("kind", ["pec_suppression", "magnetic_decay", "permeability"])
def test_patch_coefficients_reject_unqualified_updates(prepared_sheet_solver, monkeypatch, kind):
    from witwin.maxwell.fdtd.thin_sheet import _SheetRuntime
    solver = prepared_sheet_solver
    if kind == "pec_suppression":
        for component in ("ex", "ey", "ez"):
            name = f"c{component}_curl"
            monkeypatch.setattr(solver, name, torch.zeros_like(getattr(solver, name)))
        message = "PEC-suppressed"
    else:
        suffix = "decay" if kind == "magnetic_decay" else "curl"
        for component in ("hx", "hy", "hz"):
            name = f"c{component}_{suffix}"
            monkeypatch.setattr(solver, name, getattr(solver, name) * .5)
        message = "non-magnetic, magnetically lossless"
    with pytest.raises(NotImplementedError, match=message):
        _SheetRuntime(solver, solver._thin_sheet_complexes[0])


@pytest.mark.parametrize("kind, message", [
    ("foil", "axis-aligned CopperFoil"), ("cut", "PEC cut-surface"),
    ("impedance", "surface-impedance"), ("lumped", "lumped elements"),
])
def test_runtime_rejects_other_conductor_owners(prepared_sheet_solver, kind, message):
    from witwin.maxwell.fdtd.thin_sheet import _reject_overlaps
    runtime = prepared_sheet_solver._thin_sheets[0]
    c = runtime.complex
    dof = runtime.patch_e[0]
    name = ("Ex", "Ey", "Ez")[int(c.e_component[dof])]
    indices = torch.as_tensor(c.e_index[dof:dof + 1])
    solver = SimpleNamespace()
    if kind == "foil":
        solver._copper_foils = [{"electric_name": name, "indices": indices}]
    elif kind == "cut":
        solver._cut_surface = {"groups": {name: {"indices": indices}}}
    elif kind == "impedance":
        solver._surface_impedance = {"active": True}
    else:
        solver._lumped_element_runtimes = (object(),)
    with pytest.raises(NotImplementedError, match=message):
        _reject_overlaps(solver, [runtime])


def test_native_batch_admission_rejects_thin_sheet_state(prepared_sheet_solver):
    from witwin.maxwell.fdtd.runtime.batch import validate_native_solver
    with pytest.raises(NotImplementedError, match="_thin_sheets"):
        validate_native_solver(prepared_sheet_solver)


@pytest.mark.parametrize("case, message", [
    ("parallel", "single-device"), ("trainable", "forward-only"), ("cache", "FDTDMaterialCache"),
    ("port_sweep", "port sweeps"), ("waveport", "wave-port"), ("initial", "initial conditions"),
    ("absorber", "CPML absorber only"), ("periodic", "non-periodic"), ("coordinates", "Cartesian coordinates"),
])
def test_explicit_cpu_reference_profile_rejects_unqualified_state(case, message):
    from witwin.maxwell.fdtd.cpu_profile import admit_cpu_reference_scene
    config = SimpleNamespace(parallel=None, initial_condition=None, absorber="cpml")
    simulation = SimpleNamespace(config=config, has_trainable_parameters=False, material_cache=None,
                                 port_sweep=None, _waveport_excitation=lambda: None)
    scene = SimpleNamespace(device="cpu", boundary=mw.BoundarySpec.faces(default="pec"))
    if case == "parallel":
        config.parallel = object()
    elif case == "trainable":
        simulation.has_trainable_parameters = True
    elif case == "cache":
        simulation.material_cache = object()
    elif case == "port_sweep":
        simulation.port_sweep = object()
    elif case == "waveport":
        simulation._waveport_excitation = lambda: object()
    elif case == "initial":
        config.initial_condition = object()
    elif case == "absorber":
        config.absorber = "mur"
    elif case == "periodic":
        scene.boundary = mw.BoundarySpec.periodic()
    else:
        scene.coordinates = object()
    with pytest.raises(NotImplementedError, match=message):
        admit_cpu_reference_scene(simulation, scene)


@pytest.mark.parametrize("periodic_face", ["low", "high"])
def test_cpu_reference_difference_rejects_periodic_faces(periodic_face):
    from witwin.maxwell.fdtd.boundary.common import BOUNDARY_PEC, BOUNDARY_PERIODIC
    from witwin.maxwell.fdtd.cpu_reference import _backward
    low, high = ((BOUNDARY_PERIODIC, BOUNDARY_PEC) if periodic_face == "low"
                 else (BOUNDARY_PEC, BOUNDARY_PERIODIC))
    with pytest.raises(NotImplementedError, match="does not support periodic boundaries"):
        _backward(torch.zeros(2, 2, 2), (3, 2, 2), 0, low, high, torch.ones(3))
