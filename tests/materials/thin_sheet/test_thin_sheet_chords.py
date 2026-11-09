"""Named chord gates: PEC identity/bound, prism modes and the implicit inverse."""
import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.thin_sheet import compile_thin_sheet
from witwin.maxwell.scene import prepare_scene

from . import harness as h
from . import test_thin_sheet_runtime as rt

EPS0, MU0 = 8.8541878128e-12, 1.25663706212e-6


def prism_scene(material, side_cells=10):
    dl = .005
    radius = side_cells * dl / math.sqrt(2)
    extent = math.ceil((radius + 5*dl) / dl) * dl
    pts = h.polygon_points(radius, 4, center=(.37*dl, .21*dl), phase=.5)
    kw = dict(sheet_model="conformal_two_sided") if material.is_pec else {}
    mesh = h.extruded_polyline_mesh(pts, (-2*dl, 4*dl), closed=True, **kw)
    return mw.Scene(domain=mw.Domain(bounds=((-extent, extent), (-extent, extent), (0., 2*dl))),
                    grid=mw.GridSpec.uniform(dl), boundary=mw.BoundarySpec.faces(default="pec", z="pmc"),
                    device="cpu", structures=[mw.Structure(geometry=mesh, material=material)])


def _matrix(c):
    return sp.csr_matrix((c.k_vals, (c.k_rows, c.k_cols)), shape=(len(c.h_area), len(c.e_volume)))


def _radius(c):
    K = _matrix(c)[:, ~c.e_fixed]
    scaled = sp.diags(np.sqrt(c.h_dual / (MU0*c.h_area))) @ K @ sp.diags(1/np.sqrt(EPS0*c.e_volume[~c.e_fixed]))
    A = scaled.T @ scaled
    return float(eigsh(A, k=1, which="LA", v0=np.random.default_rng(0).standard_normal(A.shape[0]),
                       tol=1e-10, return_eigenvectors=False)[0])


@pytest.mark.parametrize("scene_name", ["board", "plate", "prism"])
def test_explicit_chord_bound_equals_pec(scene_name):
    complexes = []
    for kind in ("pec", "copper"):
        material = h.pec_material() if kind == "pec" else mw.CopperFoil(thickness=35e-6, frequency_range=(.3e9, 4e9))
        if scene_name == "prism":
            scene = prism_scene(material)
        else:
            normal = (.04, .025, 1.) if scene_name == "board" else rt.NORMAL
            kw = dict(sheet_model="conformal_two_sided") if kind == "pec" else {}
            structure = mw.Structure(geometry=h.rectangle_mesh(rt.CENTER, normal, .03, .025, **kw), material=material)
            scene = rt._cavity(structure)
        scene = prepare_scene(scene)
        complexes.append(compile_thin_sheet(scene, scene.structures[0]))
    pec, cu = complexes
    assert np.array_equal(pec.e_volume, cu.e_volume)
    assert np.array_equal(pec.h_area, cu.h_area)
    for suffix in ("rows", "cols", "vals"):
        assert np.array_equal(getattr(pec, "k_"+suffix), getattr(cu, "k_x_"+suffix))
    radii = [_radius(c) for c in complexes]
    ratio = math.sqrt(radii[0]/radii[1])
    print(dict(scene=scene_name, pec_radius=radii[0], copper_explicit_radius=radii[1],
               pec_bound=2/math.sqrt(radii[0]), dt_relative_to_pec=ratio, added_newmark_dispersion=0.))
    assert abs(ratio - 1) <= 1e-12


@pytest.mark.parametrize("side_cells", [10, 30])
def test_prism_held_chord_modes_equal_pec(side_cells):
    operators = []
    for kind in ("pec", "copper"):
        material = h.pec_material() if kind == "pec" else mw.CopperFoil(thickness=35e-6, frequency_range=(.3e9, 2e9))
        scene = prepare_scene(prism_scene(material, side_cells))
        c = compile_thin_sheet(scene, scene.structures[0])
        keep = np.flatnonzero(~c.e_fixed & (c.e_side == -1))
        K = _matrix(c)[:, keep]
        inv_mass = 1/np.sqrt(EPS0*c.e_volume[keep])
        A = (K.T @ sp.diags(c.h_dual/(MU0*c.h_area)) @ K).toarray() * inv_mass[:, None]*inv_mass[None]
        operators.append(A)
    assert np.array_equal(*operators)
    vals, vecs = np.linalg.eigh(operators[0])
    target = (math.pi/(side_cells*.005))**2/(EPS0*MU0)
    positive = vals > 1e-8*target
    freq = np.sqrt(vals[positive]/target)
    hz = (vecs[c.e_component[keep] == 2][:, positive]**2).sum(0) < .5
    assert freq.min() >= .5
    print(dict(side_cells=side_cells, first_hz_modes=freq[hz][:3].tolist(),
               modes_below_half=int((freq < .5).sum()), added_newmark_dispersion=0.))


@pytest.mark.parametrize("scene_name", ["plate", "prism"])
@pytest.mark.parametrize("material_name,condition_gate", [("copper", 1.05), ("resistive", 6.), ("transparent", 12.)])
def test_chord_implicit_inverse(scene_name, material_name, condition_gate, monkeypatch):
    material = (h.resistive_foil(.5/h.ETA0, band=(.3e9, 4e9)) if material_name == "resistive"
                else mw.CopperFoil(thickness=35e-6, frequency_range=(.3e9, 4e9)))
    if material_name == "transparent":
        from types import SimpleNamespace
        from witwin.maxwell.compiler import copper_foil
        response = SimpleNamespace(rates=torch.zeros(0, dtype=torch.float64), state_space=SimpleNamespace(
            B=torch.zeros((0, 2), dtype=torch.float64), D=torch.zeros((2, 2), dtype=torch.float64)))
        monkeypatch.setattr(copper_foil, "compile_copper_response", lambda material: response)
    scene = rt._cavity(rt._plate(material)) if scene_name == "plate" else prism_scene(material)
    dt = rt.pec_bound_dt(scene)
    solver = mw.Simulation.fdtd(scene, frequencies=[1e9], run_time=mw.TimeConfig(time_steps=1, dt=dt),
                                backend="cpu_reference").prepare().solver
    sheet = solver._thin_sheets[0]
    cu = sheet.copper
    S = cu["S"].to_dense()
    eig = torch.linalg.eigvalsh(S)
    assert float(eig[0]) > 0
    low, high = cu["spectrum"]
    assert high/low <= condition_gate
    generator = torch.Generator().manual_seed(19)
    vectors = torch.randn((len(S), 20), dtype=torch.float64, generator=generator)
    exact = torch.linalg.solve(S, vectors)
    approximate = torch.stack([sheet.chord_solve(v) for v in vectors.T], 1)
    error = float(torch.linalg.vector_norm(approximate-exact)/torch.linalg.vector_norm(exact))
    defect = (vectors*(exact-approximate)).sum(0)
    scale = (vectors*exact).sum(0)
    step_errors = []
    polynomial_solve = sheet.chord_solve

    def checked_increment(rhs):
        reference = torch.linalg.solve(S, rhs)
        increment = polynomial_solve(rhs)
        step_errors.append(float(torch.linalg.vector_norm(increment-reference)/torch.linalg.vector_norm(reference)))
        return increment

    monkeypatch.setattr(sheet, "chord_solve", checked_increment)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        field = getattr(solver, name)
        field.copy_(torch.randn(field.shape, generator=generator, dtype=field.dtype))
    sheet.begin(solver)
    sheet.magnetic(solver)
    sheet.electric(solver)
    print(dict(scene=scene_name, material=material_name, min_eigenvalue=float(eig[0]), condition=high/low,
               increment_error=error, step_increment_error=step_errors[0], min_inverse_defect=float((defect/scale).min()),
               degree=cu["chebyshev"][2], added_newmark_phase_rad_per_step=rt.newmark_phase_error(dt, 1e9)))
    assert max(error, step_errors[0]) <= 1e-10
    # The physical CN matrix has a stiff odd thickness mode; dense inversion
    # loses condition(S)*eps in this quadratic-form comparison.
    roundoff = 10*torch.finfo(torch.float64).eps*float(eig[-1]/eig[0])
    assert float((defect/scale).min()) >= -roundoff
