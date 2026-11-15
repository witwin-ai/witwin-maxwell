"""Construction-time refusals for inputs no solver consumes or differentiates (0.7.0 p2e).

Scene constructors that store a value as a Python float refuse a grad-carrying tensor
instead of detaching it, and `Structure.phase_screen`/`uv` and `MaterialRegion.symmetry`
are refused instead of being ignored. Fixed inputs keep their stored types and values.
"""
import struct

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import PhaseScreen


def _leaf(value):
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _region(**kwargs):
    return mw.MaterialRegion(
        name="r", geometry=mw.Box(position=(0, 0, 0), size=(0.3, 0.3, 0.3)),
        density=torch.ones(2, 2, 2), **kwargs,
    )


# (label, constructor taking the trial value, refused name)
FLOAT_FIELDS = (
    ("kerr_chi3", lambda v: mw.Material(eps_r=2.0, kerr_chi3=v), 1e-3, "Material.kerr_chi3"),
    ("chi2", lambda v: mw.media.NonlinearSusceptibility(chi2=v), 1e-12, "NonlinearSusceptibility.chi2"),
    ("chi3", lambda v: mw.media.NonlinearSusceptibility(chi3=v), 1e-20, "NonlinearSusceptibility.chi3"),
    ("tpa_beta", lambda v: mw.media.TwoPhotonAbsorption(beta=v), 1e-11, "TwoPhotonAbsorption.beta"),
    ("tpa_n0", lambda v: mw.media.TwoPhotonAbsorption(beta=1e-11, n0=v), 1.4, "TwoPhotonAbsorption.n0"),
    ("diag_xx", lambda v: mw.DiagonalTensor3(v, 2.0, 3.0), 2.5, "DiagonalTensor3 entries"),
    ("diag_zz", lambda v: mw.DiagonalTensor3(2.0, 3.0, v), 2.5, "DiagonalTensor3 entries"),
    ("tensor_entry", lambda v: mw.media.Tensor3x3(((v, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))), 2.5, "Tensor3x3 rows"),
    ("sigma_s", lambda v: mw.media.Medium2D(sigma_s=v), 0.01, "Medium2D.sigma_s"),
    ("wire_conductivity", lambda v: mw.rf.WireConductor.finite(v), 5.8e7, "WireConductor.conductivity"),
    ("wire_permeability", lambda v: mw.rf.WireConductor.finite(5.8e7, permeability=v), 1.3e-6, "WireConductor.permeability"),
    ("region_bounds", lambda v: _region(bounds=(0.0, v)), 1.0, "MaterialRegion.bounds"),
    ("region_eps_bounds", lambda v: _region(eps_bounds=(1.0, v)), 3.0, "MaterialRegion.eps_bounds"),
    ("region_mu_bounds", lambda v: _region(mu_bounds=(1.0, v)), 2.0, "MaterialRegion.mu_bounds"),
    ("region_filter_radius", lambda v: _region(filter_radius=v), 0.05, "MaterialRegion.filter_radius"),
    ("region_projection_beta", lambda v: _region(projection_beta=v), 4.0, "MaterialRegion.projection_beta"),
)
_IDS = [case[0] for case in FLOAT_FIELDS]


@pytest.mark.parametrize("label,ctor,value,name", FLOAT_FIELDS, ids=_IDS)
@pytest.mark.parametrize("form", ("leaf", "derived"))
def test_trainable_float_field_is_refused_not_detached(label, ctor, value, name, form):
    leaf = _leaf(value)
    trial = leaf if form == "leaf" else leaf * 1.0
    with pytest.raises(NotImplementedError, match=f"Trainable {name} is not supported"):
        ctor(trial)


def test_whole_trainable_tensor3x3_is_refused():
    rows = torch.eye(3, dtype=torch.float64).mul(2.0).requires_grad_(True)
    with pytest.raises(NotImplementedError, match="Trainable Tensor3x3 rows is not supported"):
        mw.media.Tensor3x3(rows)


def _bits(value):
    """Exact stored representation: the type and the IEEE-754 bit pattern of each float."""
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_bits(item) for item in value))
    if value is None:
        return ("none",)
    raise AssertionError(f"unexpected stored type {type(value).__name__}")


_STORED = {
    "kerr_chi3": lambda o: o.kerr_chi3,
    "chi2": lambda o: (o.chi2, o.chi3),
    "chi3": lambda o: (o.chi2, o.chi3),
    "tpa_beta": lambda o: (o.beta, o.n0),
    "tpa_n0": lambda o: (o.beta, o.n0),
    "diag_xx": lambda o: o.as_tuple(),
    "diag_zz": lambda o: o.as_tuple(),
    "tensor_entry": lambda o: o.rows,
    "sigma_s": lambda o: o.sigma_s,
    "wire_conductivity": lambda o: (o.conductivity, o.permeability),
    "wire_permeability": lambda o: (o.conductivity, o.permeability),
}
for _label in ("region_bounds", "region_eps_bounds", "region_mu_bounds", "region_filter_radius", "region_projection_beta"):
    _STORED[_label] = lambda o: (o.bounds, o.eps_bounds, o.mu_bounds, o.filter_radius, o.projection_beta)


@pytest.mark.parametrize("label,ctor,value,name", FLOAT_FIELDS, ids=_IDS)
def test_fixed_inputs_keep_python_float_storage_and_values(label, ctor, value, name):
    """A Python float, a fixed float64 tensor and a fixed float32 tensor store exactly float(x)."""
    reference = _bits(_STORED[label](ctor(float(value))))
    assert _bits(_STORED[label](ctor(torch.tensor(value, dtype=torch.float64)))) == reference
    fp32 = torch.tensor(value, dtype=torch.float32)
    expected32 = _bits(_STORED[label](ctor(float(fp32))))
    assert _bits(_STORED[label](ctor(fp32))) == expected32
    # A detached copy of a leaf is fixed and stores the same float as the Python value.
    assert _bits(_STORED[label](ctor(_leaf(value).detach()))) == reference


def test_fixed_tensor3x3_and_kerr_storage_types():
    rows = torch.eye(3, dtype=torch.float64) * 2.0
    stored = mw.media.Tensor3x3(rows).rows
    assert all(type(item) is float for row in stored for item in row)
    assert stored == mw.media.Tensor3x3(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))).rows
    assert mw.Material(eps_r=2.0).kerr_chi3 is None


@pytest.mark.parametrize("symmetry", ("x", "mirror_x", "anything-goes", ""))
def test_material_region_symmetry_is_refused_until_implemented(symmetry):
    with pytest.raises(NotImplementedError, match="MaterialRegion.symmetry is not consumed"):
        _region(symmetry=symmetry)


def test_material_region_without_symmetry_is_unchanged():
    region = _region(eps_bounds=(1.0, 3.0), filter_radius=0.05, projection_beta=4.0)
    assert region.symmetry is None
    assert (region.bounds, region.eps_bounds, region.mu_bounds) == ((0.0, 1.0), (1.0, 3.0), (1.0, 1.0))
    assert (region.filter_radius, region.projection_beta) == (0.05, 4.0)


def _scene(**structure_kwargs):
    geometry = structure_kwargs.pop("geometry", mw.Box(position=(0, 0, 0), size=(0.36, 0.36, 0.36)))
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.6, 0.6),) * 3), grid=mw.GridSpec.uniform(0.12),
                boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                structures=(
            mw.Structure(name="slab", geometry=geometry, material=mw.Material(eps_r=3.0), **structure_kwargs),
            ),
            )
    return scene


def _screen():
    return PhaseScreen(height=torch.rand(4, 4), height_scale_m=0.05)


def _uv_kwargs():
    vertices = torch.tensor(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2)))
    faces = torch.tensor(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    uv = torch.tensor(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    face_uv = torch.tensor(((0, 1, 2),) * 4)
    return dict(geometry=mw.Mesh(vertices, faces), uv=uv, face_uv=face_uv)


_SOLVERS = {
    "fdtd": lambda scene: mw.Simulation.fdtd(scene, frequencies=(1e9,)),
    "fdfd": lambda scene: mw.Simulation.fdfd(scene, frequencies=1e9),
    "mom": lambda scene: mw.Simulation.mom(scene, frequencies=1e9),
    "electrostatic": lambda scene: mw.Simulation.electrostatic(scene),
    "capacitance": lambda scene: mw.Simulation.capacitance(scene),
}


@pytest.mark.parametrize("solver", tuple(_SOLVERS))
@pytest.mark.parametrize("assignment", ("phase_screen", "uv"))
def test_optical_structure_assignment_is_refused_by_every_solver(solver, assignment):
    kwargs = dict(phase_screen=_screen()) if assignment == "phase_screen" else _uv_kwargs()
    with pytest.raises(NotImplementedError, match="optical phase_screen or uv/face_uv assignment"):
        _SOLVERS[solver](_scene(**kwargs))


def test_optical_assignment_added_by_update_is_refused():
    scene = _scene()
    scene = scene.updated(structures=(
        *scene.structures,
        mw.Structure(name="screened", geometry=mw.Box(position=(0, 0, 0.3), size=(0.1, 0.1, 0.1)),
                     material=mw.Material(eps_r=2.0), phase_screen=_screen()),
    ))
    with pytest.raises(NotImplementedError, match="optical phase_screen or uv/face_uv assignment"):
        mw.Simulation.fdtd(scene, frequencies=(1e9,), config=mw.FDTDConfig(device="cpu")).prepare()
