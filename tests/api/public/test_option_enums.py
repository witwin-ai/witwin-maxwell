"""String-valued public options are ``str`` Enums parsed once at construction (0.7.0, Q2/Q11)."""
from __future__ import annotations

import hashlib
from enum import Enum

import pytest

import witwin.maxwell as mw
from witwin.maxwell.errors import SetupError

# (namespace, Enum name, member values) of every option Enum, in its single public home.
OPTION_ENUMS = (
    (mw, "BoundaryKind", ("none", "pml", "periodic", "bloch", "pec", "pmc", "mur", "mode_abc", "mixed")),
    (mw, "SimulationMethod", ("fdfd", "fdtd", "mom", "electrostatic", "capacitance")),
    (mw.monitors, "FieldComponent", ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")),
    (mw.monitors, "Direction", ("+", "-")),
    (mw.monitors, "ModePolarization", ("auto", "Ex", "Ey", "Ez")),
    (mw.sources, "DipoleProfile", ("gaussian", "ideal")),
    (mw.grid, "SubpixelAveraging", ("arithmetic", "polarized", "polarized_tensor", "kottke_edge", "kottke_smooth")),
    (mw.grid, "PECTreatment", ("staircase", "conformal")),
    (mw.fdfd, "FDFDSolverType", ("gmres", "cg", "bicgstab", "tfqmr", "idr", "sqmr", "direct")),
    (mw.fdfd, "FDFDPreconditioner", ("jacobi", "ssor", "ilu", "ams", "none")),
    (mw.fdfd, "Precision", ("single", "double")),
    (mw.fdtd, "AbsorberKind", ("auto", "cpml", "pml", "absorber", "stablepml")),
    (mw.fdtd, "SpectralWindowKind", ("none", "hanning", "ramp")),
    (mw.fdtd, "CPMLMemoryMode", ("auto", "dense", "slab")),
    (mw.sar, "AveragingProfile", ("cubical-prefix-v1",)),
)


@pytest.mark.parametrize("namespace, name, values", OPTION_ENUMS, ids=[entry[1] for entry in OPTION_ENUMS])
def test_option_enum_is_a_str_enum_in_its_namespace(namespace, name, values):
    enum_type = getattr(namespace, name)
    assert name in namespace.__all__
    assert issubclass(enum_type, str) and issubclass(enum_type, Enum)
    assert tuple(member.value for member in enum_type) == values
    for member in enum_type:
        assert member == member.value and str(member) == member.value and f"{member}" == member.value
        assert hash(member) == hash(member.value) and repr(member) == repr(member.value)


def test_new_option_enums_stay_off_the_top_level():
    for name in ("FieldComponent", "Direction", "ModePolarization", "DipoleProfile", "SubpixelAveraging",
                 "PECTreatment", "FDFDSolverType", "FDFDPreconditioner", "Precision", "AveragingProfile"):
        assert name not in mw.__all__
    assert "AVERAGING_PROFILE" not in mw.sar.__all__


# (constructor, keyword, exact value string, expected stored member, rejected spellings)
def _cases():
    Monitors, Fdfd = mw.monitors, mw.fdfd
    return (
        (lambda v: mw.BoundarySpec(kind=v, num_layers=4).kind, "pml", mw.BoundaryKind.PML, ("PML", " pml", "mixed")),
        (lambda v: mw.BoundarySpec.faces(num_layers=4, x_low=v).face_kind("x", "low"), "pml",
         mw.BoundaryKind.PML, ("Pml",)),
        (lambda v: mw.SubpixelSpec(averaging=v).averaging, "polarized", mw.grid.SubpixelAveraging.POLARIZED,
         ("Polarized", "polarized ")),
        (lambda v: mw.SubpixelSpec(pec=v).pec, "staircase", mw.grid.PECTreatment.STAIRCASE, ("STAIRCASE",)),
        (lambda v: mw.PointDipole((0, 0, 0), profile=v).profile, "ideal", mw.sources.DipoleProfile.IDEAL,
         ("IDEAL",)),
        (lambda v: mw.PointMonitor("p", (0, 0, 0), fields=(v,)).fields[0], "Hy", Monitors.FieldComponent.HY,
         ("hy", "HY", "Hy ")),
        (lambda v: mw.PlaneMonitor("p", normal_direction=v).normal_direction, "-", Monitors.Direction.MINUS,
         (" -", "minus")),
        (lambda v: mw.ModeSource(size=(1, 1, 0), direction=v).direction, "-", Monitors.Direction.MINUS, (" - ",)),
        (lambda v: mw.fdtd.ModeABC(polarization=v).polarization, "Ey", Monitors.ModePolarization.EY,
         ("ey", "y", "AUTO")),
        (lambda v: mw.rf.WaveModeSpec("te", polarization=v).polarization, "Ex", Monitors.ModePolarization.EX,
         ("ex", "x")),
        (lambda v: Fdfd.FDFDSolverConfig(solver_type=v).solver_type, "direct", Fdfd.FDFDSolverType.DIRECT,
         ("DIRECT",)),
        (lambda v: Fdfd.FDFDSolverConfig(preconditioner=v).preconditioner, "ssor", Fdfd.FDFDPreconditioner.SSOR,
         ("SSOR",)),
        (lambda v: Fdfd.FDFDSolverConfig(precision=v).precision, "double", Fdfd.Precision.DOUBLE,
         ("Double", "float64")),
        (lambda v: Fdfd.FieldProjection("p", component=v, position=(0, 0, 0)).component, "Hz",
         Monitors.FieldComponent.HZ, ("hz",)),
        (lambda v: mw.FDTDConfig(absorber=v).absorber, "stablepml", mw.fdtd.AbsorberKind.STABLE_PML,
         ("CPML", " cpml")),
        (lambda v: mw.fdtd.SpectralSampler(window=v).window, "ramp", mw.fdtd.SpectralWindowKind.RAMP,
         ("Hanning",)),
        (lambda v: mw.sar.SARAveraging(1e-3, profile=v).profile, "cubical-prefix-v1",
         mw.sar.AveragingProfile.CUBICAL_PREFIX_V1, ("CUBICAL-PREFIX-V1",)),
    )


@pytest.mark.parametrize("index", range(17))
def test_exact_value_string_is_parsed_once_and_other_spellings_raise(index):
    build, exact, member, rejected = _cases()[index]
    stored = build(exact)
    assert stored is member
    assert build(member) is member
    for spelling in rejected:
        with pytest.raises(SetupError):
            build(spelling)


def test_point_dipole_polarization_takes_an_electric_component():
    FieldComponent = mw.monitors.FieldComponent
    assert mw.PointDipole((0, 0, 0), polarization=FieldComponent.EY).polarization == (0.0, 1.0, 0.0)
    assert mw.PointDipole((0, 0, 0), polarization="Ex").polarization == (1.0, 0.0, 0.0)
    for spelling in ("ez", "EZ", "Hx"):
        with pytest.raises(SetupError):
            mw.PointDipole((0, 0, 0), polarization=spelling)


def test_mode_polarization_members_and_vectors_select_the_axis():
    ModePolarization = mw.monitors.ModePolarization
    assert mw.ModeSource(size=(1, 2, 0)).polarization_axis == "y"
    assert mw.ModeSource(size=(1, 2, 0), polarization=ModePolarization.EX).polarization_axis == "x"
    assert mw.ModeMonitor("m", size=(1, 2, 0), polarization="Ex").polarization_axis == "x"
    assert mw.ModePort("p", size=(1, 2, 0), polarization=(0.0, 1.0, 0.0)).polarization_axis == "y"
    with pytest.raises(ValueError, match="tangential"):
        mw.ModeSource(size=(1, 2, 0), polarization="Ez")
    with pytest.raises(SetupError):
        mw.rf.ModalPortSweep(modes={"left": (("auto", 0),)})


def test_boundary_spec_stores_members_and_reports_mixed():
    spec = mw.BoundarySpec.faces(default="pml", num_layers=4, z="periodic")
    assert spec.kind is mw.BoundaryKind.MIXED
    assert spec.axis_kind("z") is mw.BoundaryKind.PERIODIC
    assert all(kind is mw.BoundaryKind.PML for kind in spec.axis_face_kinds("x"))
    assert spec.uses_kind("periodic") and not spec.uses_kind(mw.BoundaryKind.BLOCH)


def test_resume_fingerprint_hashes_an_enum_as_its_value_string():
    from witwin.maxwell.fdtd.resume import _hash_value

    def digest(value):
        hasher = hashlib.sha256()
        _hash_value(hasher, "v", value)
        return hasher.hexdigest()

    assert digest(mw.BoundaryKind.PML) == digest("pml")
    assert digest({mw.monitors.FieldComponent.EZ: (mw.monitors.Direction.PLUS,)}) == digest({"Ez": ("+",)})


def test_scene_dict_writes_option_enums_as_plain_strings():
    from witwin.maxwell.io.scene_dict import scene_to_dict

    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
                grid=mw.GridSpec.uniform(0.1),
                boundary=mw.BoundarySpec.pml(num_layers=4),
                monitors=(mw.PlaneMonitor("p", fields=("Ex", "Hy"), normal_direction="-"),),
            )

    def strings(value):
        if isinstance(value, dict):
            return [item for entry in value.values() for item in strings(entry)]
        if isinstance(value, list):
            return [item for entry in value for item in strings(entry)]
        return [value] if isinstance(value, str) else []

    assert all(type(item) is str for item in strings(scene_to_dict(scene)))
