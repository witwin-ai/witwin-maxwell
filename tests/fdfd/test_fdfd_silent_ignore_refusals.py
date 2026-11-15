"""FDFD refuses, at Simulation.fdfd(...) construction, every scene feature it cannot represent.

Each case was previously accepted and silently dropped or approximated (unification
doc rows 1-9), or refused only after CUDA work in the solver (gyromagnetic, nonlinear,
full tensor, face kind and symmetry). All checks are CPU-visible.
"""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdfd_sources
from witwin.maxwell.scene import prepare_scene

F = 1.0e9
HEADER = r"Simulation\.fdfd\(\.\.\.\) cannot represent"


def _scene(*, source_time=None, **kwargs):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.1),
                     boundary=kwargs.pop("boundary", mw.BoundarySpec.pml(num_layers=4)), **kwargs,
                     sources=(
                         mw.PointDipole(position=(0, 0, 0), polarization="Ez",
                                        source_time=mw.CW(frequency=F) if source_time is None else source_time,
                                        name="d"),
                     ))
    return scene


def _box(material, name="target"):
    return mw.Structure(name=name, geometry=mw.Box(position=(0.2, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
                        material=material)


def _lumped_port(termination=None):
    return mw.LumpedPort("p1", positive=(0.3, 0, 0.1), negative=(0.3, 0, -0.1), voltage_path=mw.AxisPath("z"),
                         current_surface=mw.Box(position=(0.3, 0, 0), size=(0.1, 0.1, 0.0)),
                         reference_impedance=50.0, termination=termination)


def _surface_impedance():
    from tests.oracles.surface_impedance_reference import good_conductor_surface_impedance

    band = (0.5e9, 2.0e9)
    frequencies = torch.logspace(math.log10(band[0]), math.log10(band[1]), 120, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(50.0, frequencies)).to(torch.complex128)
    model = mw.media.RationalSurfaceImpedance.fit(frequencies, admittance, order=6, band=band)
    return mw.media.SurfaceImpedanceMedium(impedance=model, name="coating")


def _tilted_pec_sheet():
    vertices = torch.tensor([[-0.2, -0.2, -0.05], [0.2, -0.2, 0.05], [0.0, 0.2, 0.0]], dtype=torch.float64)
    return mw.Structure(name="sheet", geometry=mw.geometry.SurfaceMesh(vertices, torch.tensor([[0, 1, 2]])),
                        material=mw.Material(pec=True))


def _with_source(scene, source):
    scene = scene.updated(sources=(source, *scene.sources[1:]))
    return scene


def _ferrite():
    return mw.media.GyromagneticFerrite(eps_r=14.5, saturation_magnetization=1.40e5, bias_field=(0.0, 0.0, 1.75e5),
                                  gilbert_damping=2.0e-3)


def _full_tensor():
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
    tensor = rotation @ torch.diag(torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)) @ rotation.T
    return mw.Material(epsilon_tensor=mw.media.Tensor3x3(tensor.tolist()))


# (case id, scene builder, solve frequencies, message fragment)
REFUSED = (
    ("1-lossy-metal", lambda: _scene(structures=[_box(mw.media.LossyMetalMedium(conductivity=5.8e7))]), (F,),
     "no LossyMetalMedium conductor model and would solve it as vacuum"),
    ("1-surface-impedance", lambda: _scene(structures=[_box(_surface_impedance())]), (F,),
     "no SurfaceImpedanceMedium conductor model"),
    ("1-copper-foil", lambda: _scene(structures=[_box(mw.media.CopperFoil(thickness=35e-6, frequency_range=(0.5e9, 2e9)))]),
     (F,), "no CopperFoil conductor model"),
    ("2-modulation", lambda: _scene(structures=[_box(mw.Material(
        eps_r=2.0, modulation=mw.media.ModulationSpec(frequency=1e8, amplitude=0.1)))]), (F,),
     "time-modulated (ModulationSpec) material"),
    ("3-lumped-element", lambda: _scene(lumped_elements=[mw.rf.Resistor(
        "r1", positive=(0, 0, 0.1), negative=(0, 0, -0.1), resistance=50.0)]), (F,),
     "Lumped elements ('r1') are not stamped into the FDFD operator"),
    ("4-terminated-port", lambda: _scene(ports=[_lumped_port(mw.rf.SeriesRLC(r=50.0))]), (F,),
     "Port 'p1' (LumpedPort): the termination load is not stamped"),
    ("5-mode-port", lambda: _scene(ports=[mw.ModePort("mp", position=(0, 0, -0.3), size=(0.6, 0.6, 0.0))]), (F,),
     "Port 'mp' (ModePort): FDFD neither launches nor projects port modes"),
    ("5-wave-port", lambda: _scene(ports=[mw.rf.WavePort("wp", position=(0., 0., -0.3), size=(0.6, 0.6, 0.), direction="+",
                                                      reference_plane=-0.3, modes=(mw.rf.WaveModeSpec("te"),))]), (F,),
     "Port 'wp' (WavePort): FDFD neither launches nor projects port modes"),
    ("6-internal-absorber", lambda: _scene(internal_absorbers=[mw.fdtd.InternalAbsorber(
        position=(0, 0, 0.3), size=(1.0, 1.0, 0.0))]), (F,), "InternalAbsorber(s) are not applied by FDFD"),
    ("7-surface-mesh", lambda: _scene(structures=[_tilted_pec_sheet()]), (F,),
     "Structure 'sheet': FDFD has no zero-volume SurfaceMesh trace or thin-sheet operator"),
    ("8-conformal-pec", lambda: _scene(structures=[mw.Structure(
        geometry=mw.Sphere(position=(0.2, 0, 0), radius=0.12), material=mw.Material(pec=True))],
        subpixel=mw.SubpixelSpec(pec="conformal")), (F,), 'SubpixelSpec(pec="conformal")'),
    ("9-cw-frequency", lambda: _scene(source_time=mw.CW(frequency=3 * F)), (F,),
     "Source 'd' declares CW(frequency=3000000000) but FDFD solves at (1000000000.0,) Hz"),
    ("9-cw-frequency-sweep", lambda: _scene(), (F, 1.1 * F), "declares CW(frequency=1000000000)"),
    ("9-cw-uniform-current", lambda: _with_source(_scene(), mw.UniformCurrentSource(
        size=(0.2, 0.2, 0.2), source_time=mw.CW(frequency=2 * F), name="j")), (F,), "Source 'j' declares CW"),
    ("moved-gyromagnetic", lambda: _scene(structures=[_box(_ferrite())]), (F,),
     "FDFD does not support GyromagneticFerrite"),
    ("moved-kerr", lambda: _scene(structures=[_box(mw.Material(eps_r=2.0, kerr_chi3=1e-10))]), (F,),
     "FDFD does not support Kerr nonlinear media (chi3, chi2, TPA)"),
    ("moved-breakdown", lambda: _scene(structures=[_box(mw.Material(eps_r=2.0, breakdown=mw.breakdown.DielectricBreakdown(
        critical_field=1e6, post_breakdown_conductivity=1.0)))]), (F,),
     "Structure 'target': DielectricBreakdown is a time-domain feedback model"),
    ("moved-full-tensor", lambda: _scene(structures=[_box(_full_tensor())]), (F,),
     "FDFD does not support full off-diagonal Tensor3x3"),
    ("moved-face-kind", lambda: _scene(boundary=mw.BoundarySpec.periodic()), (F,),
     "Boundary faces ['periodic']: FDFD supports per-face 'none', 'pml' and 'pec'"),
    ("moved-symmetry", lambda: _scene(symmetry=("PMC", None, None)), (F,),
     "Scene(symmetry=...) is not supported by FDFD"),
)


@pytest.mark.parametrize("build, frequencies, fragment", [case[1:] for case in REFUSED],
                         ids=[case[0] for case in REFUSED])
def test_fdfd_refuses_scene_feature_at_construction(build, frequencies, fragment):
    with pytest.raises(mw.UnsupportedFeatureError, match=HEADER) as error:
        mw.Simulation.fdfd(build(), frequencies=list(frequencies), config=mw.FDFDConfig(device="cpu"))
    message = str(error.value)
    assert "cannot represent 1 scene feature(s)" in message
    assert fragment in message, message


def test_fdfd_lists_every_unsupported_feature_in_one_error():
    scene = _scene(source_time=mw.CW(frequency=2 * F), boundary=mw.BoundarySpec.periodic(),
                   structures=[_box(mw.media.LossyMetalMedium(conductivity=5.8e7), name="metal"), _tilted_pec_sheet()],
                   lumped_elements=[mw.rf.Resistor("r1", positive=(0, 0, 0.1), negative=(0, 0, -0.1), resistance=50.0)],
                   internal_absorbers=[mw.fdtd.InternalAbsorber(position=(0, 0, 0.3), size=(1.0, 1.0, 0.0))],
                   monitors=[mw.PointMonitor("pm", position=(0.1, 0, 0))])
    with pytest.raises(mw.UnsupportedFeatureError) as error:
        mw.Simulation.fdfd(scene, frequencies=F, config=mw.FDFDConfig(device="cpu"))
    message = str(error.value)
    assert "cannot represent 7 scene feature(s)" in message
    for fragment in ("Monitor 'pm' (PointMonitor)", "Structure 'metal'", "Structure 'sheet'", "'r1'",
                     "InternalAbsorber", "['periodic']", "Source 'd'"):
        assert fragment in message
    assert message.count("\n- ") == 7 == len(error.value.items)
    assert error.value.method == "fdfd" and error.value.alternative is None


def test_fdfd_refuses_lumped_elements_added_by_update():
    scene = _scene()
    scene = scene.updated(lumped_elements=(
        *scene.lumped_elements,
        mw.rf.Resistor("r1", positive=(0, 0, 0.1), negative=(0, 0, -0.1), resistance=50.0),
    ))
    for call in ("prepare", "run"):
        with pytest.raises(NotImplementedError, match="Lumped elements"):
            getattr(mw.Simulation.fdfd(scene, frequencies=F, config=mw.FDFDConfig(device="cpu")), call)()


@pytest.mark.parametrize("build, frequencies", [
    (lambda: _scene(), (F,)),
    (lambda: _scene(source_time=mw.CW(frequency=F * (1 + 1e-14))), (F,)),
    (lambda: _with_source(_scene(), mw.PointDipole(position=(0, 0, 0), polarization="Ez", name="d")), (F, 1.1 * F)),
    (lambda: _scene(ports=[_lumped_port()]), (F,)),
    (lambda: _scene(structures=[_box(mw.Material(eps_r=4.0, sigma_e=0.1)), _box(mw.Material(pec=True), name="pec"),
                                mw.Structure(name="off", geometry=_tilted_pec_sheet().geometry,
                                             material=mw.Material(pec=True), enabled=False)]), (F,)),
], ids=["dipole", "cw-within-1e-12", "source-time-none-sweep", "probe-lumped-port", "dielectric-pec-disabled-sheet"])
def test_fdfd_admits_supported_scenes(build, frequencies):
    simulation = mw.Simulation.fdfd(build(), frequencies=list(frequencies), config=mw.FDFDConfig(device="cpu"))
    with pytest.raises(ValueError, match="requires config.device to be CUDA"):
        simulation.prepare()


def test_fdfd_source_compiler_refuses_cw_frequency_mismatch():
    prepared = prepare_scene(_scene(), device="cpu")
    assert compile_fdfd_sources(prepared.scene, default_frequency=F)[0]["name"] == "d"
    with pytest.raises(ValueError, match=r"declares CW\(frequency=1000000000\) but is solved at 1500000000 Hz"):
        compile_fdfd_sources(prepared.scene, default_frequency=1.5 * F)
