"""Homogeneous MoM refuses, at Simulation.mom(...) construction, declarations it would ignore."""
import pytest
import torch

import witwin.core as core
import witwin.maxwell as mw

pytestmark = pytest.mark.host_validation

FREQUENCY = 1e9


def _plane_wave(**kwargs):
    return mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0), source_time=mw.CW(frequency=FREQUENCY), **kwargs)


def _scene(material=None, *, sources=None, **kwargs):
    material = mw.Material(eps_r=4.0) if material is None else material
    return mw.Scene(structures=[mw.Structure(name="body", geometry=mw.Sphere(radius=0.05), material=material)],
                    sources=[_plane_wave()] if sources is None else sources, **kwargs)


def _refused(scene, *fragments):
    with pytest.raises(mw.UnsupportedFeatureError, match=r"Simulation\.mom\(\.\.\.\) cannot represent") as error:
        mw.Simulation.mom(scene, frequencies=FREQUENCY)
    for fragment in fragments:
        assert fragment in str(error.value), (fragment, str(error.value))


def test_supported_scene_still_constructs():
    mw.Simulation.mom(_scene(), frequencies=FREQUENCY)


def test_pmc_body_is_refused_instead_of_solved_as_vacuum():
    _refused(_scene(mw.Material(pmc=True)), "Structure 'body': Material(pmc=True)")


@pytest.mark.parametrize("fields,expected", [
    (dict(conductor_model="perfect"), "conductor_model"),
    (dict(sigma_e=0.01, dispersion=core.PowerLawDispersion(reference_frequency_hz=1e8, eps_r_exponent=-0.5,
                                                           sigma_e_exponent=1.0)), "dispersion"),
    (dict(layers=(core.MaterialLayer(thickness_m=0.01, eps_r=2.0),), thickness_m=0.02), "thickness_m, layers"),
    (dict(geometry_mode="volumetric"), "geometry_mode"),
])
def test_core_surface_fields_are_refused(fields, expected):
    _refused(_scene(core.PhysicalMaterial(eps_r=4.0, **fields)), f"core material field(s) {expected}")


def test_breakdown_is_refused():
    breakdown = mw.breakdown.DielectricBreakdown(critical_field=1e6, post_breakdown_conductivity=1.0)
    _refused(_scene(mw.Material(eps_r=4.0, breakdown=breakdown)),
             "Structure 'body': DielectricBreakdown is a time-domain feedback model")


def test_internal_absorbers_are_refused():
    absorber = mw.fdtd.InternalAbsorber(position=(0, 0, 0.5), size=(float("inf"), float("inf"), 0))
    _refused(_scene(internal_absorbers=[absorber]), "Scene.internal_absorbers (1)")


def test_tfsf_and_injection_axis_are_refused():
    tfsf = mw.TFSF(bounds=((-0.1, 0.1),) * 3)
    _refused(_scene(sources=[_plane_wave(injection=tfsf)]), "Source #0: injection=TFSF(...)")
    _refused(_scene(sources=[_plane_wave(injection_axis="z")]), "Source #0: injection_axis")


@pytest.mark.parametrize("material_type", [mw.Material, core.PhysicalMaterial])
def test_trainable_mu_r_is_refused_instead_of_detached(material_type):
    mu_r = torch.tensor(2.0, requires_grad=True)
    _refused(_scene(material_type(eps_r=4.0, mu_r=mu_r)), "Structure 'body': trainable mu_r")


def test_trainable_maxwell_plane_modes_are_refused():
    k = 2 * torch.pi * FREQUENCY / 299792458.0
    electric = torch.tensor([[1.0 + 0j, 0, 0]], dtype=torch.complex128, requires_grad=True)
    modes = mw.mom.MaxwellPlaneModes(FREQUENCY, torch.tensor([[0, 0, k + 0j]], dtype=torch.complex128),
                                 electric, torch.tensor([[0, 1.0 / 376.73 + 0j, 0]], dtype=torch.complex128))
    scene = mw.Scene(background=mw.Material(),
                     structures=[mw.Structure(geometry=mw.Sphere(radius=0.05), material=mw.Material(eps_r=4.0))],
                     sources=[modes])
    _refused(scene, "Source #0: trainable MaxwellPlaneModes tensors")


def test_one_error_lists_every_item():
    absorber = mw.fdtd.InternalAbsorber(position=(0, 0, 0.5), size=(float("inf"), float("inf"), 0))
    tfsf = mw.TFSF(bounds=((-0.1, 0.1),) * 3)
    _refused(_scene(mw.Material(pmc=True), sources=[_plane_wave(injection=tfsf)], internal_absorbers=[absorber]),
             "Material(pmc=True)", "Scene.internal_absorbers (1)", "injection=TFSF(...)")
