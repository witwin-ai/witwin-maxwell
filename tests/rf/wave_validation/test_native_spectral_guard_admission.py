"""Additional negatives for proposed private spectral capability boundaries."""
from types import SimpleNamespace
import pytest
import torch
from tests.rf.wave_validation.test_native_product_public_prototype import _synthetic_copper_result
from witwin.maxwell.fdtd.plane_product import _require_uniform_nonmagnetic


def test_unknown_permeability_representation_requires_explicit_scalar_contract():
    scene=SimpleNamespace(structures=[SimpleNamespace(material=SimpleNamespace(mu_r=object()))],material_regions=[])
    with pytest.raises(NotImplementedError,match='explicit fixed scalar'):
        _require_uniform_nonmagnetic(scene)


def test_missing_retained_contact_edge_is_not_fabricated_by_native_flux():
    result=_synthetic_copper_result()
    trace=next(t for t in result._metadata['conductor_losses']['foil']['bulk_electric_traces'] if t['component']=='Ey')
    raw=result._monitors['surface']['components']['Ey'];normal=raw['plane_indices'][0]
    # Remove an actual compiled Cu support row belonging to the observed plane.
    missing=int(torch.nonzero(trace['indices'][:,0]==normal)[0,0])
    keep=torch.ones(len(trace['indices']),dtype=torch.bool);keep[missing]=False
    trace['indices']=trace['indices'][keep]
    trace['electric_fields']=trace['electric_fields'][...,keep]
    with pytest.raises(NotImplementedError,match='unresolved PEC-contact or missing side-trace'):
        result.monitor('surface')

@pytest.mark.parametrize('declaration',['diagonal_mu','full_mu','ferrite','magnetic_loss','magnetic_pole'])
def test_actual_public_magnetic_material_declarations_reject_single_normal_H_contract(declaration):
    import witwin.maxwell as mw
    if declaration=='diagonal_mu':material=mw.Material(mu_tensor=mw.DiagonalTensor3(1.,1.,2.))
    elif declaration=='full_mu':material=mw.Material(mu_tensor=mw.Tensor3x3(((2.,.1,0.),(.1,1.,0.),(0.,0.,1.))))
    elif declaration=='ferrite':material=mw.GyromagneticFerrite(saturation_magnetization=1.e5,bias_field=(0.,0.,8.e4))
    elif declaration=='magnetic_loss':material=mw.Material(sigma_m=.1)
    else:material=mw.Material(mu_lorentz_poles=(mw.LorentzPole(delta_eps=1.,resonance_frequency=1e9,gamma=.1e9),))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.005),device='cpu',
        structures=(mw.Structure(name='actual_magnetic',geometry=mw.Box(size=(.01,.01,.01)),material=material),))
    with pytest.raises(NotImplementedError,match='nonmagnetic|magnetic loss'):
        _require_uniform_nonmagnetic(scene)


def test_disabled_actual_magnetic_structure_does_not_change_compiled_background_contract():
    import witwin.maxwell as mw
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.005),device='cpu',
        structures=(mw.Structure(name='disabled_magnet',geometry=mw.Box(size=(.01,.01,.01)),
            material=mw.Material(mu_r=2.),enabled=False),))
    _require_uniform_nonmagnetic(scene)


@pytest.mark.parametrize('kind',['real_scalar','fixed_real_tensor','actual_material_region'])
def test_actual_scalar_unit_permeability_background_and_density_region_are_admitted(kind):
    import witwin.maxwell as mw
    material=mw.Material(mu_r=torch.tensor(1.,dtype=torch.float64) if kind=='fixed_real_tensor' else 1.)
    region=mw.MaterialRegion(name='dielectric_design',geometry=mw.Box(size=(.01,.01,.01)),
        density=torch.ones((2,2,2),requires_grad=True),eps_bounds=(1.,3.),mu_bounds=(1.,1.))
    scene=mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"), domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.005),device='cpu',
        structures=(mw.Structure(name='dielectric',geometry=mw.Box(size=(.01,.01,.01)),material=material),),
        material_regions=(region,) if kind=='actual_material_region' else ())
    _require_uniform_nonmagnetic(scene)
