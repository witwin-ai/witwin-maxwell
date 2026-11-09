"""Physical dual-face material quadrature with the existing center metric Hodge."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import PreparedScene
from witwin.maxwell.compiler.materials import _compile_edge_component, _mapped_dual_face_weights


class SharpBox(mw.Box):
    """Isolate geometric quadrature from the existing Box's smooth SDF profile."""
    def to_mask(self,x,y,z,*,offset=0.,beta=None):
        return (self.signed_distance(x,y,z)<=offset).to(x.dtype)


def layered(curvature,*,eps=4.,sigma=.02,grid=None,sharp=True):
    domain=mw.Domain(bounds=((-0.03,.03),(-.03,.03),(-.03,.03)))
    scene=mw.Scene(domain=domain,grid=mw.GridSpec.uniform(.01) if grid is None else grid,
        coordinates=None if curvature is None else mw.BendCoordinates(curvature=curvature),device='cpu',
        structures=(mw.Structure(name='upper',geometry=(SharpBox if sharp else mw.Box)(
            position=torch.tensor((0.,.015,0.),dtype=torch.float64),size=torch.tensor((.2,.03,.2),dtype=torch.float64)),
            material=mw.Material(eps_r=eps,sigma_e=sigma)),),
        subpixel_samples=mw.SubpixelSpec(samples=(2,2,2),averaging='arithmetic'))
    return PreparedScene(scene)


def sample(scene,component='Ez',*,samples=(2,2,2),averaging='arithmetic',selection=None):
    return _compile_edge_component(scene,component,eps_background=1.,mu_background=1.,averaging=averaging,
        samples=samples,region_densities=(),geometry_periodic_shifts={},selection=selection)


@pytest.mark.parametrize('curvature',(-8.,-3.,3.,8.))
def test_radial_dual_face_layer_integral_matches_positive_area_quadrature(curvature):
    scene=layered(curvature)
    model=scene.compile_materials()['edge_components']
    # At u=0 the E_z physical dual face has area h_s(u) ds du.
    # Midpoint samples integrate both constant layers times the linear h exactly.
    expected_eps=.5*(1+4)+curvature*.01*(1-4)/8
    expected_sigma=.5*.02-curvature*.01*.02/8
    assert float(model['eps']['Ez'][3,3,2])==pytest.approx(expected_eps,abs=3e-7)
    assert float(model['sigma_e']['Ez'][3,3,2])==pytest.approx(expected_sigma,abs=3e-9)
    # The E_s dual face du dz carries no varying s scale factor.
    assert float(model['eps']['Ex'][2,3,3])==pytest.approx(2.5,abs=3e-7)
    assert torch.isfinite(model['eps']['Ez']).all()
    assert (model['eps']['Ez']>0).all()
    assert (model['sigma_e']['Ez']>=0).all()


def test_cartesian_and_zero_curvature_retain_bitwise_identical_coefficients():
    cartesian=layered(None).compile_materials()
    zero=layered(0.).compile_materials()
    for family in ('eps','mu','sigma_e','sigma_m'):
        for component,values in cartesian['edge_components'][family].items():
            torch.testing.assert_close(zero['edge_components'][family][component],values,rtol=0,atol=0)
    for family in ('eps_components','mu_components','sigma_e_components','sigma_m_components'):
        for axis,values in cartesian[family].items():
            torch.testing.assert_close(zero[family][axis],values,rtol=0,atol=0)


@pytest.mark.parametrize('curvature',(-5.,5.))
def test_dual_face_weighted_eps_and_sigma_vjp_matches_independent_difference_steps(curvature):
    eps=torch.tensor(4.,dtype=torch.float64,requires_grad=True)
    sigma=torch.tensor(.02,dtype=torch.float64,requires_grad=True)
    values,conductivity=sample(layered(curvature,eps=eps,sigma=sigma))
    objective=values[3,3,2]+20*conductivity[3,3,2]
    actual=torch.autograd.grad(objective,(eps,sigma))
    fraction=.5-curvature*.01/8
    torch.testing.assert_close(actual[0],torch.tensor(fraction,dtype=torch.float64),rtol=0,atol=2e-7)
    torch.testing.assert_close(actual[1],torch.tensor(20*fraction,dtype=torch.float64),rtol=0,atol=2e-6)
    for parameter,steps in ((0,(.003,.01)),(1,(.0001,.0003))):
        for step in steps:
            outputs=[]
            for direction in (-1,1):
                arguments={'eps':4.,'sigma':.02};arguments[('eps','sigma')[parameter]]+=direction*step
                e,s=sample(layered(curvature,**arguments));outputs.append(float(e[3,3,2]+20*s[3,3,2]))
            difference=(outputs[1]-outputs[0])/(2*step)
            assert difference==pytest.approx(float(actual[parameter]),rel=4e-4)


def test_weighted_component_selection_retains_global_native_samples():
    scene=layered(5.)
    selected=(slice(1,5),slice(1,5),slice(1,4))
    for averaging in ('arithmetic','polarized'):
        complete=sample(scene,averaging=averaging)
        partial=sample(scene,averaging=averaging,selection=selected)
        for actual,expected in zip(partial,complete):
            torch.testing.assert_close(actual,expected[selected],rtol=0,atol=0)


def test_graded_dual_face_integral_uses_shifted_control_volume_center():
    x=np.linspace(-.03,.03,7);u=np.array((-.03,-.02,-.01,0.,.02,.025,.03))
    scene=layered(5.,grid=mw.GridSpec.custom(x,u,x))
    values,sigma=sample(scene,samples=(1,6,1))
    lower,upper=-.005,.01
    integral=lambda a,b:(b-a)-5*(b*b-a*a)/2
    denominator=integral(lower,upper)
    expected=(integral(lower,0)+4*integral(0,upper))/denominator
    expected_sigma=.02*integral(0,upper)/denominator
    assert float(values[3,3,2])==pytest.approx(expected,abs=7e-7)
    assert float(sigma[3,3,2])==pytest.approx(expected_sigma,abs=5e-9)


def test_metric_quad_rejects_nonpositive_support_before_coefficient_assembly():
    scene=layered(5.)
    with pytest.raises(ValueError,match='positive finite dual-face'):
        _mapped_dual_face_weights(scene,'Ez',torch.tensor((.21,),dtype=torch.float64),[0.,.01])


def test_regular_box_retains_existing_smooth_interface_profile():
    """Half-filled bend edge keeps the full metric curvature correction.

    Re-anchored 2026-10-29. The old expectation was 2.5+c*.01*(1-4)/8*tanh(5)
    = 2.4812517; the new one is 2.48125. The tanh(5) factor was an artefact of
    the former .05-cell tanh occupancy, whose sub-samples sat 2.5 tanh widths
    from the face and so attenuated the correction. The exact sub-sample
    coverage fill has no such attenuation, and the symmetric half-fill now
    carries the analytic correction exactly.
    """
    curvature=5.
    actual,_=sample(layered(curvature,sharp=False))
    expected=2.5+curvature*.01*(1-4)/8
    assert float(actual[3,3,2])==pytest.approx(expected,abs=3e-7)
