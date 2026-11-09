"""CPU boundary-coefficient properties; no CPU field or pretend PDE advance."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import PreparedScene
from witwin.maxwell.constants import EPSILON_0,MU_0
from witwin.maxwell.fdtd.boundary.cpml import build_cpml_profiles,DEFAULT_CPML_CONFIG,_cpml_profile_1d
from witwin.maxwell.fdtd.boundary.common import BOUNDARY_PML


def compile_coefficients(nodes,thickness,config=None,shift_rate=None,shift_floor=0.):
    # Genuine Scene metadata and a coefficient-only compilation context. The
    # zero-length Ex carries dtype only; this object has no field/solve API.
    scene=PreparedScene(mw.Scene(domain=mw.Domain(bounds=((-0.2,.2),)*3),
      grid=mw.GridSpec.custom(nodes,nodes,nodes),boundary=mw.BoundarySpec.pml(num_layers=thickness),device='cpu'),lazy_meshgrid=True)
    context=SimpleNamespace(scene=scene,device='cpu',Ex=torch.empty(0,dtype=torch.float32),
      dt=8.666249213476036e-12,eps0=EPSILON_0,mu0=MU_0,
      cpml_config=dict(DEFAULT_CPML_CONFIG) if config is None else config,boundary_code=BOUNDARY_PML)
    build_cpml_profiles(context,magnetic_alpha_scale=1. if shift_rate is None else MU_0/EPSILON_0,
                        shift_rate=shift_rate,shift_floor=shift_floor)
    return scene,context


@pytest.mark.parametrize('thickness',[4,12,24])
@pytest.mark.parametrize('axis',['x','y','z'])
def test_real_builder_uses_mirror_symmetric_physical_H_halfpoints(axis,thickness):
    scene,profile=compile_coefficients(np.arange(-40,41,dtype=float)*.005,thickness)
    for letter in('e','h'):
      for kind in('kappa','b','c'):
        value=getattr(profile,f'cpml_{kind}_{letter}_{axis}')
        torch.testing.assert_close(value,value.flip(0),rtol=0,atol=0)


def test_nonuniform_actual_halfpoint_depth_matches_independent_conductivity():
    nodes=np.array([-.2,-.16,-.125,-.10,-.07,-.04,0.,.025,.06,.105,.15,.2])
    scene,context=compile_coefficients(nodes,3)
    x=scene.x_nodes64;half=scene.x_half64;dt=context.dt;m=DEFAULT_CPML_CONFIG['grading_order']
    # Reference physical depth and impedance matching are independent of the
    # builder's optional high-side sampling argument.
    for high in(False,True):
      interface=x[-4]if high else x[3];length=(x[-1]-interface)if high else(interface-x[0])
      depth=np.clip((half-interface)/length if high else(interface-half)/length,0,1)
      sigma_peak=-(m+1)*np.log(1e-6)/(2*np.sqrt(MU_0/EPSILON_0)*length)
      expected=np.exp(-sigma_peak*depth**m*dt/EPSILON_0)
      select=depth>0
      np.testing.assert_allclose(context.cpml_b_h_x.numpy()[select],expected[select],rtol=2e-7,atol=0)


def test_low_only_and_high_only_reverse_under_physical_mirroring():
    nodes=np.r_[-.26,-.24,-.23,-.215,-.2,-.18,-.12,0.,.12,.18,.2,.215,.23,.24,.26]
    half=(nodes[:-1]+nodes[1:])/2;scale=torch.tensor(MU_0/EPSILON_0,dtype=torch.float32)
    a=_cpml_profile_1d(half,nodes,4,8e-12,MU_0,scale,np.sqrt(MU_0/EPSILON_0),DEFAULT_CPML_CONFIG,apply_low=True,apply_high=False)
    b=_cpml_profile_1d(half,nodes,4,8e-12,MU_0,scale,np.sqrt(MU_0/EPSILON_0),DEFAULT_CPML_CONFIG,apply_low=False,apply_high=True)
    for left,right in zip(a,b):torch.testing.assert_close(left,right.flip(0),rtol=0,atol=0)


@pytest.mark.parametrize('uniform',[False,True])
@pytest.mark.parametrize('letter',['e','h'])
@pytest.mark.parametrize('high',[False,True])
def test_shifted_alpha_sigma_kappa_share_actual_Yee_coordinates(uniform,letter,high):
    nodes=np.linspace(-.2,.2,13) if uniform else np.array(
        [-.2,-.16,-.125,-.10,-.07,-.04,0.,.025,.06,.105,.15,.2])
    rate=2*np.pi*2.7e9
    config=dict(DEFAULT_CPML_CONFIG,alpha_max=EPSILON_0*rate,grading_order=2.,kappa_max=1.5)
    scene,context=compile_coefficients(nodes,3,config,rate,.25)
    x=scene.x_nodes64;positions=x if letter=='e' else scene.x_half64
    vacuum=EPSILON_0 if letter=='e' else MU_0
    strength=1. if letter=='e' else float(torch.tensor(MU_0/EPSILON_0,dtype=torch.float32))
    interface=x[-4] if high else x[3]
    length=x[-1]-interface if high else interface-x[0]
    depth=np.clip((positions-interface)/length if high else (interface-positions)/length,0,1)
    select=depth>0;graded=depth[select]**2
    sigma_peak=-3*np.log(config['reflection'])*strength/(2*np.sqrt(MU_0/EPSILON_0)*length)
    sigma=2*sigma_peak*graded
    alpha=vacuum*rate*(.25+.75*(1-graded))
    kappa=1+.5*graded
    u=(sigma/kappa+alpha)*context.dt/vacuum
    pole=(1-u/2+u*u/12)/(1+u/2+u*u/12)
    actual_b=getattr(context,f'cpml_b_{letter}_x').numpy()[select].astype(float)
    actual_c=getattr(context,f'cpml_c_{letter}_x').numpy()[select].astype(float)
    direct=1/getattr(context,f'cpml_kappa_{letter}_x').numpy()[select].astype(float)
    np.testing.assert_allclose(actual_b,pole,rtol=3e-7,atol=1e-9)
    # DC response and first time moment are independent of the memory-state
    # representation and identify all three physical profiles at these samples.
    np.testing.assert_allclose(direct+actual_c/(1-actual_b),alpha/(sigma+alpha*kappa),
                               rtol=3e-6,atol=3e-7)
    weight=sigma/((sigma+alpha*kappa)*kappa)
    np.testing.assert_allclose(-actual_c*actual_b/(1-actual_b)**2,weight/u,rtol=3e-6,atol=3e-7)
