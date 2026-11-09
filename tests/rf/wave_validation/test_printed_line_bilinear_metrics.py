"""Self-contained analytic native endpoint and source-bilinear flux contracts."""
import copy
import numpy as np
import pytest
import torch
from scripts.dev.fdtd.printed_line_loss_assertions import actual_basis_bilinear_flux_metrics
# Independent Cartesian cross-product endpoint layout for analytic fixtures.
_PAIRS={'x':(('Ey','Hz',1,1.),('Ez','Hy',0,-1.)),
        'y':(('Ez','Hx',0,1.),('Ex','Hz',1,-1.)),
        'z':(('Ex','Hy',1,1.),('Ey','Hx',0,-1.))}
_COORDS={'x':('y','z'),'y':('x','z'),'z':('x','y')}

class Observation:
    frequencies=(1e9,)
    def __init__(self,faces):self.faces=faces
    def stats(self):return dict(normalize_source=False)
    def monitor(self,name,*,frequency):assert frequency==1e9;return dict(faces=self.faces)

def observed(E=1.,H=1.,*,copper=False,reactive=False):
    nodes=torch.tensor([-.01,0,.01],dtype=torch.float64);half=torch.tensor([-.005,.005],dtype=torch.float64);faces={}
    for axis in'xyz':
        components={};head=[]
        for pair,(e,h,average,sign)in enumerate(_PAIRS[axis]):
            coords=(nodes,half)if average==0 else(half,nodes);shape=tuple(len(x)for x in coords)
            for component,value in((e,E*(1j if reactive else 1.)),(h,H)):
                components[component]=dict(coords=coords,data=torch.full(shape,complex(value)if pair==0 else 0.,dtype=torch.complex128))
            if copper and pair==0:
                indices=torch.tensor([[1,0],[1,1]]if average==0 else[[0,1],[1,1]])
                head.append(dict(component=e,averaging_axis=average,indices=indices,
                    negative=torch.full((2,),complex(E)*(1j if reactive else 1.),dtype=torch.complex128),
                    positive=torch.full((2,),3*complex(E)*(1j if reactive else 1.),dtype=torch.complex128)))
        for side,direction in(('neg','-'),('pos','+')):
            names=_COORDS[axis];faces[axis+'_'+side]=dict(axis=axis,normal_direction=direction,
                **{names[0]:half,names[1]:half},cell_widths={name:torch.full((2,),.01,dtype=torch.float64)for name in names},
                native_components=copy.deepcopy(components),native_flux_heads=copy.deepcopy(head))
    return Observation(faces)

def mixed(bases,c):
    result=copy.deepcopy(bases[0])
    for name,face in result.faces.items():
        for comp,value in face['native_components'].items():value['data']=sum(x*b.faces[name]['native_components'][comp]['data']for x,b in zip(c,bases))
        for i,head in enumerate(face['native_flux_heads']):
            for side in('negative','positive'):head[side]=sum(x*b.faces[name]['native_flux_heads'][i][side]for x,b in zip(c,bases))
    return result

def fixture(*,copper=False,reactive=False):
    bases=[observed(1,2,copper=copper,reactive=reactive),observed(2,-1,copper=copper,reactive=reactive)];c=torch.tensor([[1.,.5]],dtype=torch.complex128)
    result=mixed(bases,c[0]);return bases,c,result

@pytest.mark.parametrize('copper',(False,True))
def test_independent_positive_endpoint_norm_including_both_actual_Cu_sides(copper):
    bases,c,actual=fixture(copper=copper);report=actual_basis_bilinear_flux_metrics(actual,actual,bases,c)
    # Independent magnitudes: E=1+.5*2=2, H=2+.5*1=2.5.
    # Six physical faces each have area(.02m)^2. Cu endpoint1/3
    # averages against outer endpoints1 to increase the E integral by1.5.
    expected=6*.5*2*2.5*(.02**2)*(1.5 if copper else 1.)
    np.testing.assert_allclose(report['frequency_records'][0]['actual_input_basis_bilinear_W_scale'],expected,rtol=2e-15)
    assert report['passed']and report['rtol']==3e-5 and report['original_entering_power_metrics']['passed']


def test_wrong_phase_rejected_with_same_original_relative_threshold():
    bases,c,actual=fixture(copper=True,reactive=True);direct=copy.deepcopy(actual)
    for face in direct.faces.values():
        for component,value in face['native_components'].items():
            if component.startswith('H'):value['data']*=complex(np.exp(.001j))
    report=actual_basis_bilinear_flux_metrics(actual,direct,bases,c)
    assert report['rtol']==3e-5 and not report['passed']


def test_wrong_signed_orientation_rejected():
    bases,c,actual=fixture();direct=copy.deepcopy(actual);direct.faces['x_neg']['normal_direction']='+'
    with pytest.raises(ValueError,match='orientations'):actual_basis_bilinear_flux_metrics(actual,direct,bases,c)


def test_wrong_H_sign_not_hidden_by_closed_net_cancellation():
    bases,c,actual=fixture();direct=copy.deepcopy(actual)
    for face in direct.faces.values():
        for component,value in face['native_components'].items():
            if component.startswith('H'):value['data']*=-1
    report=actual_basis_bilinear_flux_metrics(actual,direct,bases,c)
    assert not report['passed']and report['frequency_records'][0]['signed_face_error_l1_w']>0


def test_omitted_face_rejected():
    bases,c,actual=fixture();direct=copy.deepcopy(actual);del direct.faces['z_pos']
    with pytest.raises(ValueError,match='six'):actual_basis_bilinear_flux_metrics(actual,direct,bases,c)


def test_hidden_zero_replacement_of_face_rejected():
    bases,c,actual=fixture();direct=copy.deepcopy(actual)
    for value in direct.faces['z_pos']['native_components'].values():value['data'].zero_()
    assert not actual_basis_bilinear_flux_metrics(actual,direct,bases,c)['passed']


def test_bilinear_scale_independent_of_observed_output_error():
    bases,c,actual=fixture();reference=actual_basis_bilinear_flux_metrics(actual,actual,bases,c)
    direct=copy.deepcopy(actual)
    for face in direct.faces.values():
        for value in face['native_components'].values():value['data']*=10
    changed=actual_basis_bilinear_flux_metrics(actual,direct,bases,c)
    assert changed['frequency_records'][0]['actual_input_basis_bilinear_W_scale']==reference['frequency_records'][0]['actual_input_basis_bilinear_W_scale']
    assert not changed['passed']


def test_zero_input_scale_rejected_without_floor():
    bases=[observed(0,0),observed(0,0)];c=torch.tensor([[1.,-.5]],dtype=torch.complex128);actual=mixed(bases,c[0])
    with pytest.raises(ValueError,match='Positive actual'):actual_basis_bilinear_flux_metrics(actual,actual,bases,c)


def test_absolute_interpolation_weights_match_independent_endpoint_operator():
    bases,c,actual=fixture(copper=True)
    for owner in(*bases,actual):
        for name in('x_neg','x_pos'):
            owner.faces[name]['y']=torch.tensor([-.0075,.005],dtype=torch.float64)
            owner.faces[name]['cell_widths']['y']=torch.tensor([.005,.01],dtype=torch.float64)
    report=actual_basis_bilinear_flux_metrics(actual,actual,bases,c)
    # At each x face, the first other-axis interpolation row is(1.25,-.25).
    # Absolute coefficient length=.005*1.5+.01*1=.0175m, not physical.015m.
    # Other four Cu faces retain area(.02m)^2 and E-side multiplier1.5.
    expected=4*.5*2*2.5*1.5*(.02**2)+2*.5*2*2.5*1.5*(.0175*.02)
    np.testing.assert_allclose(report['frequency_records'][0]['actual_input_basis_bilinear_W_scale'],expected,rtol=2e-15)
    assert report['passed']


def test_non_native_frame_is_rejected():
    bases,c,actual=fixture();direct=copy.deepcopy(actual)
    direct.faces['x_neg']['native_product_disabled']=True
    with pytest.raises(ValueError,match='Cartesian native-product'):actual_basis_bilinear_flux_metrics(actual,direct,bases,c)

def test_missing_actual_input_basis_is_rejected():
    actual=observed()
    with pytest.raises(ValueError,match='source coefficients'):actual_basis_bilinear_flux_metrics(actual,actual,[],torch.empty((1,0)))
