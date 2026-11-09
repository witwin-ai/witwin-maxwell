"""Independent MP100 diagnostics of the original covariance material tail."""
import json
import math
from pathlib import Path

import mpmath as mp
import pytest
import torch
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import compile_plane_modes
from witwin.maxwell.constants import EPSILON_0,MU_0

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _mp_plane(q,eps,mu,frequency,distance):
    """Resolve actual binary constitutive data at100digits, not a proxy medium."""
    with mp.workdps(100):
        def matrix(values):return mp.matrix([[mp.mpc(v.real,v.imag) for v in row] for row in values])
        electric,magnetic=matrix(eps),matrix(mu)
        x,y=map(mp.mpf,q)
        curl=mp.matrix([[0,0,y],[0,0,-x],[-y,x,0]])
        symbol=mp.matrix(6)
        for i in range(3):
            for j in range(3):
                symbol[i,j]=electric[i,j];symbol[i+3,j+3]=-magnetic[i,j]
                symbol[i,j+3]=symbol[i+3,j]=curl[i,j]
        tangential=[0,1,3,4];longitudinal=[2,5]
        def block(rows,cols):return mp.matrix([[symbol[i,j] for j in cols] for i in rows])
        tt,tl,lt,ll=block(tangential,tangential),block(tangential,longitudinal),block(longitudinal,tangential),block(longitudinal,longitudinal)
        reduced=tt-tl*(ll**-1)*lt
        derivative=mp.matrix([[0,0,0,-1],[0,0,1,0],[0,-1,0,0],[1,0,0,0]])
        generator=-(derivative**-1)*reduced
        roots,vectors=mp.eig(generator,left=False,right=True)
        identity=mp.eye(4);embedding=mp.matrix(6,4);forcing=mp.matrix(4,6)
        bottom=-(ll**-1)*lt;right=-tl*(ll**-1)
        for i in range(4):
            for j in range(4):embedding[i,j]=forcing[i,j]=identity[i,j]
        for i in range(2):
            for j in range(4):embedding[i+4,j]=bottom[i,j];forcing[j,i+4]=right[j,i]
        coefficients=(vectors**-1)*(derivative**-1)*forcing
        eta=mp.sqrt(mp.mpf(MU_0)/mp.mpf(EPSILON_0))
        k0=2*mp.pi*mp.mpf(frequency)*mp.sqrt(mp.mpf(EPSILON_0)*mp.mpf(MU_0))
        output=mp.matrix(6);outgoing=[]
        for column,root in enumerate(roots):
            flux=mp.re(vectors[0,column]*mp.conj(vectors[3,column])-vectors[1,column]*mp.conj(vectors[2,column]))
            if mp.im(root)>mp.mpf('1e-70') or (abs(mp.im(root))<=mp.mpf('1e-70') and flux>0):
                outgoing.append(column)
                for i in range(6):
                    for j in range(6):
                        output[i,j]+=mp.j*k0*(embedding*vectors)[i,column]*coefficients[column,j]*mp.exp(mp.j*k0*root*mp.mpf(distance))
        assert len(outgoing)==2
        reorder=[0,1,4,2,3,5]
        source=[-mp.j*eta/k0]*3+[mp.j/k0]*3;field=[1]*3+[1/eta]*3
        return [[complex(field[i]*output[reorder[i],reorder[j]]*source[j]) for j in range(6)] for i in range(6)]


def test_original_covariance_material_native_plane_tail_mp100_diagnostic(tmp_path):
    metric=torch.diag(torch.tensor([1.25,1.,.75],device='cuda',dtype=torch.float64))
    angle=.37
    rotation=metric.new_tensor([[math.cos(angle),-math.sin(angle),0.],[math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    points=[(1e2,0.),(1e4,0.),(1e6,0.),(1e8,0.),(1e4,1e4),(1e6,1e6)]
    record=dict(version='original_covariance_plane_tail_mp100_diagnostic_v1',frequency_Hz=3e7,
        original_covariance_failure_preserved=True,actual_constitutive_projection=False,MP_digits=100,
        native_production_modules_unchanged=True,original_covariance_solve_or_quadrature_replayed=False,
        standalone_plane_diagnostic_only=True,full_spatial_or_covariance_qualification=False,
        production_qualification=False,points=[])
    def save():(tmp_path/'plane-tail.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    for orientation,frame in enumerate((eye,rotation)):
        eps=(frame@(2*metric)@frame.T).to(torch.complex128)
        mu=(frame@(1.3*metric)@frame.T).to(torch.complex128)
        medium=TensorMedium(3e7,eps,mu)
        before=(eps.clone(),mu.clone())
        eta=math.sqrt(MU_0/EPSILON_0);k0=2*math.pi*3e7*math.sqrt(EPSILON_0*MU_0)
        for q in points:
            distance=1/(k0*math.hypot(*q));reference=_mp_plane(q,eps.tolist(),mu.tolist(),3e7,distance)
            expected=eps.new_tensor(reference)
            local=dict(orientation=orientation,transverse=list(q),positive_distance_m=distance,
                exactly_proportional=torch.equal(.65*eps,mu),native_evaluated=False,native_error=None)
            try:
                modes=compile_plane_modes(metric.new_tensor([q]),medium)
                actual=modes.spectrum(distance)[0]
                scale=actual.new_tensor([1.]*3+[eta]*3)
                error=float(((actual-expected)*scale[:,None]*scale[None,:]).norm()/(expected*scale[:,None]*scale[None,:]).norm())
                local.update(native_evaluated=True,balanced_relative_error=error if math.isfinite(error) else None,
                    finite_balanced_relative_error=math.isfinite(error),
                    native_error=None if math.isfinite(error) else 'nonfinite_balanced_plane_relative_error',
                    selected_plane_error_below_original_point005=math.isfinite(error) and error<.005,
                    native_eigensolver=modes.eigensolver)
            except (ValueError,RuntimeError) as error:
                local['native_error']=str(error)
            record['points'].append(local);save();print(local,flush=True)
        assert torch.equal(eps,before[0]) and torch.equal(mu,before[1])
    record['completed']=True
    record['all_selected_plane_points_below005']=all(p.get('selected_plane_error_below_original_point005',False) for p in record['points'])
    save();assert len(record['points'])==12
