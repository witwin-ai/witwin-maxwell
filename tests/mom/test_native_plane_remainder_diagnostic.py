"""Diagnose actual dynamic-minus-static anisotropic plane blocks."""
import json
import math
import pytest
import torch
from mp_plane_remainder_reference import mp_plane_remainder_reference
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import compile_plane_modes,compile_electrostatic_plane_modes
from witwin.maxwell.constants import EPSILON_0,MU_0

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_original_covariance_material_regular_plane_blocks_mp100_diagnostic(tmp_path):
    metric=torch.diag(torch.tensor([1.25,1.,.75],device='cuda',dtype=torch.float64))
    angle=.37
    rotation=metric.new_tensor([[math.cos(angle),-math.sin(angle),0.],[math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    frames=(torch.eye(3,device='cuda',dtype=torch.float64),rotation)
    # These three pairs were not in the closed full-plane diagnostic.
    points=((1e3,1e3),(5e5,7e5),(1e8,1e8))
    record=dict(version='original_covariance_regular_plane_mp100_diagnostic_v1',MP_digits=100,
        no_constitutive_projection_or_damping=True,all396_production_unchanged=True,
        old_complete_plane_diagnostic_not_replayed=True,original_covariance_failure_preserved=True,
        original_spatial_quadrature_HB_or_archive_not_replayed=True,points=[],
        production_qualification=False,full_spatial_or_covariance_qualification=False)
    def save():(tmp_path/'plane-remainder.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    samples=[];save()
    k0=2*math.pi*3e7*math.sqrt(EPSILON_0*MU_0)
    eta=math.sqrt(MU_0/EPSILON_0)
    for orientation,frame in enumerate(frames):
        eps=(frame@(2*metric)@frame.T).to(torch.complex128)
        mu=(frame@(1.3*metric)@frame.T).to(torch.complex128)
        medium=TensorMedium(3e7,eps,mu);before=(eps.clone(),mu.clone())
        for q in points:
            distance=1/(k0*math.hypot(*q))
            reference=mp_plane_remainder_reference(q,eps.tolist(),mu.tolist(),3e7,distance)
            expected={name:eps.new_tensor(value) for name,value in reference.items()}
            local=dict(orientation=orientation,transverse=list(q),distance_m=distance,
                selected_block_relative_gate=.005,native_evaluated=False,native_error=None)
            try:
                modes=compile_plane_modes(metric.new_tensor([q]),medium)
                static_modes=compile_electrostatic_plane_modes(metric.new_tensor([q]),medium)
                complete=modes.spectrum(distance)[0];static=static_modes.spectrum(distance)[0]
                remainder=complete-static
                blocks=[]
                for name,i,j in [('EE',0,0),('EH',0,3),('HE',3,0),('HH',3,3)]:
                    actual=remainder[i:i+3,j:j+3];wanted=expected['remainder'][i:i+3,j:j+3]
                    reference_norm=float(wanted.norm());native_norm=float(actual.norm())
                    valid_reference=math.isfinite(reference_norm) and reference_norm>0
                    error=float((actual-wanted).norm()/wanted.norm())
                    blocks.append(dict(block=name,finite_error=math.isfinite(error),
                        relative_error=error if math.isfinite(error) else None,
                        selected_block_gate_passed=valid_reference and math.isfinite(error) and error<.005,
                        reference_remainder_nonzero_finite=valid_reference,
                        reference_remainder_norm=reference_norm if math.isfinite(reference_norm) else None,
                        native_remainder_norm=native_norm if math.isfinite(native_norm) else None))
                scale=complete.new_tensor([1.]*3+[eta]*3)
                error=float(((complete-expected['complete'])*scale[:,None]*scale[None,:]).norm()/(expected['complete']*scale[:,None]*scale[None,:]).norm())
                local.update(native_evaluated=True,complete_balanced_relative_error=error if math.isfinite(error) else None,
                    native_eigensolver=modes.eigensolver,blocks=blocks,
                    all_selected_block_gates_passed=all(b['selected_block_gate_passed'] for b in blocks))
                samples.append(dict(orientation=orientation,transverse=list(q),distance_m=distance,
                    expected=expected,native_complete=complete,native_static=static,native_remainder=remainder))
            except (ValueError,RuntimeError) as error:
                local.update(native_error=str(error),all_selected_block_gates_passed=False)
            record['points'].append(local);save();print(local,flush=True)
        assert torch.equal(eps,before[0]) and torch.equal(mu,before[1])
    torch.save(samples,tmp_path/'reference-native-spectra.pt')
    record.update(completed=True,all_selected_block_gates_passed=all(p['all_selected_block_gates_passed'] for p in record['points']))
    save();assert len(record['points'])==6
