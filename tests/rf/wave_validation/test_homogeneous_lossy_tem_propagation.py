"""Actual native lossy TEM propagation against independent homogeneous physics."""
import cmath
import math
import pytest
import torch
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene, analytic_homogeneous_line

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Native FDTD requires CUDA')


def _chain(s):
    a,b,c,d=s[0,0],s[0,1],s[1,0],s[1,1]
    return torch.stack((torch.stack((((1+a)*(1-d)+b*c)/(2*c),50*((1+a)*(1+d)-b*c)/(2*c))),
                        torch.stack((((1-a)*(1-d)-b*c)/(100*c),((1-a)*(1+d)+b*c)/(2*c)))))


def test_static_conductivity_tem_two_length_propagation_and_roundtrip(tmp_path):
    frequency=1e9
    results=[]
    for length in (.04,.08):
        scene=printed_line_scene('stripline',sigma_e=.01,port_separation=length,device='cuda:0')
        result=mw.Simulation.fdtd(scene,frequencies=(frequency,),excitations=mw.PortSweep(),
            run_time=mw.TimeConfig.auto(transient_cycles=16,steady_cycles=6),
            spectral_sampler=mw.SpectralSampler(window='hanning'),full_field_dft=False).run()
        results.append(result)
        assert result.port('in').beta.imag.item()>0
        assert result.port('in').characteristic_impedance.imag.item()<0
        real_s=result.network.renormalize(50.).s[0]
        assert torch.linalg.svdvals(real_s).max()<1.001
    transfer=_chain(results[1].network.renormalize(50.).s[0]) @ torch.linalg.inv(_chain(results[0].network.renormalize(50.).s[0]))
    eigenvalues=torch.linalg.eigvals(transfer)
    decaying=eigenvalues[eigenvalues.abs().argmin()]
    gamma=-cmath.log(complex(decaying))/.04
    reference=analytic_homogeneous_line('stripline',(frequency,),sigma_e=.01)
    print({'measured_alpha_np_per_m':gamma.real,'measured_beta_rad_per_m':abs(gamma.imag),'reference_alpha_np_per_m':float(reference['alpha_np_per_m'][0]),'reference_beta_rad_per_m':float(reference['beta_rad_per_m'][0])})
    assert gamma.real==pytest.approx(reference['alpha_np_per_m'][0],rel=.03)
    assert abs(gamma.imag)==pytest.approx(reference['beta_rad_per_m'][0],rel=.005)
    path=tmp_path/'lossy-tem.pt'
    results[-1].save(path)
    restored=mw.Result.load(path,scene=results[-1].scene,map_location='cuda:0')
    torch.testing.assert_close(restored.network.s,results[-1].network.s)
    torch.testing.assert_close(restored.port('in').characteristic_impedance,results[-1].port('in').characteristic_impedance)
    torch.testing.assert_close(restored.port('in').beta,results[-1].port('in').beta)
