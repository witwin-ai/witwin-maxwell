"""Joint Yee/mortar/skin-state elimination versus an independent full CN solve."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.curved_copper import compile_copper_channels
from witwin.maxwell.fdtd.curved_copper import JointCopperMortar


@pytest.mark.parametrize("device",("cpu","cuda"))
def test_joint_two_current_response_matches_full_passive_field_state_system(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype=torch.float64
    generator=torch.Generator(device=device).manual_seed(901)
    dt=5e-12
    diagonal_e=torch.tensor((1.1,.9,1.2,.8),device=device,dtype=dtype)*1e-12
    diagonal_h=torch.tensor((1.,.8,1.3),device=device,dtype=dtype)*1e-12
    me,mh=torch.diag(diagonal_e),torch.diag(diagonal_h)
    normalized=torch.randn((3,4),device=device,dtype=dtype,generator=generator)
    normalized *= 1.5/torch.linalg.matrix_norm(normalized,ord=2)
    curl=diagonal_h.sqrt()[:,None]*normalized*diagonal_e.sqrt()[None]/dt
    sigma_e,sigma_h=me*.03/dt,mh*.04/dt
    ge=torch.randn((2,4),device=device,dtype=dtype,generator=generator)*.4
    gh=torch.randn((2,3),device=device,dtype=dtype,generator=generator)*.4
    mass=torch.tensor(((1.,.2),(.2,1.3)),device=device,dtype=dtype)
    material=mw.CopperFoil(thickness=torch.tensor(2e-6,device=device,dtype=dtype),frequency_range=(.5e9,2e9))
    channel=compile_copper_channels(material)
    rates=channel.rates
    coefficients=channel.midpoint_coefficients(dt)
    modes=len(rates)
    tail=channel.magnetic_inductance*gh.T@torch.linalg.solve(mass,gh)
    effective_h=mh+tail
    q=torch.cat((torch.cat((me,-dt*curl.T/2),1),torch.cat((-dt*curl/2,effective_h),1)),0)
    a=torch.cat((torch.cat((-sigma_e,curl.T),1),torch.cat((-curl,-sigma_h),1)),0)
    assert torch.linalg.eigvalsh(q).min() > 0
    trace=torch.block_diag(ge,gh)
    field_factor=q-dt*a/2
    response=dt/2*trace@torch.linalg.solve(field_factor,trace.T)
    assert torch.linalg.eigvalsh((response+response.T)/2).min() > 0
    assert response[:2,2:].norm() > 0
    torch.testing.assert_close(response[2:,:2],torch.zeros_like(response[2:,:2]),rtol=0,atol=1e-14)
    state_mass=torch.kron(torch.eye(modes,device=device,dtype=dtype),mass)
    full_q=torch.block_diag(q,state_mass,state_mass)
    full_a=torch.zeros_like(full_q)
    full_a[:7,:7]=a
    full_a[:4,:4] -= channel.electric_direct*ge.T@torch.linalg.solve(mass,ge)
    full_a[4:7,4:7] -= channel.magnetic_residue.sum()*gh.T@torch.linalg.solve(mass,gh)
    for mode in range(modes):
        electric=slice(7+2*mode,7+2*mode+2)
        magnetic=slice(7+2*modes+2*mode,7+2*modes+2*mode+2)
        electric_coupling=channel.electric_residue[mode].sqrt()
        magnetic_coupling=(channel.magnetic_residue[mode]*rates[mode]).sqrt()
        full_a[:4,electric]=-electric_coupling*ge.T
        full_a[electric,:4]=electric_coupling*ge
        full_a[4:7,magnetic]=magnetic_coupling*gh.T
        full_a[magnetic,4:7]=magnetic_coupling*gh
        full_a[electric,electric]=-rates[mode]*mass
        full_a[magnetic,magnetic]=-rates[mode]*mass
    # Exact energy-Riesz scaling avoids a misleading condition number caused by
    # mixing physical field and modal-storage units; it adds no regularization.
    riesz=torch.linalg.cholesky(full_q)
    left=torch.linalg.solve_triangular(riesz,full_a,upper=False)
    generator_a=torch.linalg.solve_triangular(riesz,left.T,upper=False).T
    identity=torch.eye(len(full_q),device=device,dtype=dtype)
    field=torch.randn(7,device=device,dtype=dtype,generator=generator)
    ze=torch.randn((modes,2),device=device,dtype=dtype,generator=generator)*1e-6
    zm=torch.randn((modes,2),device=device,dtype=dtype,generator=generator)*1e-6
    native = None if device == "cpu" else JointCopperMortar(diagonal_e,diagonal_h,
        sigma_e.diag(),sigma_h.diag(),curl.T,ge,gh,mass,material,dt)
    if native is not None:
        native.ze.copy_(ze)
        native.zm.copy_(zm)
    ge_gain,gm_gain=coefficients["electric_gain"],coefficients["magnetic_gain"]
    channel_resistance=torch.block_diag(mass/ge_gain,mass/gm_gain)
    surface_factor=channel_resistance+response
    for step in range(32):
        old=torch.cat((field,ze.flatten(),zm.flatten()))
        transformed=riesz.T@old
        full_next=torch.linalg.solve(identity-dt*generator_a/2,(identity+dt*generator_a/2)@transformed)
        full_next=torch.linalg.solve_triangular(riesz.T,full_next[:,None],upper=True)[:,0]
        # Independently apply the native conductive H-then-E predictor.
        predicted_h=torch.linalg.solve(effective_h+dt*sigma_h/2,
            (effective_h-dt*sigma_h/2)@field[4:]-dt*curl@field[:4])
        predicted_e=torch.linalg.solve(me+dt*sigma_e/2,
            (me-dt*sigma_e/2)@field[:4]+dt*curl.T@predicted_h)
        predicted=torch.cat((predicted_e,predicted_h))
        torch.testing.assert_close(predicted,torch.linalg.solve(field_factor,(q+dt*a/2)@field),rtol=1e-12,atol=1e-13)
        free_voltage=trace@(field+predicted)/2
        history=torch.cat(((coefficients["electric_history"][:,None]*ze).sum(0),
                           (coefficients["magnetic_history"][:,None]*zm).sum(0)))
        current=torch.linalg.solve(surface_factor,free_voltage+channel_resistance@history)
        next_field=predicted-dt*torch.linalg.solve(field_factor,trace.T@current)
        midpoint_voltage=trace@(field+next_field)/2
        ve=torch.linalg.solve(mass,midpoint_voltage[:2])
        vh=torch.linalg.solve(mass,midpoint_voltage[2:])
        next_ze=coefficients["decay"][:,None]*ze+coefficients["electric_drive"][:,None]*ve
        next_zm=coefficients["decay"][:,None]*zm+coefficients["magnetic_drive"][:,None]*vh
        reduced_next=torch.cat((next_field,next_ze.flatten(),next_zm.flatten()))
        if native is not None:
            native.begin(field[:4],field[4:])
            plain_h=torch.linalg.solve(mh+dt*sigma_h/2,(mh-dt*sigma_h/2)@field[4:]-dt*curl@field[:4])
            native.correct_magnetic_predictor(plain_h)
            plain_e=torch.linalg.solve(me+dt*sigma_e/2,(me-dt*sigma_e/2)@field[:4]+dt*curl.T@plain_h)
            native.advance(plain_e,plain_h)
            actual=torch.cat((plain_e,plain_h,native.ze.flatten(),native.zm.flatten()))
            native_relative=(riesz.T@(actual-full_next)).norm()/(riesz.T@full_next).norm()
            assert float(native_relative) < 2e-11
        relative=(riesz.T@(reduced_next-full_next)).norm()/(riesz.T@full_next).norm()
        assert float(relative) < 2e-11
        average=(field+next_field)/2
        average_ze,average_zm=(ze+next_ze)/2,(zm+next_zm)/2
        magnetic_heat=channel.magnetic_residue.sqrt()[:,None]*vh-rates.sqrt()[:,None]*average_zm
        heat=dt*(average[:4]@sigma_e@average[:4]+average[4:]@sigma_h@average[4:])
        heat+=dt*(channel.electric_direct*(ve@mass@ve)+
            (rates*torch.einsum("ni,ij,nj->n",average_ze,mass,average_ze)).sum()+
            torch.einsum("ni,ij,nj->",magnetic_heat,mass,magnetic_heat))
        change=(reduced_next@full_q@reduced_next-old@full_q@old)/2
        torch.testing.assert_close(change,-heat,rtol=2e-10,atol=1e-25)
        assert heat >= 0
        if step == 0:
            omitted=response.clone()
            omitted[:2,2:]=0
            wrong=torch.linalg.solve(channel_resistance+omitted,free_voltage+channel_resistance@history)
            wrong_effort=free_voltage-response@wrong
            wrong_constitutive=history+torch.cat((ge_gain*torch.linalg.solve(mass,wrong_effort[:2]),
                                                 gm_gain*torch.linalg.solve(mass,wrong_effort[2:])))
            assert float((wrong-wrong_constitutive).norm()/wrong.norm()) > 1e-4
        field,ze,zm=next_field,next_ze,next_zm
