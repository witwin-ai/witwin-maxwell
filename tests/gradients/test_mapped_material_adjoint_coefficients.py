"""CPU local coefficient/VJP controls; mapped public FDTD AD stays rejected."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import PreparedScene
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.fdtd.copper_foil import initialize_copper_foils
from witwin.maxwell.fdtd.adjoint.copper_foil import differentiable_copper_entries
from witwin.maxwell.fdtd.material_pullback import pullback_material_input_gradients
from witwin.maxwell.fdtd.adjoint.copper_spectra import prepare_spectral_outputs,pack_spectral_outputs,rebuild_spectral_outputs


def coefficient_solver(curvature, *, epsilon=2.5, sigma=.2, thickness=2e-6, copper_sigma=1e6):
    chart=None if curvature is None else mw.BendCoordinates(curvature=curvature)
    scene=PreparedScene(mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),(0.,.04),(-.02,.02))),
        grid=mw.GridSpec.uniform(.01),boundary=mw.BoundarySpec.faces(default='none'),device='cpu',coordinates=chart,
        structures=(mw.Structure(name='foil',geometry=mw.Box(position=torch.tensor((0.,.02,0.),dtype=torch.float64),
            size=torch.tensor((.02,0.,.02),dtype=torch.float64)),material=mw.CopperFoil(thickness=thickness,
                conductivity=copper_sigma,frequency_range=(.5e9,2e9),mode_count=16)),)))
    solver=SimpleNamespace(scene=scene,device=torch.device('cpu'),dt=2e-12,_spectral_integration_dt=None,_spectral_reference_frequency=None,source_frequency=1e9,
        full_aniso_enabled=False,nonlinear_enabled=False,complex_fields_enabled=False)
    for name in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        axes=component_coords(scene,name);shape=tuple(len(axis)for axis in axes)
        own='xyz'.index(name[1].lower());u=torch.as_tensor(axes[1],dtype=torch.float64)
        h=torch.ones_like(u)if chart is None else 1-curvature*u
        metric=(1/h if own==0 else h)[None,:,None].expand(shape)
        setattr(solver,name,torch.zeros(shape,dtype=torch.float64))
        if name.startswith('E'):
            setattr(solver,'eps_'+name,epsilon*mw.constants.EPSILON_0*metric)
            setattr(solver,'sigma_e_'+name,sigma*metric)
            setattr(solver,'c'+name.lower()+'_curl',torch.ones(shape,dtype=torch.float64))
        else:
            setattr(solver,'c'+name.lower()+'_curl',solver.dt/(mw.constants.MU_0*metric))
    for axis in 'xyz':
        setattr(solver,'inv_d'+axis+'_h',torch.as_tensor(getattr(scene,'d'+axis+'_primal64'),dtype=torch.float64).reciprocal())
    with torch.no_grad():initialize_copper_foils(solver)
    return solver


def entries(solver):
    return differentiable_copper_entries(solver,
        {name:getattr(solver,'eps_'+name)for name in ('Ex','Ey','Ez')},
        {name:getattr(solver,'sigma_e_'+name)for name in ('Ex','Ey','Ez')})[0]


@pytest.mark.parametrize('curvature',(-3.,0.,3.))
def test_replay_rebuild_matches_actual_initialized_split_metric_matrices(curvature):
    solver=coefficient_solver(curvature)
    for actual,replayed in zip(solver._copper_foils,entries(solver)):
        assert 'material_metric_factors'in actual and 'response_alpha'in actual
        torch.testing.assert_close(replayed['inverse'],actual['inverse'],rtol=3e-14,atol=1e-15)
        torch.testing.assert_close(replayed['rhs_matrix'],actual['rhs_matrix'],rtol=3e-14,atol=1e-15)


def test_zero_curvature_reduces_bitwise_to_cartesian_coefficient_path():
    cartesian=coefficient_solver(None);zero=coefficient_solver(0.)
    for first,second in zip(cartesian._copper_foils,zero._copper_foils):
        assert 'material_metric_factors'not in first
        for key in ('B','rates','D','capacity','conductance','inverse','rhs_matrix'):
            assert torch.equal(first[key],second[key]),key


def voltage_objective(entry):
    old=torch.tensor((.2,-.1),dtype=torch.float64)[:,None].expand(2,len(entry['indices']))
    forcing=torch.tensor((.03,.07),dtype=torch.float64)[:,None].expand_as(old)
    rhs=torch.einsum('mij,jm->im',entry['rhs_matrix'],old)+forcing
    updated=torch.einsum('mij,jm->im',entry['inverse'],rhs)
    return updated.square().sum()


@pytest.mark.parametrize('curvature',(-3.,3.))
@pytest.mark.parametrize('component',('Ex','Ez'))
def test_half_metric_epsilon_sigma_vjp_matches_ordinary_local_forward_scan(curvature,component):
    theta=torch.tensor([math.log(2.5),math.log(.2)],dtype=torch.float64,requires_grad=True)
    solver=coefficient_solver(curvature,epsilon=theta[0].exp(),sigma=theta[1].exp())
    selected=next(entry for entry in entries(solver)if entry['electric_name']==component)
    gradient=torch.autograd.grad(voltage_objective(selected),theta)[0]
    for step in (1e-3,3e-4,1e-4):
        for index in range(2):
            delta=torch.zeros_like(theta);delta[index]=step
            def ordinary(value):
                candidate=coefficient_solver(curvature,epsilon=value[0].exp(),sigma=value[1].exp())
                return voltage_objective(next(entry for entry in candidate._copper_foils if entry['electric_name']==component))
            finite=(ordinary(theta.detach()+delta)-ordinary(theta.detach()-delta))/(2*step)
            assert abs(float((gradient[index]-finite)/finite))<3e-5


def admittance_objective(B,rates,D):
    vector=torch.tensor([1+.3j,-.2+.7j],dtype=torch.complex128)
    Y=D.to(torch.complex128)+B.T.to(torch.complex128)@torch.diag(1/(-2j*math.pi*1.2e9+rates))@B.to(torch.complex128)
    return torch.vdot(vector,Y@vector).real


@pytest.mark.parametrize('curvature',(-3.,3.))
@pytest.mark.parametrize('component',('Ex','Ez'))
def test_physical_copper_parameter_pullback_includes_mapped_response_scaling(curvature,component):
    theta=torch.tensor([math.log(2e-6),math.log(1e6)],dtype=torch.float64,requires_grad=True)
    solver=coefficient_solver(curvature,thickness=theta[0].exp(),copper_sigma=theta[1].exp())
    entry=next(value for value in solver._copper_foils if value['electric_name']==component)
    leaves=tuple(entry[key].detach().requires_grad_(True)for key in ('B','rates','D'))
    bars=torch.autograd.grad(admittance_objective(*leaves),leaves)
    gradients=dict(zip(('B','rates','D'),bars),structure_index=0,response_alpha=entry['response_alpha'])
    result=pullback_material_input_gradients(solver.scene._public_scene,inputs=(theta,),
        grad_eps_ex=torch.zeros_like(solver.eps_Ex),grad_eps_ey=torch.zeros_like(solver.eps_Ey),
        grad_eps_ez=torch.zeros_like(solver.eps_Ez),eps0=mw.constants.EPSILON_0,
        grad_copper_coefficients=(gradients,))[0]
    for step in (1e-3,3e-4,1e-4):
        for index in range(2):
            delta=torch.zeros_like(theta);delta[index]=step
            def ordinary(value):
                candidate=coefficient_solver(curvature,thickness=value[0].exp(),copper_sigma=value[1].exp())
                local=next(entry for entry in candidate._copper_foils if entry['electric_name']==component)
                return admittance_objective(*(local[key]for key in ('B','rates','D')))
            finite=(ordinary(theta.detach()+delta)-ordinary(theta.detach()-delta))/(2*step)
            assert abs(float((result[index]-finite)/finite))<3e-5


@pytest.mark.parametrize('curvature',(-3.,3.))
def test_spectral_trace_sigma_coefficient_uses_actual_side_metric(curvature):
    solver=coefficient_solver(curvature)
    for entry in solver._copper_foils:
        count=len(entry['indices'])
        entry.update(spectral_real=torch.ones((1,2,count),dtype=torch.float64),
            spectral_imag=torch.zeros((1,2,count),dtype=torch.float64),spectral_norm=torch.ones(1,dtype=torch.float64),
            spectral_frequencies=torch.tensor([1e9],dtype=torch.float64),
            # Prescribed side-spectrum clock, recorded as sampling provenance.
            spectral_starts=torch.zeros(1,dtype=torch.float64),spectral_end=1,
            spectral_step=torch.ones((),dtype=torch.float64),spectral_window='none')
    raw={};prepare_spectral_outputs(solver,raw);outputs=[]
    copper,materials=pack_spectral_outputs(raw,outputs)
    leaves=[value.detach().requires_grad_(True)for value in outputs]
    records={'foil':{}}
    rebuild_spectral_outputs(SimpleNamespace(copper_spectral_templates=copper,loss_material_templates=materials),leaves,records)
    traces=records['foil']['bulk_electric_traces']
    for entry,trace in zip(solver._copper_foils,traces):
        expected=entry['conductance']*entry['area'][None]
        torch.testing.assert_close(trace['conduction_coefficient'][0],expected,rtol=2e-15,atol=1e-30)
    objective=sum(trace['conduction_coefficient'].sum()for trace in traces)
    gradients=torch.autograd.grad(objective,[leaves[index]for index in materials.values()],allow_unused=True)
    for component,gradient in zip(materials,gradients):
        expected=torch.zeros_like(getattr(solver,'sigma_e_'+component))
        for entry in solver._copper_foils:
            if entry['electric_name']!=component:continue
            for side,offset in enumerate((-1,1)):
                indices=entry['indices'].clone();indices[:,1]+=offset
                volume=entry['material_widths'][side]*entry['area']*entry['material_metric_factors'][side]
                expected.index_put_(tuple(indices.T),volume,accumulate=True)
        if component=='Ey':
            assert gradient is None  # There is no normal-E copper trace loss record.
        else:
            torch.testing.assert_close(gradient,expected,rtol=2e-15,atol=1e-30)


def test_bulk_conduction_prefactor_has_custom_outputs_without_copper():
    monitor=mw.PowerLossMonitor('bulk',position=(0,0,0),size=(.02,.02,.02))
    solver=SimpleNamespace(_copper_foils=(),scene=SimpleNamespace(monitors=(monitor,)))
    for component in ('Ex','Ey','Ez'):setattr(solver,'sigma_e_'+component,torch.full((2,2,2),.2,dtype=torch.float64))
    raw={};prepare_spectral_outputs(solver,raw)
    outputs=[];copper,materials=pack_spectral_outputs(raw,outputs)
    assert not copper and set(materials)=={'Ex','Ey','Ez'}
    leaves=[value.detach().requires_grad_(True)for value in outputs]
    rebuilt=rebuild_spectral_outputs(SimpleNamespace(copper_spectral_templates=copper,loss_material_templates=materials),leaves,{})
    objective=sum(.5*value.sum()for value in rebuilt.values())
    gradients=torch.autograd.grad(objective,leaves)
    assert all(torch.equal(value,torch.full_like(value,.5))for value in gradients)


def test_unused_bulk_conductivity_heads_are_not_allocated():
    solver=SimpleNamespace(_copper_foils=(),scene=SimpleNamespace(monitors=()))
    raw={};prepare_spectral_outputs(solver,raw)
    assert 'loss_materials'not in raw


def test_unqualified_mapped_adaptive_time_stays_rejected_by_narrow_admission():
    epsilon=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.01),
        boundary=mw.BoundarySpec.faces(default='none'),coordinates=mw.BendCoordinates(curvature=3.),device='cpu',
        structures=(mw.Structure(geometry=mw.Box(size=(.02,.02,.02)),material=mw.Material(eps_r=epsilon)),))
    with pytest.raises(NotImplementedError,match='explicit fixed dt and time_steps'):
        mw.Simulation.fdtd(scene,frequency=1e9)


@pytest.mark.parametrize('curvature',(-3.,3.))
def test_fixed_chart_world_field_jacobian_matches_independent_numpy_difference(curvature):
    chart=mw.BendCoordinates(curvature=curvature,origin=(.2,-.1,.3))
    points=torch.tensor(((.007,.01,.003),(-.009,-.004,.006)),dtype=torch.float64)
    theta=torch.tensor((1.2,-.3,.7),dtype=torch.float64,requires_grad=True)
    covariant=theta.to(torch.complex128)[None]*torch.tensor([[1+.2j,.3-.7j,.4+.8j],
        [-.2+.4j,.6+.9j,.7-.1j]],dtype=torch.complex128)
    world=chart.physical_field(points,covariant)
    objective=world.real.sum()+.31*world.imag.sum();gradient=torch.autograd.grad(objective,theta)[0]
    def ordinary(values):
        q=points.numpy();angle=curvature*q[:,0];h=1-curvature*q[:,1]
        tangent=np.stack((np.cos(angle),np.sin(angle),np.zeros(len(q))),-1)
        normal=np.stack((-np.sin(angle),np.cos(angle),np.zeros(len(q))),-1)
        axial=np.tile((0.,0.,1.),(len(q),1))
        cov=values[None]*np.asarray([[1+.2j,.3-.7j,.4+.8j],[-.2+.4j,.6+.9j,.7-.1j]])
        value=cov[:,0,None]/h[:,None]*tangent+cov[:,1,None]*normal+cov[:,2,None]*axial
        return value.real.sum()+.31*value.imag.sum()
    for step in (1e-3,3e-4,1e-4):
        for index in range(3):
            delta=np.eye(3)[index]*step;finite=(ordinary(theta.detach().numpy()+delta)-ordinary(theta.detach().numpy()-delta))/(2*step)
            assert abs(float(gradient[index])-finite)<1e-10


@pytest.mark.parametrize('curvature',(-3.,3.))
def test_curved_world_radiation_field_seed_matches_independent_numpy_phase(curvature):
    from witwin.maxwell.postprocess.stratton_chu import PlanarEquivalentCurrents,_mapped_surface_currents
    from witwin.maxwell.postprocess.nfft import NearFieldFarFieldTransformer
    chart=mw.BendCoordinates(curvature=curvature,origin=(.2,-.1,.3))
    axis=torch.tensor((-.01,0.,.01),dtype=torch.float64)
    s,z=torch.meshgrid(axis,axis,indexing='ij');normal=torch.zeros((3,3,3),dtype=torch.complex128);normal[...,1]=1
    theta=torch.tensor((1.2,-.3,.7,.9),dtype=torch.float64,requires_grad=True)
    pattern=torch.exp(1j*(s*17+z*23))
    def currents(values):
        e=torch.stack((values[0]*pattern,torch.zeros_like(pattern),values[1]*pattern),-1)
        h=torch.stack((values[2]*pattern,torch.zeros_like(pattern),values[3]*pattern),-1)
        return PlanarEquivalentCurrents(axis='y',position=.02,frequency=3e9,u=axis,v=axis,
            J=torch.cross(normal,h,dim=-1),M=-torch.cross(normal,e,dim=-1))
    directions=torch.tensor(((.3,.4,math.sqrt(.75)),(-.4,.7,math.sqrt(.35))),dtype=torch.float64)
    actual=NearFieldFarFieldTransformer(_mapped_surface_currents(currents(theta),chart),
        device='cpu',c=mw.constants.C_0,eps0=mw.constants.EPSILON_0,mu0=mw.constants.MU_0).transform_directions(directions,radius=2.)['E']
    objective=actual.real.sum()+.31*actual.imag.sum();gradient=torch.autograd.grad(objective,theta)[0]
    def ordinary(values):
        ss,zz=np.meshgrid(axis.numpy(),axis.numpy(),indexing='ij');angle=curvature*ss;hscale=1-curvature*.02
        tangent=np.stack((np.cos(angle),np.sin(angle),np.zeros_like(angle)),-1)
        unit=np.stack((-np.sin(angle),np.cos(angle),np.zeros_like(angle)),-1)
        axial=np.zeros_like(unit);axial[...,2]=1
        profile=np.exp(1j*(ss*17+zz*23))
        e=values[0]*profile[...,None]/hscale*tangent+values[1]*profile[...,None]*axial
        magnetic=values[2]*profile[...,None]/hscale*tangent+values[3]*profile[...,None]*axial
        J=np.cross(unit,magnetic);M=-np.cross(unit,e)
        points=np.stack((np.sin(angle)/curvature-.02*np.sin(angle),
            (1-np.cos(angle))/curvature+.02*np.cos(angle),zz),-1)+np.asarray((.2,-.1,.3))
        widths=np.asarray((.005,.01,.005));weights=widths[:,None]*widths[None,:]*hscale
        direction=directions.numpy();k=2*math.pi*3e9/mw.constants.C_0
        phase=np.exp(-1j*k*(direction@points.reshape(-1,3).T))
        nvec=phase@(J*weights[...,None]).reshape(-1,3);lvec=phase@(M*weights[...,None]).reshape(-1,3)
        eta=math.sqrt(mw.constants.MU_0/mw.constants.EPSILON_0)
        field=(-1j*k*np.exp(1j*k*2)/(4*math.pi*2))*(eta*np.cross(direction,np.cross(direction,nvec))+np.cross(direction,lvec))
        return field.real.sum()+.31*field.imag.sum()
    assert float(objective)==pytest.approx(ordinary(theta.detach().numpy()),rel=2e-13,abs=1e-15)
    for step in (1e-3,3e-4,1e-4):
        for index in range(4):
            delta=np.eye(4)[index]*step;finite=(ordinary(theta.detach().numpy()+delta)-ordinary(theta.detach().numpy()-delta))/(2*step)
            assert abs(float(gradient[index])-finite)<1e-9
