"""Independent watt, volume and partition controls for physical loss windows."""
import math
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.postprocess.power_loss import compute_power_loss_data
from witwin.maxwell.postprocess.fdtd_loss import apply_split_trace_densities
from witwin.maxwell.scene import prepare_scene


def scene(graded=False, material=None, device='cpu'):
    axes=((0.,1.,3.,6.), (0.,.5,2.,6.), (0.,2.,3.,6.)) if graded else ((0.,2.,4.,6.),)*3
    return prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0.,6.),)*3),
        grid=mw.GridSpec.custom(*axes), device=device,
        structures=[mw.Structure(mw.Box(position=(3.,)*3,size=(100.,)*3),
            material or mw.Material(sigma_e_tensor=mw.DiagonalTensor3(2.,3.,5.)))]))


def compile_window(prepared, bounds, quadrature='physical_intersection'):
    return compile_power_loss_monitor(prepared, mw.PowerLossMonitor('roi',
        position=tuple((a+b)/2 for a,b in bounds),size=tuple(b-a for a,b in bounds),
        frequencies=(1e9,),quadrature=quadrature))


def fields(compiled, amplitudes=(1.,2.,3.), requires_grad=False):
    return {component:torch.full((1,*shape),complex(amplitude),device=compiled.device,
        dtype=torch.complex128,requires_grad=requires_grad)
        for (component,shape),amplitude in zip(compiled.full_component_shapes.items(),amplitudes)}


def loss(compiled, electric):
    return compute_power_loss_data(compiled,torch.tensor([1e9],device=compiled.device),electric_fields=electric)


@pytest.mark.parametrize('graded',[False,True])
@pytest.mark.parametrize('bounds',[
    ((.1,2.4),(1.2,3.4),(.7,4.3)),
    ((0.,6.),)*3,
    ((.01,.07),(1.01,1.08),(2.01,2.09)),
])
def test_constant_anisotropic_native_work_integrates_exact_physical_box(graded,bounds):
    prepared=scene(graded);compiled=compile_window(prepared,bounds)
    volume=math.prod(b-a for a,b in bounds)
    for actual in compiled.component_volumes.values():
        assert actual.sum().item()==pytest.approx(volume,rel=2e-14)
    data=loss(compiled,fields(compiled))
    assert data.conduction.item()==pytest.approx(.5*(2*1**2+3*2**2+5*3**2)*volume,rel=2e-14)
    assert all(v.dtype==torch.float64 for v in compiled.component_volumes.values())


@pytest.mark.parametrize('graded',[False,True])
def test_adjacent_roi_additivity_and_native_field_autograd(graded):
    prepared=scene(graded);whole=((.1,5.3),(.4,4.7),(.7,4.8))
    left=((.1,2.37),*whole[1:]);right=((2.37,5.3),*whole[1:])
    full=compile_window(prepared,whole);electric=fields(full,requires_grad=True)
    total=loss(full,electric).total.sum()
    parts=sum(loss(compile_window(prepared,b),electric).total.sum() for b in (left,right))
    torch.testing.assert_close(parts,total,rtol=2e-14,atol=1e-12)
    gradients=torch.autograd.grad(total,tuple(electric.values()))
    volume=math.prod(b-a for a,b in whole)
    for gradient,sigma,amplitude in zip(gradients,(2.,3.,5.),(1.,2.,3.)):
        assert gradient.sum().real.item()==pytest.approx(sigma*amplitude*volume,rel=2e-14)


def test_native_conductivity_parameter_gradient_has_independent_watt_and_difference_oracles():
    bounds=((.1,2.4),(1.2,3.4),(.7,4.3));volume=math.prod(b-a for a,b in bounds)
    def objective(value):
        material=mw.Material(sigma_e=value)
        compiled=compile_window(scene(True,material),bounds)
        return loss(compiled,fields(compiled)).total.sum()
    sigma=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    power=objective(sigma);gradient,=torch.autograd.grad(power,sigma)
    expected=7*volume
    assert gradient.item()==pytest.approx(expected,rel=2e-7)
    # The objective is built from float32 compiled conductivity. A central difference
    # divides that rounding, about (prod(samples)+1)*eps32*sigma, by the step, so steps
    # at or below 1e-4 (relative 1.4e-4) sit on the float32 cancellation floor. Measured
    # FD error, 4^3 kottke_edge default / one sample: 1e-2: 1.6e-5 / 9.5e-7;
    # 1e-3: 4.3e-5 / 1.3e-5; 1e-4: 1.6e-3 / 1.7e-4; 3e-5: 3.7e-4 / 3.7e-4;
    # 1e-5: 1.4e-3 / 1.4e-3. Only truncation-dominated steps are an independent oracle.
    errors=[]
    for step in (1e-2,1e-3):
        central=(objective(float(sigma.detach())+step)-objective(float(sigma.detach())-step))/(2*step)
        errors.append(abs(float(central)/expected-1))
    assert max(errors)<5e-5


def test_intersection_integrates_homogenized_cell_density_without_claiming_subcell_material_fill():
    prepared=prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0.,2.),)*3),
        grid=mw.GridSpec.uniform(1.),device='cpu',structures=[mw.Structure(
            mw.Box(position=(1.,)*3,size=(1.,10.,10.)),mw.Material(sigma_e=1.))]))
    compiled=compile_window(prepared,((.5,1.5),(0.,2.),(0.,2.)))
    # The two native Ex centers lie exactly on slab faces: each compiled sigma
    # is 1/2. Cutting their already homogenized cells does not recover the
    # continuum material's subcell distribution or its larger exact integral.
    torch.testing.assert_close(compiled.conductivity['Ex'],torch.full_like(compiled.conductivity['Ex'],.5))
    data=loss(compiled,fields(compiled,amplitudes=(1.,0.,0.)))
    assert data.conduction.item()==pytest.approx(.5*.5*1.*2.*2.,rel=1e-14)


def test_selection_includes_intersecting_cells_with_exterior_centers_and_retains_native_option():
    prepared=scene(True);bounds=((1.1,1.3),(1.1,1.3),(1.1,1.3))
    compiled=compile_window(prepared,bounds)
    assert all(mask.any() for mask in compiled.component_masks.values())
    # This tiny box contains no Ex x half-center (the enclosing primal is 1..3).
    assert not any(1.1<=x<=1.3 for x in prepared.x_half64)
    with pytest.raises(ValueError,match='contains no Ex'):
        compile_window(prepared,bounds,'native_cells')
    assert mw.PowerLossMonitor('default',(3.,)*3,(1.,)*3).quadrature=='native_cells'
    with pytest.raises(ValueError,match='quadrature'):
        mw.PowerLossMonitor('bad',(3.,)*3,(1.,)*3,quadrature='center_guess')


@pytest.mark.parametrize('boundary',[mw.BoundarySpec.periodic(),mw.BoundarySpec.bloch((.1,0.,0.))])
def test_physical_intersection_rejects_wrapped_cells_without_changing_native_ownership(boundary):
    prepared=prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0.,6.),)*3),
        grid=mw.GridSpec.uniform(2.),boundary=boundary,device='cpu'))
    native=compile_window(prepared,((0.,6.),)*3,'native_cells')
    assert native.component_cell_bounds is None
    with pytest.raises(NotImplementedError,match='wrapped control-volume fragments'):
        compile_window(prepared,((0.,6.),)*3)


def test_mapped_default_cells_remain_available_but_new_physical_window_is_explicitly_bounded():
    prepared=prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0.,6.),)*3),
        grid=mw.GridSpec.uniform(2.),device='cpu',coordinates=mw.BendCoordinates(curvature=.01)))
    assert compile_window(prepared,((0.,6.),)*3,'native_cells').component_cell_bounds is None
    with pytest.raises(NotImplementedError,match='mapped partial-volume metrics'):
        compile_window(prepared,((0.,6.),)*3)


@pytest.mark.parametrize('normal_bounds',[(2.6,3.4),(2.4,2.8),(3.2,3.7)])
def test_split_foil_bulk_work_clips_left_and_right_physical_volumes_separately(normal_bounds):
    prepared=prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0.,6.),)*3),
        grid=mw.GridSpec.custom(*((0.,1.,3.,6.),)*3),device='cpu',
        structures=[mw.Structure(mw.Box(position=(3.,)*3,size=(100.,)*3),mw.Material(sigma_e=.5))]))
    bounds=((.1,.7),(.6,1.4),normal_bounds);compiled=compile_window(prepared,bounds)
    electric=fields(compiled,amplitudes=(2.,0.,0.))
    indices=torch.tensor([[0,1,2]])
    trace=dict(component='Ex',indices=indices,normal_axis='z',normal_coordinate=3.,
        normal_half_widths=torch.tensor([[1.],[1.5]],dtype=torch.float64),
        electric_fields=torch.tensor([[[2.],[7.]]],dtype=torch.complex128),
        conduction_coefficient=torch.tensor([[[.2*1.*1.5],[.7*1.5*1.5]]],dtype=torch.float64))
    records={'foil':dict(frequencies=torch.tensor([1e9]),bulk_electric_traces=[trace])}
    actual=apply_split_trace_densities(loss(compiled,electric),compiled,electric,records,correct_dispersion=False)
    lo,hi=normal_bounds
    left=max(0.,min(hi,3.)-max(lo,2.));right=max(0.,min(hi,4.5)-max(lo,3.))
    expected=.5*.6*.8*(.2*4*left+.7*49*right)
    assert actual.conduction.item()==pytest.approx(expected,rel=2e-14)
    assert actual.total.item()==pytest.approx(expected,rel=2e-14)
    del trace['normal_half_widths']
    with pytest.raises(ValueError,match='retained foil half-volume geometry'):
        apply_split_trace_densities(loss(compiled,electric),compiled,electric,records,correct_dispersion=False)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_cuda_physical_loss_constant_watts_and_cpu_parity():
    bounds=((.1,2.4),(1.2,3.4),(.7,4.3))
    cpu=compile_window(scene(True),bounds);gpu=compile_window(scene(True,device='cuda'),bounds)
    a,b=loss(cpu,fields(cpu)).total,loss(gpu,fields(gpu)).total.cpu()
    torch.testing.assert_close(a,b,rtol=2e-14,atol=1e-12)
