"""Finite physical cells are intersected at their true bounds, before reduction."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.result import _crop_plane_monitor_payload
from witwin.maxwell.fdtd.quadrature import plane_normal_poynting


def payload(axis, backend, sign, sampled=False, amplitude=None):
    tangent = [a for a in 'xyz' if a != axis]
    nodes = (np.array([-2., -1., -.4, .1, .7, 1.5, 2.]),
             np.array([-1.8, -.8, -.2, .4, 1., 1.9]))
    if sampled:
        nodes = (np.linspace(-2, 2, 9), np.linspace(-1.8, 1.9, 10))
    coords = tuple(.5*(n[1:]+n[:-1]) for n in nodes)
    widths = tuple(np.diff(n) for n in nodes)
    u, v = np.meshgrid(*coords, indexing='ij')
    e = 2.+.3*u-.2*v
    h = np.ones_like(e)*3.
    if backend == 'torch':
        coords = tuple(torch.tensor(c, dtype=torch.float64) for c in coords)
        widths = tuple(torch.tensor(w, dtype=torch.float64) for w in widths)
        e, h = torch.tensor(e, dtype=torch.complex128), torch.tensor(h, dtype=torch.complex128)
    if amplitude is not None:
        e = e*amplitude
    fields = {'E'+tangent[0]: e, 'H'+tangent[1]: h*(-1 if axis == 'y' else 1),
              'E'+tangent[1]: e*0, 'H'+tangent[0]: h*0}
    result = dict(axis=axis, position=.17, frequency=1e9, frequencies=(1e9,), normal_direction=sign,
                  fields=tuple(fields), components={key: dict(data=value, coords=coords) for key, value in fields.items()},
                  cell_widths=dict(zip(tangent, widths)), coords=coords, compute_flux=True, collocated=sampled,
                  **dict(zip(tangent, coords)), **fields)
    position = [0.,0.,0.]; position['xyz'.index(axis)] = .17
    size = [1.37,1.37,1.37]; size['xyz'.index(axis)] = 0
    # Both boundary cells can have centers outside these bounds.
    position['xyz'.index(tangent[0])] = -.32
    position['xyz'.index(tangent[1])] = .14
    monitor = mw.FinitePlaneMonitor('face', position=position, size=size, fields=tuple(fields),
                                    normal_direction=sign, compute_flux=True,
                                    sample_spacing=.5 if sampled else None)
    return result, monitor


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('sign', ['+', '-'])
@pytest.mark.parametrize('sampled', [False, True])
def test_intersection_exact_area_and_affine_poynting(axis, backend, sign, sampled):
    original, monitor = payload(axis, backend, sign, sampled)
    result = _crop_plane_monitor_payload(original, monitor)
    tangent = [a for a in 'xyz' if a != axis]
    widths = result['cell_widths']
    area = np.prod([float(widths[a].sum()) for a in tangent])
    assert area == pytest.approx(1.37**2, rel=2e-15)
    center = monitor.position
    affine_at_center = 2+.3*center['xyz'.index(tangent[0])]-.2*center['xyz'.index(tangent[1])]
    expected = (1 if sign == '+' else -1)*.5*3*affine_at_center*1.37**2
    poynting, weights = plane_normal_poynting(result)
    assert float((poynting*weights).sum()) == pytest.approx(expected, rel=3e-15)
    assert float(result['flux']) == pytest.approx(expected, rel=3e-15)
    from witwin.maxwell.fdtd.observers import _align_plane_monitor_payload
    aligned = _align_plane_monitor_payload(axis,result['components'])
    for name in result['fields']:
        np.testing.assert_array_equal(aligned['fields'][name],result[name])
    # The original retained fields and areas are immutable.
    assert sum(original['cell_widths'][tangent[0]]) == pytest.approx(4.)
    source = result['finite_cell_selection']['source_indices'][tangent[0]]
    low, high = monitor.tangential_bounds[tangent[0]]
    selected_centers = np.asarray(original[tangent[0]])[np.asarray(source)]
    assert np.any((selected_centers < low) | (selected_centers > high))
    repeated = _crop_plane_monitor_payload(result, monitor)
    np.testing.assert_allclose(repeated[tangent[0]], result[tangent[0]], atol=1e-15)
    assert float(repeated['flux']) == pytest.approx(expected, rel=3e-15)


@pytest.mark.parametrize('axis','xyz')
@pytest.mark.parametrize('sampled',[False,True])
def test_finite_flux_crop_retains_native_autograd_functional(axis,sampled):
    amplitude = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    raw, monitor = payload(axis,'torch','+',sampled,amplitude)
    value = _crop_plane_monitor_payload(raw,monitor)['flux']
    gradient = torch.autograd.grad(value, amplitude)[0]
    delta=1e-5
    plus = _crop_plane_monitor_payload(payload(axis,'torch','+',sampled,.8+delta)[0],monitor)['flux']
    minus = _crop_plane_monitor_payload(payload(axis,'torch','+',sampled,.8-delta)[0],monitor)['flux']
    torch.testing.assert_close(gradient,(plus-minus)/(2*delta),rtol=3e-11,atol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
@pytest.mark.parametrize('axis','xyz')
@pytest.mark.parametrize('sign',['+','-'])
@pytest.mark.parametrize('spacing',[None,.004])
def test_actual_native_and_sampled_yee_planes_crop_affine_flux(axis,sign,spacing):
    from witwin.maxwell.fdtd.coords import component_coords
    tangent = [a for a in 'xyz' if a != axis]
    center = [.0003,-.0007,.0009]
    size = [.013,.015,.017]; size['xyz'.index(axis)] = 0
    fields = ('E'+tangent[0],'E'+tangent[1],'H'+tangent[0],'H'+tangent[1])
    monitor = mw.FinitePlaneMonitor('face',position=center,size=size,fields=fields,
                                    sample_spacing=spacing,normal_direction=sign)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.025,.025),)*3),
        grid=mw.GridSpec.custom(np.linspace(-.025,.025,25),np.linspace(-.025,.025,28),np.linspace(-.025,.025,32)),
        boundary=mw.BoundarySpec.none(),device='cuda',monitors=[monitor])
    solver = mw.Simulation.fdtd(scene,frequency=1e9,full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=1),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False)).prepare().solver
    solver._prepare_observers((1e9,),'none',1)
    for entry in solver._observer_spectral_entries:
        entry.update(start_step=0,end_step=1,phase_cos=1.,phase_sin=0.,h_offset_cos=1.,h_offset_sin=0.)
    for field in fields:
        coordinates=component_coords(solver.scene,field)
        points=torch.meshgrid(*(torch.tensor(c,device='cuda') for c in coordinates),indexing='ij')
        value=torch.zeros_like(points[0])
        if field==fields[0]:
            value=2.+3*points['xyz'.index(tangent[0])]-2*points['xyz'.index(tangent[1])]
        if field==fields[3]:
            value=value+(-3. if axis=='y' else 3.)
        getattr(solver,field).copy_(value)
    solver.accumulate_observers(0)
    result=_crop_plane_monitor_payload(solver.get_observer_results()['face'],monitor)
    poynting,weights=plane_normal_poynting(result)
    area=np.prod([size['xyz'.index(a)] for a in tangent])
    expected=4*.5*3*(2+3*center['xyz'.index(tangent[0])]-2*center['xyz'.index(tangent[1])])*area*(1 if sign=='+' else -1)
    assert np.prod([np.asarray(w).sum() for w in result['cell_widths'].values()]) == pytest.approx(area,rel=1e-13)
    assert float(weights.sum()) == pytest.approx(area,rel=2e-7)
    assert float((poynting*weights).sum()) == pytest.approx(expected,rel=3e-7)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_small_cell_large_offset_interpolation_keeps_geometry_precision(device,dtype):
    from witwin.maxwell.result import _resample_plane_samples
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('requires CUDA')
    u=1e6+np.array([0.,1e-5,2.5e-5,5e-5,8e-5])
    v=-2e6+np.array([0.,2e-5,4.5e-5,9e-5])
    target_u=1e6+np.array([.7e-5,3.7e-5,6.3e-5])
    target_v=-2e6+np.array([1.3e-5,3.4e-5,7e-5])
    # Evaluate the analytic affine field in extended precision on the ACTUAL
    # IEEE64 geometry, independently of Torch's interpolation implementation.
    ld=np.longdouble
    affine=lambda a,b: 2+ld('13000')*(a.astype(ld)-ld(1e6))[:,None]-ld('7000')*(b.astype(ld)+ld(2e6))[None,:]
    source=np.asarray(affine(u,v),dtype=np.float64)*(1+.3j)
    expected=torch.as_tensor(np.asarray(affine(target_u,target_v),dtype=np.float64)*(1+.3j),device=device,dtype=dtype)
    values=torch.as_tensor(source,device=device,dtype=dtype)
    actual=_resample_plane_samples(values,(u,v),(target_u,target_v))
    assert actual.dtype==dtype and bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual,expected,rtol=3*torch.finfo(actual.real.dtype).eps,atol=0.)
