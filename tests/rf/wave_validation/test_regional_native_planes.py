"""Exact native-plane addressing and bounded inference, without field propagation."""
import copy
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.printed_line_regional_power import (
    _native_plane_field, regional_monitors, regional_plane_power,
)
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene
from tests._fresh_interpreter import run_without_cuda


FREQUENCIES=(.8e9,1e9,1.2e9)
BOUNDS=((-0.015,.015),(-.012,.012),(-.007,.007))


def prescribed_result(scene,schema):
    prepared=prepare_scene(scene);payloads={};fields={}
    for number,(component,selection) in enumerate(schema.items()):
        axes=component_coords(prepared,component)
        x,y,z=torch.meshgrid(*(torch.tensor(axis,dtype=torch.float64) for axis in axes),indexing='ij')
        # Independent affine complex fields differ by component and frequency.
        field=(1+number+.7*x-2*y+3*z).to(torch.complex128)
        field=torch.stack([field*(1+index*.2+index*.1j) for index in range(len(FREQUENCIES))])
        fields[component]=field
        for plane in selection['planes']:
            index=plane['normal_index']
            payloads[plane['name']]={'kind':'plane','monitor_type':'plane','axis':'z',
                'fields':(component,),'frequencies':FREQUENCIES,'frequency':FREQUENCIES[0],
                'components':{component:{'data':field[:,:,:,index],
                    'coords':tuple(np.asarray(axis) for axis in axes[:2]),
                    'plane_index':index,'plane_indices':(index,),'plane_weights':(1.,),
                    'plane_positions':(plane['normal_position_m'],),'collocated':False}}}
    return mw.Result(method='fdtd',scene=scene,frequencies=FREQUENCIES,monitors=payloads,
        solver_stats={'normalize_source':False}),fields


@pytest.mark.parametrize('grading',[None,.02])
def test_native_plane_single_frequency_shared_loss_matches_full_prescribed_field(grading):
    run_without_cuda(__file__,'_check_native_plane_single_frequency_shared_loss',grading)


def _check_native_plane_single_frequency_shared_loss(grading):
    scene=printed_line_scene('stripline',dx=.0025,longitudinal_dl=.005,sigma_e=.01,
        transverse_refinement_radius=grading,transverse_padding=.02,device='cpu')
    monitors,schema=regional_monitors(scene,BOUNDS,FREQUENCIES,observation='planes',include_flux=False)
    scene=scene.clone(monitors=monitors)
    result,fields=prescribed_result(scene,schema)
    calls=[];shared_loss=mw.Result.power_loss
    def checked_loss(self,name,**kwargs):
        assert len(self.frequencies)==1
        assert all(values.shape[0]==1 for values in kwargs['electric_fields'].values())
        calls.append(self.frequencies[0])
        return shared_loss(self,name,**kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mw.Result,'power_loss',checked_loss)
        actual=regional_plane_power(result,scene,schema,BOUNDS)
    assert tuple(calls)==FREQUENCIES
    full=mw.Result(method='fdtd',scene=scene,frequencies=FREQUENCIES)
    expected=full.power_loss('line_roi_bulk',electric_fields=fields)
    torch.testing.assert_close(torch.tensor(actual['bulk_conduction_w'],dtype=torch.float64),
        expected.conduction,rtol=1e-13,atol=1e-20)
    assert actual['dense_temporary_frequency_count']==1
    compiled=compile_power_loss_monitor(prepare_scene(scene),monitors[0])
    from witwin.maxwell.compiler.monitors import compile_fdtd_observers
    from witwin.maxwell.fdtd.observers import resolve_plane_observer
    declared={plane['name']:plane for selection in schema.values() for plane in selection['planes']}
    # Exercise the real observer geometry resolver, without constructing a CPU
    # propagation runtime or initializing CUDA.
    for observer in compile_fdtd_observers(scene):
        plane=declared[observer['monitor_name']]
        samples,coords=resolve_plane_observer(prepare_scene(scene),observer)
        assert samples==({'plane_index':plane['normal_index'],
            'plane_position':plane['normal_position_m'],'weight':1.},)
    for component,selection in schema.items():
        needed=compiled.component_masks[component].clone()
        needed[compiled.component_masks[component]]=compiled.conductivity[component]!=0
        expected_z=torch.nonzero(needed.any(0).any(0)).flatten().tolist()
        assert [plane['normal_index'] for plane in selection['planes']]==expected_z
        assert all(m.fields==(component,) and not m.compute_flux
            for m in monitors if m.name.startswith(f'line_roi_{component}_z'))
        for frequency in FREQUENCIES:
            dense=_native_plane_field(result,component,selection,frequency)
            assert dense.shape==(1,*selection['full_shape'])


@pytest.fixture
def small_plane():
    component='Ex';coords=([.1,.10001,.10003],[-.2,-.1999])
    selection={'full_shape':[3,2,4],'tangential_coords_m':coords,
        'planes':[{'name':'test','normal_index':2,'normal_position_m':.003}]}
    entry={'data':torch.ones((3,2),dtype=torch.complex64),'coords':coords,
        'plane_index':2,'plane_indices':(2,),'plane_weights':(1.,),
        'plane_positions':(.003,),'collocated':False}
    class Payload:
        def monitor(self,name,frequency=None):
            return {'frequencies':(frequency,),'components':{component:entry}}
    return Payload(),component,selection,entry


@pytest.mark.parametrize('change',[
    {'collocated':True},{'plane_indices':(1,2)},{'plane_weights':(.5,.5)},
    {'plane_positions':(.00300000001,)},{'plane_index':1},
    {'coords':([.1,.10001,.1000300001],[-.2,-.1999])},
    {'data':torch.ones((2,2),dtype=torch.complex64)},
    {'data':torch.full((3,2),float('nan'),dtype=torch.complex64)},
])
def test_native_plane_rejects_interpolated_collocated_shifted_or_incomplete_data(small_plane,change):
    result,component,selection,entry=small_plane;entry.update(change)
    with pytest.raises(ValueError):_native_plane_field(result,component,selection,1e9)


def test_native_plane_preserves_values_and_rejects_duplicate_planes(small_plane):
    result,component,selection,entry=small_plane
    entry['data']=torch.arange(6).reshape(3,2).to(torch.complex64)*(1+2j)
    dense=_native_plane_field(result,component,selection,1e9)
    torch.testing.assert_close(dense[0,:,:,2],entry['data'].to(torch.complex128),rtol=0,atol=0)
    assert dense[:,:,:,:2].count_nonzero()==0
    selection['planes']*=2
    with pytest.raises(ValueError,match='unique'):_native_plane_field(result,component,selection,1e9)


def test_native_plane_schema_rejects_frequency_loss_configuration_and_normalization():
    scene=printed_line_scene('stripline',dx=.0025,longitudinal_dl=.005,sigma_e=.01,device='cpu')
    monitors,schema=regional_monitors(scene,BOUNDS,FREQUENCIES,observation='planes',include_flux=False)
    scene=scene.clone(monitors=monitors);result,_=prescribed_result(scene,schema)
    invalid=copy.deepcopy(schema);invalid['Ex']['frequencies_hz'].reverse()
    with pytest.raises(ValueError,match='frequency'):
        regional_plane_power(result,scene,invalid,BOUNDS)
    invalid=copy.deepcopy(schema);invalid['Ex']['planes'].pop()
    with pytest.raises(ValueError,match='support'):
        regional_plane_power(result,scene,invalid,BOUNDS)
    invalid_bounds=((-0.014,.015),*BOUNDS[1:])
    with pytest.raises(ValueError,match='bounds'):
        regional_plane_power(result,scene,schema,invalid_bounds)
    result._solver_stats['normalize_source']=True
    with pytest.raises(ValueError,match='raw common'):
        regional_plane_power(result,scene,schema,BOUNDS)


@pytest.mark.parametrize('sigma',[0.,.01])
def test_native_plane_schema_covers_actual_copper_left_trace_with_and_without_bulk_sigma(sigma):
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces
    scene=printed_line_scene('stripline',dx=.0025,longitudinal_dl=.005,metal='copper',
        sigma_e=sigma,transverse_refinement_radius=.02,transverse_padding=.02,device='cpu')
    monitors,schema=regional_monitors(scene,BOUNDS,FREQUENCIES,observation='planes',include_flux=False)
    prepared=prepare_scene(scene);compiled=compile_power_loss_monitor(prepared,monitors[0])
    checked=0
    for face in compile_copper_faces(prepared,device='cpu'):
        component='E'+'xyz'[face.component]
        inside=compiled.component_masks[component][tuple(face.indices.T)]
        selected=face.indices[inside]
        planes={record['normal_index'] for record in schema[component]['planes']}
        assert set(selected[:,2].tolist()).issubset(planes)
        checked+=len(selected)
    assert checked>0
    assert all(type(monitor) is not mw.PointMonitor for monitor in monitors)
