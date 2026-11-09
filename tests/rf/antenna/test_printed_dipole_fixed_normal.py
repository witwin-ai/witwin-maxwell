"""Independent normal-grid controls keep exterior samples and physical feed moments."""
from functools import lru_cache
import math

import numpy as np
import pytest
import torch

from benchmark.scenes.antenna.printed_dipole import printed_dipole_scene
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene


@lru_cache(None)
def baseline(metal='pec'):
    original=printed_dipole_scene(device='cpu',lateral_dl=.0005,substrate_cells=8,air_dl=.002,metal=metal)
    widths=[(float(a[1]-a[0]),float(a[-1]-a[-2])) for a in
            (original.grid.x_coords,original.grid.y_coords,original.grid.z_coords)]
    kwargs=dict(device='cpu',lateral_dl=.0005,substrate_cells=8,air_dl=.0005,pml_layers=48,
                metal=metal,boundary_cell_widths=widths,vacuum_reference_y_nodes=original.grid.y_coords)
    scene=printed_dipole_scene(**kwargs)
    return scene,kwargs


def component_moments(scene,component,indices,weights):
    component_axis='xyz'.index(component[-1].lower())
    half=[axis==component_axis if component[0]=='E' else axis!=component_axis for axis in range(3)]
    coords=[np.asarray(getattr(scene,axis+('_half64' if staggered else '_nodes64')))[indices[:,i].numpy()]
            for i,(axis,staggered) in enumerate(zip('xyz',half))]
    values=np.column_stack([np.ones(len(indices)),*coords])
    # Accumulate the independent affine oracle accurately, rather than losing
    # tens of float64 ulps in a sequential sum of hundreds of positive terms.
    return np.asarray([math.fsum(weights.numpy()*column) for column in values.T])


@pytest.mark.parametrize('metal',['pec','copper'])
@pytest.mark.parametrize('step,cells',[(.00025,16),(.000125,32)])
def test_combined_refinement_retains_exterior_native_node_half_support_and_real_feed(metal,step,cells):
    fixed,kwargs=baseline(metal)
    actual=printed_dipole_scene(**dict(kwargs,lateral_dl=step,substrate_cells=cells,coupon_y_dl=step,
        vacuum_reference_z_nodes=fixed.grid.z_coords))
    old,new=[prepare_scene(scene,lazy_meshgrid=True) for scene in (fixed,actual)]
    for axis,position in zip('xyz',(.032,.020,.017)):
        for part in ('nodes64','half64'):
            a,b=[np.asarray(getattr(scene,axis+'_'+part)) for scene in (old,new)]
            for sign in (-1,1):
                np.testing.assert_array_equal(a[abs(a-sign*position)<.002],b[abs(b-sign*position)<.002])
    old_z,new_z=[np.asarray(scene.grid.z_coords) for scene in (fixed,actual)]
    np.testing.assert_array_equal(old_z[abs(old_z)>.002],new_z[abs(new_z)>.002])
    assert np.diff(new_z[(new_z>=-.001)&(new_z<=0.)]).max()<=.001/cells*(1+1e-12)
    assert actual.domain.bounds==fixed.domain.bounds
    assert actual.metadata['physical_model']==fixed.metadata['physical_model']
    assert actual.monitors[0].bounds==fixed.monitors[0].bounds
    for name in ('voltage_aperture','current_surface'):
        a,b=[getattr(scene.ports[0],name) for scene in (fixed,actual)]
        assert torch.equal(a.position,b.position) and torch.equal(a.size,b.size)
    for a,b in zip(fixed.structures,actual.structures):
        for name in ('position','size','vertices'):
            if hasattr(a.geometry,name):assert torch.equal(getattr(a.geometry,name),getattr(b.geometry,name))
    ca,cb=[compile_port_geometry(scene,scene.ports[0],device='cpu') for scene in (old,new)]
    for scene,port in ((old,ca),(new,cb)):
        assert port.metadata['positive_target']=='positive_arm' and port.metadata['negative_target']=='negative_arm'
        # An affine E field has the exact fixed finite-width gap integral.
        voltage=component_moments(scene,port.voltage_component,port.voltage_indices,port.voltage_weights)
        np.testing.assert_allclose(voltage,[.002,0.,0.,0.],rtol=0,atol=2e-18)
        current={key:np.zeros(4) for key in ('Hy','Hz')}
        for component,indices,weights in zip(port.current_components,port.current_indices,port.current_weights):
            current[component]+=component_moments(scene,component,indices,weights)
        # Independently integrated affine H has circulation equal to the signed
        # 7x5 mm rectangle area; constant fields and the other moments vanish.
        np.testing.assert_allclose(current['Hy'],[0.,0.,0.,35e-6],rtol=0,atol=3e-18)
        np.testing.assert_allclose(current['Hz'],[0.,0.,-35e-6,0.],rtol=0,atol=3e-18)


@pytest.mark.parametrize('metal',['pec','copper'])
def test_retained_normal_reference_is_bitwise_idempotent_on_actual_baseline(metal):
    fixed,kwargs=baseline(metal)
    actual=printed_dipole_scene(**kwargs,vacuum_reference_z_nodes=fixed.grid.z_coords)
    for axis in 'xyz':np.testing.assert_array_equal(getattr(actual.grid,axis+'_coords'),getattr(fixed.grid,axis+'_coords'))


def test_older_unconstrained_substrate_refinement_moves_exterior_normal_samples():
    fixed,kwargs=baseline()
    actual=printed_dipole_scene(**dict(kwargs,substrate_cells=16))
    a,b=[np.asarray(scene.grid.z_coords) for scene in (fixed,actual)]
    assert not np.array_equal(a[abs(a+.017)<.002],b[abs(b+.017)<.002])


@pytest.mark.parametrize('failure',['short','nonfinite','endpoint','missing_core'])
def test_invalid_exterior_reference_is_rejected(failure):
    fixed,kwargs=baseline();reference=np.asarray(fixed.grid.z_coords).copy()
    if failure=='short':reference=reference[:2]
    elif failure=='nonfinite':reference[2]=np.nan
    elif failure=='endpoint':reference[0]+=.00001
    else:reference=reference[reference!=.002]
    with pytest.raises(ValueError,match='vacuum_reference_z_nodes'):
        printed_dipole_scene(**kwargs,vacuum_reference_z_nodes=reference)
