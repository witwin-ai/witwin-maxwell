"""A second printed antenna keeps its physical dipole and coplanar feed fixed."""
import pytest
import torch
import numpy as np
from benchmark.scenes.antenna.printed_dipole import printed_dipole_scene
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.ports import compile_port_geometry
import witwin.maxwell as mw


@pytest.mark.parametrize("metal",("pec","copper"))
@pytest.mark.parametrize("lateral,cells",((.001,4),(.0005,8),(.00025,16)))
def test_fixed_printed_dipole_native_named_feed_survives_refinement(metal,lateral,cells):
    scene=printed_dipole_scene(device="cpu",metal=metal,lateral_dl=lateral,substrate_cells=cells)
    base=printed_dipole_scene(device="cpu",metal=metal)
    assert scene.domain.bounds==base.domain.bounds
    for actual,fixed in zip(scene.structures,base.structures):
        for name in ("position","size","vertices"):
            if hasattr(actual.geometry,name):assert torch.equal(getattr(actual.geometry,name),getattr(fixed.geometry,name))
    for attribute in ("current_surface","voltage_aperture"):
        assert torch.equal(getattr(scene.ports[0],attribute).position,getattr(base.ports[0],attribute).position)
        assert torch.equal(getattr(scene.ports[0],attribute).size,getattr(base.ports[0],attribute).size)
    prepared=prepare_scene(scene)
    compiled=compile_port_geometry(prepared,prepared.ports[0],device="cpu")
    assert compiled.metadata["positive_target"]=="positive_arm"
    assert compiled.metadata["negative_target"]=="negative_arm"
    assert compiled.metadata["native_parallel_path_count"]>=3


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires CUDA")
@pytest.mark.parametrize("metal",("pec","copper"))
def test_printed_dipole_actual_coplanar_named_feed_is_active_on_gpu(metal):
    scene=printed_dipole_scene(metal=metal,device="cuda",radiation_monitor=False)
    result=mw.Simulation.fdtd(scene,frequency=3e9,run_time=mw.TimeConfig(time_steps=256),
        spectral_sampler=mw.SpectralSampler(window="none"),excitations=mw.PortExcitation("feed")).run()
    assert result.port("feed").voltage.abs().max()>0
    assert torch.isfinite(result.solver.Ex).all()
    if metal=="copper":
        assert sum(float(loss["dissipated_energy"]) for loss in result.conductor_losses.values())>0


@pytest.mark.parametrize('air_dl',[.002,.001,.0005])
def test_air_refinement_retains_actual_outer_pml_geometry_and_native_feed(air_dl):
    base=printed_dipole_scene(lateral_dl=.0005,substrate_cells=8,air_dl=.002,device='cpu')
    baseline_axes=(base.grid.x_coords,base.grid.y_coords,base.grid.z_coords)
    widths=[(float(a[1]-a[0]),float(a[-1]-a[-2])) for a in baseline_axes]
    scene=printed_dipole_scene(lateral_dl=.0005,substrate_cells=8,air_dl=air_dl,
        boundary_cell_widths=widths,vacuum_reference_y_nodes=base.grid.y_coords,device='cpu')
    assert scene.domain.bounds==base.domain.bounds
    a,b=prepare_scene(base),prepare_scene(scene)
    layers=base.boundary.num_layers
    for axis in 'xyz':
        old,new=[np.asarray(getattr(s,axis+'_nodes64')) for s in (a,b)]
        np.testing.assert_array_equal(old[:layers+1],new[:layers+1])
        np.testing.assert_array_equal(old[-layers-1:],new[-layers-1:])
        assert np.diff(new).min()>=np.diff(old).min()-1e-14
        if air_dl==.002:np.testing.assert_array_equal(old,new)
    for old,new in zip(base.structures,scene.structures):
        for key in ('position','size','vertices'):
            if hasattr(old.geometry,key):assert torch.equal(getattr(old.geometry,key),getattr(new.geometry,key))
    for old,new in zip(base.monitors[0].faces,scene.monitors[0].faces):
        assert old.position==new.position and old.tangential_bounds==new.tangential_bounds
    old_y=np.asarray(base.grid.y_coords);new_y=np.asarray(scene.grid.y_coords)
    lower=int(np.searchsorted(old_y,-.015,side='right')-1)
    upper=int(np.searchsorted(old_y,.015,side='left'))
    np.testing.assert_array_equal(old_y[lower:upper+1],new_y[(new_y>=old_y[lower])&(new_y<=old_y[upper])])
    ca,cb=[compile_port_geometry(s,s.ports[0],device='cpu') for s in (a,b)]
    assert ca.metadata['native_parallel_path_count']==cb.metadata['native_parallel_path_count']
    torch.testing.assert_close(ca.voltage_weights,cb.voltage_weights,rtol=0,atol=0)
    for va,vb in zip(ca.current_weights,cb.current_weights):torch.testing.assert_close(va,vb,rtol=0,atol=0)


@pytest.mark.parametrize('widths',[[[.002,.002]],[[0.,.002]]*3,[[float('nan'),.002]]*3,[[.020,.020]]*3])
def test_invalid_retained_boundary_cells_are_rejected(widths):
    with pytest.raises(ValueError,match='boundary_cell_widths'):
        printed_dipole_scene(boundary_cell_widths=widths,device='cpu')


@pytest.mark.parametrize('target',[.001,.0005,.00025])
def test_coupon_y_refinement_preserves_existing_interface_interval_endpoints_and_outer_air(target):
    original=printed_dipole_scene(lateral_dl=.0005,substrate_cells=8,air_dl=.002,device='cpu')
    widths=[(float(a[1]-a[0]),float(a[-1]-a[-2])) for a in
            (original.grid.x_coords,original.grid.y_coords,original.grid.z_coords)]
    kwargs=dict(lateral_dl=.0005,substrate_cells=8,air_dl=.0005,device='cpu',
                boundary_cell_widths=widths,vacuum_reference_y_nodes=original.grid.y_coords)
    base=printed_dipole_scene(**kwargs)
    refined=printed_dipole_scene(**kwargs,coupon_y_dl=target)
    np.testing.assert_array_equal(base.grid.x_coords,refined.grid.x_coords)
    np.testing.assert_array_equal(base.grid.z_coords,refined.grid.z_coords)
    old,new=map(np.asarray,(base.grid.y_coords,refined.grid.y_coords))
    lower=np.searchsorted(old,-.015,side='right')-1
    upper=np.searchsorted(old,.015,side='left')
    assert all(np.any(new==point) for point in old)
    np.testing.assert_array_equal(old[:lower+1],new[new<=old[lower]])
    np.testing.assert_array_equal(old[upper:],new[new>=old[upper]])
    assert np.diff(new[(new>=old[lower])&(new<=old[upper])]).max()<=target*(1+1e-12)
    for actual,fixed in zip(refined.structures,base.structures):
        for key in ('position','size','vertices'):
            if hasattr(actual.geometry,key):assert torch.equal(getattr(actual.geometry,key),getattr(fixed.geometry,key))
    assert refined.monitors[0].bounds==base.monitors[0].bounds

    # Validate the physical feed on the actual refined nodes, not only an
    # unchanged public Box: all current-loop boundaries must remain half-grid.
    prepared=prepare_scene(refined)
    compiled=compile_port_geometry(prepared,prepared.ports[0],device='cpu')
    assert compiled.metadata['positive_target']=='positive_arm'
    assert compiled.metadata['negative_target']=='negative_arm'
    centers=.5*(new[1:]+new[:-1])
    for value in (-.0035,.0035):assert np.min(np.abs(centers-value))<1e-14


@pytest.mark.parametrize('layers,padding',[(24,0.),(48,0.),(48,.008)])
def test_boundary_reflection_control_retains_actual_pml_widths_feed_and_huygens(layers,padding):
    reference=printed_dipole_scene(lateral_dl=.0005,substrate_cells=8,device='cpu')
    widths=[(float(a[1]-a[0]),float(a[-1]-a[-2])) for a in
            (reference.grid.x_coords,reference.grid.y_coords,reference.grid.z_coords)]
    y=list(map(float,reference.grid.y_coords))
    if padding:y=[y[0]-padding,*y,y[-1]+padding]
    scene=printed_dipole_scene(lateral_dl=.0005,substrate_cells=8,air_dl=.0005,pml_layers=layers,
        air_padding=padding,boundary_cell_widths=widths,vacuum_reference_y_nodes=y,device='cpu')
    prepared=prepare_scene(scene)
    for index,axis in enumerate('xyz'):
        coord=np.asarray(getattr(prepared,axis+'_nodes64'))
        np.testing.assert_allclose(np.diff(coord[:layers+1]),widths[index][0],rtol=0,atol=1e-15)
        np.testing.assert_allclose(np.diff(coord[-layers-1:]),widths[index][1],rtol=0,atol=1e-15)
        assert coord[layers]==pytest.approx(reference.domain.bounds[index][0]-padding,abs=1e-15)
        assert coord[-layers-1]==pytest.approx(reference.domain.bounds[index][1]+padding,abs=1e-15)
    compiled=compile_port_geometry(prepared,prepared.ports[0],device='cpu')
    assert compiled.metadata['positive_target']=='positive_arm'
    assert compiled.metadata['negative_target']=='negative_arm'
    assert scene.monitors[0].bounds==reference.monitors[0].bounds
