"""Extend actual restored mixed HB maps without rebuilding their prefix."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_harmonic_public import setup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_restored_mixed_prefix_extends_only_new_frequency(tmp_path,monkeypatch):
    scene,config,_=setup()
    config=replace(config,harmonic_balance=replace(config.harmonic_balance,harmonics=1,pump_scales=(.5,)))
    simulation=mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    result=simulation.run();path=tmp_path/'prefix.pt';result.save(path)
    restored=mw.Result.load(path,scene=scene)
    prefix=restored._harmonic_data.decoder.prepared
    assert not hasattr(prefix['operators'][0],'refresh')
    import witwin.maxwell.mom.tensor_harmonic_preparation as preparation
    response=preparation.prepare_cell_field_response;calls=[]
    def appended(operator,*args,**kwargs):
        calls.append(operator.exterior.frequency)
        return response(operator,*args,**kwargs)
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    def forbidden(*args,**kwargs):raise AssertionError('Restored DC prefix must not be rebuilt')
    from witwin.maxwell.mom.tensor_harmonic_runtime import _materials
    from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.runtime import SystemOperator
    from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
    native_config=restored._harmonic_data.config
    materials=_materials(scene,3e8,2,config.dtype)
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,material.linear_samples[1])
        for body,material in zip(prefix['operators'][0].system.bodies,materials)])
    new_operator=SystemOperator(system,exterior.linear_samples[1],[],replace(config,harmonic_balance=None))
    with monkeypatch.context() as patch:
        patch.setattr(preparation,'prepare_cell_field_response',appended)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        model,extended=preparation.prepare_mixed_tensor_harmonic_balance(
            (*prefix['operators'],new_operator),native_config.harmonic_balance.volumes,
            materials,exterior,reuse_prepared=prefix,query_config=config.query_config)
        initial=torch.cat((restored.harmonic_state.cell_electric,
            torch.zeros_like(restored.harmonic_state.cell_electric[:1])))
        solution=model.solve(initial,pump_scale=.5,**native_config.harmonic_balance.solve_settings())
        state=recover_harmonic_state(model,extended,solution)
    assert calls==[6e8]
    assert extended['static'] is prefix['static']
    assert extended['harmonics'][0] is prefix['harmonics'][0]
    assert extended['operators'][0] is prefix['operators'][0]
    assert extended['bindings'][0] is prefix['bindings'][0]
    assert state.receipt['solver']['converged']
    assert len(state.frequencies)==3
    assert not state.receipt['physical_qualification']
