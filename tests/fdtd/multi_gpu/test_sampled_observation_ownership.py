"""Physical midpoint ownership must contain every native interpolation bracket."""
from types import SimpleNamespace

import numpy as np
import pytest

import witwin.maxwell as mw
from witwin.maxwell.fdtd.distributed.observations import sampled_monitor_for_shard
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan


@pytest.mark.parametrize('ranks',(2,3))
@pytest.mark.parametrize('axis',('y','z'))
@pytest.mark.parametrize('spacing',(.007,.019,.2))
def test_nonuniform_midpoints_have_one_owner_with_all_yee_brackets(ranks,axis,spacing):
    nodes=np.array((-.05,-.041,-.03,-.022,-.013,-.005,0.,.004,.012,.025,.039,.05))
    centers=.5*(nodes[:-1]+nodes[1:])
    scene=SimpleNamespace(x_half64=centers)
    plan=FDTDPartitionPlan(global_shape=(len(nodes),7,9),devices=tuple(f'cuda:{rank}' for rank in range(ranks)))
    size=(.086,0.,.06) if axis=='y' else (.086,.06,0.)
    monitor=mw.FinitePlaneMonitor('midpoints',size=size,sample_spacing=spacing)
    low,high=monitor.tangential_bounds['x']
    count=max(2,int(np.ceil((high-low)/spacing)))
    points=low+(np.arange(count)+.5)*(high-low)/count
    owners=np.zeros(count,dtype=np.int64)
    for layout in plan.shard_layouts:
        local=sampled_monitor_for_shard(monitor,layout,scene)
        if local is None:
            continue
        selected=local._fdtd_sample_x_indices
        owners[selected]+=1
        for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            coordinate=nodes if component in ('Ey','Ez','Hx') else centers
            upper=np.clip(np.searchsorted(coordinate,points[selected],side='left'),1,len(coordinate)-1)
            allocation=layout.component(component).allocation_global_slice[0]
            assert np.all(upper-1>=allocation.start)
            assert np.all(upper<allocation.stop)
        assert monitor.sample_spacing==local.sample_spacing
        assert monitor.tangential_bounds==local.tangential_bounds
    np.testing.assert_array_equal(owners,np.ones(count,dtype=np.int64))


def test_x_normal_sampled_plane_retains_single_native_plane_owner_contract():
    monitor=mw.FinitePlaneMonitor('normal_x',size=(0.,.04,.04),sample_spacing=.007)
    assert sampled_monitor_for_shard(monitor,None,None) is monitor


def test_distributed_resume_circuit_identity_uses_declarative_graph_physics():
    from witwin.maxwell.fdtd.distributed.resume import _signature
    plan=FDTDPartitionPlan(global_shape=(9,7,9),devices=('cuda:0','cuda:1'))
    def solver(resistance):
        circuit=mw.Circuit('physical_graph')
        node=circuit.node('load')
        circuit.add(mw.Resistor('R',node,circuit.ground,resistance))
        circuit.bind_port('gap',positive=node,negative=circuit.ground)
        scene=SimpleNamespace(sources=(),ports=(),circuits=(circuit,),networks=(),thin_wires=(),monitors=())
        return SimpleNamespace(logical_scene=scene,shard_layouts=plan.shard_layouts,shards=())
    first,rebuilt=solver(50.),solver(50.)
    assert first.logical_scene.circuits[0] is not rebuilt.logical_scene.circuits[0]
    assert _signature(first)==_signature(rebuilt)
    assert _signature(first)!=_signature(solver(51.))
