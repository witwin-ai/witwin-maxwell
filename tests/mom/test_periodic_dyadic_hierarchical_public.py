"""Manufactured selected matrices isolate public H/H2 and exact state routing."""
import copy
import hashlib
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SurfaceData
from witwin.maxwell.mom.linalg import StoppingRule
from test_periodic_dyadic_public import problem

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('backend',['hmatrix','h2'])
@pytest.mark.parametrize('dimension',[1,2])
def test_public_hierarchy_schema19_native_and_bench_exact_replay(tmp_path,monkeypatch,backend,dimension):
    import witwin.maxwell.mom.periodic_dyadic_nearfield as selected
    import witwin.maxwell.mom.periodic_dyadic_operator as assembly
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicRegularKernel
    from witwin.maxwell.mom.hmatrix import HierarchicalOperator
    from witwin.maxwell.mom.h2_operator import HierarchicalH2Operator
    from maxwell_bench.lifted_state import write_lifted_states,read_lifted_states
    scene,config=problem(dimension)
    config=replace(config,operator_backend=backend,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',fallback='none'))
    calls=[]
    def block(system,i,j,rows,cols,*,reference,exterior_kernel,interior_kernels,config):
        assert isinstance(reference,QuasiperiodicDyadicRegularKernel)
        assert exterior_kernel.medium is reference.medium
        assert all(interior_kernels[b.region_id].medium is b.interior for b in system.bodies)
        value=reference.medium.epsilon_r.new_zeros(2*len(rows),2*len(cols))
        diagonal=(rows[:,None]==cols[None,:]).to(value.dtype) if i==j else value[:len(rows),:len(cols)]
        value[:len(rows),:len(cols)]=diagonal
        value[len(rows):,len(cols):]=diagonal
        calls.append((i,j))
        panel=dict(triangle_pair_counts=[0]*5,maximum_classification_pairs=0)
        return value,dict(exterior_primary=panel,local_interior=panel if i==j else None,
            nonprimary_exterior=dict(panel_pairs=0,maximum_panel_pairs=0),physical_qualification=False)
    def forbidden(*args,**kwargs):raise AssertionError('Public hierarchy must not assemble a global dense matrix')
    monkeypatch.setattr(selected,'periodic_dyadic_system_block',block)
    monkeypatch.setattr(assembly,'assemble_periodic_dyadic_system',forbidden)
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    assert set(calls)=={(0,0),(0,1),(1,0),(1,1)}
    assert result.stats()['converged']
    data=result.surface_currents
    assert torch.count_nonzero(data.scattered_coefficients)==0
    payload=data.snapshot()
    assert payload['schema_version']==19 and payload['operator_backend']==backend
    assert payload['operator_metadata']['operator_identity_payload']['version']=='periodic_dyadic_hierarchical_operator_identity_v1'
    points=torch.tensor([[.1,.2,.3],[1.1,.2,.3]],device='cuda',dtype=torch.float64)
    fields=result.query_fields(points,region_id=0,field_kind='incident')
    native=tmp_path/'result.pt';result.save(native)
    root=tmp_path/'bench'
    digest=write_lifted_states([result],root,source_ids=['mode'],physical_group_id='routing-only',split='pilot')
    monkeypatch.setattr(selected,'periodic_dyadic_system_block',forbidden)
    monkeypatch.setattr(HierarchicalOperator,'__init__',forbidden)
    monkeypatch.setattr(HierarchicalH2Operator,'__init__',forbidden)
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    restored=mw.Result.load(native,scene=scene)
    assert restored.surface_currents.config==config
    assert torch.equal(restored.surface_currents.scattered_coefficients,data.scattered_coefficients)
    for actual,expected in zip(restored.query_fields(points,region_id=0,field_kind='incident'),fields):
        assert torch.equal(actual,expected)
    manifest,arrays=read_lifted_states(root,expected_manifest_sha256=digest,device='cuda')
    assert manifest.native_schema_version==19 and not manifest.physical_qualification
    assert np.concatenate((arrays['J_scattered'][0],arrays['M_scattered'][0])).tobytes()==payload['scattered_coefficients'].numpy().tobytes()
    for mutation in ('schema','backend','metric','budget'):
        changed=copy.deepcopy(payload)
        if mutation=='schema':changed['schema_version']=18
        elif mutation=='backend':changed['operator_backend']='dense'
        elif mutation=='metric':changed['operator_metadata']['residual_metric_policy']['impedance_squared_ohm2']*=2
        else:changed['hierarchical_config']['parameters']['tolerance']*=2
        with pytest.raises(ValueError):SurfaceData.restore(changed,scene,3e8)
    saved=json.loads((root/'manifest.json').read_bytes())
    saved['native_operator']['payload']['residual_metric_policy']['impedance_squared_ohm2']*=2
    def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    saved['native_operator']['identity']=sha(saved['native_operator']['payload'])
    saved['rows'][0]['native_rhs_identity']=hashlib.sha256((saved['native_operator']['identity']+':'+saved['rows'][0]['carrier_sha256']).encode()).hexdigest()
    raw=json.dumps(saved,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    (root/'manifest.json').write_bytes(raw)
    with pytest.raises(ValueError):read_lifted_states(root,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),device='cuda')
