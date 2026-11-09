"""Factorized support-tree maps against unchanged independent physical oracles."""
import json
import pytest
import torch
import test_fmm_changing_carrier_moments as original
from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection
from witwin.maxwell.mom.fmm_expansion import multi_indices

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _select_compact(monkeypatch):
    def compact(*args,**kw):
        return NestedDirectionalRWGProjection(*args,**kw,phase_storage='compact_phase_v1')
    monkeypatch.setattr(original,'NestedDirectionalRWGProjection',compact)


@pytest.mark.parametrize('kind',('single','global','mixed_leaf'))
def test_compact_all_node_independent_four_maps_full_H_current_higher_and_support(kind,monkeypatch,tmp_path):
    _select_compact(monkeypatch)
    original.test_variable_carrier_orders_independent_all_node_four_maps_H_higher_and_support(kind,tmp_path)
    record=json.loads((tmp_path/'tree.json').read_text())
    assert record['receipt']['phase_storage']=='compact_phase_v1'
    assert record['receipt']['shared_layouts_counted_once_by_tree']


def test_compact_original_support_phase_native_guards_and_RHS_bounds(monkeypatch,tmp_path):
    _select_compact(monkeypatch)
    original.test_variable_carrier_unique_inventory_preflight_support_phase_node_mutations_and_chunk_bounds(monkeypatch,tmp_path)


@pytest.mark.parametrize('lossy',(False,True))
def test_compact_nonroot_original_high_ka_weak_and_H(lossy,monkeypatch,tmp_path):
    _select_compact(monkeypatch)
    original.test_nonroot_variable_carrier_tree_original_high_ka_weak_and_full_H(lossy,tmp_path)
    record=json.loads((tmp_path/'weak.json').read_text())
    for name in ('target_receipt','source_receipt'):
        assert record[name]['phase_storage']=='compact_phase_v1'
        # The original dense-phase case owns 49,960,112 bytes for each tree.
        assert record[name]['owned_retained_bytes']<49960112
        record[name]['historical_identical_case_dense_phase_bytes']=49960112
    (tmp_path/'weak.json').write_text(json.dumps(record,indent=2)+'\n')


def test_unique_shared_caches_original_inventory_preflight_and_saved_backward_mutations(monkeypatch,tmp_path):
    import witwin.maxwell.mom.fmm_directional_moments as module
    _select_compact(monkeypatch);owner=original._owner('single')
    assert owner.phase_layouts and owner.center_layouts
    keys={(op.source_order,op.target_order,op.phase_order) for op in owner.transfers.values()}
    assert set(owner.phase_layouts)==keys
    assert set(owner.center_layouts)=={key[1] for key in keys}
    assert len(owner.transfers)>len(owner.phase_layouts)
    for op in owner.transfers.values():
        assert op.phase_layout is owner.phase_layouts[op.source_order,op.target_order,op.phase_order]
        assert op.center_layout is owner.center_layouts[op.target_order]
        assert not op.owns_phase_layout and not op.owns_center_layout
        assert op.retained_bytes==128+32*op.phase_count
        assert op.planned_tensor_workspace_bytes<=owner.planned_tensor_workspace_bytes
    phase_bytes=sum(t.untyped_storage().nbytes() for shared in owner.phase_layouts.values() for t in shared.tensors())
    center_bytes=sum(t.untyped_storage().nbytes() for shared in owner.center_layouts.values() for t in shared.tensors())
    assert phase_bytes==owner.owned_phase_layout_bytes
    assert center_bytes==owner.owned_center_layout_bytes
    assert phase_bytes==sum(8*len(multi_indices(p))*len(multi_indices(L))+32*len(multi_indices(L)) for _,p,L in keys)
    assert center_bytes==sum(144*len(multi_indices(p))*(p+1) for p in owner.center_layouts)
    tensors=owner.tensors()
    assert len({t.untyped_storage().data_ptr() for t in tensors})==len(tensors)
    assert all(t.untyped_storage().nbytes()==t.numel()*t.element_size() for t in tensors)
    assert sum(t.untyped_storage().nbytes() for t in tensors)==owner.retained_bytes==owner.planned_retained_bytes
    hypothetical_dense=(owner.retained_bytes-phase_bytes-center_bytes
        -sum(op.retained_bytes for op in owner.transfers.values())
        +sum(128+32*op.source_count*op.target_count for op in owner.transfers.values()))
    assert hypothetical_dense==868912 and owner.retained_bytes<hypothetical_dense
    args=dict(basis=owner.basis,tree=owner.tree,direction=owner.node_directions[0],wave_number=owner.wave_number,
        order=owner.order,quadrature_order=owner.quadrature_order,face_batch=owner.face_batch,
        node_directions=owner.node_directions,node_orders=owner.node_orders,phase_orders=owner.phase_orders,
        phase_storage='compact_phase_v1')
    def forbidden(*a,**kw):raise AssertionError('Rejected capacity reached geometry, layouts, moments or tables')
    with monkeypatch.context() as patch:
        for name in ('PreparedRWGGeometry','ChangingCarrierPhaseLayout','SeparableMultipoleShift',
                'DirectionalChangingCarrierShift','DirectionalRWGProjection'):
            patch.setattr(module,name,forbidden)
        for kw in ({'retained_budget_bytes':owner.retained_bytes-1},
                {'tensor_workspace_budget_bytes':owner.planned_tensor_workspace_bytes-1}):
            altered=dict(args);altered.update(kw)
            with pytest.raises(MemoryError):NestedDirectionalRWGProjection(**altered)
        for kw in ({'phase_storage':'invalid'},dict(node_directions=None,node_orders=None,phase_orders=None)):
            altered=dict(args);altered.update(kw)
            with pytest.raises(ValueError):NestedDirectionalRWGProjection(**altered)
    for field in ('phase_storage','owned_phase_layout_bytes','owned_center_layout_bytes','phase_layouts','center_layouts'):
        previous=getattr(owner,field)
        replacement='dense_v1' if field=='phase_storage' else ({} if field.endswith('layouts') else previous+1)
        setattr(owner,field,replacement)
        with pytest.raises(ValueError,match='changed'):owner.receipt()
        setattr(owner,field,previous);owner.assert_current()
    cache_mutations=[]
    for name in ('phase_layouts','center_layouts'):
        current=torch.randn((owner.tree.size,),device='cuda',dtype=torch.complex128,requires_grad=True)
        root=owner.source_upward(current)[0]
        shared=next(iter(getattr(owner,name).values()));shared.tensors()[0].add_(0)
        with pytest.raises(ValueError,match='changed'):torch.autograd.grad(root,current,torch.ones_like(root))
        cache_mutations.append(name)
        owner=original._owner('single')
    record=dict(compact_owned_bytes=owner.retained_bytes,historical_dense_owned_bytes=hypothetical_dense,
        phase_layout_bytes=phase_bytes,center_layout_bytes=center_bytes,
        phase_layout_count=len(owner.phase_layouts),center_layout_count=len(owner.center_layouts),
        edge_count=len(owner.transfers),exact_unique_inventory=True,preflight_before_all_factories=True,
        saved_backward_cache_mutations=cache_mutations,bit_exact_actions_claimed=False,qualification=False)
    (tmp_path/'shared_inventory.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
