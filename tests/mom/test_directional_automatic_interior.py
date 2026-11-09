"""Actual automatically partitioned magnetic interiors in complete coordinates."""
import json
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks,scalar_regional_block
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from test_directional_sparse_system import _basis,_original_reference
from test_directional_far_actions import _original_global_far_block

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _provider():
    entries=[]
    for i,(z,eps,mu,se,sm) in enumerate(((-.3,2.,1.3,.01,.02),(.3,3.,1.5,.015,.005))):
        entries.append((i,str(i),_basis(z),compile_medium(mw.Material(eps_r=eps,mu_r=mu,sigma_e=se,sigma_m=sm),3e8)))
    config=IntegrationConfig(order=6,singular_order=3,pair_batch=4,far_pair_batch=16,near_max_order=32)
    return ScalarSystemBlocks(SurfaceSystem.create(entries),compile_medium(mw.Material(),3e8),config)


def _plans(provider):
    eta=abs(provider.exterior.eta)
    common=dict(kernel_error_budget=KernelErrorBudget(1e-6,1e-4),ideal_rwg_error_budget=1e-4,
        balancing_impedance=eta,leaf_size=8,separation_ratio=.1,cone_subdivisions=(2,))
    return (AutomaticDirectionalRegionPlan(provider,provider.exterior,**common),
        *(AutomaticDirectionalRegionPlan(b.basis,b.interior,geometry=g,**common)
            for b,g in zip(provider.system.bodies,provider.geometry,strict=True)))


def test_actual_automatic_magnetic_interior_far_complete_support_partition(tmp_path):
    provider=_provider();plans=_plans(provider)
    assert provider.system.total_edges==396
    for p in plans:
        assert p.far_nodes and p.near_nodes
        seen=torch.zeros((p.tree.size,p.tree.size),device='cuda',dtype=torch.int64)
        for a,b in (*p.far_nodes,*p.near_nodes):
            rows=p.tree.permutation[int(p.tree.starts[a]):int(p.tree.stops[a])]
            cols=p.tree.permutation[int(p.tree.starts[b]):int(p.tree.stops[b])]
            seen[rows[:,None],cols]+=1
        assert bool((seen==1).all())
        assert p.receipt()['ideal_bound']['absolute_bound']<=1e-4
    record=dict(phase='automatic_all_regions_admitted',complete_edges=396,
        receipts=[p.receipt() for p in plans],positive_actual_interior_far=[len(p.far_nodes) for p in plans[1:]],
        all_physical_pairs_covered_once=True,geometry_and_materials_unchanged=True,qualification=False)
    (tmp_path/'catalog.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


def test_automatic_actual_complete_magnetic_system_original_actions_and_native_current(tmp_path):
    provider=_provider();plans=_plans(provider)
    regions=[]
    for index,p in enumerate(plans):
        print(dict(stage='construct_automatic_region',index=index,far=len(p.far_nodes),near=len(p.near_nodes)),flush=True)
        regions.append(p.build_region(integration=provider.config,quadrature_order=12,face_batch=4,
            tensor_workspace_budget_bytes=2*1024**3))
        print(dict(stage='automatic_region_constructed',index=index,receipt=regions[-1].receipt()),flush=True)
    actions=DirectionalSystemActions(provider,regions[0],regions[1:],ideal_rwg_error_budget=1e-4,
        tensor_workspace_budget_bytes=2*1024**3)
    reference=_original_reference(provider);n=provider.system.total_edges;eta=abs(provider.exterior.eta)
    gen=torch.Generator(device='cuda').manual_seed(938724)
    q=torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,generator=gen).transpose(0,1).requires_grad_()
    z=torch.randn(q.shape,device='cuda',dtype=torch.complex128,generator=gen).requires_grad_()
    qbalanced=q*reference.real.new_tensor([1.]*n+[eta]*n)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        value=actions.apply(qbalanced)
    assert not saved
    l=reference.real.new_tensor([eta]*n+[1.]*n)
    forward=float(((value-qbalanced@reference.mT)*l).norm()/((qbalanced@reference.mT)*l).norm())
    h=actions.adjoint(z);original_h=z@reference.conj()
    h_error=float((h-original_h).norm()/original_h.norm())
    current,=torch.autograd.grad(value,q,z,create_graph=True)
    expected_current=h*reference.real.new_tensor([1.]*n+[eta]*n)
    native=float((current-expected_current).norm()/expected_current.norm())
    tangent=torch.randn(q.shape,device='cuda',dtype=torch.complex128,generator=gen)
    higher,=torch.autograd.grad(current,z,tangent)
    expected_higher=actions.apply(tangent*reference.real.new_tensor([1.]*n+[eta]*n))
    native_higher=float((higher-expected_higher).norm()/expected_higher.norm())
    lhs=(value.conj()*z).sum();rhs=(qbalanced.conj()*h).sum()
    dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach())
    record=dict(phase='actual_original_actions',full_edges=396,full_currents=792,noncontiguous_rhs=15,
        original_forward=forward,original_H=h_error,native_current=native,native_higher=native_higher,
        dot=dot,saved_tensors=len(saved),ideal_bound=actions.ideal_rwg_truncation_bound(),
        region_receipts=[r.receipt() for r in regions],qualification=False,
        reference_is_full_original_matrix=True,sampled_full_domain_actions_not_matrix_norm_certificate=True)
    (tmp_path/'actual.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
    assert forward<1e-8 and h_error<1e-8
    assert native<1e-11 and native_higher<1e-11 and dot<1e-11
    # Independently integrate selected physical FAR rectangles at q16. All
    # original full-domain action samples above include every retained block;
    # these separate q16 probes do not certify unsampled rectangle quadrature.
    errors=[]
    for index,r in enumerate(regions):
        for coupling,rectangle in zip(r.far.couplings,r.far.rectangles,strict=True):
            if coupling is not r.far.couplings[0] and coupling is not r.far.couplings[-1]:continue
            start,stop,low,high,a,b=rectangle
            rows=r.test.tree.permutation[start:stop];cols=r.source.tree.permutation[low:high]
            if index==0:original=_original_global_far_block(provider,rows,cols,r.medium)
            else:original,_=scalar_regional_block(r.test.basis,r.source.basis,rows,cols,(r.medium,),
                IntegrationConfig(order=16,singular_order=3),geometry=(r.test.geometry,r.source.geometry))
            count=2*r.test.tree.size;rr,cc=torch.cat((rows,rows+count//2)),torch.cat((cols,cols+count//2))
            basis=torch.zeros((len(cc),count),device='cuda',dtype=torch.complex128)
            basis[torch.arange(len(cc),device='cuda'),cc]=1
            actual=r.far.apply(basis)[:,rr].mT
            left=original.real.new_tensor([eta]*len(rows)+[1.]*len(rows));right=original.real.new_tensor([1.]*len(cols)+[eta]*len(cols))
            error=float((left[:,None]*(actual-original)*right).norm()/(left[:,None]*original*right).norm())
            errors.append(dict(region=index,test_physical=r.test.context_keys[a][0],source_physical=r.source.context_keys[b][0],error=error))
            assert error<1e-8
    record['independent_q16_far_probes']=errors
    (tmp_path/'actual.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
