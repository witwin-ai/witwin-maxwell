"""Actual catalogs beyond32 families under explicit native capacity limits."""
import json
import weakref
import mpmath as mp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh, RWGBasis, SurfaceSystem
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fmm_order import AutomaticDirectionalRegionPlan, AutomaticDirectionalDAGPlan
from witwin.maxwell.mom.fmm_directional_moments import SparseDirectionalRWGProjection, _directional_carrier_catalog_workspace_bytes
from witwin.maxwell.mom.fmm_directional_rwg import (_sparse_directional_rwg_admission,
    DirectionalFFTElectromagneticM2L, DirectionalDAGFarActions)
from witwin.maxwell.mom.direct_capacity import storage_bytes
from test_directional_sparse_coupling import _mp_integrals
from test_directional_far_actions import _original_global_far_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _provider():
    entries = []
    centers = ((0.,0.,0.),(2.,.31,.13),(-1.1,2.4,.5),(.45,-1.3,2.65),
        (3.2,1.75,-.95),(-2.7,-.45,1.15),(1.2,3.1,2.2),(-1.65,1.8,-2.4),(2.5,-2.2,-1.7))
    for i, center in enumerate(centers):
        vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
            device='cuda',dtype=torch.float64)*.015
        vertices += vertices.new_tensor(center)
        faces = torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],device='cuda')
        basis = RWGBasis(SurfaceMesh(vertices, faces))
        medium = compile_medium(mw.Material(eps_r=2.+.1*i,mu_r=1.3+.03*i,sigma_e=.001,sigma_m=.002),3e8)
        entries.append((i,str(i),basis,medium))
    return ScalarSystemBlocks(SurfaceSystem.create(entries),compile_medium(mw.Material(),3e8),
        IntegrationConfig(order=3,singular_order=3,pair_batch=4,far_pair_batch=8))


def _root(provider, limit):
    return AutomaticDirectionalRegionPlan(provider,provider.exterior,
        kernel_error_budget=mw.KernelErrorBudget(1e-10,1e-10),ideal_rwg_error_budget=1e-4,
        balancing_impedance=abs(provider.exterior.eta),leaf_size=8,separation_ratio=.15,
        cone_subdivisions=(2**20,),max_families=limit)


def _relative(actual, reference):
    return float(((actual-reference).norm()/reference.norm()).detach())


@pytest.fixture(scope='module')
def catalog():
    provider = _provider()
    root = _root(provider,512)
    assert len(root.orders)>32
    plan = AutomaticDirectionalDAGPlan(root,quadrature_order=12,face_batch=4,
        max_candidates=1,max_contexts=8192,tensor_workspace_budget_bytes=512*1024**2)
    assert len(plan.orders)>32
    far = plan.build_far()
    assert far.test is far.source
    owner = far.test
    return provider,root,plan,owner


def test_public_catalog_policy_roundtrip_and_strict_integer_capacity(tmp_path):
    config = mw.DirectionalFMMConfig(max_families=512)
    payload = json.loads(json.dumps(config.payload()))
    assert mw.FMMConfig.restore(payload)==config
    assert mw.DirectionalFMMConfig().max_families==32
    for value in (0,-1,True,1.5):
        with pytest.raises(ValueError): mw.DirectionalFMMConfig(max_families=value)
    from dataclasses import replace
    from witwin.maxwell.mom.runtime import _numerical_identity
    from witwin.maxwell.mom.linalg import StoppingRule
    options = dict(operator_backend='fmm',riesz_storage='local_rwg_v1',
        stopping=StoppingRule(preconditioner='rwg_diagonal'))
    a,b = mw.MoMConfig(fmm=config,**options),mw.MoMConfig(fmm=replace(config,max_families=513),**options)
    assert _numerical_identity(a)!=_numerical_identity(b)
    (tmp_path/'configuration.json').write_text(json.dumps(dict(payload=payload,
        unchanged_default_capacity=32,configured_catalog_bound_in_numerical_identity=True,
        no_new_physics_or_scaling_qualification=True,qualification=False),indent=2)+'\n')


def test_actual_all_root_material_MP100_and_tail_family_original_q16(catalog,tmp_path):
    provider,root,plan,owner = catalog
    assert len(owner.carriers)==len(plan.orders)>32
    assert storage_bytes(owner.tensors())==owner.retained_bytes==plan.planned_projection_retained_bytes
    lookup = {key:i for i,key in enumerate(owner.context_keys)}
    errors = []
    with mp.workdps(100):
        physical = {node:_mp_integrals(owner,node) for node in {key[0] for key in lookup}}
        integrals = {key:physical[key[0]] for key in lookup}
        z = mp.mpf(root.balancing_impedance)
        k = abs(mp.mpc(root.medium.k))
        e,m = abs(mp.mpc(1j*root.medium.omega*root.medium.epsilon)),abs(mp.mpc(-1j*root.medium.omega*root.medium.mu))
        for (a,b),family in zip(root.far_nodes,plan.family_ids,strict=True):
            ta,sb = lookup[a,family],lookup[b,family]
            _,_,bound = _sparse_directional_rwg_admission(owner,ta,owner,sb,root.medium,
                root.balancing_impedance,128*1024**2,dag=True)
            left,right = integrals[a,family],integrals[b,family]
            dsum = sum((u*v+q*r/k**2)**2 for u,q in left for v,r in right)
            usum = sum((u*v)**2 for u,q in left for v,r in right)
            expected = mp.sqrt((z**4*e**2+m**2)*dsum*mp.mpf(bound['effective_green_absolute'])**2
                +6*z**2*usum*mp.mpf(bound['effective_gradient_component_absolute'])**2)
            errors.append(float(abs(mp.mpf(bound['absolute_bound'])/expected-1)))
    assert errors and max(errors)<1e-11
    mp_maximum = max(errors)
    chosen = max(range(len(root.far_nodes)),key=lambda i:plan.family_ids[i])
    a,b = root.far_nodes[chosen];family = plan.family_ids[chosen]
    assert family>=32
    ta,sb = lookup[a,family],lookup[b,family]
    cone = root.plans[chosen]
    coupling = DirectionalFFTElectromagneticM2L.from_dag_contexts(owner,ta,owner,sb,root.medium,
        cone.normalization_lengths[0],kernel_error_budget=cone.kernel_error_budget,
        ideal_rwg_error_budget=root.per_pair_budget,balancing_impedance=root.balancing_impedance)
    rows = root.tree.permutation[int(root.tree.starts[a]):int(root.tree.stops[a])]
    cols = root.tree.permutation[int(root.tree.starts[b]):int(root.tree.stops[b])]
    n = root.tree.size
    actions = DirectionalDAGFarActions(owner,owner,root.medium,(coupling,),
        ideal_rwg_error_budget=root.per_pair_budget,balancing_impedance=root.balancing_impedance,
        tensor_workspace_budget_bytes=512*1024**2)
    cc = torch.cat((cols,cols+n));rr = torch.cat((rows,rows+n))
    vectors = torch.eye(2*n,device='cuda',dtype=torch.complex128)[cc]
    actual = actions.apply(vectors)[:,rr].mT
    reference = _original_global_far_block(provider,rows,cols,root.medium)
    eta = root.balancing_impedance
    left,right = actual.real.new_tensor([eta]*len(rows)+[1.]*len(rows)),actual.real.new_tensor([1.]*len(cols)+[eta]*len(cols))
    weak = float((left[:,None]*(actual-reference)*right).norm()/(left[:,None]*reference*right).norm())
    assert weak<1e-8
    full = actual.new_zeros((2*n,2*n)); full[rr[:,None],cc[None,:]] = actual
    original = full.new_zeros(full.shape); original[rr[:,None],cc[None,:]] = reference
    generator = torch.Generator(device='cuda').manual_seed(720413)
    x = torch.randn((3,5,2*n),device='cuda',dtype=torch.complex128,
        generator=generator).transpose(0,1).requires_grad_()
    dual = torch.randn(x.shape,device='cuda',dtype=torch.complex128,
        generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t,lambda t:t):
        y = actions.apply(x)
    assert not saved and not x.is_contiguous()
    H = actions.adjoint(dual)
    gradient, = torch.autograd.grad(y,x,dual,retain_graph=True)
    adjoint_gradient, = torch.autograd.grad(H,dual,x,retain_graph=True)
    first, = torch.autograd.grad(y.abs().square().sum(),x,create_graph=True)
    tangent = torch.randn_like(x); second, = torch.autograd.grad(first,x,tangent)
    lhs,rhs = (y.conj()*dual).sum(),(x.conj()*H).sum()
    errors = dict(forward=_relative(y,x@full.mT),H=_relative(H,dual@full.conj()),
        current=_relative(gradient,H),adjoint_current=_relative(adjoint_gradient,y),
        higher=_relative(second,2*actions.adjoint(actions.apply(tangent))),
        dot=float(((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs())).detach()))
    original_H = _relative(H,dual@original.conj())
    assert original_H<1e-8 and max(errors.values())<1e-11
    (tmp_path/'catalog.json').write_text(json.dumps(dict(actual_bodies=9,physical_edges=n,
        root_families=len(root.orders),selected_DAG_families=len(plan.orders),actual_FAR=len(root.far_nodes),
        all_root_material_MP100_max_relative=mp_maximum,MP_absolute_tolerance=0,
        tail_family_id=family,tail_original_q16_weak_relative=weak,
        maps_owned_bytes=owner.retained_bytes,configured_capacity=512,
        fifteen_noncontiguous_RHS=True,native_errors=errors,original_H_relative=original_H,
        saved_internal_tensors=len(saved),full_system_solver_physics_and_scaling_qualified=False,
        qualification=False),indent=2)+'\n')


def test_actual_catalog_limit_duplicate_keys_and_workspace_before_maps(monkeypatch,tmp_path):
    provider = _provider()
    root = _root(provider,512)
    plan = root
    assert len(plan.orders)>32
    def forbidden(*args,**kwargs): raise AssertionError('Rejected catalog reached moments or CUDA unique')
    with monkeypatch.context() as patch:
        patch.setattr('witwin.maxwell.mom.fmm_directional_moments.DirectionalRWGProjection',forbidden)
        patch.setattr('witwin.maxwell.mom.fmm_directional_moments.SeparableMultipoleShift',forbidden)
        with pytest.raises(MemoryError,match='family capacity'): _root(provider,32)
        carriers = torch.cat((plan.carriers,plan.carriers[:1]))
        orders = (*plan.orders,plan.orders[0])
        requests = tuple((0,family) for family in range(len(orders)))
        with pytest.raises(ValueError,match='canonical and distinct'):
            SparseDirectionalRWGProjection(root.basis,root.tree,carriers,orders,requests,root.medium.k,
                quadrature_order=12,face_batch=4)
        patch.setattr(torch,'unique',forbidden)
        with pytest.raises(MemoryError,match='before canonical validation'):
            SparseDirectionalRWGProjection(root.basis,root.tree,plan.carriers,plan.orders,
                tuple((0,family) for family in range(len(plan.orders))),
                root.medium.k,quadrature_order=12,face_batch=4,
                tensor_workspace_budget_bytes=_directional_carrier_catalog_workspace_bytes(len(plan.orders))-1)
    (tmp_path/'capacity.json').write_text(json.dumps(dict(actual_default_limit_rejected=True,
        no_rejected_FAR_relabelled_near=True,duplicate_carrier_order_bits_rejected=True,
        catalog_workspace_preflight_before_unique_and_moments=True,
        qualification=False),indent=2)+'\n')


def test_native_catalog_preserves_signed_zero_and_order_bit_identity(catalog,monkeypatch,tmp_path):
    _,root,_,_ = catalog
    carriers = root.carriers.new_tensor([[1.,0.,0.],[1.,-0.,0.],[1.,0.,0.]])
    orders = (1,1,2)
    calls = []
    temporaries = []
    original_unique = torch.unique
    original_cat = torch.cat
    def cat(values,**kwargs):
        if len(values)==2 and values[0].dtype==torch.int64 and values[0].shape==(3,3):
            temporaries.extend((weakref.ref(values[0]),weakref.ref(values[1]._base)))
        return original_cat(values,**kwargs)
    def unique(keys,**kwargs):
        assert keys.is_cuda and keys.dtype==torch.int64 and keys.shape==(3,4)
        temporaries.append(weakref.ref(keys))
        calls.append(dict(device=str(keys.device),rows=keys.shape[0],columns=keys.shape[1]))
        return original_unique(keys,**kwargs)
    class ReachedCapacity(RuntimeError): pass
    def reached(*args,**kwargs):
        assert len(temporaries)==3 and all(reference() is None for reference in temporaries)
        raise ReachedCapacity('Canonical catalog admitted')
    with monkeypatch.context() as patch:
        patch.setattr(torch,'unique',unique)
        patch.setattr(torch,'cat',cat)
        patch.setattr('witwin.maxwell.mom.fmm_directional_moments._directional_context_capacity',reached)
        with pytest.raises(ReachedCapacity):
            SparseDirectionalRWGProjection(root.basis,root.tree,carriers,orders,
                ((0,0),(0,1),(0,2)),root.medium.k,quadrature_order=12,face_batch=4)
    assert len(calls)==1
    (tmp_path/'bit_identity.json').write_text(json.dumps(dict(native_unique_calls=calls,
        signed_zero_distinct=True,same_carrier_different_order_distinct=True,
        catalog_temporaries_released_before_context_capacity=True,
        validation_stopped_before_maps=True,qualification=False),indent=2)+'\n')
