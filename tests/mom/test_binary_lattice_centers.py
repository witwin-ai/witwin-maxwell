"""Actual binary-grid centers, complete native pairs and one new public solve."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.cluster import build_support_octree, support_spheres, sphere_block_partition
from witwin.maxwell.mom.fmm_storage import shared_displacement_catalog
from witwin.maxwell.mom.fmm_capacity import storage_bytes, multipole_action_tensors
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def write_raw(path, value):
    def encode(item):
        if isinstance(item,float) and not __import__('math').isfinite(item):
            return {'nonfinite_value':repr(item)}
        if isinstance(item,dict):return {k:encode(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [encode(v) for v in item]
        return item
    path.write_text(json.dumps(encode(value),indent=2,allow_nan=False)+'\n')


def new_boxes():
    axes=[torch.arange(n,device='cuda',dtype=torch.float64) for n in (6,5,4)]
    grid=torch.cartesian_prod(*axes)
    generator=torch.Generator(device='cuda').manual_seed(10237)
    centers=grid*.173+grid.new_tensor([.137,-.219,.083])
    centers+=.009*torch.rand(grid.shape,device='cuda',dtype=grid.dtype,generator=generator)
    widths=.001+.003*torch.rand(grid.shape,device='cuda',dtype=grid.dtype,generator=generator)
    return centers-widths,centers+widths


def test_new_global_integer_grid_supports_and_complete_native_catalog(tmp_path):
    raw=dict(version='new_binary_lattice_geometry_v1',completed=False,production_qualification=False)
    path=tmp_path/'binary-geometry.json';write_raw(path,raw)
    source=os.environ.get('GSP_BINARY_ORIGINAL_CLUSTER')
    digest=os.environ.get('GSP_BINARY_ORIGINAL_CLUSTER_SHA256')
    if source is None and digest is None:pytest.skip('Requires the pinned original cluster module')
    assert source and digest,'Original source and SHA must be supplied together'
    original=Path(source)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location('_gsp_binary_original_cluster',original)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    lower,upper=new_boxes()
    baseline=module.build_support_octree(lower,upper,leaf_size=2)
    tree=build_support_octree(lower,upper,leaf_size=2,expansion_center_policy='binary_lattice_v1')
    assert len(tree.tensors())==8 and tree.version=='cuda_support_octree_binary_lattice_v1'
    for a,b in zip(tree.tensors()[:7],baseline.tensors()[:7],strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    centers,radii=support_spheres(tree)
    exponent=torch.tensor(tree.binary_lattice_exponent,device=centers.device,dtype=torch.int64)
    indices=torch.ldexp(centers,-exponent)
    assert torch.equal(indices,indices.round()) and bool((indices.abs()<=2**51).all())
    integer=indices.to(torch.int64)
    native=centers[:,None]-centers[None]
    independent=torch.ldexp((integer[:,None]-integer[None]).to(torch.float64),exponent)
    torch.testing.assert_close(native,independent,rtol=0,atol=0)
    selectors=torch.tensor([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)],device='cuda',dtype=torch.bool)
    for node in range(tree.node_count):
        edges=tree.permutation[tree.starts[node]:tree.stops[node]]
        corners=torch.where(selectors[None],upper[edges,None],lower[edges,None])
        assert bool(((corners-centers[node]).norm(dim=-1)<=radii[node]).all())
    coverage=torch.zeros((len(lower),len(lower)),device='cuda',dtype=torch.int64)
    far_count=0
    for aa,bb,flags in sphere_block_partition(tree,separation_ratio=.25,pair_batch=13):
        for a,b,far in zip(aa.tolist(),bb.tolist(),flags.tolist(),strict=True):
            left=tree.permutation[tree.starts[a]:tree.stops[a]]
            right=tree.permutation[tree.starts[b]:tree.stops[b]]
            coverage[left[:,None],right[None]]+=1
            if far:
                far_count+=1
                assert bool(radii[a]+radii[b]<=.25*(centers[a]-centers[b]).norm())
    assert far_count and bool((coverage==1).all())
    coordinates=integer.tolist()
    all_pairs=[(a,b) for a in range(tree.node_count) for b in range(tree.node_count) if a!=b]
    pairs=[(a,b) for a,b in all_pairs if coordinates[a]!=coordinates[b]]
    zero_pairs=[(a,b) for a,b in all_pairs if coordinates[a]==coordinates[b]]
    for a,b in zero_pairs:
        assert bool(radii[a]+radii[b]>.25*(centers[a]-centers[b]).norm())
    if zero_pairs:
        with pytest.raises(ValueError,match='nonzero'):
            shared_displacement_catalog(centers,[zero_pairs[0]])
    owners,bindings=shared_displacement_catalog(centers,pairs)
    keys=set()
    for a,b in pairs:
        delta=tuple(x-y for x,y in zip(coordinates[a],coordinates[b],strict=True))
        keys.add(min(delta,tuple(-x for x in delta)))
    assert len(owners)==len(keys)<len(pairs)
    pair_indices=torch.tensor(pairs,device='cuda',dtype=torch.int64)
    actual=centers[pair_indices[:,0]]-centers[pair_indices[:,1]]
    representatives=torch.tensor([pairs[owners[owner]] for owner,_ in bindings],device='cuda',dtype=torch.int64)
    expected=centers[representatives[:,0]]-centers[representatives[:,1]]
    reverse=torch.tensor([flag for _,flag in bindings],device='cuda',dtype=torch.bool)
    expected=torch.where(reverse[:,None],-expected,expected)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # This NEW cube-diagonal fixture has distinct nodes with the same
    # actual rounded center. Their supports and every physical pair remain.
    points=lower.new_tensor([[.2,.2,.2],[.71,.71,.71]])
    zero_tree=build_support_octree(points-.001,points+.001,leaf_size=1,
        expansion_center_policy='binary_lattice_v1')
    zc,zr=support_spheres(zero_tree)
    zero=(zc[:,None]==zc[None]).all(-1)
    zero.fill_diagonal_(False)
    same_nodes=zero.nonzero();assert len(same_nodes)>0
    za,zb=same_nodes[0].tolist()
    with pytest.raises(ValueError,match='nonzero'):
        shared_displacement_catalog(zc,[(za,zb)])
    zero_coverage=torch.zeros((2,2),device='cuda',dtype=torch.int64)
    for aa,bb,flags in sphere_block_partition(zero_tree,separation_ratio=.25,pair_batch=2):
        for a,b,far in zip(aa.tolist(),bb.tolist(),flags.tolist(),strict=True):
            left=zero_tree.permutation[zero_tree.starts[a]:zero_tree.stops[a]]
            right=zero_tree.permutation[zero_tree.starts[b]:zero_tree.stops[b]]
            zero_coverage[left[:,None],right[None]]+=1
            assert not far
    assert bool((zero_coverage==1).all())
    raw.update(completed=True,physical_supports=len(lower),nodes=tree.node_count,
        original_seven_tree_tensors_exact=True,complete_member_corners_enclosed=True,
        all_physical_ordered_pairs_once=True,far_blocks=far_count,
        binary_lattice_exponent=tree.binary_lattice_exponent,
        logical_node_pairs=len(all_pairs),nonzero_catalog_pairs=len(pairs),
        same_center_node_pairs=len(zero_pairs),same_center_physical_pairs_preserved=True,
        explicit_same_center_fixture_pairs=4,native_catalog_owners=len(owners),
        independent_integer_catalog_owners=len(keys),all_native_differences_exact=True,
        coefficient_displacement_projection=False,physical_qualification=False)
    write_raw(path,raw)


def config(**options):
    return mw.FMMConfig(order=6,max_order=12,table_storage='compact_shared_displacement_v1',
        shift_storage='separable_v1',near_storage='scalar_reciprocal_dk_v1',far_action_batch=32,
        leaf_size=8,separation_ratio=.25,expansion_center_policy='binary_lattice_v1',
        kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        kernel_bound_rule='direct_first_wave_min_v1',**options)


def test_new_binary_center_identity_and_strict_public_refusals(monkeypatch):
    cfg=config();payload=cfg.payload()
    assert payload['version']=='cartesian_fmm_binary_lattice_config_v1'
    assert mw.FMMConfig.restore(json.loads(json.dumps(payload)))==cfg
    assert cfg.construction()['expansion_center_policy']=='binary_lattice_v1'
    for kind in ('policy','version','missing','extra'):
        bad=copy.deepcopy(payload)
        if kind=='policy':bad['parameters']['expansion_center_policy']='dyadic_cell_v1'
        elif kind=='version':bad['version']='cartesian_fmm_dyadic_centers_config_v1'
        elif kind=='missing':bad['parameters'].pop('expansion_center_policy')
        else:bad['parameters']['coordinate_projection']=True
        with pytest.raises(ValueError):mw.FMMConfig.restore(bad)
    with pytest.raises(NotImplementedError,match='free space'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(preconditioner='none'),periodic=mw.MoMPeriodicConfig(((1.,0.,0.),)))
    with pytest.raises(ValueError,match='complex128'):
        mw.MoMConfig(operator_backend='fmm',fmm=cfg,riesz_storage='local_rwg_v1',
            stopping=StoppingRule(preconditioner='none'),dtype=torch.complex64)
    lower,upper=new_boxes()
    def forbidden(*args,**kwargs):raise AssertionError('Invalid precision reached native sort')
    monkeypatch.setattr(torch,'argsort',forbidden)
    with pytest.raises(ValueError,match='float64'):
        build_support_octree(lower.float(),upper.float(),expansion_center_policy='binary_lattice_v1')


@pytest.mark.parametrize('mutation',['tensor_version','gradient_flag','lattice_exponent'])
def test_new_binary_center_owned_state_mutation_refused(mutation):
    lower,upper=new_boxes()
    tree=build_support_octree(lower,upper,leaf_size=2,expansion_center_policy='binary_lattice_v1')
    if mutation=='tensor_version':tree.expansion_centers.add_(0)
    elif mutation=='gradient_flag':tree.expansion_centers.requires_grad_()
    else:object.__setattr__(tree,'binary_lattice_exponent',tree.binary_lattice_exponent+1)
    with pytest.raises(ValueError,match='changed'):support_spheres(tree)


def test_new_public_binary_centers_all_columns_native_H_gradients_and_solve(tmp_path,monkeypatch):
    from witwin.maxwell.mom.runtime import SystemOperator,SurfaceData
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    raw=dict(version='new_public_binary_lattice_centers_v1',completed=False,
        production_qualification=False,physical_qualification=False)
    path=tmp_path/'binary-public.json';write_raw(path,raw)
    scene=mw.Scene(structures=[
        mw.Structure(mw.Sphere(radius=.012,position=(-.287,.019,0.)),mw.Material(eps_r=2.2+.01j,mu_r=1.25)),
        mw.Structure(mw.Sphere(radius=.013,position=(.293,-.017,.011)),mw.Material(eps_r=3.1+.02j,mu_r=1.35))],
        sources=[mw.PlaneWave()])
    cfg=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=config(),riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    try:
        prepared=mw.Simulation.mom(scene,frequency=1.05e8,config=cfg).prepare()
        op=prepared.solver.handle.actions
        raw.update(capacity_plan=op.capacity_plan,order_plan=op.order_plan,fmm_config=cfg.fmm.payload())
        write_raw(path,raw)
        assert op.exterior.far and op.expansion_center_policy=='binary_lattice_v1'
        for region in (op.exterior,*op.interiors):
            assert region.test.tree.expansion_center_policy=='binary_lattice_v1'
            torch.testing.assert_close(region.test.centers,region.test.tree.expansion_centers,rtol=0,atol=0)
        edges=torch.arange(op.shape[0]//2,device='cuda')
        reference=op.provider(edges,edges)
        reference_artifact=tmp_path/'binary-native-reference.pt'
        saved=dict(reference=reference.detach(),frequency_Hz=1.05e8)
        torch.save(saved,reference_artifact)
        relative=lambda a,b:float(((a-b).norm()/b.norm()).detach())
        columns=op.apply(torch.eye(op.shape[0],device='cuda',dtype=op.dtype)).mT
        generator=torch.Generator(device='cuda').manual_seed(10537)
        x=torch.randn((3,5,op.shape[0]),device='cuda',dtype=op.dtype,generator=generator).transpose(0,1).requires_grad_()
        y=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        forward=op.apply(x);hermitian=op.adjoint(y)
        lhs=(forward.conj()*y).sum();rhs=(x.conj()*hermitian).sum()
        native_error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()))
        errors=dict(all_columns=relative(columns,reference),forward=relative(forward,x@reference.mT),
            adjoint=relative(hermitian,y@reference.conj()),native_H=native_error)
        saved.update(columns=columns.detach(),x=x.detach(),y=y.detach(),forward=forward.detach(),adjoint=hermitian.detach())
        torch.save(saved,reference_artifact)
        raw.update(errors=errors,reference_constructed_once=True,reference_artifact=str(reference_artifact))
        write_raw(path,raw)
        assert max(errors[k] for k in ('all_columns','forward','adjoint'))<1e-8 and native_error<1e-11
        captured=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:captured.append(t) or t,lambda t:t):value=op.apply(x)
        assert not captured
        gradient=torch.autograd.grad(value,x,y)[0]
        first=torch.autograd.grad(op.apply(x).abs().square().sum(),x,create_graph=True)[0]
        direction=torch.randn(x.shape,device='cuda',dtype=op.dtype,generator=generator)
        second=torch.autograd.grad(first,x,direction)[0]
        gradient_error=relative(gradient,hermitian)
        second_error=relative(second,2*op.adjoint(op.apply(direction)))
        raw.update(current_gradient_error=gradient_error,higher_current_gradient_error=second_error,saved_forward_tensors=0)
        write_raw(path,raw)
        assert gradient_error<1e-11 and second_error<1e-11
        owned=storage_bytes(multipole_action_tensors(op))
        assert owned==op.capacity_plan['planned_bytes']
        result=prepared.run()
        assert result.stats()['converged'] and result.stats()['error_budget']['checked']
        current=result.surface_currents.scattered_coefficients
        residual=relative(current@reference.mT,prepared.solver.rhs)
        raw.update(complete_original_residual=residual,residual_relative_gate=1e-10,
            residual_absolute_gate=0.,actual_owned_bytes=owned)
        write_raw(path,raw);assert residual<=1e-10
        artifact=tmp_path/'binary-centers-result.pt';result.save(artifact)
        def forbidden(*args,**kwargs):raise AssertionError('Binary-center restore assembled or solved')
        with monkeypatch.context() as patch:
            for cls in (MultipoleOperator,MultipoleSystemActions,SystemOperator):patch.setattr(cls,'__init__',forbidden)
            patch.setattr(torch.linalg,'solve',forbidden)
            restored=mw.Result.load(artifact,scene=scene)
            assert restored.surface_currents.config.fmm==cfg.fmm
            torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
            bad=copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters']['expansion_center_policy']='dyadic_cell_v1'
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,1.05e8)
        raw.update(completed=True,artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            exact_restore=True,whole_peak_qualified=False,original_scalar_reference_is_not_independent_EM_reference=True)
        write_raw(path,raw)
    except Exception as error:
        raw.update(error=str(error),failure_receipt=getattr(error,'receipt',None))
        write_raw(path,raw)
        raise
