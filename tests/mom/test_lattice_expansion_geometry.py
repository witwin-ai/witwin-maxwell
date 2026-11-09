"""Fixed lattice centers preserve original supports and complete physical pairs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from witwin.maxwell.mom.cluster import build_support_octree, sphere_block_partition, support_spheres
from witwin.maxwell.mom.fmm_storage import shared_displacement_catalog

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def boxes():
    generator=torch.Generator(device='cuda').manual_seed(774)
    centers=torch.randn((45,3),device='cuda',dtype=torch.float64,generator=generator)
    width=.02+.12*torch.rand(centers.shape,device='cuda',dtype=centers.dtype,generator=generator)
    return centers-width,centers+width


def test_default_original_bytes_and_all_member_support_corners(tmp_path):
    path=os.environ.get('GSP_LATTICE_ORIGINAL_CLUSTER');digest=os.environ.get('GSP_LATTICE_ORIGINAL_CLUSTER_SHA256')
    if path is None and digest is None:
        pytest.skip('Requires exact original default geometry source')
    assert path and digest and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location('_gsp_original_cluster',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    lower,upper=boxes();original=module.build_support_octree(lower,upper,leaf_size=3)
    default=build_support_octree(lower,upper,leaf_size=3)
    assert default.receipt()==original.receipt() and len(default.tensors())==7
    for current,prior in zip(default.tensors(),original.tensors(),strict=True):
        torch.testing.assert_close(current,prior,rtol=0,atol=0)
    center,radius=support_spheres(default)
    torch.testing.assert_close(center,original.lower*.5+original.upper*.5,rtol=0,atol=0)
    torch.testing.assert_close(radius,torch.linalg.vector_norm((original.upper-original.lower)*.5,dim=-1),rtol=0,atol=0)
    lattice=build_support_octree(lower,upper,leaf_size=3,expansion_center_policy='dyadic_cell_v1')
    assert len(lattice.tensors())==8
    for current,prior in zip(lattice.tensors()[:7],default.tensors(),strict=True):
        torch.testing.assert_close(current,prior,rtol=0,atol=0)
    center,radius=support_spheres(lattice)
    selectors=torch.tensor([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)],device='cuda',dtype=torch.bool)
    for node in range(lattice.node_count):
        edges=lattice.permutation[lattice.starts[node]:lattice.stops[node]]
        corners=torch.where(selectors[None],upper[edges,None],lower[edges,None])
        distance=(corners-center[node]).norm(dim=-1)
        assert bool((distance<=radius[node]).all())
    torch.testing.assert_close(center,lattice.expansion_centers,rtol=0,atol=0)
    assert center.data_ptr()!=lattice.expansion_centers.data_ptr()
    (tmp_path/'geometry.json').write_text(json.dumps(dict(nodes=lattice.node_count,physical_supports=45,
        original_seven_tensors_exact=True,default_original_bytes_exact=True,complete_member_corners_enclosed=True,
        owned_center_tensor=True,physical_qualification=False),indent=2)+'\n')


def test_lattice_partition_covers_every_original_ordered_pair_once():
    lower,upper=boxes();tree=build_support_octree(lower,upper,leaf_size=3,expansion_center_policy='dyadic_cell_v1')
    coverage=torch.zeros((45,45),device='cuda',dtype=torch.int64)
    center,radius=support_spheres(tree);far_count=0
    for aa,bb,flags in sphere_block_partition(tree,separation_ratio=.5,pair_batch=11):
        for a,b,far in zip(aa.tolist(),bb.tolist(),flags.tolist(),strict=True):
            left=tree.permutation[tree.starts[a]:tree.stops[a]]
            right=tree.permutation[tree.starts[b]:tree.stops[b]]
            coverage[left[:,None],right[None]]+=1
            if far:
                far_count+=1
                assert bool(radius[a]+radius[b]<=.5*(center[a]-center[b]).norm())
            else:
                assert bool((tree.children[a]<0).all() & (tree.children[b]<0).all())
    assert far_count>0 and bool((coverage==1).all())


def test_irregular_supports_share_only_exact_native_lattice_displacements(tmp_path):
    grid=torch.cartesian_prod(*[torch.arange(4,device='cuda',dtype=torch.float64)]*3)
    generator=torch.Generator(device='cuda').manual_seed(815)
    points=grid+.08*torch.rand(grid.shape,device='cuda',dtype=grid.dtype,generator=generator)
    lower,upper=points-.001,points+.001
    default=build_support_octree(lower,upper,leaf_size=1)
    lattice=build_support_octree(lower,upper,leaf_size=1,expansion_center_policy='dyadic_cell_v1')
    leaves=((lattice.children<0).all(-1)).nonzero().flatten().tolist()
    pairs=[(a,b) for a in leaves for b in leaves if a<b]
    original_center,_=support_spheres(default);center,_=support_spheres(lattice)
    old,_=shared_displacement_catalog(original_center,pairs)
    owners,bindings=shared_displacement_catalog(center,pairs)
    assert len(owners)<len(old)
    for (a,b),(owner,reverse) in zip(pairs,bindings,strict=True):
        left,right=pairs[owners[owner]]
        expected=center[left]-center[right]
        if reverse:expected=-expected
        torch.testing.assert_close(center[a]-center[b],expected,rtol=0,atol=0)
    (tmp_path/'sharing.json').write_text(json.dumps(dict(logical_pairs=len(pairs),
        original_support_box_owners=len(old),native_lattice_owners=len(owners),exact_native_equivalence=True,
        approximation_or_pair_removal=False,physical_qualification=False),indent=2)+'\n')


@pytest.mark.parametrize('mutation',['version','trainable'])
def test_owned_lattice_center_mutations_reject(mutation):
    lower,upper=boxes();tree=build_support_octree(lower,upper,leaf_size=3,expansion_center_policy='dyadic_cell_v1')
    if mutation=='version':tree.expansion_centers.add_(0)
    else:tree.expansion_centers.requires_grad_()
    with pytest.raises(ValueError,match='changed'):support_spheres(tree)


def test_lattice_invalid_precision_or_policy_reject_before_sort(monkeypatch):
    lower,upper=boxes()
    def forbidden(*args,**kwargs):raise AssertionError('invalid geometry reached native Morton sort')
    monkeypatch.setattr(torch,'argsort',forbidden)
    with pytest.raises(ValueError):build_support_octree(lower.float(),upper.float(),expansion_center_policy='dyadic_cell_v1')
    with pytest.raises(ValueError):build_support_octree(lower,upper,expansion_center_policy='approximate_bins')


@pytest.mark.parametrize('kind',['directional','sparse_directional','periodic','periodic_capacity','periodic_bound'])
def test_unimplemented_center_consumers_refuse_before_native_projection(kind):
    from types import SimpleNamespace
    from witwin.maxwell.mom.fmm_directional_moments import NestedDirectionalRWGProjection, SparseDirectionalRWGProjection
    from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
    from witwin.maxwell.mom.fmm_periodic_regional import PeriodicMultipoleRegionalOperator
    from witwin.maxwell.mom.fmm_periodic_capacity import plan_periodic_storage
    from witwin.maxwell.mom.fmm_periodic_error import periodic_hierarchy_error_bound
    lower,upper=boxes();tree=build_support_octree(lower,upper,leaf_size=3,expansion_center_policy='dyadic_cell_v1')
    direction=torch.tensor([1.,0.,0.],device='cuda',dtype=torch.float64)
    with pytest.raises(NotImplementedError,match='support-box centers|free space'):
        if kind=='directional':
            NestedDirectionalRWGProjection(None,tree,direction,1.,0,quadrature_order=2)
        elif kind=='sparse_directional':
            SparseDirectionalRWGProjection(None,tree,direction[None],(0,),((0,0),),1.,quadrature_order=2)
        elif kind=='periodic':
            # Reject the center policy before any numerical projection is
            # required; an existing class identity supplies only that boundary.
            projection=NestedRWGMultipoles.__new__(NestedRWGMultipoles);projection.tree=tree
            PeriodicMultipoleRegionalOperator(projection,projection,None,None)
        elif kind=='periodic_capacity':
            identity=SimpleNamespace(assert_current=lambda:None)
            plan_periodic_storage(identity,(tree,),identity,order=0,shells=1,separation_ratio=.5,budget_bytes=1024**2)
        else:
            periodic_hierarchy_error_bound(tree,tree,None,1.,order=0,shells=1)
