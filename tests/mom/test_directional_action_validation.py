"""Shared-graph validation without repeated per-M2L projection traversal."""
import functools
import hashlib
import json
from pathlib import Path
import pytest
import torch
from witwin.maxwell.mom.fmm_config import DirectionalFMMConfig
from witwin.maxwell.mom.fmm_system import DirectionalSystemActions
from witwin.maxwell.mom.fmm_operator import DirectionalMultipoleOperator
from witwin.maxwell.mom.linalg import residual_loss
from witwin.maxwell.mom.fmm_capacity import directional_retained_inventory
from test_directional_automatic_dag import _physical_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(scope='module')
def case():
    provider = _physical_plan().basis
    actions = DirectionalSystemActions.from_automatic(provider,
        DirectionalFMMConfig(leaf_size=4, separation_ratio=.15, cone_subdivisions=(2**20,),
            max_contexts=4096, tensor_workspace_budget_bytes=512*1024**2))
    import witwin.maxwell.mom.fmm_directional_rwg as module
    path = Path(__file__).parent/'fixtures'/'directional_action_validation_legacy_far.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '557fd9632ae8b307254d2cf13725eac810938c174c3cdce57caf2c640f23233c'
    namespace = dict(vars(module))
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    return provider, actions, namespace['_action'], namespace['assert_current']


def test_actual_shared_graph_validation_owner_visits_once(case, monkeypatch, tmp_path):
    _, actions, _, legacy_guard = case
    far = actions.exterior.far
    owners = {id(p):p for p in (far.test,far.source,*far.targets,*far.origins)}
    calls = {key:0 for key in owners}
    def counted(owner, original):
        calls[id(owner)] += 1
        return original()
    with monkeypatch.context() as patch:
        for key, owner in owners.items():
            patch.setattr(owner, 'assert_current', functools.partial(counted, owner, owner.assert_current))
        far.assert_current()
        optimized = dict(calls)
        assert all(n == 1 for n in optimized.values())
        for key in calls: calls[key] = 0
        legacy_guard(far)
        previous = dict(calls)
    assert sum(previous.values()) > sum(optimized.values())
    (tmp_path/'visits.json').write_text(json.dumps(dict(actual_FAR=len(far.couplings),
        distinct_actual_context_owners=len(owners), optimized_owner_visits=list(optimized.values()),
        original_owner_visits=list(previous.values()), every_translation_table_still_validated=True,
        graph_validation_only_no_timing_scaling_claim=True, qualification=False), indent=2)+'\n')


def test_complete_actual_original_dispatch_columns_H_current_higher_and_Riesz(case, monkeypatch, tmp_path):
    provider, actions, legacy_action, _ = case
    inventory = directional_retained_inventory(actions)
    size = actions.shape[0]
    eye = torch.eye(size, device='cuda', dtype=torch.complex128)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with monkeypatch.context() as patch:
            for region in (actions.exterior, *actions.interiors):
                patch.setattr(region.far, '_action', functools.partial(legacy_action, region.far))
            original = actions.apply(eye).mT
        actual = actions.apply(eye).mT
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    generator = torch.Generator(device='cuda').manual_seed(190542)
    x = torch.randn((3,5,size), generator=generator, device='cuda', dtype=torch.complex128).transpose(0,1).requires_grad_()
    z = torch.randn(x.shape, generator=generator, device='cuda', dtype=torch.complex128)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t:saved.append(t) or t, lambda t:t):
        y = actions.apply(x)
    H = actions.adjoint(z)
    grad, = torch.autograd.grad(y, x, z, retain_graph=True)
    first, = torch.autograd.grad(y.abs().square().sum(), x, create_graph=True)
    tangent = torch.randn_like(x)
    higher, = torch.autograd.grad(first, x, tangent)
    relative = lambda a,b:float(((a-b).norm()/b.norm()).detach())
    errors = dict(forward=relative(y,x@original.mT),original_dispatch_H=relative(H,z@original.conj()),
        native_current=relative(grad,H),higher_current=relative(higher,2*actions.adjoint(actions.apply(tangent))))
    operator = DirectionalMultipoleOperator(provider, actions)
    rhs = torch.randn_like(x)
    riesz, = torch.autograd.grad(residual_loss(operator,x,rhs),x)
    errors['Riesz_current'] = relative(riesz,operator.adjoint(operator.riesz_inverse(operator.apply(x)-rhs)))
    lhs, right = (y.conj()*z).sum(),(x.conj()*H).sum()
    errors['dot'] = float(((lhs-right).abs()/torch.maximum(lhs.abs(),right.abs())).detach())
    assert max(errors.values()) < 1e-11 and not saved
    assert directional_retained_inventory(actions) == inventory
    (tmp_path/'complete.json').write_text(json.dumps(dict(actual_bodies=3,physical_currents=size,
        all_columns_original_dispatch_numerically_exact=True,
        column_comparison_deterministic_algorithms=True,
        previous_deterministic_mode_restored=True,
        default_execution_elementwise_determinism_qualified=False,
        noncontiguous_RHS=15,errors=errors,
        saved_internal_tensors=len(saved),retained_inventory_unchanged=inventory,
        original_near_and_all_actual_regions_included=True,no_original_weak_matrix_reintegrated=True,
        no_new_solve_or_whole_timing_scaling_claim=True,qualification=False),indent=2)+'\n')


def test_saved_full_graph_standalone_M2L_and_exit_mutation_guards(case, monkeypatch, tmp_path):
    _, actions, _, _ = case
    region = actions.exterior
    plan = region.test.plan_guard
    coupling = region.far.couplings[0]
    x = torch.ones((2,actions.shape[0]),device='cuda',dtype=torch.complex128,requires_grad=True)
    y = actions.apply(x)
    moments = torch.ones((1,2,coupling.count,4),device='cuda',dtype=torch.complex128,requires_grad=True)
    translated = coupling.apply(moments)
    trips = []
    for kind in ('statistics_identity','path_mode','translation_identity','translation_mode','translation_context'):
        old = None
        if kind == 'statistics_identity': old = plan._statistics; plan._statistics = old.clone()
        elif kind == 'path_mode': plan.path_products.requires_grad_(True)
        elif kind == 'translation_identity': old = coupling.scalar.coefficients; coupling.scalar.coefficients = old.clone()
        elif kind == 'translation_mode': coupling.scalar.coefficients.requires_grad_(True)
        else: old = coupling._rwg_context; coupling._rwg_context = (actions.interiors[0].test, old[1])
        try:
            with pytest.raises(ValueError): region.far.assert_current()
            with pytest.raises(ValueError): actions.receipt()
            with pytest.raises(ValueError): torch.autograd.grad(y,x,torch.ones_like(y),retain_graph=True)
            with pytest.raises(ValueError): coupling.assert_current()
            with pytest.raises(ValueError): torch.autograd.grad(translated,moments,torch.ones_like(translated),retain_graph=True)
        finally:
            if kind == 'statistics_identity': plan._statistics = old
            elif kind == 'path_mode': plan.path_products.requires_grad_(False)
            elif kind == 'translation_identity': coupling.scalar.coefficients = old
            elif kind == 'translation_mode': coupling.scalar.coefficients.requires_grad_(False)
            else: coupling._rwg_context = old
        actions.assert_current(); trips.append(kind)
    original = region._action
    def mutate_after_exterior(value, *, adjoint):
        result = original(value, adjoint=adjoint)
        plan._statistics.add_(0)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(region,'_action',mutate_after_exterior)
        with pytest.raises(ValueError): actions.apply(x.detach())
    (tmp_path/'guards.json').write_text(json.dumps(dict(restored_complete_saved_current_and_standalone_M2L_guards=trips,
        statistics_version_mutation_after_exterior_rejected_before_system_return=True,
        original_standalone_context_guards_preserved=True,qualification=False),indent=2)+'\n')


def test_packaged_original_dispatch_reference_lifecycle():
    import ast
    import witwin.maxwell.mom.fmm_directional_rwg as module
    path = Path(__file__).parent/'fixtures'/'directional_action_validation_legacy_far.py'
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '557fd9632ae8b307254d2cf13725eac810938c174c3cdce57caf2c640f23233c'
    tree = ast.parse(path.read_text())
    assert [node.name for node in tree.body] == ['assert_current','_action']
    namespace = dict(vars(module))
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    assert callable(namespace['_action']) and callable(namespace['assert_current'])
    # The numerical fixture resolves the bundled reference in a fresh checkout.
    constants = case.__wrapped__.__code__.co_consts
    assert 'directional_action_validation_legacy_far.py' in constants
    assert all(not isinstance(value,str) or not value.startswith('/tmp/') for value in constants)
