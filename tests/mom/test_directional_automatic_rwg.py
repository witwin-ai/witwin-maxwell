"""Automatic actual-support PMCHWT blocks versus original weak integration."""
import json
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import PreparedRWGGeometry, IntegrationConfig
from witwin.maxwell.mom.nearfield import scalar_regional_block
from witwin.maxwell.mom.fmm_directional_rwg import AutomaticDirectionalRWGBlock
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget
from witwin.maxwell.mom.direct_capacity import storage_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _fixture(lossy=False):
    bases = [RWGBasis(sphere_mesh(radius=.02, center=c, subdivisions=0, device='cuda', dtype=torch.float64))
        for c in ((-.5, .1, 0.), (.5, -.1, .05))]
    geometries = [PreparedRWGGeometry(b) for b in bases]
    edges = [torch.tensor(v, device='cuda') for v in ([17, 2, 11, 29, 7], [8, 1, 22])]
    centers = [b.mesh.vertices.mean(0)+b.mesh.vertices.new_tensor([.0007, -.0003, .0002]) for b in bases]
    medium = compile_medium(mw.Material(eps_r=2., mu_r=1.3, sigma_e=.01 if lossy else 0.,
        sigma_m=.02 if lossy else 0.), 3e8)
    return bases, geometries, edges, centers, medium


def _build(fixture, **kwargs):
    bases, geometries, edges, centers, medium = fixture
    options = dict(kernel_error_budget=KernelErrorBudget(1e-6, 1e-4), ideal_rwg_error_budget=1e-4,
        balancing_impedance=abs(compile_medium(mw.Material(), 3e8).eta), quadrature_order=12)
    options.update(kwargs)
    return AutomaticDirectionalRWGBlock(bases[0], edges[0], centers[0], geometries[0],
        bases[1], edges[1], centers[1], geometries[1], medium, **options)


def _relative(a, b):
    return float(((a-b).norm()/b.norm()).detach())


@pytest.mark.parametrize('lossy', (False, True))
def test_auto_actual_support_original_weak_full_h_current_higher_and_inventory(lossy, tmp_path):
    fixture = _fixture(lossy); block = _build(fixture)
    bases, geometries, edges, centers, medium = fixture
    receipt = block.receipt(); order = block.test.order
    radii = []
    for b, e, c in zip(bases, edges, centers):
        triangles = b.mesh.vertices[b.mesh.faces]
        corners = torch.cat((triangles[b.plus_faces[e]], triangles[b.minus_faces[e]]), 1)
        actual = (corners-c).norm(dim=-1).amax()
        radii.append(float(actual)); assert float(actual) <= float(block.plan.test_radii[0] if not len(radii)-1 else block.plan.source_radii[0])
    assert storage_bytes(block.tensors()) == receipt['owned_retained_bytes']
    assert block.test.owns_geometry is False and block.source.owns_geometry is False
    # A distinct1e-10 absolute balanced ideal budget must tighten the originally
    # loose kernel budgets before selecting a higher-order admitted block.
    tighter = _build(fixture, ideal_rwg_error_budget=1e-10, tensor_workspace_budget_bytes=512*1024**2)
    assert tighter.test.order > order
    assert tighter.coupling.ideal_rwg_truncation_bound()['absolute_bound'] <= 1e-10
    actual = block.apply(torch.eye(2*len(edges[1]), device='cuda', dtype=torch.complex128)).mT
    reference, _ = scalar_regional_block(*bases, *edges, (medium,), IntegrationConfig(order=16, singular_order=3))
    eta = receipt['balancing_impedance']
    left = actual.real.new_tensor([eta]*len(edges[0])+[1.]*len(edges[0]))
    right = actual.real.new_tensor([1.]*len(edges[1])+[eta]*len(edges[1]))
    weak = _relative(left[:, None]*actual*right, left[:, None]*reference*right)
    record = dict(lossy=lossy, phase='original_weak_computed', original_weak=weak,
        selected_order=order, tighter_ideal_order=tighter.test.order, radii=radii, receipt=receipt, qualification=False)
    torch.save(dict(actual=actual.detach(), reference=reference.detach()), tmp_path/'matrices.pt')
    (tmp_path/'weak.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
    assert weak < 1e-8
    generator = torch.Generator(device='cuda').manual_seed(92817)
    q = torch.randn((3, 5, 2*len(edges[1])), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    z = torch.randn((*q.shape[:-1], 2*len(edges[0])), device='cuda', dtype=torch.complex128, generator=generator).requires_grad_()
    value = block.apply(q); adjoint = block.adjoint(z)
    physical_h = _relative(adjoint, z@reference.conj()); assert physical_h < 1e-8
    current, = torch.autograd.grad(value, q, z, create_graph=True)
    native_current = _relative(current, adjoint); assert native_current < 1e-11
    tangent = torch.randn(q.shape, device='cuda', dtype=torch.complex128, generator=generator)
    higher, = torch.autograd.grad(current, z, tangent)
    native_higher = _relative(higher, block.apply(tangent)); assert native_higher < 1e-11
    lhs = (value.conj()*z).sum(); rhs = (q.conj()*adjoint).sum()
    dot = float(((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs())).detach()); assert dot < 1e-11
    saved = []
    moments = block.source.source_moments(torch.stack((q[..., :len(edges[1])], q[..., len(edges[1]):]), -2))
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        block.coupling.apply(moments)
    assert not saved
    record.update(physical_h=physical_h, native_current=native_current, native_higher=native_higher,
        dot=dot, noncontiguous_rhs=15, primitive_saved_tensors=0, owned_bytes=storage_bytes(block.tensors()))
    (tmp_path/'actual.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_automatic_preallocation_capacity_budget_and_actual_support_guards(monkeypatch, tmp_path):
    import witwin.maxwell.mom.fmm_directional_rwg as rwg
    import witwin.maxwell.mom.fmm_directional_fft as fft
    fixture = _fixture(); block = _build(fixture)
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected automatic admission reached moment/coefficients')
    with monkeypatch.context() as patch:
        patch.setattr(rwg, '_rwg_weighted_moments', forbidden)
        patch.setattr(fft, 'directional_helmholtz_taylor', forbidden)
        for opts in ({'retained_budget_bytes': block.retained_bytes-1},
                     {'tensor_workspace_budget_bytes': block.planned_tensor_workspace_bytes-1}):
            with pytest.raises(MemoryError):
                _build(fixture, **opts)
        with pytest.raises(ValueError):
            _build(fixture, ideal_rwg_error_budget=1e-300)
        with pytest.raises(ValueError):
            _build(fixture, quadrature_order=1)
        same = list(fixture); same[3] = [fixture[3][0], fixture[3][0]]
        with pytest.raises(ValueError):
            _build(tuple(same))
    tripwires = []
    for kind in ('center', 'edges', 'geometry', 'plan', 'medium', 'budget'):
        f = _fixture(); b = _build(f)
        if kind == 'center': f[3][0].add_(.001)
        elif kind == 'edges': f[2][0][0] = 1
        elif kind == 'geometry': f[1][0].triangles.add_(.001)
        elif kind == 'plan': b.plan.normalization_lengths.mul_(2)
        elif kind == 'medium': object.__setattr__(f[4], 'k', f[4].k*1.01)
        else: b.ideal_rwg_error_budget *= 2
        with pytest.raises(ValueError): b.receipt()
        with pytest.raises(ValueError): b.apply(torch.ones(6, device='cuda', dtype=torch.complex128))
        tripwires.append(kind)
    record = dict(before_factories_rejections=True, caller_and_owned_tripwires=tripwires,
        budget=block.receipt(), qualification=False)
    (tmp_path/'guards.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_independent_mp_actual_integral_weights_and_half_squared_budget(tmp_path):
    import mpmath as mp
    fixture = _fixture(True)
    block = _build(fixture, ideal_rwg_error_budget=1e-10, tensor_workspace_budget_bytes=512*1024**2)
    bases, geometries, edges, _, medium = fixture
    with mp.workdps(100):
        integral_sides = []
        # Independent positive closed triangle second moment. No production
        # integral logs, block_error_log, contour or budget planner is called.
        for basis, geometry, selected in zip(bases, geometries, edges):
            values = []
            for edge in selected.tolist():
                U, Q = mp.mpf(0), mp.mpf(0)
                for face in (int(basis.plus_faces[edge]), int(basis.minus_faces[edge])):
                    position = int((basis.face_edges[face] == edge).nonzero()[0])
                    triangle = [list(map(mp.mpf, v)) for v in geometry.triangles[face].tolist()]
                    origin = triangle[position]
                    other = [v for j, v in enumerate(triangle) if j != position]
                    a, b = [[x-y for x, y in zip(v, origin)] for v in other]
                    cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
                    area = mp.sqrt(sum(v*v for v in cross))/2
                    mean = (sum(v*v for v in a)+sum(v*v for v in b)+sum(x*y for x, y in zip(a, b)))/6
                    slope = abs(mp.mpf(float(geometry.slopes[face, position])))
                    U += area*slope*mp.sqrt(mean); Q += 2*area*slope
                values.append((U, Q))
            integral_sides.append(values)
        eta = mp.mpf(block.balancing_impedance); k = abs(mp.mpc(medium.k))
        e, m = abs(mp.mpc(1j*medium.omega*medium.epsilon)), abs(mp.mpc(-1j*medium.omega*medium.mu))
        dsum = sum((u*v+q*r/k**2)**2 for u, q in integral_sides[0] for v, r in integral_sides[1])
        usum = sum((u*v)**2 for u, _ in integral_sides[0] for v, _ in integral_sides[1])
        A = mp.sqrt((eta**4*e**2+m**2)*dsum); B = mp.sqrt(6*eta**2*usum)
        original = block.kernel_error_budget
        expected = [min(mp.mpf(original.green_absolute), mp.mpf('1e-10')/(mp.sqrt(2)*A)),
                    min(mp.mpf(original.gradient_component_absolute), mp.mpf('1e-10')/(mp.sqrt(2)*B))]
        actual = block.plan.kernel_error_budget
        effective = [actual.green_absolute, actual.gradient_component_absolute]
        relative = [float(abs(mp.mpf(x)/y-1)) for x, y in zip(effective, expected)]
        assert max(relative) < 1e-11
        g, h = mp.mpf(float(block.plan.green_bounds[0])), mp.mpf(float(block.plan.gradient_bounds[0]))
        independent = mp.sqrt((A*g)**2+(B*h)**2)
        native = block.coupling.ideal_rwg_truncation_bound()['absolute_bound']
        bound_error = float(abs(mp.mpf(native)/independent-1)); assert bound_error < 1e-11
        assert independent <= mp.mpf('1e-10')
        # Removing the sqrt(2) split may violate the declared absolute budget.
        unsplit = mp.sqrt((A*(mp.mpf('1e-10')/A))**2+(B*(mp.mpf('1e-10')/B))**2)
        assert unsplit > mp.mpf('1.4e-10')
        record = dict(mp_digits=100, effective_budget_relative_errors=relative,
            combined_ideal_relative_error=bound_error, independent_selected_ideal_bound=float(independent),
            unsplit_counterexample=float(unsplit), receipt=block.receipt(), qualification=False)
    (tmp_path/'mp_budget.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


@pytest.mark.parametrize('lossy', (False, True))
def test_changed_bounded_wrapper_reuses_frozen_original_matrix_full_h_grad_and_lifetime(lossy, tmp_path, monkeypatch):
    import hashlib
    import weakref
    from pathlib import Path
    import os
    artifact = _frozen_reference_artifact(lossy, os.environ.get('GSP_DIRECTIONAL_RWGMATRIX_MANIFEST'))
    fixture = _fixture(lossy)
    if artifact is None:
        # Ordinary regression has no external historical collector dependency.
        bases, _, edges, _, medium = fixture
        reference, _ = scalar_regional_block(*bases, *edges, (medium,), IntegrationConfig(order=16, singular_order=3))
        before = None
    else:
        before = hashlib.sha256(artifact.read_bytes()).hexdigest()
        reference = torch.load(artifact, weights_only=True)['reference']
    block = _build(fixture); observations = []; last = {}
    for owner, names in ((block.source, ('source_moments', 'source_adjoint')),
                         (block.test, ('receiver_evaluate', 'receiver_adjoint'))):
        for name in names:
            original = getattr(owner, name)
            def checked(value, original=original, name=name):
                assert value.shape[0] <= 4
                if name in last:
                    assert last[name]() is None, 'A previous chunk map result remains live'
                output = original(value)
                observations.append((name, value.shape[0])); last[name] = weakref.ref(output)
                return output
            monkeypatch.setattr(owner, name, checked)
    actual = block.apply(torch.eye(6, device='cuda', dtype=torch.complex128)).mT
    eta = block.balancing_impedance
    left = actual.real.new_tensor([eta]*5+[1.]*5); right = actual.real.new_tensor([1.]*3+[eta]*3)
    weak = _relative(left[:, None]*actual*right, left[:, None]*reference*right); assert weak < 1e-8
    generator = torch.Generator(device='cuda').manual_seed(449127)
    q = torch.randn((3, 5, 6), device='cuda', dtype=torch.complex128, generator=generator).transpose(0, 1).requires_grad_()
    z = torch.randn((*q.shape[:-1], 10), device='cuda', dtype=torch.complex128, generator=generator).requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        value = block.apply(q); adjoint = block.adjoint(z)
    assert not saved
    physical_h = _relative(adjoint, z@reference.conj()); assert physical_h < 1e-8
    forward = _relative(value, q@actual.mT); assert forward < 1e-11
    native_h = _relative(adjoint, z@actual.conj()); assert native_h < 1e-11
    derivative, = torch.autograd.grad(value, q, z, create_graph=True)
    current = _relative(derivative, adjoint); assert current < 1e-11
    tangent = torch.randn(q.shape, device='cuda', dtype=torch.complex128, generator=generator)
    higher, = torch.autograd.grad(derivative, z, tangent)
    higher_error = _relative(higher, block.apply(tangent)); assert higher_error < 1e-11
    empty = torch.empty((0, 6), device='cuda', dtype=torch.complex128, requires_grad=True)
    empty_value = block.apply(empty); assert empty_value.shape == (0, 10)
    empty_gradient, = torch.autograd.grad(empty_value.sum().real, empty); assert empty_gradient.shape == empty.shape
    assert all(t() is None for t in last.values())
    assert max(rows for _, rows in observations) == 4
    assert any(rows == 3 for _, rows in observations)
    if artifact is not None:
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
    record = dict(lossy=lossy, unchanged_reference_artifact=str(artifact), sha256=before,
        original_weak=weak, physical_h=physical_h, native_forward=forward, native_h=native_h,
        native_current=current, native_higher=higher_error, whole_wrapper_saved_tensors=0,
        max_physical_chunk=4, previous_map_results_released=True, observations=observations,
        empty_forward_backward=True, qualification=False)
    (tmp_path/'bounded.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def test_changed_shared_backward_guards_borrowed_and_owned_state_after_saved_output(tmp_path):
    tripwires = []
    for mode in ('forward', 'adjoint'):
        for kind in ('center', 'edges', 'ideal', 'retained', 'workspace', 'plan', 'medium'):
            fixture = _fixture(); block = _build(fixture)
            value = torch.ones(6 if mode == 'forward' else 10, device='cuda', dtype=torch.complex128, requires_grad=True)
            saved_output = block.apply(value) if mode == 'forward' else block.adjoint(value)
            if kind == 'center': fixture[3][0].add_(.001)
            elif kind == 'edges': fixture[2][0][0] = 1
            elif kind == 'ideal': block.ideal_rwg_error_budget *= 2
            elif kind == 'retained': block.retained_budget_bytes += 1
            elif kind == 'workspace': block.tensor_workspace_budget_bytes += 1
            elif kind == 'plan': block.plan.orders.add_(1)
            else: object.__setattr__(fixture[4], 'k', fixture[4].k*1.01)
            with pytest.raises(ValueError): torch.autograd.grad(saved_output, value, torch.ones_like(saved_output))
            tripwires.append((mode, kind))
    record = dict(saved_output_backward_tripwires=tripwires, shared_fixed_action_protocol=True, qualification=False)
    (tmp_path/'saved_backward.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)


def _frozen_reference_artifact(lossy, manifest_path):
    """Optional strict reuse; unconfigured regression integrates a fresh oracle."""
    import hashlib
    from pathlib import Path
    if manifest_path is None:
        return None
    manifest = json.loads(Path(manifest_path).read_text())
    hashes = manifest['evidence_sha256']
    matches = []
    for name, digest in hashes.items():
        diagnostic = Path(name)
        if diagnostic.name != 'actual.json':
            continue
        if hashlib.sha256(diagnostic.read_bytes()).hexdigest() != digest:
            raise ValueError('Configured frozen diagnostic SHA mismatch')
        if json.loads(diagnostic.read_text())['lossy'] == lossy:
            matches.append(diagnostic.parent/'matrices.pt')
    if len(matches) != 1 or str(matches[0]) not in hashes:
        raise ValueError('Configured frozen matrix is missing or ambiguous')
    artifact = matches[0]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != hashes[str(artifact)]:
        raise ValueError('Configured frozen matrix SHA mismatch')
    return artifact


def test_optional_reference_artifact_lifecycle_without_numeric_repetition(tmp_path):
    import hashlib
    from pathlib import Path
    assert _frozen_reference_artifact(False, None) is None
    with pytest.raises(FileNotFoundError):
        _frozen_reference_artifact(False, tmp_path/'absent.json')
    diagnostic = tmp_path/'actual.json'; artifact = tmp_path/'matrices.pt'; manifest = tmp_path/'manifest.json'
    diagnostic.write_text(json.dumps(dict(lossy=False))); artifact.write_bytes(b'controlled-lifecycle-only')
    hashes = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (diagnostic, artifact)}
    manifest.write_text(json.dumps(dict(evidence_sha256=hashes)))
    assert _frozen_reference_artifact(False, manifest) == artifact
    with pytest.raises(ValueError): _frozen_reference_artifact(True, manifest)
    artifact.write_bytes(b'tampered')
    with pytest.raises(ValueError): _frozen_reference_artifact(False, manifest)
    artifact.unlink()
    with pytest.raises(FileNotFoundError): _frozen_reference_artifact(False, manifest)
    # Actual historical artifacts used by v3 resolve in place under explicit
    # configuration with their previously accepted SHA; no numerical rerun.
    original = Path('/tmp/gsp-directional-actual-support-evidence-v1/summary.json')
    if original.is_file():
        for lossy in (False, True): assert _frozen_reference_artifact(lossy, original).is_file()
    record = dict(default_fresh_reference=True, explicit_missing_partial_tampered_fail=True,
        optional_strict_reference_reuse=True, numerical_checks_repeated=False, qualification=False)
    (tmp_path/'reference_lifecycle.json').write_text(json.dumps(record, indent=2)+'\n'); print(record, flush=True)
