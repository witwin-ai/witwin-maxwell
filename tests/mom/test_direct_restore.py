"""One-shot CUDA storage lifecycle; host I/O tests never solve on the CPU."""
from dataclasses import asdict
import errno
import hashlib
import fcntl
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom import linalg as m
from witwin.maxwell.mom.operator import IntegrationConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
rule = lambda: m.StoppingRule(method='direct', direct_storage='sealed_host_restore_v1', rtol=1e-10)


def test_explicit_configuration_and_historical_restore():
    assert m.StoppingRule().direct_storage == 'resident'
    assert asdict(rule())['direct_storage'] == 'sealed_host_restore_v1'
    historical=asdict(m.StoppingRule(method='direct'));historical.pop('direct_storage')
    assert m.StoppingRule(**historical).direct_storage=='resident'
    for value in (None,False,1,[], 'auto'):
        with pytest.raises(ValueError):m.StoppingRule(method='direct', direct_storage=value)
    for method,fallback in [('gmres','none'),('block_gmres','none'),('direct','direct')]:
        with pytest.raises(ValueError):m.StoppingRule(method=method,fallback=fallback,direct_storage='sealed_host_restore_v1')


def cgroup_fixture(tmp_path,child='max',parent='3000',used='400'):
    mem=tmp_path/'meminfo';mem.write_text('MemAvailable: 100 kB\n')
    group=tmp_path/'selfgroup';group.write_text('0::/parent/child\n')
    root=tmp_path/'cgroup';(root/'parent/child').mkdir(parents=True)
    for where,maximum in [(root/'parent',parent),(root/'parent/child',child)]:
        (where/'memory.max').write_text(maximum);(where/'memory.current').write_text(used)
    mounts=tmp_path/'mountinfo';mounts.write_text('1 0 0:1 / '+str(root)+' rw - cgroup2 cgroup2 rw\n')
    return dict(meminfo=mem,cgroup=group,root=root,mountinfo=mounts)


@pytest.mark.parametrize('child,parent,expected',[('max','3000',2600),('1000','3000',600),('max','max',102400),('100','3000',0)])
def test_host_capacity_uses_tightest_real_ancestor_limit(tmp_path,child,parent,expected):
    record=m._host_backup_capacity(**cgroup_fixture(tmp_path,child,parent))
    assert record['available_bytes']==expected and len(record['cgroup_limits'])==2


@pytest.mark.parametrize('change',['missing_usage','bad_group','missing_memavailable','no_accounting'])
def test_unverifiable_host_capacity_rejects_before_mutation(tmp_path,change):
    paths=cgroup_fixture(tmp_path)
    if change=='missing_usage':(paths['root']/'parent/memory.current').unlink()
    elif change=='bad_group':paths['cgroup'].write_text('0::/../../outside\n')
    elif change=='missing_memavailable':paths['meminfo'].write_text('MemTotal: 10 kB\n')
    else:
        for path in paths['root'].rglob('memory.max'):path.unlink()
    with pytest.raises((OSError,RuntimeError)):m._host_backup_capacity(**paths)


def test_actual_host_capacity_has_visible_cgroup_limits_without_cuda():
    initialized=torch.cuda.is_initialized();record=m._host_backup_capacity()
    assert record['available_bytes']>0 and record['cgroup_limits']
    assert torch.cuda.is_initialized() is initialized


def test_insufficient_host_capacity_is_a_preflight_error(monkeypatch):
    monkeypatch.setattr(m,'_host_backup_capacity',lambda:{'available_bytes':1})
    with pytest.raises(RuntimeError,match='Insufficient'):m._require_host_backup_capacity(100)


def test_kernel_seals_reject_writes_and_truncation():
    fd=m._create_sealed_fd()
    try:
        os.write(fd,b'original byte evidence')
        flags=m._REQUIRED_SEALS
        fcntl.fcntl(fd,m._F_ADD_SEALS,flags)
        for mutation in (lambda:os.pwrite(fd,b'x',0),lambda:os.ftruncate(fd,0),lambda:os.ftruncate(fd,100)):
            with pytest.raises(OSError) as error:mutation()
            assert error.value.errno==errno.EPERM
        assert os.pread(fd,100,0)==b'original byte evidence'
    finally:os.close(fd)


def lease_handle():
    handle=SimpleNamespace(_matrix_lease_lock=threading.Lock(),_matrix_readers={},_matrix_borrower=None,_direct_storage_state='ready')
    handle.read_access=lambda:m.DenseOperator.read_access(handle)
    return handle


def test_normal_readers_stay_concurrent_and_exclusive_borrow_rejects():
    handle=lease_handle();entered=threading.Event();finish=threading.Event();errors=[]
    @m._matrix_operation
    def read(h,wait=False):
        if wait:entered.set();assert finish.wait(5)
        return 'read'
    def worker():
        try:read(handle,True)
        except BaseException as error:errors.append(error)
    task=threading.Thread(target=worker);task.start();assert entered.wait(5)
    try:assert read(handle)=='read'
    finally:finish.set();task.join(5)
    assert not errors and not handle._matrix_readers
    handle._direct_storage_state='borrowed'
    with pytest.raises(RuntimeError):read(handle)
    handle._direct_storage_state='unrestored'
    with pytest.raises(RuntimeError):read(handle)


def system(n=17,dtype=torch.complex128):
    generator=torch.Generator(device='cuda').manual_seed(881)
    a=torch.randn(n,n,dtype=dtype,device='cuda',generator=generator)+n*torch.eye(n,dtype=dtype,device='cuda')
    b=torch.randn(8,n,dtype=dtype,device='cuda',generator=generator)
    return m.DenseOperator(a,torch.eye(n,dtype=dtype,device='cuda')),b


@cuda
@pytest.mark.parametrize('n',[17,64])
def test_eight_rhs_true_solution_alias_restore_and_no_lu_cache(n):
    handle,b=system(n);original=handle.matrix.clone();pointer=handle.matrix.data_ptr();version=handle.matrix._version
    expected=torch.linalg.solve(original,b.mT).mT
    result,log=m.solve_reference(handle,b,rule())
    assert log['converged'] and len(log['rhs_logs'])==8
    torch.testing.assert_close(result,expected,rtol=1e-11,atol=1e-12)
    assert torch.equal(handle.matrix,original) and handle.matrix.data_ptr()==pointer and handle.matrix._version>version
    assert handle._lu is None and handle._direct_storage_backup is None
    receipt=log['direct_storage'];assert receipt['state']=='succeeded_restored' and receipt['factor_output_alias']
    assert receipt['backup_sha256']==receipt['restored_sha256']==hashlib.sha256(original.detach().cpu().numpy().tobytes()).hexdigest()
    assert receipt['sealed_backup_bytes']==16*n*n
    assert receipt['matrix_restored_exact'] and not receipt['backup_descriptor_open']
    assert all(row['relative_residual']<1e-10 for row in log['rhs_logs'])
    torch.testing.assert_close(handle.apply(result),b,rtol=1e-10,atol=1e-10)
    # Ordinary resident mode still creates and reuses its independent LU cache.
    _,resident=m.solve_reference(handle,b,m.StoppingRule(method='direct'))
    assert resident['converged'] and handle._lu is not None and 'direct_storage' not in resident


@cuda
def test_singular_factor_is_failed_but_original_matrix_is_exact():
    handle,b=system();handle.matrix[-1].zero_();original=handle.matrix.clone()
    _,log=m.solve_reference(handle,b,rule())
    assert not log['converged'] and all(row['reason']=='singular_matrix' for row in log['rhs_logs'])
    assert torch.equal(handle.matrix,original) and handle._lu is None
    assert log['direct_storage']['state']=='failed_restored' and log['direct_storage']['factor_info']>0


@cuda
@pytest.mark.parametrize('stage',['factor','solve','alias'])
def test_native_exception_or_wrong_alias_restores_before_escape(monkeypatch,stage):
    handle,b=system();original=handle.matrix.clone();factor=torch.linalg.lu_factor_ex
    def changed(a,*args,**kwargs):
        result=factor(a,*args,**kwargs)
        if stage=='factor':raise RuntimeError('injected factor failure after mutation')
        if stage=='alias':return (result.LU.clone(),result.pivots,result.info)
        return result
    monkeypatch.setattr(torch.linalg,'lu_factor_ex',changed)
    if stage=='solve':monkeypatch.setattr(torch.linalg,'lu_solve',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('injected solve failure')))
    with pytest.raises(RuntimeError):m.solve_reference(handle,b,rule())
    assert torch.equal(handle.matrix,original) and handle._direct_storage_state=='ready'
    assert handle.direct_storage_report['state']=='failed_restored' and handle._direct_storage_backup is None
    torch.testing.assert_close(handle.apply(b),b@original.mT)


@cuda
def test_restore_failure_poison_retains_sealed_descriptor_and_explicit_recovery(monkeypatch):
    handle,b=system();original=handle.matrix.clone();restore=m._SealedMatrixBackup.restore
    monkeypatch.setattr(m._SealedMatrixBackup,'restore',lambda self:(_ for _ in ()).throw(OSError('injected restore failure')))
    with pytest.raises(RuntimeError,match='poisoned'):m.solve_reference(handle,b,rule())
    backup=handle._direct_storage_backup;assert backup is not None and backup.fd is not None
    assert os.fstat(backup.fd).st_size==original.numel()*original.element_size()
    assert fcntl.fcntl(backup.fd,m._F_GET_SEALS)==backup.required_seals
    for operation in (lambda:handle.apply(b),lambda:handle.adjoint(b),lambda:m.residual_loss(handle,b,b),
                      lambda:m.apply_test_riesz_inverse(handle,b),lambda:m.condition_diagnostics(handle),lambda:m.solve_reference(handle,b)):
        with pytest.raises(RuntimeError):operation()
    monkeypatch.setattr(m._SealedMatrixBackup,'restore',restore)
    receipt=handle.restore_direct_storage()
    assert receipt['state']=='restored_original_matrix_only' and torch.equal(handle.matrix,original)
    assert handle._direct_storage_backup is None and handle._direct_storage_state=='ready' and handle._lu is None
    assert handle.direct_storage_report['state']=='unrestored'  # Historical failure is not rewritten.
    assert handle.direct_storage_report['current_operator_state']=='ready'


@cuda
@pytest.mark.parametrize('kind',['matrix','rhs','previous_graph','resident_lu','alias_rhs','complex64'])
def test_explicit_mode_rejects_unsafe_ownership_without_mutation(kind):
    handle,b=system(dtype=torch.complex64 if kind=='complex64' else torch.complex128)
    if kind=='matrix':handle.matrix.requires_grad_()
    elif kind=='rhs':b.requires_grad_()
    elif kind=='previous_graph':handle.apply(b.clone().requires_grad_())
    elif kind=='resident_lu':m.solve_reference(handle,b)
    elif kind=='alias_rhs':b=handle.matrix[:8]
    original=handle.matrix.detach().clone();version=handle.matrix._version
    with pytest.raises(ValueError):m.solve_reference(handle,b,rule())
    assert torch.equal(handle.matrix,original) and handle.matrix._version==version and handle._direct_storage_backup is None


@cuda
def test_restored_matrix_supports_fresh_public_vector_gradients():
    handle,b=system();original=handle.matrix.clone();m.solve_reference(handle,b,rule())
    x=b.clone().requires_grad_();loss=m.residual_loss(handle,x,b)
    grad,=torch.autograd.grad(loss,x)
    expected=(x.detach()@original.mT-b)@original.conj()
    torch.testing.assert_close(grad,expected,rtol=1e-11,atol=1e-11)
    with pytest.raises(ValueError,match='autograd'):m.solve_reference(handle,b,rule())


@cuda
def test_insufficient_host_memory_and_busy_callbacks_never_mutate_matrix(monkeypatch):
    handle,b=system();original=handle.matrix.clone();version=handle.matrix._version
    def check(requested):
        with pytest.raises(RuntimeError):handle.apply(b)
        raise RuntimeError('insufficient host fixture')
    monkeypatch.setattr(m,'_require_host_backup_capacity',check)
    with pytest.raises(RuntimeError,match='host fixture'):m.solve_reference(handle,b,rule())
    assert torch.equal(handle.matrix,original) and handle.matrix._version==version
    assert handle.direct_storage_report['state']=='failed_before_mutation'


@cuda
def test_public_scene_simulation_result_uses_explicit_storage():
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1),mw.Material(eps_r=2.25,sigma_e=.002))],sources=[mw.PlaneWave()])
    simulation=mw.Simulation.mom(scene,frequency=3e8,config=mw.MoMConfig(subdivisions=0,riesz_storage='compact_rwg_v1',
        integration=IntegrationConfig(order=3,singular_order=3),stopping=rule()))
    result=simulation.run()
    assert result.solver_stats['converged'] and result.solver_stats['direct_storage']['matrix_restored_exact']
    assert result.surface_currents.handle._lu is None


@cuda
def test_nested_and_other_thread_operations_reject_during_storage_loan(monkeypatch):
    handle,b=system();original=handle.matrix.clone();factor=torch.linalg.lu_factor_ex;observations=[]
    def blocked():
        try:handle.apply(b)
        except RuntimeError:observations.append('rejected')
        else:observations.append('unexpected_success')
    def checked(*args,**kwargs):
        blocked();thread=threading.Thread(target=blocked);thread.start();thread.join(5)
        assert not thread.is_alive()
        return factor(*args,**kwargs)
    monkeypatch.setattr(torch.linalg,'lu_factor_ex',checked)
    _,log=m.solve_reference(handle,b,rule())
    assert log['converged'] and observations==['rejected','rejected'] and torch.equal(handle.matrix,original)


@cuda
def test_preconditioner_cache_invalidation_is_explicit_and_metric_cache_survives():
    handle,b=system();handle._refresh_matrix();handle._scale=torch.ones(handle.shape[0],device=handle.device);metric=handle._riesz_factor
    _,log=m.solve_reference(handle,b,rule())
    assert log['converged'] and log['direct_storage']['invalidated_matrix_caches']==['_scale']
    assert handle._scale is None and handle._riesz_factor is metric and handle._lu is None


@pytest.mark.parametrize('change',['missing_current','unrelated_mount','hidden_ancestor_root'])
def test_host_cgroup_mapping_cannot_silently_fall_back_to_visible_ancestor(tmp_path,change):
    paths=cgroup_fixture(tmp_path)
    if change=='missing_current':paths['cgroup'].write_text('0::/parent/absent\n')
    elif change=='unrelated_mount':paths['mountinfo'].write_text('1 0 0:1 / /unrelated rw - cgroup2 cgroup2 rw\n')
    else:paths['mountinfo'].write_text('1 0 0:1 /hidden '+str(paths['root'])+' rw - cgroup2 cgroup2 rw\n')
    with pytest.raises(RuntimeError):m._host_backup_capacity(**paths)


@cuda
@pytest.mark.parametrize('shape',[(17,),(1,17),(2,4,17)])
def test_rank_one_and_multidimensional_rhs_keep_original_shape(shape):
    handle,b=system();rhs=b[0] if len(shape)==1 else b[:1] if shape[0]==1 else b.reshape(shape)
    original=handle.matrix.clone();expected=torch.linalg.solve(original,rhs.reshape(-1,17).mT).mT.reshape(shape)
    actual,log=m.solve_reference(handle,rhs,rule())
    assert actual.shape==rhs.shape and log['converged'] and torch.equal(handle.matrix,original)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
