"""CUDA surface-operator actions, dense direct algebra and residual-certified solves."""
from __future__ import annotations
from contextlib import contextmanager
from .operator_protocol import KrylovView, IterativeState


from dataclasses import dataclass, field
import math
import copy
import ctypes
import sys
import functools
import hashlib
import os
import re
from pathlib import Path
import threading
import time
from typing import Any

import torch


# Linux UAPI fcntl.h/memfd.h constants; the pinned Python build omits these
# names while its libc and kernel provide the same native byte-storage API.
_F_ADD_SEALS, _F_GET_SEALS = 1033, 1034
_REQUIRED_SEALS = 1 | 2 | 4 | 8


def _create_sealed_fd():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Explicit direct storage requires Linux sealed memory files')
    library = ctypes.CDLL(None, use_errno=True)
    create = getattr(library, 'memfd_create', None)
    if create is None:
        raise RuntimeError('The native immutable byte-storage API is unavailable')
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    descriptor = create(b'mom-original-matrix', 1 | 2)  # MFD_CLOEXEC | MFD_ALLOW_SEALING
    if descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return descriptor


_HOST_CHUNK_BYTES = 32 * 1024**2
_HOST_RESERVE_BYTES = 1024**3
_VALIDATION_ROWS = 512


def _row_blocks(tensor):
    """Row slices that bound validation temporaries to one block, never N^2."""
    for start in range(0, tensor.shape[0], _VALIDATION_ROWS):
        yield start, tensor[start:start + _VALIDATION_ROWS]


def _all_finite(tensor):
    return all(bool(torch.isfinite(block).all()) for _, block in _row_blocks(tensor))


def _absolute_maximum(tensor):
    return max(float(block.abs().max()) for _, block in _row_blocks(tensor))


def _self_adjoint(tensor, *, conjugate, rtol, atol):
    """Row-blocked ``allclose(tensor, tensor.mH)`` (or ``.mT``); same elementwise predicate."""
    for start, block in _row_blocks(tensor):
        columns = tensor[:, start:start + block.shape[0]]
        if not torch.allclose(block, columns.mH if conjugate else columns.mT, rtol=rtol, atol=atol):
            return False
    return True


def _host_backup_capacity(*, meminfo='/proc/meminfo', cgroup='/proc/self/cgroup', root='/sys/fs/cgroup', mountinfo='/proc/self/mountinfo'):
    """Configuration/I/O only: available RAM and every visible cgroup-v2 limit."""
    values = dict(line.split(':', 1) for line in Path(meminfo).read_text().splitlines() if ':' in line)
    if 'MemAvailable' not in values:
        raise RuntimeError('Available host memory cannot be verified')
    available = int(values['MemAvailable'].strip().split()[0]) * 1024
    groups = [line[3:] for line in Path(cgroup).read_text().splitlines() if line.startswith('0::')]
    if len(groups) != 1 or not groups[0].startswith('/') or '..' in Path(groups[0]).parts:
        raise RuntimeError('A visible unified cgroup-v2 memory hierarchy is required')
    root = Path(root).resolve()
    mounts = []
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)
    for line in Path(mountinfo).read_text().splitlines():
        before, separator, after = line.partition(' - ')
        fields = before.split()
        if separator and after.split()[0] == 'cgroup2' and len(fields) >= 5:
            if Path(unescape(fields[4])).resolve() == root:
                mounts.append(dict(mount_root=unescape(fields[3]), mount_point=str(root)))
    if len(mounts) != 1 or mounts[0]['mount_root'] != '/':
        raise RuntimeError('Cannot verify full visible cgroup-v2 root mapping for the process')
    current = root / groups[0].lstrip('/')
    if not current.is_dir():
        raise RuntimeError('Current process cgroup is absent from the verified visible mount')
    observations = []
    while True:
        maximum, used = current/'memory.max', current/'memory.current'
        if maximum.is_file():
            text = maximum.read_text().strip()
            usage = int(used.read_text().strip())
            limit = None if text == 'max' else int(text)
            if usage < 0 or (limit is not None and limit < 0):
                raise RuntimeError('Invalid cgroup memory accounting')
            if limit is not None:
                available = min(available, max(0, limit - usage))
            observations.append(dict(path=str(current), maximum_bytes=limit, current_bytes=usage))
        if current == root:
            break
        current = current.parent
    if not observations:
        raise RuntimeError('No visible cgroup memory accounting could be verified')
    return dict(available_bytes=available, mem_available_bytes=int(values['MemAvailable'].strip().split()[0])*1024,
                cgroup_limits=observations, cgroup_membership=groups[0], cgroup_mount=mounts[0],
                visibility_scope='reported membership and ancestors in the visible full-root cgroup2 mount',
                accounting_scope='sealed backup memory is separate from process RSS')


def _require_host_backup_capacity(requested):
    record = _host_backup_capacity()
    needed = requested + 2*_HOST_CHUNK_BYTES + _HOST_RESERVE_BYTES
    record.update(requested_backup_bytes=requested, scratch_and_headroom_bytes=needed-requested)
    if record['available_bytes'] < needed:
        raise RuntimeError('Insufficient verified host/cgroup memory for immutable matrix backup')
    return record


class _SealedMatrixBackup:
    """Process-local immutable byte storage; no CPU numerical computation."""
    def __init__(self, matrix):
        import fcntl
        self.matrix = matrix
        self.fd = _create_sealed_fd()
        self.identity = (id(matrix), matrix.data_ptr(), tuple(matrix.shape), tuple(matrix.stride()), matrix.dtype)
        self.bytes = matrix.numel()*matrix.element_size()
        self.rows = max(1, _HOST_CHUNK_BYTES//(matrix.shape[1]*matrix.element_size()))
        self.sha256 = None
        self.required_seals = _REQUIRED_SEALS

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def populate(self):
        import fcntl
        digest = hashlib.sha256()
        for start in range(0, self.matrix.shape[0], self.rows):
            # This host tensor is a transient byte-I/O buffer, never an operand
            # of a CPU operator, factorization, solve, norm or field evaluation.
            host = self.matrix[start:start+self.rows].detach().view(torch.uint8).cpu()
            buffer = memoryview(host.numpy()).cast('B')
            digest.update(buffer)
            offset = 0
            while offset < len(buffer):
                written = os.write(self.fd, buffer[offset:])
                if written <= 0:
                    raise OSError('Matrix backup byte write made no progress')
                offset += written
            del buffer, host
        if os.fstat(self.fd).st_size != self.bytes:
            raise OSError('Matrix backup is incomplete')
        fcntl.fcntl(self.fd, _F_ADD_SEALS, self.required_seals)
        if fcntl.fcntl(self.fd, _F_GET_SEALS) != self.required_seals:
            raise OSError('Matrix backup did not receive all immutable seals')
        self.sha256 = digest.hexdigest()
        verified = hashlib.sha256()
        for offset in range(0, self.bytes, _HOST_CHUNK_BYTES):
            data = os.pread(self.fd, min(_HOST_CHUNK_BYTES, self.bytes-offset), offset)
            if len(data) != min(_HOST_CHUNK_BYTES, self.bytes-offset):
                raise OSError('Matrix backup byte verification was incomplete')
            verified.update(data)
        if verified.hexdigest() != self.sha256:
            raise OSError('Immutable matrix backup differs from original transferred bytes')

    def restore(self):
        import fcntl
        if (id(self.matrix), self.matrix.data_ptr(), tuple(self.matrix.shape), tuple(self.matrix.stride()), self.matrix.dtype) != self.identity:
            raise RuntimeError('Borrowed CUDA storage/layout changed; original sealed bytes remain retained')
        if (self.fd is None or self.sha256 is None or os.fstat(self.fd).st_size != self.bytes
                or fcntl.fcntl(self.fd, _F_GET_SEALS) != self.required_seals):
            raise RuntimeError('A complete sealed original-matrix backup is unavailable')
        digest = hashlib.sha256()
        row_bytes = self.matrix.shape[1]*self.matrix.element_size()
        for start in range(0, self.matrix.shape[0], self.rows):
            count = min(self.rows, self.matrix.shape[0]-start)
            data = os.pread(self.fd, count*row_bytes, start*row_bytes)
            if len(data) != count*row_bytes:
                raise OSError('Immutable matrix restore read was incomplete')
            digest.update(data)
            host = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(count, row_bytes)
            self.matrix[start:start+count].view(torch.uint8).copy_(host, non_blocking=False)
            del data, host
        torch.cuda.synchronize(self.matrix.device)
        if digest.hexdigest() != self.sha256:
            raise RuntimeError('Restored input bytes differ from sealed original digest')
        actual = hashlib.sha256()
        for start in range(0, self.matrix.shape[0], self.rows):
            host = self.matrix[start:start+self.rows].detach().view(torch.uint8).cpu()
            actual.update(memoryview(host.numpy()).cast('B'))
            del host
        if actual.hexdigest() != self.sha256:
            raise RuntimeError('CUDA original matrix failed exact restored-byte verification')
        return actual.hexdigest()


def _matrix_operation(function):
    """Use the backend's reentrant read lease around operator operations."""
    @functools.wraps(function)
    def wrapped(handle, *args, **kwargs):
        with handle.read_access():
            return function(handle, *args, **kwargs)
    return wrapped


def _direct_storage_error(error):
    return dict(type=type(error).__name__, message=str(error) or repr(error))


def _restore_borrowed_matrix(handle):
    backup = handle._direct_storage_backup
    if backup is None or handle.matrix is not backup.matrix:
        raise RuntimeError('Original borrowed tensor identity changed; sealed backup must be retained')
    restored = backup.restore()
    handle._invalidate_matrix_caches()
    # Exact original byte identity certifies the previously validated finite
    # matrix. Avoid a new full-size validation mask or an unrecorded matrix copy.
    handle._matrix_version = (id(handle.matrix), handle.matrix._version)
    handle._direct_storage_state = 'ready'
    return restored


def _one_shot_direct(handle, b, x0):
    """One CUDA LU, all RHS, then exact original-matrix restoration."""
    a = handle.matrix
    if (a.dtype != torch.complex128 or not a.is_contiguous() or a.storage_offset() != 0
            or a.untyped_storage().nbytes() != a.numel()*a.element_size() or a.is_conj() or a.is_neg()):
        raise ValueError('Explicit direct storage requires a full contiguous plain complex128 CUDA matrix')
    metric = handle.test_riesz
    if isinstance(metric, BlockCompactRWGRiesz):
        grad_metric = [value for block in metric.blocks for value in (block.gram, block.impedance_squared)]
    else:
        grad_metric = ([metric.gram, metric.impedance_squared] if isinstance(metric, CompactRWGRiesz) else [metric])
    if (a.requires_grad or b.requires_grad or x0.requires_grad
            or any(isinstance(value, torch.Tensor) and value.requires_grad for value in grad_metric)
            or handle._matrix_autograd_identity == (id(a), a.data_ptr())):
        raise ValueError('Explicit direct storage requires exclusive non-autograd matrix ownership')
    if any(value.untyped_storage().data_ptr() == a.untyped_storage().data_ptr() for value in (b, x0)):
        raise ValueError('Direct-storage RHS and initial guess must not alias the matrix')
    if handle._lu is not None:
        raise ValueError('Explicit direct storage requires no resident LU cache')
    thread = threading.get_ident()
    with handle._matrix_lease_lock:
        if (handle._direct_storage_state != 'ready' or handle._matrix_borrower is not None
                or any(owner != thread for owner in handle._matrix_readers)):
            raise RuntimeError('Cannot borrow a matrix with another active operation')
        handle._matrix_borrower = thread
        handle._direct_storage_state = 'borrowed'
    report = dict(storage='sealed_host_restore_v1', state='started', phases={}, matrix_bytes=a.numel()*a.element_size(),
                  matrix_ptr=a.data_ptr(), original_version=a._version, rhs_count=b.numel()//b.shape[-1],
                  cpu_scope='immutable byte I/O and hashes only; all numerical work is CUDA',
                  host_memory_scope='sealed backup bytes plus transient buffers; not included by RSS alone',
                  backup_recoverability='process-local descriptor while retained; not durable across process loss')
    handle._direct_storage_last_report = report
    primary = restore_error = None
    mutated = False
    result = None
    failure = None
    start = time.perf_counter()
    def phase(name, operation):
        started = time.perf_counter()
        try:
            value = operation()
        except BaseException as error:
            report['phases'][name] = dict(state='failed', seconds=time.perf_counter()-started, error=_direct_storage_error(error))
            raise
        report['phases'][name] = dict(state='completed', seconds=time.perf_counter()-started)
        return value
    try:
        report['host_preflight'] = phase('host_preflight', lambda: _require_host_backup_capacity(report['matrix_bytes']))
        phase('cuda_quiescence', lambda: torch.cuda.synchronize(a.device))
        backup = _SealedMatrixBackup(a)
        handle._direct_storage_backup = backup
        phase('backup', backup.populate)
        report.update(sealed_backup_bytes=backup.bytes, backup_sha256=backup.sha256, required_seals=backup.required_seals,
                      backup_descriptor=backup.fd, backup_descriptor_path=f'/proc/{os.getpid()}/fd/{backup.fd}')
        report['host_before_mutation'] = phase('host_before_mutation', lambda: _require_host_backup_capacity(0))
        if handle.matrix is not a or a.data_ptr() != report['matrix_ptr'] or a._version != report['original_version']:
            raise RuntimeError('Exclusive matrix ownership changed during backup; no storage mutation was started')
        report['invalidated_matrix_caches'] = [name for name in ('_scale','_column_scale','_balanced_row_scale','_recycle_u','_recycle_c') if getattr(handle,name) is not None]
        handle._invalidate_matrix_caches()
        at = a.mT
        pivots = torch.empty(a.shape[0], device=a.device, dtype=torch.int32)
        info = torch.empty((), device=a.device, dtype=torch.int32)
        report['factor_input'] = dict(pointer=at.data_ptr(), storage_pointer=at.untyped_storage().data_ptr(),
                                      strides=list(at.stride()), tensor_identity=at._cdata)
        mutated = True
        factor, returned_pivots, returned_info = phase('factor', lambda: torch.linalg.lu_factor_ex(at, out=(at, pivots, info)))
        phase('factor_synchronize', lambda: torch.cuda.synchronize(a.device))
        report['factor_info'] = int(info)
        report['factor_output_alias'] = (factor._cdata == report['factor_input']['tensor_identity'] and factor.data_ptr() == report['matrix_ptr'] == a.data_ptr()
            and factor.untyped_storage().data_ptr() == report['factor_input']['storage_pointer']
            and tuple(factor.stride()) == (1, a.shape[0]) and returned_pivots.data_ptr() == pivots.data_ptr()
            and returned_info.data_ptr() == info.data_ptr())
        if not report['factor_output_alias']:
            raise RuntimeError('CUDA LU did not preserve the exact reviewed input/output alias')
        if report['factor_info'] < 0:
            raise RuntimeError('CUDA LU rejected an argument')
        if report['factor_info'] > 0:
            failure = 'singular_matrix'
            result = torch.zeros_like(b)
        else:
            result = phase('solve', lambda: torch.linalg.lu_solve(at, pivots, b.reshape(-1, b.shape[-1]), left=False).reshape(b.shape))
            phase('solve_synchronize', lambda: torch.cuda.synchronize(a.device))
    except BaseException as error:
        primary = error
        report['primary_error'] = _direct_storage_error(error)
    finally:
        if mutated:
            try:
                report['restored_sha256'] = phase('restore', lambda: _restore_borrowed_matrix(handle))
                report['restored_version'] = a._version
                report['matrix_restored_exact'] = True
            except BaseException as error:
                restore_error = error
                handle._direct_storage_state = 'unrestored'
                report['restore_error'] = _direct_storage_error(error)
                report['matrix_restored_exact'] = False
        else:
            handle._direct_storage_state = 'ready'
            report['matrix_not_mutated'] = True
        if handle._direct_storage_state == 'ready' and handle._direct_storage_backup is not None:
            try:
                phase('close_backup', handle._direct_storage_backup.close)
                handle._direct_storage_backup = None
                report['backup_descriptor_open'] = False
            except BaseException as error:
                report['close_error'] = _direct_storage_error(error)
                if primary is None:
                    primary = error
        else:
            report['backup_descriptor_open'] = handle._direct_storage_backup is not None
        with handle._matrix_lease_lock:
            handle._matrix_borrower = None
        report['elapsed_seconds'] = time.perf_counter()-start
        report['backup_descriptor_open'] = handle._direct_storage_backup is not None and handle._direct_storage_backup.fd is not None
        report['state'] = ('unrestored' if restore_error else 'failed_before_mutation' if primary and not mutated
                           else 'failed_restored' if primary or failure else 'succeeded_restored')
    if restore_error is not None:
        raise RuntimeError('Original matrix restoration failed; operator is poisoned and sealed backup remains attached') from (primary or restore_error)
    if primary is not None:
        raise primary
    return result, failure


@dataclass
class CompactRWGRiesz:
    """Exact ``diag(G / impedance_squared, G)`` stored as a real CUDA Gram.

    This explicitly selected representation is not a Tensor and never
    materializes the full block metric implicitly. The operator owns the
    real Cholesky cache; mutations of either operand invalidate that cache.
    """

    gram: torch.Tensor
    impedance_squared: float | torch.Tensor


@dataclass(frozen=True)
class BlockCompactRWGRiesz:
    """Per-body Gram storage with global all-J then all-M coefficient order."""
    blocks: tuple[CompactRWGRiesz, ...]

    def __post_init__(self):
        if not isinstance(self.blocks, tuple) or not self.blocks or any(not isinstance(b, CompactRWGRiesz) for b in self.blocks):
            raise ValueError('Block Riesz requires a nonempty tuple of compact body metrics')


def _compact_factor(metric, n, dtype, device):
    gram, scale = metric.gram, metric.impedance_squared
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    if not isinstance(gram, torch.Tensor) or gram.shape != (n, n) or gram.device != device or gram.dtype != real_dtype:
        raise ValueError('Compact body Gram must match its real CUDA coefficient space')
    if isinstance(scale, torch.Tensor):
        if scale.ndim != 0 or scale.device != device or scale.dtype != real_dtype:
            raise ValueError('Compact body impedance must match Gram dtype/device')
    elif type(scale) not in (float, int):
        raise ValueError('Compact body impedance must be a positive real scalar')
    value = torch.as_tensor(scale, dtype=real_dtype, device=device)
    if not bool(torch.isfinite(value) & (value > 0)) or not _all_finite(gram):
        raise ValueError('Compact body metric must be finite and positive')
    eps = torch.finfo(real_dtype).eps
    if not _self_adjoint(gram, conjugate=False, rtol=32*eps, atol=32*eps*_absolute_maximum(gram.detach())):
        raise ValueError('Compact body Gram must be symmetric')
    factor, info = torch.linalg.cholesky_ex(gram.detach())
    if bool(info != 0):
        raise ValueError('Compact body Gram must be positive definite')
    return factor


@dataclass
class DenseOperator:
    """Fixed-geometry operator with an explicitly named test-space Riesz map.

    Vectors use the last dimension for coefficients. ``test_riesz`` is the
    Hermitian positive definite Gram matrix defining the residual dual norm,
    or an explicitly selected exact ``CompactRWGRiesz`` representation.
    In-place changes invalidate numerical caches through tensor version checks.
    """

    matrix: torch.Tensor
    test_riesz: torch.Tensor | CompactRWGRiesz | BlockCompactRWGRiesz
    metadata: dict[str, Any] = field(default_factory=dict)
    _lu: Any = field(default=None, init=False, repr=False)
    _scale: Any = field(default=None, init=False, repr=False)
    _column_scale: Any = field(default=None, init=False, repr=False)
    _balanced_row_scale: Any = field(default=None, init=False, repr=False)
    _riesz_factor: Any = field(default=None, init=False, repr=False)
    _matrix_version: Any = field(default=None, init=False, repr=False)
    _riesz_version: Any = field(default=None, init=False, repr=False)
    iterative_state: IterativeState = field(default_factory=IterativeState, init=False, repr=False)
    _apply_calls: int = field(default=0, init=False, repr=False)
    _adjoint_calls: int = field(default=0, init=False, repr=False)
    _validation_seconds: float = field(default=0., init=False, repr=False)
    _matrix_lease_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _matrix_readers: dict = field(default_factory=dict, init=False, repr=False)
    _matrix_borrower: Any = field(default=None, init=False, repr=False)
    _direct_storage_state: str = field(default='ready', init=False, repr=False)
    _direct_storage_backup: Any = field(default=None, init=False, repr=False)
    _direct_storage_last_report: Any = field(default=None, init=False, repr=False)
    _matrix_autograd_identity: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        started = time.perf_counter()
        a = self.matrix
        if a.device.type != 'cuda' or a.dtype not in (torch.complex64, torch.complex128):
            raise ValueError('MoM dense operators require complex64/complex128 CUDA tensors')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError('matrix must have nonempty square shape [N, N]')
        if not _all_finite(a):
            raise ValueError('operator must be finite')
        self._validate_riesz()
        self._validation_seconds = time.perf_counter()-started

    def _metric_version(self):
        r = self.test_riesz
        if isinstance(r, BlockCompactRWGRiesz):
            versions = []
            for block in r.blocks:
                if not isinstance(block.gram, torch.Tensor):
                    raise ValueError('Block Riesz Gram must be a CUDA tensor')
                scale = block.impedance_squared
                if isinstance(scale, torch.Tensor):
                    scale_version = (id(scale), scale._version)
                elif type(scale) in (int, float):
                    scale_version = scale
                else:
                    raise ValueError('Block Riesz impedance must be real')
                versions.append((id(block), id(block.gram), block.gram._version, scale_version))
            return (id(r), tuple(versions), self.shape, self.dtype, self.device)
        if isinstance(r, CompactRWGRiesz):
            if not isinstance(r.gram, torch.Tensor):
                raise ValueError('compact Riesz Gram must be a real CUDA tensor')
            scale = r.impedance_squared
            if isinstance(scale, torch.Tensor):
                scale_version = (id(scale), scale._version)
            elif type(scale) in (int, float):
                scale_version = scale
            else:
                raise ValueError('compact Riesz impedance_squared must be a positive real scalar')
            return (id(r), id(r.gram), r.gram._version, scale_version,
                    self.matrix.shape, self.matrix.dtype, self.matrix.device)
        if not isinstance(r, torch.Tensor):
            raise ValueError('test_riesz must be a tensor or CompactRWGRiesz')
        return (id(r), r._version)

    def _validate_riesz(self):
        r = self.test_riesz
        if isinstance(r, BlockCompactRWGRiesz):
            version = self._metric_version()
            sizes = [block.gram.shape[0] if block.gram.ndim == 2 else 0 for block in r.blocks]
            if any(size < 1 for size in sizes) or 2*sum(sizes) != self.shape[0]:
                raise ValueError('Block Riesz dimensions do not match global J/M space')
            self._riesz_factor = tuple(_compact_factor(block, size, self.dtype, self.device) for block, size in zip(r.blocks, sizes))
            self._riesz_version = version
            return
        if isinstance(r, CompactRWGRiesz):
            version = self._metric_version()
            gram, scale = r.gram, r.impedance_squared
            n = self.matrix.shape[0]
            if (n % 2 or gram.shape != (n // 2, n // 2)
                    or gram.device != self.matrix.device or gram.dtype != self.matrix.real.dtype):
                raise ValueError('compact Riesz Gram must match half the matrix shape, real dtype, and CUDA device')
            if isinstance(scale, torch.Tensor):
                if scale.ndim != 0 or scale.device != gram.device or scale.dtype != gram.dtype:
                    raise ValueError('compact Riesz impedance_squared must match Gram real dtype and CUDA device')
            scale_value = (scale.detach() if isinstance(scale, torch.Tensor)
                           else torch.as_tensor(scale, dtype=gram.dtype, device=gram.device))
            if not bool(torch.isfinite(scale_value) & (scale_value > 0)):
                raise ValueError('compact Riesz impedance_squared must be positive and finite')
            if not _all_finite(gram):
                raise ValueError('compact Riesz Gram must be finite')
            eps = torch.finfo(gram.dtype).eps
            if not _self_adjoint(gram, conjugate=False, rtol=32 * eps, atol=32 * eps * _absolute_maximum(gram.detach())):
                raise ValueError('compact Riesz Gram must be symmetric')
            factor, info = torch.linalg.cholesky_ex(gram.detach())
            if bool(info != 0):
                raise ValueError('compact Riesz Gram must be positive definite')
            self._riesz_factor = factor
            self._riesz_version = version
            return
        self._metric_version()
        if r.shape != self.matrix.shape or r.device != self.matrix.device or r.dtype != self.matrix.dtype:
            raise ValueError('test_riesz must match matrix shape, dtype, and CUDA device')
        if not _all_finite(r):
            raise ValueError('test Riesz map must be finite')
        eps = torch.finfo(r.real.dtype).eps
        if not _self_adjoint(r, conjugate=True, rtol=32 * eps, atol=32 * eps * _absolute_maximum(r.detach())):
            raise ValueError('test_riesz must be Hermitian')
        factor, info = torch.linalg.cholesky_ex(r)
        if bool(info != 0):
            raise ValueError('test_riesz must be positive definite')
        self._riesz_factor = factor
        self._riesz_version = (id(r), r._version)

    def _invalidate_matrix_caches(self):
        self._lu = self._scale = self._column_scale = self._balanced_row_scale = None
        self._recycle_u = self._recycle_c = None

    @property
    def direct_storage_report(self):
        """Copy of the most recent storage lifecycle, including retained failure."""
        report = copy.deepcopy(self._direct_storage_last_report)
        if report is not None:
            report['current_operator_state'] = self._direct_storage_state
            report['current_backup_descriptor_open'] = self._direct_storage_backup is not None and self._direct_storage_backup.fd is not None
        return report

    @torch.no_grad()
    def restore_direct_storage(self):
        """Explicitly retry byte restoration of an unrestored process-local loan.

        This never solves or changes the failed solve outcome. The exact sealed
        descriptor remains attached if recovery itself fails.
        """
        with self._matrix_lease_lock:
            if (self._direct_storage_state != 'unrestored' or self._matrix_borrower is not None
                    or self._matrix_readers):
                raise RuntimeError('Explicit restoration requires an unrestored idle operator')
            self._matrix_borrower = threading.get_ident()
        attempt = dict(started_unix=time.time())
        self._direct_storage_last_report.setdefault('restoration_retries', []).append(attempt)
        start = time.perf_counter()
        try:
            attempt['restored_sha256'] = _restore_borrowed_matrix(self)
            self._direct_storage_backup.close()
            self._direct_storage_backup = None
            attempt['state'] = 'restored_original_matrix_only'
        except BaseException as error:
            self._direct_storage_state = 'unrestored'
            attempt.update(state='failed', error=_direct_storage_error(error))
            raise
        finally:
            attempt['seconds'] = time.perf_counter()-start
            with self._matrix_lease_lock:
                self._matrix_borrower = None
        return copy.deepcopy(attempt)

    def _refresh_matrix(self):
        if self._direct_storage_state != 'ready':
            raise RuntimeError('Operator matrix is borrowed or unrestored')
        version = (id(self.matrix), self.matrix._version)
        if version != self._matrix_version:
            a = self.matrix
            if a.device.type != 'cuda' or a.dtype not in (torch.complex64, torch.complex128):
                raise ValueError('MoM dense operators require complex64/complex128 CUDA tensors')
            if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
                raise ValueError('matrix must have nonempty square shape [N, N]')
            if not _all_finite(a):
                raise ValueError('operator matrix must be finite')
            self._lu = None
            self._scale = None
            self._column_scale = None
            self._balanced_row_scale = None
            self._recycle_u = None
            self._recycle_c = None
            self._matrix_version = version

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def device(self):
        return self.matrix.device

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def _recycle_u(self): return self.iterative_state.recycle_u
    @_recycle_u.setter
    def _recycle_u(self, value): self.iterative_state.recycle_u = value
    @property
    def _recycle_c(self): return self.iterative_state.recycle_c
    @_recycle_c.setter
    def _recycle_c(self, value): self.iterative_state.recycle_c = value

    def operator_stats(self):
        tensors = [self.matrix]
        if isinstance(self.test_riesz, BlockCompactRWGRiesz):
            tensors.extend(block.gram for block in self.test_riesz.blocks)
            tensors.extend(self._riesz_factor)
            factor = None
        else:
            tensors.extend([self.test_riesz.gram] if isinstance(self.test_riesz, CompactRWGRiesz) else [self.test_riesz])
            factor = self._riesz_factor
        tensors.extend(t for t in (factor, self._recycle_u, self._recycle_c,
                                  self._scale, self._column_scale, self._balanced_row_scale) if t is not None)
        if self._lu is not None:
            tensors.extend(self._lu)
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
        return {'operator_backend': 'dense', 'apply_calls': self._apply_calls, 'adjoint_calls': self._adjoint_calls,
                'storage_bytes': sum(storages.values()), 'storage_scope': 'unique owned CUDA storages including numerical caches',
                'setup_seconds': self.metadata.get('assembly_seconds', 0.)+self._validation_seconds,
                'apply_seconds': None, 'workspace_bytes': None,
                'timing_scope': 'No per-action CUDA synchronization; unmeasured values remain null'}

    def persistence_receipt(self):
        def tensor_record(tensor):
            checksum = hashlib.sha256()
            rows = max(1, (16*1024**2)//max(1, tensor[0].numel()*tensor.element_size()))
            for block in tensor.detach().split(rows):
                checksum.update(block.contiguous().cpu().numpy().tobytes())
            return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': checksum.hexdigest()}
        with self.read_access():
            self.refresh()
            metric = self.test_riesz
            if isinstance(metric, BlockCompactRWGRiesz):
                riesz = {'storage': 'block_compact_rwg_v1', 'order': 'all_J_then_all_M',
                    'blocks': [{'gram': tensor_record(block.gram), 'impedance_squared': float(block.impedance_squared)} for block in metric.blocks]}
            elif isinstance(metric, CompactRWGRiesz):
                riesz = {'storage': 'compact_rwg_v1', 'gram': tensor_record(metric.gram),
                         'impedance_squared': float(metric.impedance_squared)}
            else:
                riesz = {'storage': 'dense', 'tensor': tensor_record(metric)}
            return {'version': 'surface_operator_receipt_v1', 'operator_backend': 'dense',
                    'kernel_id': self.metadata.get('kernel_id', 'unspecified'),
                    'kernel_version': self.metadata.get('kernel_version', 'unspecified'),
                    'formulation_id': self.metadata.get('formulation_id', 'unspecified'),
                    'numerical_identity': copy.deepcopy(self.metadata.get('numerical_identity', self.metadata.get('quadrature', {}))),
                    'matrix': tensor_record(self.matrix), 'test_riesz': riesz}

    supports_direct = True

    def refresh(self):
        self._refresh_matrix()

    @contextmanager
    def read_access(self):
        thread = threading.get_ident()
        with self._matrix_lease_lock:
            if self._direct_storage_state != 'ready' or self._matrix_borrower not in (None, thread):
                raise RuntimeError('Operator matrix is exclusively borrowed or unrestored')
            self._matrix_readers[thread] = self._matrix_readers.get(thread, 0)+1
        try:
            yield
        finally:
            with self._matrix_lease_lock:
                self._matrix_readers[thread] -= 1
                if not self._matrix_readers[thread]:
                    del self._matrix_readers[thread]

    def riesz_inverse(self, residual):
        return apply_test_riesz_inverse(self, residual)

    def preconditioner_scales(self, kind):
        self.refresh()
        ones = torch.ones(self.shape[0], device=self.device, dtype=self.matrix.real.dtype)
        cached = False
        if kind == 'none':
            return ones, ones, cached
        if kind == 'row_norm':
            cached = self._scale is not None
            if not cached:
                norms = torch.linalg.vector_norm(self.matrix, dim=1)
                self._scale = torch.where(norms > 0, norms.reciprocal(), ones)
            return self._scale, ones, cached
        if kind == 'row_column_norm':
            cached = self._column_scale is not None
            if not cached:
                norms = torch.linalg.vector_norm(self.matrix, dim=0)
                self._column_scale = torch.where(norms > 0, norms.reciprocal(), ones)
                balanced = torch.cat([torch.linalg.vector_norm(rows*self._column_scale[None], dim=1)
                                      for rows in self.matrix.split(256)])
                self._balanced_row_scale = torch.where(balanced > 0, balanced.reciprocal(), ones)
            return self._balanced_row_scale, self._column_scale, cached
        raise ValueError('Unsupported operator preconditioner')

    def apply(self, x):
        return apply_A(self, x)

    def adjoint(self, y):
        return apply_A_adjoint(self, y)


def _vector(handle, x):
    handle.refresh()
    if x.ndim < 1 or x.shape[-1] != handle.shape[0]:
        raise ValueError('coefficient vectors must have final dimension N')
    if x.dtype != handle.dtype or x.device != handle.device:
        raise ValueError('vectors must match the operator dtype and CUDA device')


@_matrix_operation
def apply_A(handle: DenseOperator, x: torch.Tensor) -> torch.Tensor:
    _vector(handle, x)
    if not isinstance(handle, DenseOperator):
        return handle.apply(x)
    if torch.is_grad_enabled() and (x.requires_grad or handle.matrix.requires_grad):
        handle._matrix_autograd_identity = (id(handle.matrix), handle.matrix.data_ptr())
    handle._apply_calls += 1
    return x @ handle.matrix.mT


@_matrix_operation
def apply_A_adjoint(handle: DenseOperator, y: torch.Tensor) -> torch.Tensor:
    _vector(handle, y)
    if not isinstance(handle, DenseOperator):
        return handle.adjoint(y)
    # Conjugate only vectors: a conjugated matrix operand may require a full
    # matrix copy in the backend, even though Tensor.conj itself is a view.
    if torch.is_grad_enabled() and (y.requires_grad or handle.matrix.requires_grad):
        handle._matrix_autograd_identity = (id(handle.matrix), handle.matrix.data_ptr())
    handle._adjoint_calls += 1
    return (y.conj() @ handle.matrix).conj().resolve_conj()


@_matrix_operation
def apply_test_riesz_inverse(handle: DenseOperator, r: torch.Tensor) -> torch.Tensor:
    _vector(handle, r)
    if not isinstance(handle, DenseOperator):
        return handle.riesz_inverse(r)
    if handle._metric_version() != handle._riesz_version:
        handle._validate_riesz()
    if isinstance(handle.test_riesz, BlockCompactRWGRiesz):
        total = handle.shape[0]//2
        result = torch.empty_like(r)
        offset = 0
        for block, cached in zip(handle.test_riesz.blocks, handle._riesz_factor):
            n = block.gram.shape[0]
            factor = torch.linalg.cholesky(block.gram) if block.gram.requires_grad else cached
            columns = torch.stack((r[..., offset:offset+n], r[..., total+offset:total+offset+n]), dim=-2)
            shape = columns.shape
            columns = columns.reshape(-1, n).mT
            solved = torch.complex(torch.cholesky_solve(columns.real, factor), torch.cholesky_solve(columns.imag, factor))
            solved = solved.mT.reshape(shape)
            result[..., offset:offset+n] = solved[..., 0, :]*block.impedance_squared
            result[..., total+offset:total+offset+n] = solved[..., 1, :]
            offset += n
        return result
    if isinstance(handle.test_riesz, CompactRWGRiesz):
        metric = handle.test_riesz
        factor = (torch.linalg.cholesky(metric.gram) if metric.gram.requires_grad
                  else handle._riesz_factor)
        n = metric.gram.shape[0]
        columns = r.reshape(-1, 2, n).reshape(-1, n).mT
        solved = torch.complex(torch.cholesky_solve(columns.real, factor),
                               torch.cholesky_solve(columns.imag, factor))
        solved = solved.mT.reshape(*r.shape[:-1], 2, n)
        return torch.cat((solved[..., 0, :] * metric.impedance_squared,
                          solved[..., 1, :]), dim=-1)
    # Rebuild the factor when differentiating the metric itself, so separate
    # backward calls do not reuse a consumed autograd graph.
    factor = (torch.linalg.cholesky(handle.test_riesz) if handle.test_riesz.requires_grad
              else handle._riesz_factor)
    shape = r.shape
    return torch.cholesky_solve(r.reshape(-1, shape[-1]).mT, factor).mT.reshape(shape)


@_matrix_operation
def residual_loss(handle: DenseOperator, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One half the squared dual norm, summed over all batch dimensions."""
    _vector(handle, b)
    r = apply_A(handle, x) - b
    return 0.5 * (r.conj() * apply_test_riesz_inverse(handle, r)).real.sum()


@_matrix_operation
def condition_diagnostics(handle: DenseOperator) -> dict:
    """Small dense SVD diagnostic; this is not a certified stability bound."""
    if not isinstance(handle, DenseOperator):
        from .operator_protocol import estimate_condition
        return estimate_condition(handle)
    handle._refresh_matrix()
    s = torch.linalg.svdvals(handle.matrix)
    return {'sigma_max': float(s[0]), 'sigma_min': float(s[-1]),
            'condition_number': float(s[0] / s[-1]),
            'certified_stability_bound': False, 'method': 'dense_svd', 'exact_svd': True}


@dataclass(frozen=True)
class StoppingRule:
    rtol: float = 1e-8
    atol: float = 1e-12
    maxiter: int = 500
    restart: int = 40
    stagnation_window: int | None = None
    stagnation_rtol: float = 1e-4
    method: str = 'gmres'
    preconditioner: str = 'row_norm'
    recycle_size: int = 0
    fallback: str = 'none'
    direct_storage: str = 'resident'

    def __post_init__(self):
        if self.stagnation_window is None:
            object.__setattr__(self, 'stagnation_window', self.restart)
        if not math.isfinite(self.rtol) or not math.isfinite(self.atol) or self.rtol < 0 or self.atol < 0:
            raise ValueError('residual tolerances must be finite and nonnegative')
        if self.rtol == self.atol == 0:
            raise ValueError('at least one residual tolerance must be positive')
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
               for v in (self.maxiter, self.restart, self.stagnation_window)):
            raise ValueError('iteration, restart, and stagnation budgets must be positive integers')
        if not 0 <= self.stagnation_rtol < 1:
            raise ValueError('stagnation_rtol must be in [0, 1)')
        if self.direct_storage not in ('resident', 'sealed_host_restore_v1') or (self.direct_storage != 'resident' and (self.method != 'direct' or self.fallback != 'none')):
            raise ValueError('direct_storage must be resident, or sealed_host_restore_v1 with direct and no fallback')
        if self.method not in ('direct', 'gmres', 'block_gmres'):
            raise ValueError('method must be direct, gmres, or block_gmres')
        if self.preconditioner not in ('none', 'row_norm', 'row_column_norm', 'rwg_diagonal'):
            raise ValueError('preconditioner must be none, row_norm, row_column_norm, or backend-supported rwg_diagonal')
        if not isinstance(self.recycle_size, int) or isinstance(self.recycle_size, bool) or self.recycle_size < 0:
            raise ValueError('recycle_size must be a nonnegative integer')
        if self.recycle_size and self.method != 'gmres':
            raise ValueError('recycled correction subspaces require method=gmres')
        if self.fallback not in ('none', 'direct') or (self.fallback != 'none' and self.method not in ('gmres', 'block_gmres')):
            raise ValueError('fallback must be none, or direct with an iterative method')


def _norm(x):
    return float(torch.linalg.vector_norm(x))


def _recycle_initial_correction(handle, b, initial, capacity):
    """Minimize the true residual over stored correction directions.

    C = A U has orthonormal columns; U itself need not be orthonormal.
    This is an initial augmentation followed by ordinary restarted GMRES,
    not a block solve or a Krylov method with projection at every iteration.
    """
    if handle._recycle_c is None:
        return initial.clone(), 0, None
    if handle._recycle_c.shape[1] > capacity:
        handle._recycle_u = handle._recycle_u[:, -capacity:].contiguous()
        handle._recycle_c = handle._recycle_c[:, -capacity:].contiguous()
    residual = b-apply_A(handle, initial)
    coordinates = handle._recycle_c.mH @ residual
    corrected = initial+handle._recycle_u @ coordinates
    return corrected, 1, _norm(residual)


def _recycle_update(handle, correction, capacity):
    """Retain independent solution corrections using twice-reorthogonalized A U."""
    direction = correction.clone()
    image = apply_A(handle, direction)
    initial_norm = torch.linalg.vector_norm(image)
    if not bool(torch.isfinite(image).all()) or float(initial_norm) == 0:
        return False
    if handle._recycle_c is not None:
        for _ in range(2):
            coefficients = handle._recycle_c.mH @ image
            image -= handle._recycle_c @ coefficients
            direction -= handle._recycle_u @ coefficients
    norm = torch.linalg.vector_norm(image)
    if bool(norm <= 128*torch.finfo(image.real.dtype).eps*initial_norm):
        return False
    direction, image = direction/norm, image/norm
    if handle._recycle_c is None:
        handle._recycle_u, handle._recycle_c = direction[:, None], image[:, None]
    else:
        handle._recycle_u = torch.cat((handle._recycle_u, direction[:, None]), 1)[:, -capacity:].contiguous()
        handle._recycle_c = torch.cat((handle._recycle_c, image[:, None]), 1)[:, -capacity:].contiguous()
    return True


def _single_gmres(handle, b, x, rule, scale, right_scale, *, action=None):
    # The private action hook reuses the same residual-certified algorithm for
    # benchmark operator prototypes without widening the public dense contract.
    action = (lambda value: apply_A(handle, value)) if action is None else action
    norm_b = _norm(b)
    threshold = max(rule.atol, rule.rtol * norm_b)
    history = []
    matvecs = iterations = restarts = 0
    reason = 'maxiter'

    def residual(value):
        nonlocal matvecs
        r = b - action(value)
        matvecs += 1
        norm = _norm(r)
        history.append({'iteration': iterations, 'absolute': norm,
                        'relative': norm / norm_b if norm_b > 0 else None})
        return r, norm

    r, true_norm = residual(x)
    while iterations < rule.maxiter:
        if not math.isfinite(true_norm):
            reason = 'nonfinite_residual'
            break
        if true_norm <= threshold:
            reason = 'converged'
            break
        z = scale * r
        beta = torch.linalg.vector_norm(z)
        if float(beta) == 0:
            reason = 'breakdown'
            break
        length = min(rule.restart, rule.maxiter - iterations, handle.shape[0])
        basis = torch.zeros((length + 1, b.numel()), dtype=b.dtype, device=b.device)
        h = torch.zeros((length + 1, length), dtype=b.dtype, device=b.device)
        rhs = torch.zeros(length + 1, dtype=b.dtype, device=b.device)
        rhs[0] = beta
        basis[0] = z / beta
        base = x.clone()
        cycle_residual_norm = true_norm
        interrupted = False
        for j in range(length):
            w = scale * action(right_scale*basis[j])
            matvecs += 1
            original_norm = torch.linalg.vector_norm(w)
            # Twice-modified Gram-Schmidt prevents complex128 residual drift.
            for _ in range(2):
                coefficients = basis[:j + 1].conj() @ w
                h[:j + 1, j] += coefficients
                w = w - coefficients @ basis[:j + 1]
            hn = torch.linalg.vector_norm(w)
            h[j + 1, j] = hn
            breakdown = bool(hn <= torch.finfo(b.real.dtype).eps * original_norm)
            if not breakdown:
                basis[j + 1] = w / hn
            iterations += 1
            try:
                y = torch.linalg.lstsq(h[:j + 2, :j + 1], rhs[:j + 2]).solution
            except torch.linalg.LinAlgError:
                reason = 'breakdown'
                interrupted = True
                break
            candidate = base + right_scale*(y @ basis[:j + 1])
            if not bool(torch.isfinite(candidate).all()):
                reason = 'nonfinite_iterate'
                interrupted = True
                break
            x = candidate
            r, true_norm = residual(x)
            if not math.isfinite(true_norm):
                reason = 'nonfinite_residual'
                interrupted = True
                break
            if true_norm <= threshold:
                reason = 'converged'
                interrupted = True
                break
            stagnated = False
            if len(history) > rule.stagnation_window:
                previous = history[-rule.stagnation_window - 1]['absolute']
                stagnated = true_norm >= previous * (1 - rule.stagnation_rtol)
            if breakdown:
                # Finite precision can close the Arnoldi basis before the true
                # residual reaches tolerance. Restart only after measured
                # progress, within the original budget and stagnation rule.
                if (iterations < rule.maxiter and not stagnated and
                        true_norm < cycle_residual_norm * (1 - rule.stagnation_rtol)):
                    break
                reason = 'breakdown'
                interrupted = True
                break
            if stagnated:
                reason = 'stagnation'
                interrupted = True
                break
        if interrupted:
            break
        if iterations < rule.maxiter:
            restarts += 1
    # A last-budget iteration may already have reached tolerance.
    converged = math.isfinite(true_norm) and true_norm <= threshold
    return x, {'converged': converged, 'reason': 'converged' if converged else reason,
               'iterations': iterations, 'matvecs': matvecs, 'restarts': restarts,
               'residual_checks': len(history), 'residual_history': history,
               'absolute_residual': true_norm,
               'relative_residual': true_norm / norm_b if norm_b > 0 else None,
               'threshold': threshold}


def _flexible_gmres(handle, b, x, rule, scale, preconditioner):
    """Restarted flexible GMRES for L A P with a possibly varying right action P.

    Arnoldi vectors live in L-scaled residual coordinates. The preconditioner
    receives ``v / L`` (physical residual units) and returns a trial
    correction z; every z is stored, so x = x0 + Z y holds for any P. The
    unscaled true residual b - A x is recomputed after every step, exactly as
    in ``_single_gmres``; only that residual admits convergence.
    """
    norm_b = _norm(b)
    threshold = max(rule.atol, rule.rtol * norm_b)
    history = []
    matvecs = iterations = restarts = applications = 0
    preconditioner_seconds = 0.0
    reason = 'maxiter'

    def residual(value):
        nonlocal matvecs
        r = b - apply_A(handle, value)
        matvecs += 1
        norm = _norm(r)
        history.append({'iteration': iterations, 'absolute': norm,
                        'relative': norm / norm_b if norm_b > 0 else None})
        return r, norm

    def precondition(v):
        nonlocal applications, preconditioner_seconds
        torch.cuda.synchronize(handle.device)
        start = time.perf_counter()
        z = preconditioner(v / scale)
        torch.cuda.synchronize(handle.device)
        preconditioner_seconds += time.perf_counter() - start
        applications += 1
        if (not isinstance(z, torch.Tensor) or z.shape != v.shape or z.dtype != v.dtype
                or z.device != v.device or z.requires_grad):
            raise ValueError('Right preconditioner must return a detached tensor matching the residual')
        return z

    r, true_norm = residual(x)
    while iterations < rule.maxiter:
        if not math.isfinite(true_norm):
            reason = 'nonfinite_residual'
            break
        if true_norm <= threshold:
            reason = 'converged'
            break
        v0 = scale * r
        beta = torch.linalg.vector_norm(v0)
        if float(beta) == 0:
            reason = 'breakdown'
            break
        length = min(rule.restart, rule.maxiter - iterations, handle.shape[0])
        basis = torch.zeros((length + 1, b.numel()), dtype=b.dtype, device=b.device)
        directions = torch.zeros((length, b.numel()), dtype=b.dtype, device=b.device)
        h = torch.zeros((length + 1, length), dtype=b.dtype, device=b.device)
        rhs = torch.zeros(length + 1, dtype=b.dtype, device=b.device)
        rhs[0] = beta
        basis[0] = v0 / beta
        base = x.clone()
        cycle_residual_norm = true_norm
        interrupted = False
        for j in range(length):
            z = precondition(basis[j])
            if not bool(torch.isfinite(z).all()):
                reason = 'nonfinite_preconditioner'
                interrupted = True
                break
            directions[j] = z
            w = scale * apply_A(handle, z)
            matvecs += 1
            original_norm = torch.linalg.vector_norm(w)
            for _ in range(2):
                coefficients = basis[:j + 1].conj() @ w
                h[:j + 1, j] += coefficients
                w = w - coefficients @ basis[:j + 1]
            hn = torch.linalg.vector_norm(w)
            h[j + 1, j] = hn
            breakdown = bool(hn <= torch.finfo(b.real.dtype).eps * original_norm)
            if not breakdown:
                basis[j + 1] = w / hn
            iterations += 1
            try:
                y = torch.linalg.lstsq(h[:j + 2, :j + 1], rhs[:j + 2]).solution
            except torch.linalg.LinAlgError:
                reason = 'breakdown'
                interrupted = True
                break
            candidate = base + y @ directions[:j + 1]
            if not bool(torch.isfinite(candidate).all()):
                reason = 'nonfinite_iterate'
                interrupted = True
                break
            x = candidate
            r, true_norm = residual(x)
            if not math.isfinite(true_norm):
                reason = 'nonfinite_residual'
                interrupted = True
                break
            if true_norm <= threshold:
                reason = 'converged'
                interrupted = True
                break
            stagnated = False
            if len(history) > rule.stagnation_window:
                previous = history[-rule.stagnation_window - 1]['absolute']
                stagnated = true_norm >= previous * (1 - rule.stagnation_rtol)
            if breakdown:
                if (iterations < rule.maxiter and not stagnated and
                        true_norm < cycle_residual_norm * (1 - rule.stagnation_rtol)):
                    break
                reason = 'breakdown'
                interrupted = True
                break
            if stagnated:
                reason = 'stagnation'
                interrupted = True
                break
        if interrupted:
            break
        if iterations < rule.maxiter:
            restarts += 1
    converged = math.isfinite(true_norm) and true_norm <= threshold
    return x, {'converged': converged, 'reason': 'converged' if converged else reason,
               'iterations': iterations, 'matvecs': matvecs, 'restarts': restarts,
               'residual_checks': len(history), 'residual_history': history,
               'absolute_residual': true_norm,
               'relative_residual': true_norm / norm_b if norm_b > 0 else None,
               'threshold': threshold, 'krylov_method': 'flexible_right_gmres',
               'preconditioner_applications': applications,
               'preconditioner_seconds': preconditioner_seconds}


def _block_rank_factor(values, *, reference_norm=None, max_rank=None):
    """CUDA rank-revealing factorization, including empty and dependent blocks."""
    u, singular, vh = torch.linalg.svd(values, full_matrices=False)
    scale = singular[0] if reference_norm is None else reference_norm
    cutoff = torch.finfo(values.real.dtype).eps * max(values.shape) * scale
    rank = int(torch.count_nonzero(singular > cutoff))
    if max_rank is not None:
        rank = min(rank, max_rank)
    return u[:, :rank], singular[:rank, None]*vh[:rank], rank


def _block_gmres(handle, b, x, rule, scale, right_scale):
    """Variable-block Arnoldi with rank deflation and independent RHS locking.

    The shared Krylov basis belongs to L A R. Small block-Hessenberg least
    squares minimizes each L-scaled residual independently. Only the original
    A x - b certifies convergence. No operator action is silently attributed
    once per RHS: block products and vector-equivalent work are both recorded.
    """
    rhs = b.reshape(-1, b.shape[-1]).mT.contiguous()
    solution = x.reshape(-1, x.shape[-1]).mT.contiguous().clone()
    n, count = rhs.shape
    norms_b = torch.linalg.vector_norm(rhs, dim=0).tolist()
    thresholds = [max(rule.atol, rule.rtol*value) for value in norms_b]
    logs = [dict(converged=False, reason='maxiter', iterations=0, restarts=0,
                 matvecs=0, residual_checks=0, residual_history=[], threshold=threshold,
                 shared_arnoldi_cost_scope='reported_once_in_parent_block_log')
            for threshold in thresholds]
    active = list(range(count))
    block_steps = restarts = products = vector_actions = residual_products = 0
    arnoldi_actions = 0
    rank_history = []
    max_basis = 0

    def residual(indices):
        nonlocal products, vector_actions, residual_products
        values = rhs[:, indices] - apply_A(handle, solution[:, indices].mT).mT
        products += 1
        residual_products += 1
        vector_actions += len(indices)
        norms = torch.linalg.vector_norm(values, dim=0).tolist()
        remaining = []
        for position, column in enumerate(indices):
            log = logs[column]
            norm = norms[position]
            log['absolute_residual'] = norm
            log['relative_residual'] = norm/norms_b[column] if norms_b[column] > 0 else None
            log['residual_history'].append(dict(iteration=block_steps, absolute=norm,
                relative=log['relative_residual']))
            log['residual_checks'] += 1
            log['matvecs'] += 1
            log['iterations'] = block_steps
            if not math.isfinite(norm):
                log['reason'] = 'nonfinite_residual'
            elif norm <= thresholds[column]:
                log.update(converged=True, reason='converged')
            elif len(log['residual_history']) > rule.stagnation_window:
                previous = log['residual_history'][-rule.stagnation_window-1]['absolute']
                if norm >= previous*(1-rule.stagnation_rtol):
                    log['reason'] = 'stagnation'
                else:
                    remaining.append(column)
            else:
                remaining.append(column)
        return values, remaining

    current_residual, active = residual(active)
    # The initial call includes every column; subsequent cycles select from the
    # already measured residual, rather than hiding an extra operator action.
    residual_columns = list(range(count))
    while active and block_steps < rule.maxiter:
        cycle = list(active)
        positions = [residual_columns.index(column) for column in cycle]
        scaled_residual = scale[:, None]*current_residual[:, positions]
        base = solution[:, cycle].clone()
        # Normalize seed columns before detecting rank, so a strong RHS cannot
        # erase an unrelated weak RHS merely through its amplitude.
        seed_norm = torch.linalg.vector_norm(scaled_residual, dim=0)
        seed = scaled_residual / seed_norm.clamp_min(torch.finfo(seed_norm.dtype).tiny)
        if not bool(torch.isfinite(seed).all()):
            for column in active: logs[column]['reason'] = 'nonfinite_residual'
            break
        try:
            basis, _, initial_rank = _block_rank_factor(seed)
        except torch.linalg.LinAlgError:
            for column in active: logs[column]['reason'] = 'factorization_failure'
            break
        rank_history.append(dict(iteration=block_steps, phase='restart_seed',
                                 input_columns=len(cycle), retained_rank=initial_rank))
        if initial_rank == 0:
            for column in active: logs[column]['reason'] = 'breakdown'
            break
        projected_rhs = basis.mH @ scaled_residual
        hessenberg = torch.zeros((initial_rank, 0), device=b.device, dtype=b.dtype)
        processed = 0
        width = initial_rank
        broke_down = False
        for _ in range(min(rule.restart, rule.maxiter-block_steps)):
            block = basis[:, processed:processed+width]
            image = scale[:, None]*apply_A(handle, (right_scale[:, None]*block).mT).mT
            products += 1
            arnoldi_actions += width
            vector_actions += width
            if not bool(torch.isfinite(image).all()):
                for column in active: logs[column]['reason'] = 'nonfinite_residual'
                broke_down = True
                break
            image_norm = torch.linalg.matrix_norm(image, ord='fro')
            coefficients = torch.zeros((basis.shape[1], width), device=b.device, dtype=b.dtype)
            for _pass in range(2):
                projection = basis.mH @ image
                coefficients += projection
                image -= basis @ projection
            if not bool(torch.isfinite(image).all()):
                for column in active: logs[column]['reason'] = 'nonfinite_residual'
                broke_down = True
                break
            try:
                following, coupling, next_rank = _block_rank_factor(image,
                    reference_norm=image_norm, max_rank=n-basis.shape[1])
            except torch.linalg.LinAlgError:
                for column in active: logs[column]['reason'] = 'factorization_failure'
                broke_down = True
                break
            rows = basis.shape[1] + next_rank
            expanded = torch.zeros((rows, processed+width), device=b.device, dtype=b.dtype)
            expanded[:hessenberg.shape[0], :processed] = hessenberg
            expanded[:basis.shape[1], processed:] = coefficients
            expanded[basis.shape[1]:, processed:] = coupling
            hessenberg = expanded
            projected_rhs = torch.cat((projected_rhs,
                torch.zeros((next_rank, len(cycle)), device=b.device, dtype=b.dtype)))
            processed += width
            if next_rank:
                basis = torch.cat((basis, following), dim=1)
            max_basis = max(max_basis, basis.shape[1])
            # CUDA gels assumes full rank. SVD explicitly handles singular or
            # exhausted block Hessenberg systems without numerical CPU fallback.
            try:
                u, singular, vh = torch.linalg.svd(hessenberg, full_matrices=False)
            except torch.linalg.LinAlgError:
                for column in active: logs[column]['reason'] = 'factorization_failure'
                broke_down = True
                break
            cutoff = torch.finfo(singular.dtype).eps*max(hessenberg.shape)*singular[0]
            rank = int(torch.count_nonzero(singular > cutoff))
            y = vh[:rank].mH @ ((u[:, :rank].mH @ projected_rhs)/singular[:rank, None])
            candidate = base + right_scale[:, None]*(basis[:, :processed] @ y)
            local = [cycle.index(column) for column in active]
            solution[:, active] = candidate[:, local]
            block_steps += 1
            rank_history.append(dict(iteration=block_steps, phase='arnoldi',
                input_columns=width, retained_rank=next_rank, least_squares_rank=rank,
                least_squares_columns=processed))
            residual_columns = list(active)
            current_residual, active = residual(residual_columns)
            if not active:
                break
            if next_rank == 0:
                # A full-rank exhausted Krylov image can leave roundoff after
                # cancelling a large x0. Restart from the measured residual
                # within the original budget; inconsistent singular systems
                # still fail instead of treating a small Hessenberg residual
                # as a physical convergence certificate.
                if rank < processed:
                    for column in active: logs[column]['reason'] = 'breakdown'
                    broke_down = True
                break
            width = next_rank
        if broke_down or not active or block_steps >= rule.maxiter:
            break
        restarts += 1
        for column in active: logs[column]['restarts'] += 1
    stats = dict(iterations=block_steps, restarts=restarts, matvecs=vector_actions,
        block_iterations=block_steps, block_operator_products=products,
        block_residual_products=residual_products, arnoldi_vector_actions=arnoldi_actions,
        residual_vector_actions=sum(log['residual_checks'] for log in logs),
        residual_checks=sum(log['residual_checks'] for log in logs),
        block_rank_history=rank_history, maximum_block_basis_columns=max_basis,
        iteration_budget_scope='block Arnoldi steps; restart is block steps per cycle',
        matvec_scope='vector-equivalent columns across all Arnoldi and true-residual products',
        block_algorithm='rank_deflated_restarted_block_gmres',
        block_cost_scope='primary block attempt; direct fallback costs are separate when present')
    return solution.mT.reshape(b.shape), logs, stats


@torch.no_grad()
@_matrix_operation
def _correct(handle, b: torch.Tensor, x0: torch.Tensor,
            stopping_rule: StoppingRule | None = None, *, right_preconditioner=None):
    """Solve from any finite initial guess and report the true residual.

    GMRES batch entries share setup; block_gmres shares a block Krylov space. Numerical failures
    are returned as explicit unsuccessful logs; invalid input raises ValueError.
    Iterative solves are not differentiable; ``residual_loss`` is the native
    differentiable fixed-operator interface. ``recycle_size > 0`` retains a
    bounded solution-correction subspace across RHS and calls for this matrix.
    Its true-residual minimizing initial augmentation precedes ordinary GMRES;
    every projection and update cost is included in the returned log.
    ``row_column_norm`` applies L A R during Arnoldi but checks A x - b in
    physical coordinates. ``fallback='direct'`` explicitly opts into an LU
    fallback after failed GMRES; both attempts and their costs remain logged.
    """
    rule = stopping_rule or StoppingRule()
    if right_preconditioner is not None and (not callable(right_preconditioner) or rule.method != 'gmres'
                                             or rule.recycle_size or rule.fallback != 'none'):
        raise ValueError('A right preconditioner requires callable action, method=gmres, no recycling and no fallback')
    if (rule.method == 'direct' or rule.fallback == 'direct' or rule.direct_storage != 'resident') and not isinstance(handle, DenseOperator):
        raise ValueError('Direct/LU requires a dense operator with supports_direct')
    _vector(handle, b)
    _vector(handle, x0)
    if b.shape != x0.shape:
        raise ValueError('b and x0 must have identical shapes')
    if not bool(torch.isfinite(b).all()) or not bool(torch.isfinite(x0).all()):
        raise ValueError('b and x0 must be finite')
    if b.numel() == 0:
        raise ValueError('RHS batch must be nonempty')
    torch.cuda.synchronize(handle.device)
    start = time.perf_counter()
    memory_start = torch.cuda.memory_allocated(handle.device)
    handle.refresh()
    cache_reused = handle._lu is not None if rule.method == 'direct' else False
    setup_failure = None
    if rule.method == 'direct':
        if rule.direct_storage == 'resident' and handle._lu is None:
            lu, pivots, info = torch.linalg.lu_factor_ex(handle.matrix)
            if bool(info != 0):
                setup_failure = 'singular_matrix'
            else:
                handle._lu = (lu, pivots)
    if rule.method != 'direct':
        scale, right_scale, cache_reused = handle.preconditioner_scales(rule.preconditioner)
        for diagonal in (scale, right_scale):
            if (diagonal.shape != (handle.shape[0],) or diagonal.device != handle.device or diagonal.dtype != b.real.dtype
                    or not bool(torch.isfinite(diagonal).all() & (diagonal > 0).all())):
                raise ValueError('Preconditioner scales must be finite positive matching CUDA real vectors')
    else:
        scale = right_scale = torch.ones(handle.shape[0], device=handle.device, dtype=b.real.dtype)
    torch.cuda.synchronize(handle.device)
    setup_end = time.perf_counter()
    logs = []
    solutions = []
    block_stats = {}
    if rule.method == 'block_gmres':
        block_solution, logs, block_stats = _block_gmres(handle, b, x0, rule, scale, right_scale)
        solutions = list(block_solution.reshape(-1, b.shape[-1]))
    if rule.method == 'direct' and setup_failure is None:
        if rule.direct_storage == 'sealed_host_restore_v1':
            result, setup_failure = _one_shot_direct(handle, b, x0)
        else:
            result = torch.linalg.lu_solve(*handle._lu, b.reshape(-1, b.shape[-1]).mT).mT.reshape(b.shape)
    else:
        result = x0.clone()
    entries = zip(b.reshape(-1, b.shape[-1]), x0.reshape(-1, b.shape[-1]), result.reshape(-1, b.shape[-1]))
    for rhs, initial, direct in (() if rule.method == 'block_gmres' else entries):
        if rule.method == 'gmres':
            augmented = initial.clone()
            recycle_dimension_before = 0
            projection_matvecs = 0
            residual_before = None
            projection_seconds = update_seconds = 0.0
            if rule.recycle_size:
                torch.cuda.synchronize(handle.device)
                projection_start = time.perf_counter()
                augmented, projection_matvecs, residual_before = _recycle_initial_correction(handle, rhs, initial, min(rule.recycle_size, handle.shape[0]))
                recycle_dimension_before = 0 if handle._recycle_c is None else handle._recycle_c.shape[1]
                torch.cuda.synchronize(handle.device)
                projection_seconds = time.perf_counter()-projection_start
            if right_preconditioner is None:
                solution, log = _single_gmres(handle, rhs, augmented, rule, scale, right_scale)
            else:
                solution, log = _flexible_gmres(handle, rhs, augmented, rule, scale, right_preconditioner)
            update_matvecs = 0
            updated = False
            if rule.recycle_size and log['converged']:
                torch.cuda.synchronize(handle.device)
                update_start = time.perf_counter()
                updated = _recycle_update(handle, solution-initial, min(rule.recycle_size, handle.shape[0]))
                update_matvecs = 1
                torch.cuda.synchronize(handle.device)
                update_seconds = time.perf_counter()-update_start
            log.update({'recycle_dimension_before': recycle_dimension_before,
                        'recycle_dimension_after': 0 if handle._recycle_c is None else handle._recycle_c.shape[1],
                        'recycle_projection_matvecs': projection_matvecs,
                        'recycle_update_matvecs': update_matvecs,
                        'recycle_projection_seconds': projection_seconds,
                        'recycle_update_seconds': update_seconds,
                        'recycle_direction_added': updated,
                        'recycle_reorthogonalization_passes': 2 if update_matvecs and recycle_dimension_before else 0,
                        'recycle_residual_before_projection': residual_before})
            log['matvecs'] += projection_matvecs+update_matvecs
            if projection_matvecs:
                log['residual_checks'] += 1
                norm_b = _norm(rhs)
                log['residual_history'].insert(0, {'iteration': 0, 'phase': 'before_recycle_projection',
                    'absolute': residual_before, 'relative': residual_before/norm_b if norm_b > 0 else None})
        else:
            solution = direct
            norm_b = _norm(rhs)
            norm_r = _norm(rhs - apply_A(handle, solution))
            threshold = max(rule.atol, rule.rtol * norm_b)
            converged = setup_failure is None and math.isfinite(norm_r) and norm_r <= threshold
            relative = norm_r / norm_b if norm_b > 0 else None
            log = {'converged': converged,
                   'reason': 'converged' if converged else (setup_failure or 'residual_above_tolerance'),
                   'iterations': 0, 'matvecs': 1, 'restarts': 0, 'residual_checks': 1,
                   'residual_history': [{'iteration': 0, 'absolute': norm_r, 'relative': relative}],
                   'absolute_residual': norm_r, 'relative_residual': relative, 'threshold': threshold}
        logs.append(log)
        solutions.append(solution)
    torch.cuda.synchronize(handle.device)
    end = time.perf_counter()
    combined = dict(logs[0]) if len(logs) == 1 else {
        'converged': all(log['converged'] for log in logs),
        'reason': 'converged' if all(log['converged'] for log in logs) else 'batch_failure',
        'rhs_logs': logs,
        **{key: sum(log[key] for log in logs)
           for key in ('iterations', 'matvecs', 'restarts', 'residual_checks')}}
    combined.update({'method': rule.method, 'preconditioner': rule.preconditioner if rule.method != 'direct' else 'none',
                     'cache_reused': cache_reused, 'setup_seconds': setup_end - start,
                     'solve_seconds': end - setup_end, 'total_seconds': end - start,
                     'peak_memory_bytes': torch.cuda.max_memory_allocated(handle.device),
                     'allocated_delta_bytes': torch.cuda.memory_allocated(handle.device) - memory_start,
                     'peak_memory_scope': 'device_allocator_since_last_external_reset',
                     'device': str(handle.device), 'dtype': str(handle.dtype),
                     'rhs_count': len(logs), 'fallback_used': False,
                     'recycle_capacity': rule.recycle_size,
                     'recycle_cache_reused': any(log.get('recycle_dimension_before', 0) > 0 for log in logs),
                     'recycle_storage_bytes': 0 if handle._recycle_c is None else (handle._recycle_u.numel()+handle._recycle_c.numel())*b.element_size(),
                     'recycle_algorithm': 'initial_true_residual_augmentation_then_restarted_gmres' if rule.recycle_size else None})
    combined.update(block_stats)
    if right_preconditioner is not None:
        combined['krylov_method'] = 'flexible_right_gmres'
        for key in ('preconditioner_applications', 'preconditioner_seconds'):
            combined[key] = sum(log[key] for log in logs)
    if rule.direct_storage != 'resident':
        combined['direct_storage'] = handle.direct_storage_report
    for key in ('recycle_projection_matvecs', 'recycle_update_matvecs', 'recycle_projection_seconds', 'recycle_update_seconds', 'recycle_reorthogonalization_passes'):
        combined[key] = sum(log.get(key, 0) for log in logs)
    solution = torch.stack(solutions).reshape(b.shape)
    if rule.fallback == 'direct' and not combined['converged']:
        # Keep the unsuccessful iteration certificate and all work. Re-solve
        # every RHS in this batch to provide one unambiguous final certificate.
        direct_rule = StoppingRule(method='direct', rtol=rule.rtol, atol=rule.atol)
        fallback_solution, fallback_log = correct(handle, b, torch.zeros_like(b), direct_rule)
        attempted = dict(combined)
        combined.update({'converged': fallback_log['converged'],
                         'reason': 'fallback_converged' if fallback_log['converged'] else 'fallback_failed',
                         'fallback_used': True, 'fallback_method': 'direct',
                         'gmres_attempt': attempted, 'fallback_log': fallback_log,
                         'final_residual_source': 'fallback_log'})
        for key in ('matvecs', 'residual_checks', 'setup_seconds', 'solve_seconds', 'total_seconds'):
            combined[key] = attempted[key]+fallback_log[key]
        for key in ('absolute_residual', 'relative_residual', 'threshold', 'rhs_logs'):
            if key in fallback_log:
                combined[key] = fallback_log[key]
        if 'residual_history' in fallback_log:
            combined['residual_history'] = [dict(entry, solver_phase='gmres') for entry in attempted['residual_history']]+[dict(entry, solver_phase='direct_fallback') for entry in fallback_log['residual_history']]
        combined['peak_memory_bytes'] = max(attempted['peak_memory_bytes'], fallback_log['peak_memory_bytes'])
        combined['allocated_delta_bytes'] = attempted['allocated_delta_bytes']+fallback_log['allocated_delta_bytes']
        torch.cuda.synchronize(handle.device)
        elapsed = time.perf_counter()-start
        combined['fallback_orchestration_seconds'] = max(0.0, elapsed-combined['total_seconds'])
        combined['total_seconds'] = elapsed
        solution = fallback_solution
    return solution, combined


def correct(handle, b: torch.Tensor, x0: torch.Tensor, stopping_rule: StoppingRule | None = None,
            *, right_preconditioner=None):
    """Residual-certified solve using operator actions; only LU requires dense storage.

    ``right_preconditioner`` (callable residual -> correction) selects flexible
    right-preconditioned GMRES; omitted, the default algorithms are unchanged.
    """
    view = handle if isinstance(handle, (DenseOperator, KrylovView)) else KrylovView(handle)
    if right_preconditioner is None:
        return _correct(view, b, x0, stopping_rule)
    return _correct(view, b, x0, stopping_rule, right_preconditioner=right_preconditioner)


def solve_reference(handle: DenseOperator, b: torch.Tensor,
                    stopping_rule: StoppingRule | None = None, *, right_preconditioner=None):
    """Reference solve with a zero initial guess; dense LU is the default."""
    rule = stopping_rule or StoppingRule(method='direct')
    return correct(handle, b, torch.zeros_like(b), rule, right_preconditioner=right_preconditioner)
