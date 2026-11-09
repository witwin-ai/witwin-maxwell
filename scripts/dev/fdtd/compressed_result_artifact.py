"""Lossless trusted Result artifacts without a second on-disk raw snapshot.

This Linux research workflow uses anonymous RAM for the unchanged Result.save
payload. It changes neither numerical observations nor the public save format.
"""
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

import witwin.maxwell as mw

BLOCK = 8 * 1024**2
SCHEMA = 'lossless-compressed-result-artifact/1'


def _memory_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    group = Path('/sys/fs/cgroup')
    if (group / 'memory.max').exists():
        limit = (group / 'memory.max').read_text().strip()
        if limit != 'max':
            available = min(available, int(limit) - int((group / 'memory.current').read_text()))
    return available


def check_artifact_resources(path, *, max_raw_bytes, reserve_bytes=16 * 1024**3):
    """Check the uncompressed upper bound before starting an expensive solve."""
    if type(max_raw_bytes) is not int or max_raw_bytes <= 0:
        raise ValueError('A positive explicit raw snapshot byte budget is required.')
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('Disk reserve must be a nonnegative byte count.')
    path = Path(path)
    if path.suffix != '.gz':
        raise ValueError('Compressed snapshots use an explicit .gz filename.')
    candidates = (path, path.with_name(path.name + '.partial'), path.with_name(path.name + '.json'))
    if any(p.exists() or p.is_symlink() for p in candidates):
        raise FileExistsError('Preserve existing complete or partial artifact outputs.')
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    worst_compressed_bytes = max_raw_bytes + max_raw_bytes // 100 + 1024**2
    if shutil.disk_usage(parent).free < reserve_bytes + worst_compressed_bytes:
        raise RuntimeError('Actual free disk is below the raw-size bound plus reserve.')
    if _memory_available() < 3 * max_raw_bytes + 1024**3:
        raise RuntimeError('Insufficient available RAM for the declared snapshot budget.')
    return dict(max_raw_bytes=max_raw_bytes, worst_compressed_bytes=worst_compressed_bytes,
                reserve_bytes=reserve_bytes, favorable_compression_ratio_assumed=False)


@contextmanager
def _anonymous_snapshot(maximum_bytes):
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError('A positive explicit raw snapshot byte budget is required.')
    if _memory_available() < 3 * maximum_bytes + 1024**3:
        raise RuntimeError('Insufficient available RAM for the declared snapshot budget.')
    library = ctypes.CDLL(None, use_errno=True)
    create = library.memfd_create
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    descriptor = create(b'pcb-result-artifact', 1)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), 'Cannot allocate the anonymous result snapshot.')
    try:
        yield Path('/proc/self/fd') / str(descriptor)
    finally:
        os.close(descriptor)


def _checksum(path, *, compressed=False):
    digest, count = hashlib.sha256(), 0
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def save_compressed_result(result, path, *, max_raw_bytes, reserve_bytes=16 * 1024**3):
    """Persist the exact public snapshot, verifying a full gzip round trip."""
    if not isinstance(result, mw.Result):
        raise TypeError('Expected an actual Result snapshot provider.')
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    receipt = path.with_name(path.name + '.json')
    # Repeat the pre-solve admission against actual resources at persistence.
    check_artifact_resources(path, max_raw_bytes=max_raw_bytes, reserve_bytes=reserve_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _anonymous_snapshot(max_raw_bytes) as raw:
        result.save(raw)
        raw_sha, raw_bytes = _checksum(raw)
        if raw_bytes > max_raw_bytes:
            raise RuntimeError('Actual public snapshot exceeded its declared raw byte budget.')
        with raw.open('rb') as source, partial.open('xb') as destination:
            with gzip.GzipFile(filename='', mode='wb', fileobj=destination, compresslevel=1, mtime=0) as archive:
                shutil.copyfileobj(source, archive, BLOCK)
            destination.flush()
            os.fsync(destination.fileno())
        if _checksum(partial, compressed=True) != (raw_sha, raw_bytes):
            raise RuntimeError('Compressed artifact failed exact serialized-byte verification.')
    compressed_sha, compressed_bytes = _checksum(partial)
    # A hard-link publish refuses a preexisting destination atomically.
    os.link(partial, path)
    partial.unlink()
    record = dict(schema=SCHEMA, raw_sha256=raw_sha, raw_bytes=raw_bytes,
        compressed_sha256=compressed_sha, compressed_bytes=compressed_bytes,
        max_raw_bytes=max_raw_bytes, reserve_bytes=reserve_bytes,
        raw_snapshot_storage='anonymous RAM; no uncompressed disk artifact',
        exact_serialized_round_trip_verified=True, numerical_payload_changed=False)
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return record


def load_compressed_result(path, *, scene, map_location='cpu'):
    """Verify and restore the original public snapshot from trusted local data."""
    path = Path(path)
    record = json.loads(path.with_name(path.name + '.json').read_text())
    if (record.get('schema') != SCHEMA or record.get('numerical_payload_changed') is not False
        or record.get('exact_serialized_round_trip_verified') is not True
        or _checksum(path) != (record['compressed_sha256'], record['compressed_bytes'])):
        raise ValueError('Compressed artifact or verification receipt differs.')
    with _anonymous_snapshot(record['raw_bytes']) as raw:
        digest, count = hashlib.sha256(), 0
        with gzip.open(path, 'rb') as source, raw.open('wb') as destination:
            while block := source.read(BLOCK):
                count += len(block)
                if count > record['raw_bytes']:
                    raise ValueError('Decompressed data exceeded its verified size.')
                digest.update(block)
                destination.write(block)
        if (digest.hexdigest(), count) != (record['raw_sha256'], record['raw_bytes']):
            raise ValueError('Decompressed public snapshot differs from its verified bytes.')
        return mw.Result.load(raw, scene=scene, map_location=map_location)
