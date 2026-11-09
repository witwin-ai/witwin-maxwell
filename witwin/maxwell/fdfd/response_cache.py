"""Checked, portable persistence of prepared GPU background responses.

Only explicit file I/O copies tensors to the host. Loaded arrays return to CUDA;
no factorization or numerical solve is performed on the CPU.
"""
from __future__ import annotations
import hashlib
import math
import os
from pathlib import Path
import tempfile
import torch

_ARRAYS = ('_green','_observation_green','_rhs','_background','_u0','_y0')


def _digest(values):
    digest = hashlib.sha256()
    for name,value in sorted(values.items()):
        tensor = value.detach().resolve_conj().contiguous().cpu()
        digest.update(repr((name,tuple(tensor.shape),str(tensor.dtype))).encode())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _fingerprint(response):
    solver = response.solver
    matrix = solver.A_matrix
    scale = torch.from_dlpack(solver._ensure_symmetrization_scale())
    rhs = torch.from_dlpack(solver._build_source_vector_yee())/scale
    tensors = {'matrix_data':torch.from_dlpack(matrix.data),
               'matrix_indices':torch.from_dlpack(matrix.indices),
               'matrix_indptr':torch.from_dlpack(matrix.indptr),
               'scale':scale,'rhs':rhs,'design':response._indices,
               'projection_indices':response._projection.indices(),
               'projection_values':response._projection.values(),
               'epsilon':response._base_eps,'permeability':response._base_mu,
               'frequency':torch.tensor([response.frequency],dtype=torch.float64)}
    tensors.update({axis:getattr(response.scene,axis) for axis in 'xyz'})
    return _digest(tensors)


def save_cache(response, path):
    response._validate()
    arrays = {name:getattr(response,name).detach().resolve_conj().contiguous().cpu() for name in _ARRAYS}
    payload = {'format':'maxwell-fdfd-local-response','version':1,'fingerprint':_fingerprint(response),
               'arrays':arrays,'checksum':_digest(arrays),'precompute_residual':response.precompute_residual,
               'precompute_seconds':response.original_precompute_seconds}
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as output:
        temporary = Path(output.name)
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_cache(response, path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('format')!='maxwell-fdfd-local-response' or payload.get('version')!=1:
        raise ValueError('Unsupported local-response cache format.')
    if payload.get('fingerprint') != _fingerprint(response):
        raise ValueError('Local-response cache does not match the current operator, source, design or observations.')
    residual = payload.get('precompute_residual',math.inf)
    if not isinstance(residual,(int,float)) or not math.isfinite(residual) or not 0 <= residual <= response.config.tolerance:
        raise ValueError('Cached response does not satisfy the requested precompute tolerance.')
    n,N,q = response._indices.numel(),response.scene.N_vector_total,len(response._names)
    shapes = ((n,n),(q,n),(N,),(N,),(n,),(q,))
    arrays = payload.get('arrays',{})
    if not isinstance(arrays,dict) or set(arrays)!=set(_ARRAYS) or any(not isinstance(arrays[name],torch.Tensor)
            or tuple(arrays[name].shape)!=shape or arrays[name].dtype!=response.dtype
            for name,shape in zip(_ARRAYS,shapes)):
        raise ValueError('Cached response tensor shapes or precision do not match.')
    if payload.get('checksum') != _digest(arrays):
        raise ValueError('Local-response cache checksum mismatch.')
    for name in _ARRAYS:
        value = arrays[name].to(response.device)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Local-response cache contains nonfinite values.')
        setattr(response,name,value)
    response.precompute_residual = float(residual)
    response.original_precompute_seconds = float(payload['precompute_seconds'])


def frequency_cache_path(directory, frequency):
    return Path(directory)/('frequency-'+float(frequency).hex()+'.pt')
