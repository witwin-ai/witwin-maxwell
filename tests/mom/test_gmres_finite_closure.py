"""True residual certification across finite precision Arnoldi closure."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from witwin.maxwell.mom import linalg

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def original_scalar():
    names=('GSP_GMRES_ORIGINAL_PRIMARY','GSP_GMRES_ORIGINAL_PRIMARY_SHA256')
    if all(os.environ.get(name) is None for name in names):
        pytest.skip('Frozen original scalar primary required for failure comparison')
    assert all(os.environ.get(name) for name in names)
    source=Path(os.environ[names[0]]).read_bytes()
    assert hashlib.sha256(source).hexdigest()==os.environ[names[1]]
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_single_gmres')
    space=dict(vars(linalg))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-original-scalar','exec'),space)
    return space['_single_gmres']


def problem(dtype):
    a=torch.tensor([[1,2,4,8],[0,3,6,12],[0,0,5,10],[0,0,0,7]],device='cuda',dtype=dtype)
    exact=torch.tensor([1+1j,-2+2j,3-3j,4+4j],device='cuda',dtype=dtype)
    b=a@exact
    return a,b,exact,SimpleNamespace(shape=a.shape)


def solve(function,a,b,handle,rule):
    ones=torch.ones_like(b.real)
    return function(handle,b,torch.zeros_like(b),rule,ones,ones,action=lambda x:a@x)


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_new_closed_basis_residual_restart_still_requires_original_tolerance(dtype,tmp_path,monkeypatch):
    a,b,exact,handle=problem(dtype)
    rtol=1e-8 if dtype==torch.complex64 else 1e-16
    rule=linalg.StoppingRule(rtol=rtol,atol=0.,maxiter=24,restart=4,preconditioner='none',fallback='none')
    def forbidden(*args,**kwargs):raise AssertionError('Residual restart used direct solve')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    old,old_log=solve(original_scalar(),a,b,handle,rule)
    new,new_log=solve(linalg._single_gmres,a,b,handle,rule)
    state=tmp_path/'native-state.pt'
    torch.save(dict(matrix=a,rhs=b,exact=exact,old=old,new=new,old_log=old_log,new_log=new_log),state)
    measured=float(torch.linalg.vector_norm(b-a@new)/torch.linalg.vector_norm(b))
    (tmp_path/'closure.json').write_text(json.dumps(dict(dtype=str(dtype),old=old_log,new=new_log,
        actual_relative_residual=measured,rtol=rtol,atol=0.,original_primary_sha256=os.environ['GSP_GMRES_ORIGINAL_PRIMARY_SHA256'],
        actual_native_state=str(state),actual_native_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        production_qualification=False),indent=2)+'\n')
    assert not old_log['converged'] and old_log['reason']=='breakdown',old_log
    assert new_log['converged'] and new_log['restarts']>0,new_log
    assert measured<=rtol and new_log['iterations']<=rule.maxiter
    assert new_log['threshold']==rtol*float(torch.linalg.vector_norm(b))


def test_new_closed_basis_cannot_extend_exhausted_iteration_budget(tmp_path):
    a,b,_,handle=problem(torch.complex128)
    rule=linalg.StoppingRule(rtol=1e-16,atol=0.,maxiter=4,restart=4,preconditioner='none',fallback='none')
    _,log=solve(linalg._single_gmres,a,b,handle,rule)
    (tmp_path/'budget.json').write_text(json.dumps(log,indent=2)+'\n')
    assert not log['converged'] and log['reason']=='breakdown'
    assert log['iterations']==4 and log['restarts']==0
