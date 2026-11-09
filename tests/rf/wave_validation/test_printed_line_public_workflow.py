"""Maintained public selected-Cu workflow contracts; no historical raw files."""
from dataclasses import replace
from pathlib import Path
import ast,sys
import numpy as np,pytest,torch
from scripts.dev.fdtd.printed_line_public_execution import (
    column_case,source_basis,actual_clock,build_public_column,run_public_column)
from scripts.dev.fdtd.printed_line_public_vi import select_frequencies,query_kwargs


def compact_case():
    return replace(column_case(0),axial_half_length=.06,separation=.06,
        padding=.01,positions=(-.015,0.,.015),duration=4e-9,checkpoint_time=2e-9)

@pytest.fixture(scope='module')
def contour_helpers():
    # The printed-line builders load their contour helpers from this evidence directory.
    if not (Path(__file__).resolve().parents[3]/'docs/assessments/printed-lines-2026-10-22').exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22')


@pytest.fixture(scope='module',params=('cpw','differential_pair'))
def geometry(request,contour_helpers):
    before=torch.cuda.is_initialized()
    context=build_public_column(request.param,compact_case());context['CPU_factory_CUDA_state_before']=before
    return request.param,context


def test_fixed_physical_geometry_owner_rows_and_regions(geometry):
    kind,context=geometry;plan=context['plan']
    assert len(plan.queries)==6 and len(context['oracle_indices'])==12
    assert len(context['groups'])==(2 if kind=='cpw'else 4)
    assert len(plan.rows)==(3 if kind=='cpw'else 6)
    assert all(len(row)==2 for row in plan.native_rows.values())
    assert {q.region_id for q in plan.queries.values()}==({'interior'}if kind=='cpw'else {'exterior'})
    assert torch.cuda.is_initialized()==context['CPU_factory_CUDA_state_before']


def test_public_requests_explicit_owner_side_and_frequency_axes(geometry):
    _,context=geometry;plan=context['plan'];freq=tuple(reversed(plan.frequencies))
    assert select_frequencies(plan,freq)==freq
    for term in plan.queries.values():
        a=query_kwargs(term,freq);b=query_kwargs(term,freq,side='positive',representation='constitutive')
        assert a['owner']==term.owner and a['region_id']==term.region_id
        assert a['trace_side']=='negative'and a['representation']=='published_native'
        assert b['trace_side']=='positive'and b['representation']=='constitutive'


def test_runtime_closure_does_not_import_private_retained_field_builder(geometry,monkeypatch):
    from types import ModuleType
    kind,_=geometry
    private=('scripts.dev.fdtd.negative_trace_query','scripts.dev.fdtd.printed_line_negative_vi')
    # Unrelated earlier imports do not become runtime dependencies or disable
    # the public workflow. Empty sentinels fail if this builder reads them.
    for name in private:
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    before=set(sys.modules)
    context=build_public_column(kind,compact_case())
    assert len(context['plan'].queries)==6
    assert not any(name in set(sys.modules)-before for name in private)
    for module in ('printed_line_public_vi','printed_line_public_vi_checks','printed_line_public_execution'):
        path=Path(__file__).resolve().parents[3]/'scripts/dev/fdtd'/f'{module}.py'
        imports=[n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,(ast.Import,ast.ImportFrom))]
        for node in imports:
            names=[node.module]if isinstance(node,ast.ImportFrom)else[a.name for a in node.names]
            assert not any(name in private for name in names)


@pytest.mark.parametrize('length',[0,1])
def test_original_full64_case_and_nonzero_source_basis(length):
    case=column_case(length);L=(.4,.48)[length]
    assert case.separation==L and case.positions==(-L/2+.025,0.,L/2-.025)
    assert case.duration==64e-9 and case.checkpoint_time==32e-9
    assert case.dx==.00125 and case.longitudinal_dl==.0025
    for n in(2,4):
        B=source_basis(n);assert np.linalg.matrix_rank(B)==n and np.all(B!=0)

@pytest.mark.parametrize('dt',[2.7797007933179227e-12,2.5017308e-12])
def test_actual_clock_controls_complete_physical_horizon(dt):
    import math
    c=actual_clock(dt,column_case(0));assert c['total_steps']==math.ceil(64e-9/dt)
    assert c['checkpoint_step']==math.ceil(32e-9/dt)

@pytest.mark.parametrize('failure',['kind','metal','reference_grid','trainable_sigma'])
def test_fixed_detached_workflow_rejects_unadmitted_templates(failure):
    case=compact_case();kind='cpw'
    if failure=='kind':kind='stripline'
    elif failure=='metal':case=replace(case,metal='pec')
    elif failure=='reference_grid':case=replace(case,probe_reference_dx=.00125)
    else:case=replace(case,sigma=torch.tensor(.01,requires_grad=True))
    with pytest.raises(NotImplementedError if failure=='trainable_sigma'else ValueError):build_public_column(kind,case)


def test_existing_pilot_CLI_optin_is_one_column_and_defaults_unchanged():
    import scripts.dev.fdtd.printed_line_loss_execution as execution
    tree=ast.parse(Path(execution.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    assert any(isinstance(n,ast.Constant)and n.value=='--public-line'for n in ast.walk(main))
    # The existing stripline two-drive branch remains separate after opt-in.
    assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='reduce_pair'for n in ast.walk(main))
