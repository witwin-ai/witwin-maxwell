"""Independent operator-layout controls for actual PEC support comparisons."""
import numpy as np
import pytest
from scripts.dev.fdtd.pcb_material_operator_audit import compare_operator


def fixture():
    common=(7,8,9);native={};reference={'vv':np.ones((3,*common)),'vi':np.ones((3,*common))}
    for component,prefix in enumerate(('cex','cey','cez')):
        shape=list(common);shape[component]-=1
        native[prefix+'_decay']=np.ones(shape);native[prefix+'_curl']=np.ones(shape)
        for key in ('vv','vi'):reference[key][component,3,4,5]=0
        for suffix in ('_decay','_curl'):native[prefix+suffix][3,4,5]=0
    return native,reference


def test_true_component_stagger_and_pml_exclusion_keep_three_edge_families():
    native,reference=fixture()
    for key in ('vv','vi'):reference[key][0,0,4,5]=0
    rows=compare_operator(native,reference,pml_layers=1)
    assert rows['identical_pec_support']
    assert [row['native_constrained_edges'] for row in rows['components']]==[1,1,1]
    assert not compare_operator(native,reference,pml_layers=0)['identical_pec_support']


@pytest.mark.parametrize('component',range(3))
def test_single_missing_constraint_is_reported_without_scalar_or_coordinate_fit(component):
    native,reference=fixture();reference['vi'][component,3,4,5]=1
    rows=compare_operator(native,reference,pml_layers=1)['components']
    assert rows[component]['different_indices']==[[3,4,5]]
    assert sum(row['different_edges'] for row in rows)==1


def test_both_operator_coefficients_are_required_to_identify_a_hard_short():
    native,reference=fixture();native['cex_curl'][3,4,5]=1
    assert compare_operator(native,reference,pml_layers=1)['components'][0]['different_edges']==1


@pytest.mark.parametrize('pml',(-1,True,1.5,4))
def test_invalid_or_empty_pml_regions_are_rejected(pml):
    with pytest.raises(ValueError):compare_operator(*fixture(),pml_layers=pml)


def test_wrong_stagger_is_rejected_before_comparing_any_field():
    native,reference=fixture();native['cex_decay']=native['cex_decay'][:-1];native['cex_curl']=native['cex_curl'][:-1]
    with pytest.raises(ValueError,match='stagger'):compare_operator(native,reference,pml_layers=1)
