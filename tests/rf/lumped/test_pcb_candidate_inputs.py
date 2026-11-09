"""Physical input transformations for explicitly unqualified antenna candidates."""
import copy
import json
import numpy as np
import pytest
import witwin.maxwell as mw
from witwin.maxwell.adapters.pcb import import_pcb_scene
from scripts.dev.fdtd.pcb_application_campaign import FIXTURES, NAMES, direct_scene
from scripts.dev.fdtd.pcb_candidate_inputs import candidate

if not FIXTURES.exists():
    pytest.skip('repository evidence not in this checkout: docs/assessments/thin-copper-2026-10-22/site-scenes', allow_module_level=True)


@pytest.mark.parametrize('name', NAMES)
def test_candidate_retains_input_and_selects_actual_complete_radiator(name):
    original = json.loads((FIXTURES/f'metahz-scene-{name}.json').read_text())
    saved = copy.deepcopy(original)
    data = candidate(original)
    assert original == saved
    assert len(data['initial_conductor_ids']) > 10
    assert set(data['initial_conductor_ids']) <= {c['id'] for c in original['conductors']}
    scene = import_pcb_scene(data, frequency=2.45e9, device='cpu')
    selected = data['initial_conductor_ids']
    direct = direct_scene(data, scene, 2.45e9, selected)
    assert [s.name for s in direct.structures] == [s.name for s in scene.structures]
    assert data['mesh'] == original['mesh']
    assert data['ports'] == original['ports']
    assert 'untuned' in data['candidate_status']


@pytest.mark.parametrize('name,thickness', [('pcb-flexible',18e-6),('pcb-rigid',35e-6)])
def test_explicit_surface_mapping_preserves_substrate_and_removes_only_copper_interval(name, thickness):
    original = json.loads((FIXTURES/f'metahz-scene-{name}.json').read_text())
    data = candidate(original, finite_copper=True)
    assert data['dielectrics'] == original['dielectrics']
    before = np.asarray(original['mesh']['axes_m']['z'])
    after = np.asarray(data['mesh']['axes_m']['z'])
    assert len(after) == len(before)-1
    assert np.diff(after).min() > thickness
    bottom = original['conductors'][0]['bounds_m'][4]
    assert all(c['bounds_m'][4:] == [bottom,bottom] for c in data['conductors'])
    scene = import_pcb_scene(data, frequency=2.45e9, device='cpu')
    assert all(s.material.thickness == pytest.approx(thickness) for s in scene.structures
               if isinstance(s.material,mw.CopperFoil))
    assert all(point[2] == bottom for point in data['ports'][0]['integration_line_m'])
    assert data['candidate_provenance']['surface_mapping']['substrate_thickness_preserved']
