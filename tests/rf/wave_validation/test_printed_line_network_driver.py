"""Actual physical grids selected by the typed printed-line network driver."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from witwin.maxwell.scene import prepare_scene

PATH = Path(__file__).resolve().parents[3] / 'docs/assessments/printed-lines-2026-10-22/validate.py'
if not PATH.exists():
    pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22/validate.py', allow_module_level=True)
SPEC = importlib.util.spec_from_file_location('printed_line_network_validation', PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def arguments(kind, *, radius=None, dx=.000625):
    values = ['--kind', kind, '--dx', str(dx), '--longitudinal-dl', '.0025',
              '--padding', '.08', '--separation', '.4', '--axial-half-length', '.3',
              '--output', '/tmp/unexecuted-printed-line-network.json']
    if radius is not None:
        values += ['--transverse-refinement-radius', str(radius)]
    return driver.argument_parser().parse_args(values)


@pytest.mark.parametrize('kind', driver.KINDS)
def test_network_driver_grades_outer_grid_without_changing_physical_fixture(kind):
    uniform = driver.scene_from_arguments(arguments(kind), device='cpu')
    graded = driver.scene_from_arguments(arguments(kind, radius=.02), device='cpu')
    assert graded.domain.bounds == uniform.domain.bounds
    assert graded.metadata['pml_physical_thickness_m'] == uniform.metadata['pml_physical_thickness_m']
    assert graded.metadata['transverse_refinement_radius_m'] == .02
    for first, second in zip(uniform.structures, graded.structures):
        assert first.name == second.name
        for name in ('position', 'size', 'vertices', 'rotation'):
            if hasattr(first.geometry, name):
                assert torch.equal(getattr(first.geometry, name), getattr(second.geometry, name))
    for first, second in zip(uniform.ports, graded.ports):
        assert first.position == second.position
        assert first.reference_plane == second.reference_plane
        assert first.size == second.size
    prepared = prepare_scene(graded)
    record = driver.grid_record(prepared)
    assert record['node_shape'] == [257, 177, 169]
    for axis in 'xyz':
        actual = getattr(prepared, axis + '_nodes64')
        np.testing.assert_array_equal(record['nodes_m'][axis], actual)
    for axis in 'yz':
        values = np.asarray(record['nodes_m'][axis])
        fine = np.diff(values[(values >= -.02) & (values <= .02)])
        outer = np.diff(values[values > .02])
        np.testing.assert_allclose(fine, .000625, rtol=0, atol=1e-15)
        np.testing.assert_allclose(outer, .0025, rtol=0, atol=1e-15)


def test_network_driver_refinement_keeps_axial_and_external_nodes_exact():
    coarse = prepare_scene(driver.scene_from_arguments(arguments('cpw', radius=.02), device='cpu'))
    fine = prepare_scene(driver.scene_from_arguments(arguments('cpw', radius=.02, dx=.0003125), device='cpu'))
    a, b = driver.grid_record(coarse), driver.grid_record(fine)
    assert a['node_sha256']['x'] == b['node_sha256']['x']
    for axis in 'yz':
        first, second = (np.asarray(row['nodes_m'][axis]) for row in (a, b))
        np.testing.assert_array_equal(first[np.abs(first) > .02], second[np.abs(second) > .02])


@pytest.mark.parametrize('radius', (.001, .0201, float('nan'), float('inf')))
def test_network_driver_rejects_invalid_radius_before_field_preparation(radius):
    with pytest.raises(ValueError, match='transverse_refinement_radius'):
        driver.scene_from_arguments(arguments('microstrip', radius=radius), device='cpu')
