"""Failed-only successors reusing the closed independent MP100 spectra."""
from pathlib import Path
import hashlib
import json
import math
import os
import pytest
import torch
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.dyadic_green import _factored_evanescent_plane_spectrum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_only_six_failed_original_tensor_cross_plane_blocks(tmp_path):
    reference_path = os.environ.get('GSP_FACTORED_PLANE_REFERENCE')
    expected_sha = os.environ.get('GSP_FACTORED_PLANE_REFERENCE_SHA256')
    if reference_path is None and expected_sha is None:
        pytest.skip('Requires the closed failed-plane reference artifact and its SHA256')
    assert reference_path and expected_sha, 'Reference path and SHA256 must be supplied together'
    reference = Path(reference_path)
    assert reference.is_file(), 'Declared failed-plane reference artifact is absent'
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == expected_sha, 'Reference SHA256 mismatch'
    samples = torch.load(reference, map_location='cuda', weights_only=True)
    metric = torch.diag(torch.tensor([1.25, 1., .75], device='cuda', dtype=torch.float64))
    angle = .37
    rotation = metric.new_tensor([[math.cos(angle), -math.sin(angle), 0.],
                                 [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    frames = (torch.eye(3, device='cuda', dtype=torch.float64), rotation)
    selected = [(0, (1e8, 1e8)), (1, (5e5, 7e5)), (1, (1e8, 1e8))]
    records = []
    receipt = dict(version='failed_only_factored_cross_plane_successor_v2',
                   reference_sha256=expected_sha,
                   independent_MP100_reference_reused=True, old_successful_planes_not_replayed=True,
                   six_closed_diagonal_block_PASS_not_retested=True,
                   original_spatial_covariance_failure_preserved=True,
                   production_qualification=False, points=records)
    def save():
        (tmp_path/'factored-remainder.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    save()
    for orientation, q in selected:
        sample = next(s for s in samples if s['orientation'] == orientation and tuple(s['transverse']) == q)
        frame = frames[orientation]
        eps, mu = (frame@(2*metric)@frame.T).to(torch.complex128), (frame@(1.3*metric)@frame.T).to(torch.complex128)
        medium = TensorMedium(3e7, eps, mu)
        before = (medium.epsilon_r.clone(), medium.mu_r.clone())
        item = dict(orientation=orientation, transverse=list(q), blocks=[], error=None)
        try:
            value, algorithm = _factored_evanescent_plane_spectrum(metric.new_tensor([q]), sample['distance_m'], medium)
            item['algorithm'] = algorithm
            for name, i, j in [('EH', 0, 3), ('HE', 3, 0)]:
                wanted = sample['expected']['remainder'][i:i+3, j:j+3]
                actual = value[0, i:i+3, j:j+3]
                error = float((actual-wanted).norm()/wanted.norm())
                old_error = float((sample['native_remainder'][i:i+3, j:j+3]-wanted).norm()/wanted.norm())
                item['blocks'].append(dict(block=name, relative_error=error if math.isfinite(error) else None,
                                            old_relative_error=old_error if math.isfinite(old_error) else None,
                                            gate=.005, passed=math.isfinite(error) and error < .005))
        except (ValueError, RuntimeError) as error:
            item['error'] = str(error)
        item['passed'] = item['error'] is None and len(item['blocks']) == 2 and all(b['passed'] for b in item['blocks'])
        assert torch.equal(medium.epsilon_r, before[0]) and torch.equal(medium.mu_r, before[1])
        records.append(item)
        save()
        print(item, flush=True)
    receipt['completed'] = True
    save()
    assert len(records) == 3 and all(r['passed'] for r in records)
