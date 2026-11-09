"""Explicit source identities cannot be silently swapped between physical RHS."""
from dataclasses import FrozenInstanceError
import copy
import json

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.carrier import IncidentCarrier
from witwin.maxwell.mom.physics import compile_medium, incident_fields


def configuration():
    exterior = compile_medium(mw.Material(), 3e8, background=True)
    sources = (mw.PlaneWave(source_time=mw.CW(frequency=3e8, phase=.37)),
               mw.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1),
                            source_time=mw.CW(frequency=3e8, phase=-.51, amplitude=.31)))
    return sources, exterior


@pytest.mark.host_validation
def test_carrier_preserves_shared_plane_wave_transverse_tolerance():
    from witwin.maxwell.mom.runtime import DecodeHandle
    from witwin.maxwell.mom.operator import IntegrationConfig
    source = mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 1e-8))
    exterior = compile_medium(mw.Material(), 3e8, background=True)
    handle = DecodeHandle(None, exterior, exterior, (source,), IntegrationConfig(), {})
    assert handle.incident_carrier.waves[0].polarization == source.polarization


@pytest.mark.host_validation
def test_carrier_payload_roundtrip_and_immutable_ownership():
    sources, exterior = configuration()
    carrier = IncidentCarrier.compile(sources, exterior)
    payload = json.loads(json.dumps(carrier.payload()))
    restored = IncidentCarrier.restore(payload, expected_identity=carrier.identity,
                                       sources=sources, exterior=exterior)
    assert restored == carrier and restored.identity == carrier.identity
    with pytest.raises(FrozenInstanceError):
        carrier.frequency = 4e8
    payload['waves'][0]['direction'][0] = 1.
    assert carrier.waves[0].direction == (0., 0., 1.)
    assert IncidentCarrier.compile(tuple(reversed(sources)), exterior).identity != carrier.identity


@pytest.mark.parametrize('mutation', ['version', 'phase', 'frequency', 'order', 'extra', 'digest', 'bool_coordinates'])
@pytest.mark.host_validation
def test_carrier_rejects_changed_or_ambiguous_snapshot(mutation):
    sources, exterior = configuration()
    carrier = IncidentCarrier.compile(sources, exterior)
    payload = copy.deepcopy(carrier.payload())
    identity = carrier.identity
    if mutation == 'version': payload['version'] = 'unversioned'
    elif mutation == 'phase': payload['waves'][0]['amplitude'][1] += .1
    elif mutation == 'frequency': payload['frequency'] *= 2
    elif mutation == 'order': payload['waves'].reverse()
    elif mutation == 'extra': payload['unstated_default'] = 1
    elif mutation == 'bool_coordinates': payload['waves'][0]['direction'] = [False, False, True]
    else: identity = '0'*64
    with pytest.raises(ValueError, match='does not match'):
        IncidentCarrier.restore(payload, expected_identity=identity, sources=sources, exterior=exterior)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_compiled_carrier_fields_match_declared_sources_on_both_devices(dtype):
    sources, exterior = configuration()
    sources += (mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 1e-8)),)
    carrier = IncidentCarrier.compile(sources, exterior)
    for device in range(torch.cuda.device_count()):
        points = torch.tensor([[.1, .2, .3], [-.2, .1, .4]], dtype=dtype, device=f'cuda:{device}')
        e, h = carrier.evaluate(points, exterior)
        reference = [incident_fields(source, points, exterior) for source in sources]
        torch.testing.assert_close(e, sum(pair[0] for pair in reference), rtol=0, atol=0)
        torch.testing.assert_close(h, sum(pair[1] for pair in reference), rtol=0, atol=0)
        assert e.device == h.device == points.device
        other = compile_medium(mw.Material(eps_r=1.1), 3e8, background=True)
        with pytest.raises(ValueError, match='background identity'):
            carrier.evaluate(points, other)
