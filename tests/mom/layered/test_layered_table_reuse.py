"""Shared spectral panels retain the fully batched Bessel values bitwise."""

import pytest
import torch

from tests.support.rfic_fixtures import fixture_fa
from witwin.maxwell.mom.layered.build import build_extension

pytestmark = pytest.mark.cuda


def test_shared_vertical_bessel_panels_match_full_batch(monkeypatch, record_property):
    ext = build_extension()
    original_nodes = ext.nodes
    current = {}
    shared_calls = []

    def nodes(*args):
        result = original_nodes(*args)
        current["nodes"] = result
        return result

    def checked(operation):
        def evaluate(q, weights, kinds, *args):
            result = operation(q, weights, kinds, *args)
            full_q, full_weights, full_kinds = current["nodes"]
            if q.shape[0] == 1 and full_q.shape[0] > 1:
                width = q.shape[1]
                inputs = tuple(x[:, -width:].contiguous()
                               for x in (full_q, full_weights, full_kinds))
                for full, shared in zip(inputs, (q, weights, kinds)):
                    assert torch.equal(full, shared.expand_as(full))
                reference = operation(*inputs, *args)
                assert torch.equal(reference, result.expand_as(reference))
                shared_calls.append(width)
            return result
        return evaluate

    monkeypatch.setattr(ext, "nodes", nodes)
    monkeypatch.setattr(ext, "bessel", checked(ext.bessel))
    monkeypatch.setattr(ext, "bessel_one", checked(ext.bessel_one))
    prepared = fixture_fa().simulation((1e8, 20e9)).prepare().solver.prepared
    assert prepared.table.vertical is not None
    assert shared_calls
    record_property("shared_panel_calls", len(shared_calls))
    record_property("shared_nodes", sum(shared_calls))
