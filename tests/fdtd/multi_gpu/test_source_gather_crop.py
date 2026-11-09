"""Ownership-cropped source terms keep routes that match their own payload.

The distributed shard build crops compiled source terms to the x range each rank
owns. Periodic and Bloch terms carry a prepared gather built for the uncropped
patch, so a crop must rebuild it: a stale gather reads entries past the cropped
patch (whatever the allocator left there) and routes the surviving planes to the
wrong x cells.
"""
from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from tests.fdtd.multi_gpu.test_review_regressions import (
    _FIELD_NAMES,
    _FREQUENCY,
    _direct_distributed,
    _distributed_component,
    _ex_interface_scene,
)
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.scene import prepare_scene


_PAYLOAD_KEYS = ("patch", "cw_cos_patch")


def _solve_pair(scene, devices):
    single = FDTD(prepare_scene(scene), frequency=_FREQUENCY, absorber_type="cpml")
    single.init_field()
    single.solve(time_steps=24, dft_frequency=_FREQUENCY, dft_window="none", full_field_dft=False,
                 normalize_source=False, use_cuda_graph=False)
    distributed = _direct_distributed(scene, devices, gather_fields=True)
    output = distributed.solve(time_steps=24, dft_frequency=_FREQUENCY, dft_window="none", full_field_dft=False,
                               normalize_source=False, use_cuda_graph=False)
    return single, distributed, output


def _assert_parity(single, distributed, output):
    assert float(torch.abs(single.Ex).max().item()) > 0.0
    for name in _FIELD_NAMES:
        torch.testing.assert_close(
            _distributed_component(distributed, output, name).to("cuda:0"),
            getattr(single, name).to("cuda:0"),
            rtol=2.0e-5,
            atol=2.0e-6,
        )


def test_cropped_periodic_terms_route_only_their_own_entries(cuda_p2p_devices, cuda_memory_cleanup):
    distributed = _direct_distributed(_ex_interface_scene("y_periodic"), cuda_p2p_devices, gather_fields=True)
    checked = 0
    for shard in distributed.shards:
        solver = shard.solver
        for term in solver._source_terms + solver._electric_source_terms + solver._magnetic_source_terms:
            gather = term.get("source_gather")
            if gather is None:
                continue
            payload = next(term[key] for key in _PAYLOAD_KEYS if term.get(key) is not None)
            rows, targets, entries = gather
            field = getattr(solver, term["field_name"])
            plane = int(payload.shape[1]) * int(payload.shape[2])
            assert int(entries[:, 0].max()) < payload.numel()
            # Entries follow their target rows (CSR). x is the sharded axis and never
            # wraps, so patch entry q must land on plane offset_x + q // plane.
            per_entry = targets[:, 1].repeat_interleave(rows[1:] - rows[:-1])
            target_x = per_entry // (field.shape[1] * field.shape[2])
            torch.testing.assert_close(target_x, int(term["offsets"][0]) + entries[:, 0] // plane, rtol=0, atol=0)
            checked += 1
    assert checked >= 2


def test_cropped_patch_gather_ignores_stale_allocator_contents(cuda_p2p_devices, cuda_memory_cleanup):
    # Leave nonzero values in freed small blocks on the second rank's device; an
    # over-reading gather then injects them, as after other tests in one process.
    blocks = [torch.full((96,), 3.0, device=cuda_p2p_devices[1]) for _ in range(4096)]
    del blocks
    _assert_parity(*_solve_pair(_ex_interface_scene("y_periodic"), cuda_p2p_devices))


def test_seam_straddling_uniform_current_matches_single_device(cuda_p2p_devices, cuda_memory_cleanup):
    # A box current across the x seam that touches the periodic y face gives the
    # second rank cropped nonzero planes, which must reach its own owned cells.
    scene = _ex_interface_scene("y_periodic")
    source_time = scene.sources[0].source_time
    scene.sources = [mw.UniformCurrentSource(size=(0.3, 0.15, 0.2), center=(0.0, -0.15, 0.0), polarization="Ex",
                                             source_time=source_time, name="seam_box_ex")]
    _assert_parity(*_solve_pair(scene, cuda_p2p_devices))
