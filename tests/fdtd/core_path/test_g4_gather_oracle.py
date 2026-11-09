"""CPU checks for the independent G4 periodic source-table oracle."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from benchmark.core_path.gates import source_patch_gather_reference, source_table_reference
from witwin.maxwell.fdtd.excitation.gather import build_patch_gather


@pytest.mark.parametrize("wrap", range(8))
def test_patch_gather_reference(wrap):
    rng = np.random.default_rng(43)
    shapes = [(1, 1, 1), (1, 3, 4), (4, 3, 2)]
    shapes += [tuple(rng.integers(1, 7, size=3)) for _ in range(12)]
    for shape in shapes:
        patch_shape = tuple(rng.integers(1, np.asarray(shape) + 3))
        offsets = [(0, 0, 0), tuple(size - 1 for size in shape), (-1, -1, -1),
                   shape, tuple(-size for size in patch_shape),
                   tuple(rng.integers(-np.asarray(patch_shape), np.asarray(shape) + 1))]
        for offset in offsets:
            expected = build_patch_gather(shape, patch_shape, offset, device="cpu", wrap=wrap)
            actual = source_patch_gather_reference(shape, patch_shape, offset, device="cpu", wrap=wrap)
            for reference, candidate in zip(expected, actual):
                assert candidate.dtype == reference.dtype == torch.int64
                assert candidate.shape == reference.shape
                assert candidate.is_contiguous()
                assert candidate.device.type == "cpu"
                assert torch.equal(reference, candidate), (shape, patch_shape, offset, wrap)


@pytest.mark.parametrize("periodic_axes", range(8))
def test_source_table_gather_predicate(periodic_axes):
    boundary = mw.BoundarySpec(**{
        axis: "periodic" if periodic_axes & (1 << index) else "pec"
        for index, axis in enumerate("xyz")
    })
    terms = []
    for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        for kind in ("scalar", "compiled", "cw", "delayed"):
            patch = torch.ones((3, 3, 3), device="cpu")
            terms.append({
                "field_name": field_name, "patch": patch, "offsets": (-1, 0, 2),
                "phase_real": 1.0, "periodic_compiled": kind == "compiled",
                "cw_cos_patch": patch if kind == "cw" else None,
                "delay_patch": patch if kind == "delayed" else None,
            })
    solver = SimpleNamespace(
        device="cpu", dt=0.1, source_omega=2.0,
        scene=SimpleNamespace(boundary=boundary),
        _source_time={"kind": "cw", "frequency": 1.0, "amplitude": 1.0, "phase": 0.0},
        _source_terms=terms, _magnetic_source_terms=terms, _electric_source_terms=terms,
        tfsf_enabled=True, _tfsf_state={"magnetic_terms": terms, "electric_terms": terms},
        **{name: torch.empty((3, 4, 5), device="cpu") for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
    )
    rows = source_table_reference(solver, 2, budget=0)
    expected_keys = {"time_h", "time_e"}
    for group in ("ordinary", "magnetic", "electric", "tfsf_magnetic", "tfsf_electric"):
        for ordinal, term in enumerate(terms):
            prefix = f"{group}/{ordinal}"
            expected_keys.add(f"{prefix}/signals")
            component = "xyz".index(term["field_name"][-1].lower())
            wrap = periodic_axes & ~(1 << component)
            if (wrap and not term["periodic_compiled"] and term["cw_cos_patch"] is None
                    and term["delay_patch"] is None):
                gather = build_patch_gather((3, 4, 5), (3, 3, 3), term["offsets"], device="cpu", wrap=wrap)
                for key, tensor in zip(("gather_rows", "gather_targets", "gather_entries"), gather):
                    expected_keys.add(f"{prefix}/{key}")
                    assert torch.equal(rows[f"{prefix}/{key}"], tensor)
    assert rows.keys() == expected_keys
