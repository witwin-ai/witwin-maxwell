"""Prepare the shipped example scenes on CUDA without advancing a design or FDTD step."""

import importlib.util
from pathlib import Path

import pytest
import torch

import witwin.maxwell as mw


_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_ENTRYPOINTS = tuple(sorted(path.relative_to(_EXAMPLES).as_posix() for path in _EXAMPLES.rglob("*.py")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="example preparation requires CUDA")
@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_examples_prepare(entrypoint):
    spec = importlib.util.spec_from_file_location("maxwell_example", _EXAMPLES / entrypoint)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    if entrypoint == "invdes/waveguide_bend_topology.py":
        model = example.BendDesign(device="cuda")
        simulation = mw.Simulation.fdtd(
            model,
            frequencies=[example.FREQUENCY],
            run_time=mw.TimeConfig.auto(steady_cycles=20, transient_cycles=25),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        )
        prepared = simulation.prepare()
        assert prepared.solver._compiled_sources[0]["prepared_mode_power"] == 1.0
        assert prepared.solver._electric_source_terms
        assert prepared.solver._magnetic_source_terms
    elif entrypoint == "fdfd/local_response_antenna.py":
        # This public prepare builds the source and background responses at all
        # three example frequencies, with no binary-search passes or design run.
        with example.build().prepare() as prepared:
            assert len(prepared.responses) == 3
            assert all(response.design_indices.numel() > 0 for response in prepared.responses)
    else:
        pytest.fail(f"Add the example's exact construction and prepare call: {entrypoint}")
