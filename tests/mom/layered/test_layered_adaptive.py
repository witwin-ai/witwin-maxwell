"""Adaptive F-A sweep versus independent 125-point uniform solves.

The reference is split into five-frequency tests to keep every GPU test within
120 seconds and 6 GiB. The shared adaptive solve runs only once per module.
"""

from dataclasses import replace
import gc
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.touchstone import read_touchstone
from tests.support.rfic_fixtures import fixture_fa
from tests.support.rfic_frontend_fixtures import process_fa, write_fa_gds

pytestmark = pytest.mark.cuda
UNIFORM = tuple(1e8 + (20e9-1e8)*i/124 for i in range(125))


@pytest.fixture(autouse=True)
def resource_gate(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    record_property("seconds",seconds)
    record_property("peak_allocated_bytes",peak)
    assert seconds < 120
    assert peak < 6*2**30


def adaptive_simulation(directory):
    fixture = fixture_fa()
    path = write_fa_gds(fixture,directory/"fa.gds")
    scene = process_fa(fixture).scene_from_gds(path)
    options = replace(fixture.config.layered,sweep=mw.AdaptiveSweep(1e8,20e9,1e-3,25))
    return mw.Simulation.mom(scene,config=mw.MoMConfig(layered=options))


@pytest.fixture(scope="module")
def adaptive_result(tmp_path_factory):
    directory = tmp_path_factory.mktemp("adaptive-gds")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = adaptive_simulation(directory).run()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    result.network.metadata["adaptive_resource_gate"] = dict(seconds=seconds,peak_allocated_bytes=peak)
    assert seconds < 120
    assert peak < 6*2**30
    return result


def test_adaptive_fa_sampling_and_passivity(adaptive_result,record_property):
    result = adaptive_result
    sweep = result.surface_currents.sweep
    assert result.network.frequencies.numel() <= 25
    assert sweep.solved_frequencies == result.network.frequencies.numel()
    assert tuple(result.network.frequencies.tolist()) == result.frequencies
    assert result.surface_currents.currents.shape[0] == len(result.frequencies)
    assert sweep.dense_network.frequencies.numel() == 2000
    assert sweep.passivity_report.passive
    assert torch.linalg.matrix_norm(sweep.dense_network.s,ord=2).max() <= 1+1e-9
    record_property("sample_frequencies_Hz",result.frequencies)
    record_property("passivity_report",str(sweep.passivity_report))
    record_property("enforcement_change",sweep.enforcement_change)
    record_property("estimated_error",sweep.estimated_error)
    record_property("adaptive_solve_resources",result.network.metadata["adaptive_resource_gate"])


@pytest.mark.parametrize("start",range(0,125,5))
def test_adaptive_fa_uniform_reference(adaptive_result,start,record_property):
    frequencies = UNIFORM[start:start+5]
    reference = fixture_fa().simulation(frequencies).run().network
    model = adaptive_result.surface_currents.sweep.evaluate_network(frequencies)
    error = float((model.s-reference.s).abs().max())
    record_property("max_S_error",error)
    record_property("reference_frequencies_Hz",frequencies)
    assert error <= 1e-3


@pytest.mark.parametrize("repeat_index",range(2))
def test_adaptive_fa_repeat_is_exact(adaptive_result,repeat_index,record_property,tmp_path):
    record_property("complete_run",repeat_index+2)
    repeat = adaptive_simulation(tmp_path).run()
    assert torch.equal(repeat.network.frequencies,adaptive_result.network.frequencies)
    assert torch.equal(repeat.network.s,adaptive_result.network.s)
    assert torch.equal(repeat.surface_currents.currents,adaptive_result.surface_currents.currents)
    assert torch.equal(repeat.surface_currents.sweep.dense_network.s,
                       adaptive_result.surface_currents.sweep.dense_network.s)


def test_adaptive_touchstone(adaptive_result,tmp_path,record_property):
    dense = adaptive_result.surface_currents.sweep.dense_network
    path = dense.to_touchstone(tmp_path/"fa.s2p")
    restored = read_touchstone(path,device=dense.s.device)
    record_property("touchstone_max_S_error",float((restored.s-dense.s).abs().max()))
    record_property("touchstone_max_frequency_error_Hz",float((restored.frequencies-dense.frequencies).abs().max()))
    record_property("touchstone_path",str(path))
    assert torch.allclose(dense.frequencies,restored.frequencies,rtol=5e-15,atol=1e-5)
    assert torch.allclose(dense.s,restored.s,rtol=5e-15,atol=5e-15)
    assert dense.port_names == restored.port_names


def test_adaptive_config_guards():
    with pytest.raises(ValueError,match="f_min"):
        mw.AdaptiveSweep(0,20e9)
    with pytest.raises(ValueError,match="tolerance"):
        mw.AdaptiveSweep(1e8,20e9,tolerance=0)
    with pytest.raises(ValueError,match="max_points"):
        mw.AdaptiveSweep(1e8,20e9,max_points=10)
    fixture = fixture_fa()
    config = mw.MoMConfig(layered=replace(fixture.config.layered,sweep=mw.AdaptiveSweep(1e8,20e9)))
    with pytest.raises(ValueError,match="defines its own frequencies"):
        mw.Simulation.mom(fixture.scene,frequency=1e9,config=config)
