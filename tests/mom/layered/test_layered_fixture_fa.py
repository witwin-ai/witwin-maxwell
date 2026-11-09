"""F-A physical gates; full-mesh convergence is recorded in the adjacent data file.

GPU-0 tests use the frozen coarse mesh. Five-frequency batches keep each
test below 120 seconds and 6 GiB. The benchmark separately measures the
complete 25-point full mesh; batch determinism is tested on three fresh
Scene -> Simulation -> Result runs, never a reused prepared solver.
"""
import gc
import json
import math
from pathlib import Path
import time

import pytest
import torch
from tests.support.rfic_fixtures import fixture_fa,SWEEP
from tests.support.rfic_fa_references import (
    greenhouse_fa,stranded_greenhouse_fa,dc_resistance_fa,mohan_spiral,extract_fa,resonance_and_peak,partial_inductance)

pytestmark=pytest.mark.cuda


@pytest.fixture(autouse=True)
def gpu_budget(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated()
    record_property("seconds",seconds)
    record_property("peak_allocated_bytes",peak)
    assert seconds<120
    assert peak<6*2**30


def test_lossless_vacuum_greenhouse(record_property):
    fixture=fixture_fa(lossy=False,substrate=False,ring=False)
    result=fixture.simulation((1e6,1e7)).run()
    data=extract_fa(result.network)
    reference=stranded_greenhouse_fa(fixture)
    for key,value in dict(extraction=data,greenhouse=reference,single_strip=greenhouse_fa(fixture),mohan=mohan_spiral()).items():
        record_property(key,json.dumps(value))
    assert max(abs(x/reference["total_H"]-1) for x in data["L_H"])<.02
    assert reference["constraint_error"]<1e-12
    assert reference["minimum_partial_eigenvalue_H"]>0
    assert abs(data["L_H"][1]/data["L_H"][0]-1)<1e-5
    # Reversing one of two parallel segments changes only the mutual sign.
    a=((0.,0.,0.),(1e-3,0.,0.),10e-6,0.)
    b=((0.,20e-6,0.),(1e-3,20e-6,0.),10e-6,0.)
    positive=partial_inductance((a,b))
    negative=partial_inductance((a,(b[1],b[0],b[2],b[3])))
    assert positive["positive_mutual_H"]>0
    assert negative["negative_mutual_H"]<0
    assert abs(positive["total_H"]+negative["total_H"]-2*positive["self_H"])<1e-20


def test_dc_resistance_and_thickness(record_property):
    fixture=fixture_fa(substrate=False,ring=False)
    result=fixture.simulation((1e6,1e9)).run()
    data=extract_fa(result.network)
    reference=dc_resistance_fa(fixture)
    record_property("extraction",json.dumps(data))
    record_property("dc_reference",json.dumps(reference))
    assert abs(data["R_ohm"][0]/reference["total_ohm"]-1)<.03
    # Bulk-wall via resistance vanishes as sqrt(f), whereas the solid via
    # has rho*h/A. Its entire missing DC contribution is below 0.3%.
    assert reference["solid_via_ohm"]/reference["total_ohm"]<.003
    del result
    gc.collect(); torch.cuda.empty_cache()
    thick=fixture_fa(substrate=False,ring=False,top_thickness_um=4.).simulation((1e9,)).run()
    resistance=extract_fa(thick.network)["R_ohm"][0]
    record_property("thicker_R_1GHz_ohm",resistance)
    assert resistance<data["R_ohm"][1]


@pytest.mark.parametrize("start",range(0,25,5))
def test_sweep_reciprocity_passivity_complete_repeat(start,record_property):
    frequencies=SWEEP[start:start+5]
    first=None
    for repeat in range(3):
        result=fixture_fa().simulation(frequencies).run()
        network=result.network
        y=network.to_y()
        symmetry=(torch.linalg.matrix_norm(y-y.transpose(-1,-2))/torch.linalg.matrix_norm(y)).max().item()
        singular=torch.linalg.matrix_norm(network.s,ord=2).max().item()
        assert symmetry<1e-10
        assert singular<=1+1e-9
        assert network.metadata["low_frequency_method"]=="loop_tree_scaled"
        if first is None:
            first=(network.s.clone(),result.surface_currents.currents.clone())
            record_property("extraction",json.dumps(extract_fa(network)))
            record_property("reciprocity",symmetry)
            record_property("max_singular_S",singular)
        else:
            assert torch.equal(network.s,first[0])
            assert torch.equal(result.surface_currents.currents,first[1])
        del result,network,y
        gc.collect(); torch.cuda.empty_cache()


def test_substrate_low_frequency_inductance(record_property):
    values=[]
    for substrate in (False,True):
        result=fixture_fa(substrate=substrate).simulation((1e8,)).run()
        values.append(extract_fa(result.network)["L_H"][0])
        del result
        gc.collect(); torch.cuda.empty_cache()
    record_property("vacuum_substrate_L_H",values)
    record_property("relative",abs(values[1]/values[0]-1))
    # sigma=10 S/m at 100 MHz gives delta=15.9 mm, far larger than the
    # 200-um device. Magnetic eddy screening is correspondingly negligible.
    assert abs(values[1]/values[0]-1)<.01


def test_substrate_loss_quality_trend(record_property):
    frequencies=SWEEP[15:20]
    peaks=[]
    for sigma in (10.,.1,0.):
        result=fixture_fa(substrate_sigma=sigma).simulation(frequencies).run()
        data=extract_fa(result.network)
        peak=resonance_and_peak(data)
        peaks.append(peak["Q_peak"])
        record_property(f"sigma_{sigma}_extraction",json.dumps(data))
        record_property(f"sigma_{sigma}_peak",json.dumps(peak))
        del result
        gc.collect(); torch.cuda.empty_cache()
    assert peaks[0]<peaks[1]<peaks[2]


def test_full_mesh_convergence_evidence(record_property):
    # The full dense work set exceeds the 6-GiB correctness allocator cap.
    # Its measured benchmark data are frozen separately, with source hashes.
    evidence=json.loads(Path(__file__).with_name("data").joinpath("fixture_fa_convergence.json").read_text())
    assert 5000<=evidence["full_unknowns"]<=10000
    assert evidence["L_relative"]<=.01
    assert evidence["Q_peak_relative"]<=.05
    assert evidence["peak_frequency_relative"]<=.05
    peak=SWEEP.index(evidence["full_peak_frequency_Hz"])
    frequencies=(SWEEP[0],*SWEEP[peak-1:peak+2])
    result=fixture_fa().simulation(frequencies).run()
    data=extract_fa(result.network)
    actual=resonance_and_peak(data)
    assert abs(data["L_H"][0]/evidence["full_L_0p1GHz_H"]-1)<=.01
    assert abs(actual["Q_peak"]/evidence["full_Q_peak"]-1)<=.05
    assert abs(actual["peak_frequency_Hz"]/evidence["full_peak_frequency_Hz"]-1)<=.05
    record_property("convergence",json.dumps(evidence))

def test_substrate_thickness_resistance_trend(record_property):
    resistance=[]
    for thickness in (2.,4.):
        result=fixture_fa(top_thickness_um=thickness).simulation((1e9,)).run()
        resistance.append(extract_fa(result.network)["R_ohm"][0])
        del result
        gc.collect(); torch.cuda.empty_cache()
    record_property("R_series_1GHz_2um_4um_ohm",resistance)
    assert resistance[1]<resistance[0]
