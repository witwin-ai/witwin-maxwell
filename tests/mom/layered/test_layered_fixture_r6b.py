"""Exact-stack scalar/vector checks for the F-A discrepancy investigation."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from tests.support.rfic_fixtures import fixture_fa
from witwin.maxwell.mom.layered.green import LayeredGreenTable

pytestmark=pytest.mark.cuda


@pytest.fixture(autouse=True)
def budget(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    yield
    torch.cuda.synchronize()
    record_property("seconds",time.perf_counter()-started)
    record_property("peak_allocated_bytes",torch.cuda.max_memory_allocated())
    assert time.perf_counter()-started<120
    assert torch.cuda.max_memory_allocated()<6*2**30


@pytest.mark.parametrize("index",(0,1))
def test_r6b_exact_stack_green_reference(index,record_property):
    # Coplanar rows retain the external-library gate. Cross-height rows use O2:
    # its real-axis result resolves the default library integration error.
    data=np.load(Path(__file__).with_name("data")/"fixture_r6b_green_reference.npz")
    planes=tuple(sorted(set(data["pairs"].ravel())))
    frequency=float(data["frequencies"][index])
    table=LayeredGreenTable.build(fixture_fa(device="cpu").scene.background,planes,
        (frequency,),float(data["rho"][-1]),device="cuda",via_spans=((planes[0],planes[-1]),))
    rho=torch.tensor(data["rho"],device="cuda",dtype=torch.float64)
    errors=[]
    for pair,(source,observer) in enumerate(data["pairs"]):
        horizontal=table.evaluate(float(source),float(observer),rho)[0].cpu().numpy()
        vertical=table.evaluate_vertical(rho,torch.full_like(rho,float(source)),
                                        torch.full_like(rho,float(observer)))[0].cpu().numpy()
        for actual,expected in ((horizontal,data["values"][index,pair]),
                                (vertical,data["vertical"][index,pair])):
            row_error=np.max(abs(actual-expected),axis=-1)/np.max(abs(expected),axis=-1)
            errors.extend(row_error.tolist())
            tolerance=1e-6 if pair in data["oracle_pair_indices"] else 1e-3
            assert np.all(row_error<=tolerance),(frequency,source,observer,row_error)
    record_property("max_relative_row_error",max(errors))
    record_property("component_errors",json.dumps(errors))


def _shunt_evidence():
    import hashlib
    data=json.loads(Path(__file__).with_name("data").joinpath("fixture_r6b_shunt_evidence.json").read_text())
    root=Path(__file__).resolve().parents[3]
    for name,digest in data["source_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    return data


def test_r6b_isolated_shunt_three_complete_runs(record_property):
    import witwin.maxwell as mw
    from tests.support.rfic_r6b_references import plate_fixture,FREQUENCIES
    evidence=_shunt_evidence()
    expected=np.array(evidence["native_coarse"]["y_real"])[:,0,0]
    first=None
    for _ in range(3):
        scene,config=plate_fixture(40.,10.)
        result=mw.Simulation.mom(scene,frequencies=FREQUENCIES,config=config).run()
        s=result.network.s
        current=result.surface_currents.currents
        if first is None:
            first=(s.clone(),current.clone())
            measured=result.network.to_y()[:,0,0].real.cpu().numpy()
            assert np.max(abs(measured/expected-1))<1e-8
            record_property("conductance_S",json.dumps(measured.tolist()))
        else:
            assert torch.equal(first[0],s)
            assert torch.equal(first[1],current)
        del result,s,current
        gc.collect()
        torch.cuda.empty_cache()


def test_r6b_shunt_against_independent_electrostatics(record_property):
    evidence=_shunt_evidence()
    native=np.array(evidence["native_fine"]["y_real"])[:,0,0]
    fine=np.array(evidence["static_fine"]["y_real"])
    previous=np.array(evidence["static_previous"]["y_real"])
    # This is an observed discretization spread, not a continuum error certificate.
    # It does not replace any frozen R6 L/Q qualification target.
    error=abs(native-fine)
    spread=abs(fine-previous)
    record_property("native_relative_error",json.dumps((error/fine).tolist()))
    record_property("independent_mesh_spread",json.dumps((spread/fine).tolist()))
    assert np.all(error<spread),(error,spread)
