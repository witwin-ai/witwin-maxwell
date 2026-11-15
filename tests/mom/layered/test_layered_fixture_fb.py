"""F-B geometry/reference contracts and coarse GPU smoke; fine physics remains open."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from tests.support.rfic_fixtures import fixture_fb
from tests.support.rfic_fb_references import geometry_components, winding_metrics, unpack
from tests.support.rfic_r6_references import geometry_digest
from witwin.maxwell.mom.layered.runtime import compile_layered_scene
from witwin.maxwell.mom.layered.mesh import validate_vias
import witwin.maxwell as mw

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")

DATA=Path(__file__).with_name("data")/"fixture_fb_reference.json"


def frozen():
    return json.loads(DATA.read_text())


@pytest.mark.parametrize("lossy",(False,True))
@pytest.mark.parametrize("substrate",(False,True))
def test_fb_geometry_ports_hash(lossy,substrate):
    f=fixture_fb(lossy=lossy,substrate=substrate)
    stack,r,p,m,_,v,vm=compile_layered_scene(f.scene)
    validate_vias(r,v)
    assert len(v)==4
    components=geometry_components(r,v)
    assert len(set(components))==3
    def owner(point):
        hits={components[i] for i,rect in enumerate(r)
              if rect[4]==point[2] and rect[0]<=point[0]<=rect[1] and rect[2]<=point[1]<=rect[3]}
        assert len(hits)==1
        return hits.pop()
    signals=[owner(port.positive) for port in p]
    grounds=[owner(port.negative) for port in p]
    assert signals[0]==signals[1] and signals[2]==signals[3]
    assert len(set(signals))==2
    assert len(set(grounds))==1 and grounds[0] not in signals
    assert [port.name for port in p]==["p1","p2","p3","p4"]
    for port,y in zip(p,(170.,70.,145.,45.)):
        np.testing.assert_allclose(np.asarray(port.positive)*1e6,(205,y,305),rtol=0,atol=1e-9)
        np.testing.assert_allclose(np.asarray(port.negative)*1e6,(215,y,305),rtol=0,atol=1e-9)
        assert port.voltage_path.axis=="x"
        assert port.reference_impedance==50.
    assert stack.layers[0].eps_r==(11.9 if substrate else 1.)
    assert stack.layers[0].sigma==(10. if substrate else 0.)
    assert [round(layer.thickness*1e6,6) for layer in stack.layers]==[300.,6.]
    assert stack.layers[1].eps_r==(4.1 if substrate else 1.)
    assert stack.bottom.eps_r==stack.layers[0].eps_r
    assert stack.bottom.sigma==stack.layers[0].sigma
    assert stack.top.eps_r==1.
    assert abs(f.config.layered.bridge_width_m-20e-6)<1e-16
    assert geometry_digest(r,m,v,vm)==frozen()["geometry_sha256"]["Cu" if lossy else "PEC"]
    assert f.config.layered.conductor_model=="sheet"


def test_fb_frozen_reference_self_consistency():
    evidence=frozen()
    assert set(evidence["quasi_static"])=={"qs-10-1-3","qs-5-1-3","qs-5-2-3","qs-5-1-5",
        "qs-2p5-1-3","qs-thin-5-1-3","qs-thin-2p5-1-3","qs-5-1-3-multi",
        "qs-2p5-1-3-order8","qs-5-1-3-long2"}
    assert set(evidence["fdtd"])=={"fdtd-10","fdtd-5","fdtd-2p5"}
    assert set(evidence["fdtd_controls"])=={"fdtd-boundary","fdtd-z","fdtd-gap"}
    assert evidence["accuracy_targets"]==dict(L_relative=.02,Q_peak_relative=.05,
        S_magnitude_absolute=.01,S_phase_degrees=1.,k_absolute=.01)
    assert evidence["native_full_qualification"]=="pending_dense_memory_task"
    primary=evidence["reference_roles"]["quasi_static"]
    assert primary["role"]=="PRIMARY"
    assert primary["quantities"]==["L1_H","L2_H","M_H","k","R1_ohm","R2_ohm"]
    assert primary["R_relative_target"]==.03
    assert primary["R_spread_exceeds_target"]
    assert .03<primary["discretization_spreads"]["width_filaments"]["R_relative"]<.039
    field=evidence["reference_roles"]["fdtd"]
    assert field["role"]=="EVIDENCE-ONLY" and field["status"]=="not_converged"
    assert field["quantities"]==["S","Q","SRF","shunt_G_S"]
    assert field["not_targets_for"]==["L1_H","L2_H","M_H","k"]
    assert all(order<0 for order in field["self_L_observed_orders"])
    for diagnostic in evidence["fdtd_convergence"].values():
        assert set(diagnostic)=={"frequency_Hz","values","observed_order","monotone","caveat"}
        a,b,c=diagnostic["values"]
        ratio=(a-b)/(b-c)
        assert diagnostic["monotone"]==(ratio>0)
        np.testing.assert_allclose(diagnostic["observed_order"],np.log2(abs(ratio)),rtol=1e-14)
    for case in evidence["quasi_static"].values():
        assert case["receipt"]["returncode"]==0
        assert len(case["receipt"]["binary_sha256"])==64
        assert case["receipt"]["input_sha256"]==case["raw_sha256"]["model.inp"]
        for row in case["table"]:
            assert 0<row["k"]<1
            assert row["R1_ohm"]>0 and row["R2_ohm"]>0
            assert row["kcl_relative"]<1e-8
            # Matrix text has six significant digits; finite expansion error is measured separately.
            assert row["reciprocity_relative"]<1e-4
            y=np.asarray(row["y_real"])+1j*np.asarray(row["y_imag"])
            full_y=np.stack((y[:,0],-y[:,0],y[:,1],-y[:,1]),-1)
            identity=np.eye(4)
            s=(identity-50*full_y)@np.linalg.inv(identity+50*full_y)
            assert np.linalg.svd(s,compute_uv=False).max()<=1+1e-8
            assert abs(row["k"]-row["M_H"]/np.sqrt(row["L1_H"]*row["L2_H"]))<1e-14
    for case in evidence["fdtd"].values():
        for number,receipt in enumerate(case["receipts"],1):
            assert receipt["returncode"]==0
            assert len(receipt["binary_sha256"])==64
            assert receipt["input_sha256"]==case["raw_sha256"][f"port{number}/model.xml"]
        assert case["checks"]["max_S_reciprocity"]<.01
        assert case["checks"]["max_singular_S"]<=1.001
        s=unpack(case["samples"],"s")
        assert s.shape[-2:]==(4,4)
        assert np.isfinite(s).all()
        assert np.max(np.abs(s-s.swapaxes(-1,-2)))<.01
        assert np.linalg.svd(s,compute_uv=False).max()<=1.001
        identity=np.eye(4)
        z=50*(identity+s)@np.linalg.inv(identity-s)
        reconstructed=winding_metrics(case["samples"]["frequency_Hz"],z)
        for key in ("L1_H","L2_H","M_H","R1_ohm","R2_ohm"):
            np.testing.assert_allclose(reconstructed[key],case["table"][key],rtol=1e-8,atol=1e-15)
        low=np.asarray(case["samples"]["frequency_Hz"])<=1e9
        coupling=np.asarray(reconstructed["k"],dtype=float)[low]
        np.testing.assert_allclose(coupling,np.asarray(case["table"]["k"],dtype=float)[low],rtol=1e-8)
        assert ((coupling>0)&(coupling<1)).all()
    for case in evidence["fdtd_controls"].values():
        assert case["receipt"]["returncode"]==0
        assert case["receipt"]["input_sha256"]==case["raw_sha256"]["port1/model.xml"]
        assert np.isfinite(case["max_loaded_column_absolute_change"])
    for group in ("quasi_static","fdtd"):
        assert evidence[group]
        for case in evidence[group].values():
            assert len(case["raw_sha256"])>=3
            assert all(len(value)==64 for value in case["raw_sha256"].values())


def test_fb_differential_extraction():
    # A reciprocal coupled-inductor matrix plus arbitrary common-mode impedance.
    d=np.array(((1,0),(-1,0),(0,1),(0,-1)),dtype=float)
    zw=np.array(((2-12j,.1-6j),(.1-6j,3-10j)))
    common=np.array(((1,0),(1,0),(0,1),(0,1)),dtype=float)
    z=(d@zw@d.T/4+common@np.eye(2)@common.T)[None]
    measured=winding_metrics(np.array([1e9]),z)
    assert measured["R1_ohm"]==[2.]
    assert measured["R2_ohm"]==[3.]
    np.testing.assert_allclose(measured["k"],[6/np.sqrt(120)],rtol=1e-15)



def test_fb_four_port_trace_extraction(monkeypatch):
    from scripts.dev.rfic.reference_fixture_fb import extract
    time=np.linspace(0.,1e-9,501)
    pulse=np.exp(-((time-.2e-9)/.04e-9)**2)
    resistance=np.array((100.,50.,100/3,25.))
    voltage=np.eye(4)+np.ones((4,4))/8
    current=voltage/resistance[:,None]
    def samples(path,comments):
        driven=int(path.parent.name[1:])
        port=int(path.name[-1])-1
        amplitudes=current if "_it" in path.name else voltage
        return np.column_stack((time,pulse*amplitudes[port,driven]))
    monkeypatch.setattr(np,"loadtxt",samples)
    frequencies=np.array((1e9,2e9))
    measured=extract([Path(f"d{d}") for d in range(4)],frequencies)
    np.testing.assert_allclose(unpack(measured,"y"),
        np.broadcast_to(np.diag(1/resistance),(2,4,4)),rtol=1e-13,atol=1e-14)
    expected=np.diag((resistance-50)/(resistance+50))
    np.testing.assert_allclose(unpack(measured,"s"),
        np.broadcast_to(expected,(2,4,4)),rtol=1e-13,atol=1e-14)

@pytest.fixture
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
    assert peak<6*2**30


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("lossy,substrate,thin",
    ((False,False,False),(False,True,False),(True,False,False),(True,True,False),(True,False,True)))
def test_fb_coarse_native_smoke(lossy,substrate,thin,gpu_budget,record_property):
    first=None
    for repeat in range(3):
        result=fixture_fb(lossy=lossy,substrate=substrate,thin=thin).simulation((1e9,)).run()
        network=result.network
        s=network.s
        currents=mw.mom.surface_currents(result).currents
        assert torch.isfinite(currents).all()
        assert (s-s.transpose(-1,-2)).abs().max()<1e-9
        assert torch.linalg.matrix_norm(s,ord=2).max()<=1+1e-9
        assert network.metadata["unknowns"]==frozen()["native_meshes"]["test"]["unknowns"]<=6000
        assert network.metadata["mesh_quality"]["min_angle_degrees"]>1.
        if first is None:
            first=s.clone(),currents.clone()
            record_property("metadata",json.dumps(network.metadata))
            measured=winding_metrics(network.frequencies.cpu().numpy(),network.to_z().cpu().numpy())
            assert measured["L1_H"][0]>0 and measured["L2_H"][0]>0
            assert 0<measured["k"][0]<1
            record_property("winding_metrics",json.dumps(measured))
        else:
            assert torch.equal(first[0],s)
            assert torch.equal(first[1],currents)
        del result,network,s,currents
        gc.collect()
        torch.cuda.empty_cache()


@requires_cuda
@pytest.mark.cuda
def test_fb_fine_dense_admission(gpu_budget,record_property):
    """Dense admission of the full F-B mesh follows the memory cap that is actually set.

    Admission refuses one frequency when its footprint (dense system, workspace and chunk
    scratch) exceeds 80% of the available device memory, before any dense operator is
    allocated. With the PH1 footprint the full mesh fits a full A6000 but not a 6 GiB test
    cap, so the test checks whichever branch the current cap selects instead of assuming one.
    """
    import re
    from witwin.maxwell.mom.layered.network import _wall_nodes
    from witwin.maxwell.mom.layered.operator import dense_chunk_bytes, dense_frequency_bytes
    full=frozen()["native_meshes"]["full"]
    unknowns,faces,ports=full["unknowns"],full["faces"],4
    device=torch.device("cuda",torch.cuda.current_device())
    cap=torch.cuda.get_per_process_memory_fraction(device)*torch.cuda.get_device_properties(device).total_memory
    try:
        prepared=fixture_fb(mesh="full").simulation((1e9,)).prepare()
    except RuntimeError as error:
        match=re.fullmatch(r"One dense layered frequency needs ~([0-9.]+) GiB plus workspace; ([0-9.]+) GiB available",
                           str(error))
        assert match,str(error)
        needed,available=(float(value)*2**30 for value in match.groups())
        # The refusal rule (frequency_chunk_capacity), within the two-decimal rounding of the message.
        assert needed>.8*available-.01*2**30
        assert needed>=dense_frequency_bytes(unknowns,faces,ports)-.01*2**30
        assert available<=cap+.01*2**30
        record_property("admission",dict(refused=str(error),cap_GiB=cap/2**30))
        return
    network=prepared.solver.prepared
    assert (network.metadata["unknowns"],network.metadata["faces"])==(unknowns,faces)
    needed=dense_frequency_bytes(unknowns,faces,ports)+dense_chunk_bytes(faces,_wall_nodes(network.table))
    # Admitted: one frequency fits 80% of the memory the process may use.
    assert needed<=.8*cap
    record_property("admission",dict(admitted_GiB=needed/2**30,cap_GiB=cap/2**30))
    del prepared,network
