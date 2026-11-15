"""F-A shield geometry, deterministic GPU contracts and independent reference gates.

Reference acquisition runs outside pytest. All tolerances below are the R6
brief targets; unresolved physical discrepancies are strict xfails tracked by R6b.
"""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from tests.support.rfic_fixtures import fixture_fa
from tests.support.rfic_r6_references import fit_pi,pi_admittance,metrics,unpack_network,geometry_digest
from witwin.maxwell.mom.layered.runtime import compile_layered_scene
from witwin.maxwell.mom.layered.mesh import validate_vias

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


@pytest.fixture(autouse=True)
def budget(record_property):
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
    # The 120 s wall budget is checked on an idle card by tests/perf/test_layered_mom_wall_budgets.py.
    assert peak<6*2**30


def test_r6_grounded_geometry():
    original=fixture_fa()
    for variant in ("pgs","solid"):
        fixture=fixture_fa(shield=variant,mesh="shield")
        stack,rectangles,ports,materials,_,vias,_=compile_layered_scene(fixture.scene)
        validate_vias(rectangles,vias)
        assert len(vias)==6
        assert [(p.positive,p.negative) for p in ports]==[(p.positive,p.negative) for p in original.scene.ports]
        shield=[(r,m) for r,m in zip(rectangles,materials) if r[4]==300.5e-6]
        assert shield
        assert all(abs(1/(m[0]*m[1])-10)<1e-12 for _,m in shield)
        assert all(v[5]-v[4]<3.01e-6 for v in vias)
        # No two via spans share an identical footprint at the middle sheet.
        assert len({v[:4] for v in vias})==len(vias)
        assert stack.layers[0].sigma==10.


def test_r6_pi_known_components():
    frequencies=np.geomspace(.5e9,8e9,40)
    components=np.array([3.3e-9,3.,5e-15,80e-15,30e-15,1000.,100e-15,40e-15,800.])
    fitted=fit_pi(frequencies,pi_admittance(frequencies,components))
    assert fitted["success"]
    assert max(fitted["relative_branch_rms"])<1e-8
    assert np.max(np.abs(np.array(list(fitted["parameters"].values()))/components-1))<1e-5


@pytest.mark.parametrize("variant",("pgs","solid"))
def test_r6_complete_repeat(variant,record_property):
    first=None
    for repeat in range(3):
        result=fixture_fa(shield=variant,mesh="shield").simulation((3.419951893e9,)).run()
        s=result.network.s
        if first is None:
            first=s.clone()
            record_property("unknowns",result.network.metadata["unknowns"])
            record_property("extraction",json.dumps(metrics(result.network.frequencies.cpu().numpy(),
                                                           result.network.to_y().cpu().numpy())))
        else:
            assert torch.equal(first,s)
        assert torch.linalg.matrix_norm(s,ord=2).max()<=1+1e-9
        del result,s
        gc.collect()
        torch.cuda.empty_cache()


def _frozen(name):
    return json.loads(Path(__file__).with_name("data").joinpath(name).read_text())


def _native_recording():
    # Values are locked numerically (D-P1): test_layered_evidence_lock checks the lock
    # records and scripts/dev/rfic/evidence_lock.py re-acquires and judges each field.
    return _frozen("fixture_r6_native_evidence.json")


def _check_live_coarse(variant,native,record_property):
    expected=native["coarse"][variant]
    frequencies=(1e9,expected["metrics"]["peak_frequency_Hz"])
    result=fixture_fa(shield=variant,mesh=expected["mesh"]).simulation(frequencies).run()
    measured=metrics(result.network.frequencies.cpu().numpy(),result.network.to_y().cpu().numpy())
    unknowns=result.network.metadata["unknowns"]
    changes={key:abs(measured[key]/expected["metrics"][key]-1) for key in ("L_1GHz_H","Q_peak")}
    del result
    gc.collect()
    torch.cuda.empty_cache()
    record_property("live_coarse_"+variant,json.dumps(dict(relative_changes=changes,unknowns=unknowns)))
    assert unknowns==expected["unknowns"]
    # This binds cached fine evidence to a live coarse solve, not a physics tolerance.
    assert max(changes.values())<1e-6


def test_r6_frozen_reference_geometry():
    reference=_frozen("fixture_r6_reference.json")
    native=_native_recording()
    targets=dict(L_1GHz_H=.03,Q_peak=.10,peak_frequency_Hz=.10,self_resonance_Hz=.05)
    assert reference["targets"]==targets
    assert reference["convention"]=="exp(-i omega t)"
    for variant in ("none","pgs","solid"):
        for level in ("coarse","finest"):
            evidence=native[level][variant]
            fixture=fixture_fa(shield=variant,mesh=evidence["mesh"])
            stack,rectangles,ports,materials,_,vias,via_materials=compile_layered_scene(fixture.scene)
            digest=geometry_digest(rectangles,materials,vias,via_materials)
            assert digest==evidence["geometry_sha256"]
            assert json.loads(json.dumps(fixture.config.layered.payload()))==evidence["config"]
            declared_ports=[dict(name=p.name,terminal_pair_um=[[round(float(v)*1e6,6) for v in point]
                            for point in (p.positive,p.negative)],axis=p.voltage_path.axis,
                            reference_impedance=p.reference_impedance) for p in ports]
            assert declared_ports==evidence["ports"]
            for resolution in (5,10):
                ref=reference["reference_variants"][f"{variant}/{resolution}"]
                assert ref["geometry_sha256"]==digest
                assert ref["ports"]==declared_ports
                assert ref["dielectrics"]["silicon_epsilon_r"]==stack.layers[0].eps_r
                assert ref["dielectrics"]["silicon_sigma_S_m"]==stack.layers[0].sigma
                assert ref["dielectrics"]["oxide_epsilon_r"]==stack.layers[1].eps_r
                assert ref["dielectrics"]["silicon_top_um"]==300.
                assert ref["dielectrics"]["oxide_top_um"]==306.
                y=unpack_network(ref["samples"])
                assert y.shape==(11,2,2)
                assert np.isfinite(y).all()


# Sheet model (locked evidence). Strict: an XPASS fails and forces removal of the marker.
OPEN_GAP=("conductor_model='sheet' (the frozen native evidence): L +6.1%, Q peak +28.8% vs the 5-um reference. "
          "Attributed (R6-1, exact pi-branch swaps): series R at the peak 3.09 vs 3.77 ohm (no lateral edge/"
          "proximity crowding) carries 2.7-2.9 of the 3.65 Q gap, the shunt branch 0.7-1.0; L +5-6% is the sheet's "
          "missing conductor thickness (R6b volume reference). The default model for this copper, "
          "surface_admittance, meets both targets: test_r6_surface_admittance_no_shield_targets")


def _no_shield_errors(native,reference):
    ours=native["finest"]["none"]["metrics"]
    ref=reference["reference_variants"]["none/5"]["metrics"]
    targets=dict(L_1GHz_H=.03,Q_peak=.10,peak_frequency_Hz=.10,self_resonance_Hz=.05)
    return {key:abs(ours[key]/ref[key]-1) for key in targets},targets,ours,ref


def test_r6_no_shield_resonance_targets(record_property):
    reference=_frozen("fixture_r6_reference.json")
    native=_native_recording()
    _check_live_coarse("none",native,record_property)
    errors,targets,ours,ref=_no_shield_errors(native,reference)
    record_property("finest_native",json.dumps(ours))
    record_property("independent_reference",json.dumps(ref))
    record_property("relative_errors",json.dumps(errors))
    assert all(errors[key]<=targets[key] for key in ("peak_frequency_Hz","self_resonance_Hz")),errors


@pytest.mark.xfail(strict=True,reason=OPEN_GAP)
def test_r6_no_shield_inductance_and_q_targets(record_property):
    errors,targets,_,_=_no_shield_errors(_frozen("fixture_r6_native_evidence.json"),_frozen("fixture_r6_reference.json"))
    record_property("relative_errors",json.dumps(errors))
    assert all(errors[key]<=targets[key] for key in ("L_1GHz_H","Q_peak")),errors


def _pgs_delta_q(native,reference):
    delta=native["finest"]["pgs"]["metrics"]["Q_peak"]-native["finest"]["none"]["metrics"]["Q_peak"]
    deltas={r:reference["reference_variants"][f"pgs/{r}"]["metrics"]["Q_peak"]-
              reference["reference_variants"][f"none/{r}"]["metrics"]["Q_peak"] for r in (5,10)}
    return delta,deltas


def test_r6_pgs_direction_and_attribution(record_property):
    # The fixture's 10-ohm/sq shield 0.5 um above silicon lowers Q in both solvers, so the
    # brief's expected Q gain is not a property of this geometry. The gates are agreement in
    # direction with the independent reference, a small L change, and the pi attribution: the
    # Cox-coupled effective shunt conductance increases.
    reference=_frozen("fixture_r6_reference.json")
    native=_native_recording()
    for variant in ("none","pgs"):
        _check_live_coarse(variant,native,record_property)
    base=native["finest"]["none"]["metrics"]
    shield=native["finest"]["pgs"]["metrics"]
    delta,deltas=_pgs_delta_q(native,reference)
    l_change=abs(shield["L_1GHz_H"]/base["L_1GHz_H"]-1)
    ref_l_change=abs(reference["reference_variants"]["pgs/5"]["metrics"]["L_1GHz_H"]/
                     reference["reference_variants"]["none/5"]["metrics"]["L_1GHz_H"]-1)
    conductance=native["pi_effective_shunt_conductance_3p4GHz_S"]
    gates=dict(same_direction=(delta<0)==(deltas[5]<0)==(deltas[10]<0),
               native_small_L_change=l_change<.02,reference_small_L_change=ref_l_change<.02,
               fitted_effective_shunt_conductance_increases=conductance["pgs"]>conductance["none"])
    record_property("comparison",json.dumps(dict(native_delta_Q=delta,reference_delta_Q=deltas,
                    L_relative=l_change,reference_L_relative=ref_l_change,pi_shunt_G_S=conductance,gates=gates)))
    assert all(gates.values()),gates


@pytest.mark.xfail(strict=True,reason="conductor_model='sheet': PGS delta-Q -3.60 vs reference -2.51 (allowance 0.35). "
                   "Attributed (R6-1): the mismatch -1.08 is the difference of the per-variant Q gaps, series-R "
                   "term -1.44 and shunt term +0.36 (one relative R error, larger absolute Q error on the higher-Q "
                   "unshielded coil). surface_admittance refuses the poly-shield/copper-via contacts; with its "
                   "unshielded series change transferred, delta-Q is about -1.14 and the mismatch +1.38 is then "
                   "shunt dominated (+1.58: shield/substrate coupling, where the 5-um reference has a documented "
                   "grid bias), series -0.20")
def test_r6_pgs_delta_q_magnitude(record_property):
    delta,deltas=_pgs_delta_q(_frozen("fixture_r6_native_evidence.json"),_frozen("fixture_r6_reference.json"))
    allowance=abs(deltas[10]-deltas[5])+.1*abs(deltas[5])
    record_property("delta_Q",json.dumps(dict(native=delta,reference=deltas,allowance=allowance)))
    assert abs(delta-deltas[5])<=allowance


def test_r6_solid_negative_control(record_property):
    reference=_frozen("fixture_r6_reference.json")
    native=_native_recording()
    for variant in ("none","solid"):
        _check_live_coarse(variant,native,record_property)
    ours={v:native["finest"][v]["metrics"] for v in ("none","solid")}
    ref={v:reference["reference_variants"][f"{v}/5"]["metrics"] for v in ("none","solid")}
    gates={f"{backend}_{quantity}_drops":data["solid"][quantity]<data["none"][quantity]
           for backend,data in (("native",ours),("reference",ref)) for quantity in ("L_1GHz_H","Q_peak")}
    record_property("negative_control",json.dumps(dict(native=ours,reference=ref,gates=gates)))
    assert all(gates.values()),gates


def _surface_admittance_evidence():
    # R6-1: F-A no shield with conductor_model="surface_admittance" (the auto default for this copper),
    # acquired outside pytest by scripts/dev/rfic/surface_admittance_fixture_r6.py (A100, 20/5-um mesh).
    return _frozen("fixture_r6_surface_admittance_evidence.json")


def _q11(y):
    zin=1/y[:,0,0]
    return -zin.imag/zin.real


def test_r6_surface_admittance_evidence_record():
    evidence=_surface_admittance_evidence()
    native=_native_recording()
    fixture=fixture_fa(mesh=evidence["mesh"],conductor_model="surface_admittance")
    _,rectangles,_,materials,_,vias,via_materials=compile_layered_scene(fixture.scene)
    assert geometry_digest(rectangles,materials,vias,via_materials)==evidence["geometry_sha256"]
    assert evidence["geometry_sha256"]==native["finest"]["none"]["geometry_sha256"]
    assert json.loads(json.dumps(fixture.config.layered.payload()))==evidence["config"]
    samples=evidence["samples"]
    y=unpack_network(samples)
    measured=metrics(np.array(samples["frequency_Hz"]),y)
    assert {k:measured[k] for k in evidence["metrics"]}==evidence["metrics"]
    assert evidence["max_singular_S"]<=1
    # The DSA body blocks are not symmetric (A1: 3-5% per block); A1 recorded 5e-5 network
    # nonreciprocity on F-B. Measured here: 1.1e-5.
    assert evidence["max_relative_Y_nonreciprocity"]<5e-5


def test_r6_surface_admittance_no_shield_targets(record_property):
    reference=_frozen("fixture_r6_reference.json")["reference_variants"]["none/5"]["metrics"]
    ours=_surface_admittance_evidence()["metrics"]
    targets=dict(L_1GHz_H=.03,Q_peak=.10,peak_frequency_Hz=.10,self_resonance_Hz=.05)
    errors={key:abs(ours[key]/reference[key]-1) for key in targets}
    record_property("relative_errors",json.dumps(errors))
    assert all(errors[key]<=targets[key] for key in targets),errors


def test_r6_q_gap_attribution_at_reference_samples(record_property):
    """Exact pi-branch swaps at the frozen 3.0/3.4-GHz reference samples (Q11 there, not the peak).

    Q11 depends only on the series branch Ys=-(Y12+Y21)/2 and the port-1 shunt Y11+Y21. Giving the sheet
    network the reference's series resistance closes most of its Q gap; surface_admittance needs no swap.
    """
    reference=_frozen("fixture_r6_reference.json")["reference_variants"]["none/5"]["samples"]
    evidence=_surface_admittance_evidence()
    f_ref=np.array(reference["frequency_Hz"])
    rows={}
    for name,samples in (("sheet",evidence["sheet_full_mesh_samples"]),("surface_admittance",evidence["samples"])):
        f=np.array(samples["frequency_Hz"])
        for frequency in (3.0e9,3.4e9):
            y=unpack_network(samples)[f==frequency]
            r=unpack_network(reference)[f_ref==frequency]
            ys,yr=-(y[:,0,1]+y[:,1,0])/2,-(r[:,0,1]+r[:,1,0])/2
            zs,zr=1/ys,1/yr
            swapped=y.copy()
            series=1/(zr.real+1j*zs.imag)
            # Replace the series branch in Y: Y11=shunt1+Ys, Y12=Y21=-Ys.
            swapped[:,0,0]+=series-ys
            swapped[:,1,0]=swapped[:,0,1]=-series
            q,qr,qs=_q11(y)[0],_q11(r)[0],_q11(swapped)[0]
            rows[f"{name}@{frequency:.3g}"]=dict(Q=q,reference_Q=qr,Q_with_reference_R=qs,
                                                 gap=q/qr-1,gap_with_reference_R=qs/qr-1)
    record_property("attribution",json.dumps(rows))
    for frequency in (3.0e9,3.4e9):
        sheet,sa=rows[f"sheet@{frequency:.3g}"],rows[f"surface_admittance@{frequency:.3g}"]
        assert sheet["gap"]>.10                                  # the sheet model misses the 10% Q target
        assert abs(sheet["gap_with_reference_R"])<.5*sheet["gap"]  # series R carries most of it
        assert abs(sa["gap"])<.10                                # surface_admittance meets it
