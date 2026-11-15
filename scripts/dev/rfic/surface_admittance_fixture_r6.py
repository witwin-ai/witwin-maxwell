"""Acquire the F-A (R6, no shield) network with conductor_model="surface_admittance" (R6-1).

Evidence acquisition outside pytest (cloud A100 or the GPU-0 runner). Frequencies
run in batches; every batch writes one JSON (Y, S, metrics of that batch, cost)
under OUT so a partial run keeps its completed batches. ``merge`` (CPU) joins the
batch files of one mesh into OUT/<variant>-<mesh>-surface_admittance.json with the
R6 metrics over all samples.

  python scripts/dev/rfic/surface_admittance_fixture_r6.py acquire OUT --mesh test --frequencies 1e9 3.2e9
  python scripts/dev/rfic/surface_admittance_fixture_r6.py merge OUT --mesh test
  python scripts/dev/rfic/surface_admittance_fixture_r6.py campaign   (OUT = $RFIC_OUT; cloud A100)
  python scripts/dev/rfic/surface_admittance_fixture_r6.py freeze MERGED.json --sheet S.json.. --attribution A.json --target T.json

campaign (--meshes, default test): mesh "test" (20/5 um) at 1 GHz, the Q-peak bands (unshielded and PGS) and the SRF band, then
mesh "full" (10/4.9 um, the sheet model's finest R6 mesh) at 1 / 3.3 GHz.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import os

CAMPAIGN = (("test", (1e9, 2.2e9, 2.4e9, 2.6e9, 2.8e9, 3.0e9, 3.2e9, 3.3e9, 3.4e9, 3.6e9, 14e9, 14.5e9, 15e9)),
            ("full", (1e9, 3.3e9)))


def acquire(out, variant, mesh, frequencies, batch, lossy, cap_gib=None):
    import torch
    if cap_gib is not None:
        torch.cuda.set_per_process_memory_fraction(cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    import witwin.maxwell as mw
    from tests.support.rfic_fixtures import fixture_fa
    from tests.support.rfic_r6_references import metrics
    assert mw.__file__.startswith(str(ROOT)), mw.__file__
    print("witwin.maxwell", mw.__file__, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(frequencies), batch):
        chunk = tuple(frequencies[start:start + batch])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fixture = fixture_fa(shield=variant, mesh=mesh, lossy=lossy)
        config = replace(fixture.config, layered=replace(fixture.config.layered, conductor_model="surface_admittance"))
        started = time.perf_counter()
        result = mw.Simulation.mom(fixture.scene, frequencies=chunk, config=config).run()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        network = result.network
        y = network.to_y().cpu().numpy()
        s = network.s.cpu().numpy()
        f = network.frequencies.cpu().numpy()
        data = dict(variant=variant, mesh=mesh, lossy_metal=lossy, conductor_model=network.metadata["conductor_model"],
                    frequency_Hz=f.tolist(), y_real=y.real.tolist(), y_imag=y.imag.tolist(),
                    s_real=s.real.tolist(), s_imag=s.imag.tolist(),
                    unknowns=network.metadata["unknowns"], faces=network.metadata["faces"],
                    seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    max_singular_S=float(np.linalg.svd(s, compute_uv=False).max()),
                    max_relative_Y_nonreciprocity=float(np.max(abs(y[:, 0, 1]-y[:, 1, 0])/abs(y[:, 0, 1]))),
                    regime=network.metadata.get("regime"),
                    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                   ("tests/support/rfic_fixtures.py", "witwin/maxwell/mom/layered/admittance.py",
                                    "witwin/maxwell/mom/layered/network.py", "witwin/maxwell/mom/layered/runtime.py")})
        if len(f) > 1:
            data["metrics_batch"] = {k: v for k, v in metrics(f, y).items() if not isinstance(v, list)}
        name = f"{variant}-{mesh}-{'cu' if lossy else 'pec'}-sa-{start:03d}.json"
        (out/name).write_text(json.dumps(data, indent=1, default=str)+"\n")
        series = -1/y[:, 1, 0]
        print(json.dumps(dict(file=name, unknowns=data["unknowns"], seconds=round(seconds, 1),
                              peak_GiB=round(data["peak_allocated_bytes"]/2**30, 2),
                              L_nH=(-series.imag/(2*np.pi*f)*1e9).round(6).tolist(),
                              R_ohm=series.real.round(6).tolist(),
                              Q=(-(1/y[:, 0, 0]).imag/(1/y[:, 0, 0]).real).round(5).tolist())), flush=True)
        del result, network
        gc.collect()


def merge(out, variant, mesh, lossy):
    from tests.support.rfic_r6_references import metrics
    tag = f"{variant}-{mesh}-{'cu' if lossy else 'pec'}-sa"
    parts = [json.loads(p.read_text()) for p in sorted(out.glob(f"{tag}-[0-9][0-9][0-9].json"))]
    f = np.concatenate([p["frequency_Hz"] for p in parts])
    y = np.concatenate([np.array(p["y_real"])+1j*np.array(p["y_imag"]) for p in parts])
    order = np.argsort(f)
    f, y = f[order], y[order]
    assert len(set(f.tolist())) == len(f)
    assert len({p["unknowns"] for p in parts}) == 1
    data = dict(variant=variant, mesh=mesh, lossy_metal=lossy, conductor_model="surface_admittance",
                unknowns=parts[0]["unknowns"], frequency_Hz=f.tolist(), y_real=y.real.tolist(),
                y_imag=y.imag.tolist(), metrics=metrics(f, y),
                max_peak_allocated_bytes=max(p["peak_allocated_bytes"] for p in parts),
                max_singular_S=max(p["max_singular_S"] for p in parts),
                max_relative_Y_nonreciprocity=max(p["max_relative_Y_nonreciprocity"] for p in parts),
                sources=sorted(p.name for p in out.glob(f"{tag}-[0-9][0-9][0-9].json")))
    (out/f"{tag}.json").write_text(json.dumps(data, indent=1)+"\n")
    print(json.dumps({k: data["metrics"][k] for k in ("L_1GHz_H", "Q_peak", "peak_frequency_Hz", "self_resonance_Hz")}))


def freeze(merged, sheet, attribution, target):
    """Committed evidence: merged surface_admittance samples, metrics, config/geometry record and the
    sheet samples at the frozen reference's 3.0/3.4 GHz points (pi-swap attribution test)."""
    from tests.support.rfic_fixtures import fixture_fa
    from tests.support.rfic_r6_references import geometry_digest, metrics
    from witwin.maxwell.mom.layered.runtime import compile_layered_scene
    data = json.loads(Path(merged).read_text())
    f = np.array(data["frequency_Hz"])
    y = np.array(data["y_real"]) + 1j*np.array(data["y_imag"])
    fixture = fixture_fa(mesh=data["mesh"], conductor_model="surface_admittance")
    _, rectangles, _, materials, _, vias, via_materials = compile_layered_scene(fixture.scene)
    sheet_f, sheet_y = [], []
    for path in sheet:
        part = json.loads(Path(path).read_text())
        for i, frequency in enumerate(part["frequency_Hz"]):
            if frequency in (3.0e9, 3.4e9) and frequency not in sheet_f:
                sheet_f.append(frequency)
                sheet_y.append(np.array(part["y_real"][i]) + 1j*np.array(part["y_imag"][i]))
    order = np.argsort(sheet_f)
    sheet_y = np.array(sheet_y)[order]
    report = json.loads(Path(attribution).read_text())
    keep = ("native", "reference_cu5", "native_series_from_reference", "native_shunt_from_reference",
            "reference_cu5_plus_pec_grid_shift_series_and_shunt")
    evidence = dict(
        description="F-A no shield, conductor_model='surface_admittance' (R6-1). Acquired outside pytest "
                    "(scripts/dev/rfic/surface_admittance_fixture_r6.py); metrics over every sample.",
        mesh=data["mesh"], unknowns=data["unknowns"],
        config=json.loads(json.dumps(fixture.config.layered.payload())),
        geometry_sha256=geometry_digest(rectangles, materials, vias, via_materials),
        samples=dict(frequency_Hz=f.tolist(), y_real=y.real.tolist(), y_imag=y.imag.tolist()),
        metrics={k: v for k, v in metrics(f, y).items() if k in ("L_1GHz_H", "Q_peak", "peak_frequency_Hz",
                                                                    "self_resonance_Hz")},
        max_singular_S=data["max_singular_S"], max_relative_Y_nonreciprocity=data["max_relative_Y_nonreciprocity"],
        sources=data["sources"],
        sheet_full_mesh_samples=dict(frequency_Hz=sorted(sheet_f), y_real=sheet_y.real.tolist(),
                                     y_imag=sheet_y.imag.tolist()),
        attribution={k: {q: report["cases"][k][q] for q in ("L_1GHz_H", "Q_peak", "peak_frequency_Hz")}
                     for k in keep})
    Path(target).write_text(json.dumps(evidence, indent=1)+"\n")
    print(json.dumps(evidence["metrics"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("acquire", "merge", "campaign", "freeze"))
    parser.add_argument("--sheet", nargs="+", help="freeze: sheet sample files (3.0/3.4 GHz)")
    parser.add_argument("--attribution", help="freeze: attribution_fixture_r6.py report")
    parser.add_argument("--target", help="freeze: committed evidence file")
    parser.add_argument("out", type=Path, nargs="?", default=Path(os.environ.get("RFIC_OUT", "")))
    parser.add_argument("--variant", default="none", choices=("none", "pgs", "solid"))
    parser.add_argument("--mesh", default="test")
    parser.add_argument("--frequencies", type=float, nargs="+")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--pec-metal", action="store_true")
    parser.add_argument("--cap-gib", type=float)
    parser.add_argument("--meshes", nargs="+", default=["test"], help="campaign meshes (test, full)")
    args = parser.parse_args()
    assert str(args.out), "OUT or $RFIC_OUT is required"
    if args.mode == "campaign":
        for mesh, frequencies in (c for c in CAMPAIGN if c[0] in args.meshes):
            acquire(args.out, args.variant, mesh, frequencies, args.batch, not args.pec_metal, args.cap_gib)
            merge(args.out, args.variant, mesh, not args.pec_metal)
    elif args.mode == "freeze":
        freeze(args.out, args.sheet, args.attribution, args.target)
    elif args.mode == "acquire":
        acquire(args.out, args.variant, args.mesh, args.frequencies, args.batch, not args.pec_metal, args.cap_gib)
    else:
        merge(args.out, args.variant, args.mesh, not args.pec_metal)


if __name__ == "__main__":
    main()
