"""Validate against read-only native artifacts; exit nonzero on any mismatch.

Run from the repository root with the maxwell environment. No GPU is accessed.
"""
import os
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from model import Cell, native_matrix, oscillatory_spectrum

EVIDENCE = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "instamax-line-modulated-growth-20261101"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,default=EVIDENCE)
    p.add_argument("--output",type=Path)
    p.add_argument("--variants",action="store_true",help="also reproduce four independent variant-probe radii")
    p.add_argument("--candidate-check",action="store_true",help="also check candidates on the full native-basis projection")
    args = p.parse_args()
    rows=[]
    for pol,q,tag,target in (("TM",1,"q1_TM",1.0001352599793165),
                             ("TE",1,"q1",.999963662082867),
                             ("TE",0,"q0",.9999480673476152)):
        cell = Cell(polarization=pol,order=q)
        native = np.load(args.evidence/f"native_matrix_default_{tag}.npy")
        model64 = native_matrix(cell,rounded=False)
        replay = native_matrix(cell,rounded=True)
        spec64 = oscillatory_spectrum(model64,cell.dt)
        spec32 = oscillatory_spectrum(replay,cell.dt)
        # Absolute tolerance is tied to binary32 basis/operation roundoff; a
        # relative tolerance alone cannot check nominally zero matrix entries.
        entry_error = float(np.max(np.abs(replay-native)))
        entry_error64 = float(np.max(np.abs(model64-native)))
        row = dict(case=tag,reference_rho=target,float64=spec64,float32_replay=spec32,
                   float64_rho_error=abs(spec64["rho"]-target),
                   float64_max_entry_error=entry_error64,
                   replay_max_entry_error=entry_error,
                   replay_frobenius_error=float(np.linalg.norm(replay-native)),
                   passed=bool(abs(spec64["rho"]-target)<1e-8 and
                               abs(spec32["rho"]-target)<1e-8 and entry_error<6e-8 and
                               np.allclose(model64,native,rtol=1.2e-7,atol=6e-8)))
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(passed=all(r["passed"] for r in rows),cases=rows,
                python=sys.executable,numpy=np.__version__)
    root=Path(__file__).resolve().parents[4]
    paths=[p for p in args.evidence.glob("native_matrix_default_*.npy") if p.stem in
           ("native_matrix_default_q1_TM","native_matrix_default_q1","native_matrix_default_q0")]
    paths += [args.evidence/"native_matrix.py",args.evidence/"REPORT.md"]
    paths += [root/"witwin/maxwell/fdtd"/p for p in ("boundary/cpml.py","cuda/kernels/electric.cu","cuda/kernels/magnetic.cu")]
    report["input_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if args.variants:
        variants=[]
        for name,cell,target in (("scale1",Cell(alpha_scale=1.,grading=2.),1+2.4641691342752736e-5),
            ("scale4",Cell(alpha_scale=4.,grading=2.),1+5.802260139020632e-8),
            ("pecq1",Cell(wall="pec"),1+5.60227506398725e-5),
            ("pmcq1",Cell(wall="pmc"),1+5.503727780009626e-5)):
            spec=oscillatory_spectrum(native_matrix(cell),cell.dt)
            row=dict(case=name,reference_rho=target,model=spec,error=spec["rho"]-target,
                     passed=bool(abs(spec["rho"]-target)<1e-8))
            variants.append(row)
            print(json.dumps(row),flush=True)
        report["variants"]=variants
        report["passed"] &= all(r["passed"] for r in variants)
    if args.candidate_check:
        report["native_basis_candidates"]={}
        for pol in ("TE","TM"):
            cell=Cell(polarization=pol)
            report["native_basis_candidates"][pol]={c:oscillatory_spectrum(
                native_matrix(cell,rounded=False,candidate=c),cell.dt)
                for c in ("c1","c3_old","c3_average")}
        print(json.dumps(report["native_basis_candidates"]),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
