"""Measure an external precomputation executable without modifying its sources."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--frequency", type=float, default=30e9)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--cpus", default="0-27")
    parser.add_argument("--schur-size", type=int, default=8066)
    parser.add_argument("--label", default="30ghz-28t")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    prefix = args.work / args.label
    matrix = prefix.with_suffix(".bin").resolve()
    if matrix.exists() or prefix.with_suffix(".json").exists():
        raise FileExistsError("Choose a new label to preserve earlier measurements")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.blas_threads),
               OMP_PROC_BIND="close", OMP_PLACES="cores",
               OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
    command = ["/usr/bin/time", "-v", "-o", str(prefix.with_suffix(".time.txt")),
               "taskset", "-c", args.cpus, str(args.executable.resolve()),
               str(matrix), str(args.frequency), str(args.threads)]
    report = {
        "command": command, "frequency_hz": args.frequency,
        "omp_threads": args.threads, "openblas_threads": args.blas_threads,
        "cpu_affinity": args.cpus, "schur_size": args.schur_size,
        "runtime_environment": {key: env.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
            "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS", "LD_PRELOAD", "LD_LIBRARY_PATH")},
        "executable_sha256": digest(args.executable),
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_status": subprocess.check_output(
            ["git", "-C", str(args.upstream), "status", "--porcelain"], text=True),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "linked_libraries": subprocess.check_output(["ldd", str(args.executable)], text=True),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Process launch through exit, including assembly, APF, output write and teardown; excludes build and output validation. No optimizer calls.",
    }
    print(f"Running {args.label}; log: {prefix.with_suffix('.log')}", flush=True)
    with prefix.with_suffix(".log").open("w") as log:
        start = time.perf_counter()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        report["process_wall_s"] = time.perf_counter() - start
    report["exit_code"] = result.returncode
    log = prefix.with_suffix(".log").read_text()
    for field, pattern in {
        "apf_analysis_s": r"MUMPS APF analyze time elapsed:\s*([\d.eE+-]+)",
        "apf_factorization_s": r"MUMPS APF factorize time elapsed:\s*([\d.eE+-]+)",
        "apf_solve_s": r"APF solve time:\s*([\d.eE+-]+)",
    }.items():
        match = re.search(pattern, log)
        report[field] = float(match[1]) if match else None
    usage = prefix.with_suffix(".time.txt").read_text()
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", usage)
    report["peak_rss_kib"] = int(match[1]) if match else None
    expected_bytes = args.schur_size**2 * np.dtype("complex128").itemsize
    report["output_expected_bytes"] = expected_bytes
    report["output_bytes"] = matrix.stat().st_size if matrix.exists() else 0
    report["output_finite"] = False
    if report["output_bytes"] == expected_bytes:
        values = np.memmap(matrix, dtype="complex128", mode="r")
        report["output_finite"] = all(bool(np.isfinite(values[i:i + 1024**2]).all())
                                      for i in range(0, values.size, 1024**2))
        report["output_max_abs"] = max(float(np.abs(values[i:i + 1024**2]).max())
                                       for i in range(0, values.size, 1024**2))
        report["output_sha256"] = digest(matrix)
    report["valid_measurement"] = (result.returncode == 0 and report["output_finite"]
                                   and report["apf_factorization_s"] is not None
                                   and "An error has occured" not in log)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "process_wall_s", "apf_analysis_s", "apf_factorization_s", "apf_solve_s",
        "peak_rss_kib", "valid_measurement")}, indent=2), flush=True)
    if not report["valid_measurement"]:
        raise RuntimeError("Precomputation failed; inspect the retained log and report")


if __name__ == "__main__":
    main()
