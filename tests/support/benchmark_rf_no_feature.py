"""Reproducible single-device FDTD timing with every RF feature disabled.

The ``sample`` command times one checkout.  The ``compare`` command launches
the sampler against two checkouts in ABBA order so both revisions see similar
thermal and clock conditions.  Timing uses CUDA events after an untimed warmup;
the ordinary pytest suite deliberately does not assert a wall-clock threshold.
Comparison baselines must be exact non-Git extractions of ``--baseline-archive``;
Git-checkout baselines and every extra extracted file are rejected.

Examples::

    python tests/support/benchmark_rf_no_feature.py sample --label candidate
    python tests/support/benchmark_rf_no_feature.py compare \
        --baseline-root E:/tmp/maxwell-baseline \
        --candidate-root E:/Code/witwin-platform/maxwell \
        --baseline-commit 0a69fc8 \
        --baseline-archive E:/tmp/maxwell-baseline.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 2
FREQUENCY_HZ = 1.0e9


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_metadata(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    repository_text = _git_output(resolved_root, "rev-parse", "--show-toplevel")
    repository_root = None if repository_text is None else Path(repository_text).resolve()
    is_checkout_root = repository_root == resolved_root
    status = (
        _git_output(resolved_root, "status", "--short", "--untracked-files=no")
        if is_checkout_root
        else None
    )
    return {
        "resolved_root": str(resolved_root),
        "repository_root": None if repository_root is None else str(repository_root),
        "is_checkout_root": is_checkout_root,
        "commit": _git_output(resolved_root, "rev-parse", "HEAD") if is_checkout_root else None,
        "dirty_tracked_files": bool(status) if is_checkout_root else None,
        "tracked_status": status.splitlines() if status else [],
    }


def _is_generated_untracked_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return True
    if parts[0].lower() in {
        "build",
        ".build_phase4",
        ".cuda_cache_phase4",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }:
        return True
    lowered = {part.lower() for part in parts}
    return bool(
        lowered.intersection({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})
        or PurePosixPath(path).suffix.lower() in {".pyc", ".pyo"}
    )


def _manifest_sha256(records: list[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"witwin-maxwell-manifest-v1\0")
    for record in records:
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _raw_git_blob_oid(path: Path, algorithm: str) -> str:
    size = path.stat().st_size
    digest = hashlib.new(algorithm)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_baseline_zip(
    archive: Path,
    entries: list[tuple[str, str, str]],
    algorithm: str,
) -> dict[str, Any]:
    expected = {path: (mode, object_id) for path, mode, object_id in entries}
    actual: dict[str, tuple[str, str]] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename).as_posix()
            archive_mode = (info.external_attr >> 16) & 0xFFFF
            file_type = archive_mode & 0o170000
            if file_type in {0, 0o100000}:
                git_mode = 0o100755 if archive_mode & 0o111 else 0o100644
            elif file_type == 0o120000:
                git_mode = 0o120000
            else:
                raise ValueError(
                    f"Baseline archive contains unsupported mode {archive_mode:06o} "
                    f"for {path!r}."
                )
            mode = f"{git_mode:06o}"
            digest = hashlib.new(algorithm)
            digest.update(f"blob {info.file_size}\0".encode("ascii"))
            with bundle.open(info) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            actual[path] = (mode, digest.hexdigest())
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        path for path in set(expected).intersection(actual) if expected[path] != actual[path]
    )
    if missing or extra or changed:
        raise ValueError(
            "Baseline archive does not match baseline_commit: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}."
        )
    return {
        "verified": True,
        **_sha256_file(archive),
        "file_count": len(actual),
    }


def _git_tree_entries(repository_root: Path, commit: str) -> list[tuple[str, str, str]]:
    payload = _git_bytes(repository_root, "ls-tree", "-r", "-z", "--full-tree", commit)
    if payload is None:
        raise ValueError(f"Could not read the Git tree for baseline commit {commit}.")
    entries: list[tuple[str, str, str]] = []
    for raw_entry in payload.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            raise ValueError(
                f"Baseline tree contains unsupported {object_type} entry "
                f"{raw_path.decode('utf-8', errors='replace')!r}."
            )
        path = raw_path.decode("utf-8", errors="surrogateescape")
        entries.append((path, mode, object_id))
    entries.sort()
    return entries


def _verify_baseline_snapshot(
    repository_root: Path,
    baseline_root: Path,
    commit: str,
    archive: Path | None = None,
) -> dict[str, Any]:
    resolved_baseline = baseline_root.resolve()
    algorithm = _git_output(repository_root, "rev-parse", "--show-object-format")
    if algorithm not in {"sha1", "sha256"}:
        raise ValueError(f"Unsupported Git object format: {algorithm!r}.")
    entries = _git_tree_entries(repository_root, commit)
    checkout = _git_metadata(resolved_baseline)
    if checkout["is_checkout_root"]:
        raise ValueError("baseline_root must be a non-Git directory extracted from baseline_archive.")
    if archive is None:
        raise ValueError("An extracted baseline requires --baseline-archive.")
    archive_binding = _verify_baseline_zip(archive.resolve(), entries, algorithm)
    expected_paths = {path for path, _, _ in entries}
    actual_paths = {
        path.relative_to(resolved_baseline).as_posix()
        for path in resolved_baseline.rglob("*")
        if (path.is_file() or path.is_symlink())
    }
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    present_entries = [entry for entry in entries if entry[0] not in missing]
    actual_object_ids = [
        _raw_git_blob_oid(
            resolved_baseline.joinpath(*PurePosixPath(relative).parts),
            algorithm,
        )
        for relative, _, _ in present_entries
    ]
    changed = []
    for (relative, _, expected_oid), actual_oid in zip(
        present_entries,
        actual_object_ids,
        strict=True,
    ):
        parts = PurePosixPath(relative).parts
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"Unsafe path in baseline Git tree: {relative!r}.")
        if actual_oid != expected_oid:
            changed.append(relative)
    if missing or extra or changed:
        raise ValueError(
            "Extracted baseline does not match baseline_commit: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}."
        )
    manifest_records = [(path, mode, object_id) for path, mode, object_id in entries]
    return {
        "verified": True,
        "resolved_root": str(resolved_baseline),
        "object_format": algorithm,
        "content_hash_method": "raw extracted bytes",
        "file_count": len(entries),
        "manifest_sha256": _manifest_sha256(manifest_records),
        "archive": archive_binding,
    }


def _worktree_content_metadata(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    git = _git_metadata(resolved_root)
    if not git["is_checkout_root"] or git["commit"] is None:
        raise ValueError("Content provenance requires the root of a Git checkout with a valid HEAD.")
    staged = _git_bytes(resolved_root, "ls-files", "--stage", "-z")
    untracked = _git_bytes(resolved_root, "ls-files", "--others", "--exclude-standard", "-z")
    if staged is None or untracked is None:
        raise ValueError("Could not enumerate candidate tracked and untracked files.")
    records: list[tuple[str, ...]] = []
    missing_tracked = []
    tracked_count = 0
    for raw_entry in staged.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, _, stage = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if stage != "0":
            raise ValueError(f"Candidate checkout has an unresolved index entry for {path!r}.")
        tracked_count += 1
        absolute = resolved_root.joinpath(*PurePosixPath(path).parts)
        if not (absolute.is_file() or absolute.is_symlink()):
            missing_tracked.append(path)
            records.append(("tracked", path, mode, "missing"))
            continue
        metadata_hash = _sha256_file(absolute)
        records.append(
            (
                "tracked",
                path,
                mode,
                str(metadata_hash["bytes"]),
                str(metadata_hash["sha256"]),
            )
        )
    relevant_untracked = sorted(
        raw_path.decode("utf-8", errors="surrogateescape")
        for raw_path in untracked.split(b"\0")
        if raw_path
        and not _is_generated_untracked_path(
            raw_path.decode("utf-8", errors="surrogateescape")
        )
    )
    for path in relevant_untracked:
        absolute = resolved_root.joinpath(*PurePosixPath(path).parts)
        if not (absolute.is_file() or absolute.is_symlink()):
            raise ValueError(f"Relevant untracked source disappeared during hashing: {path!r}.")
        metadata_hash = _sha256_file(absolute)
        records.append(
            (
                "untracked",
                path,
                str(metadata_hash["bytes"]),
                str(metadata_hash["sha256"]),
            )
        )
    records.sort()
    return {
        **git,
        "content_manifest": {
            "format": "witwin-maxwell-worktree-v1",
            "sha256": _manifest_sha256(records),
            "tracked_file_count": tracked_count,
            "missing_tracked_paths": missing_tracked,
            "relevant_untracked_paths": relevant_untracked,
        },
    }


def _sha256_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"SHA-256 input is not a file: {resolved}")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": size,
    }


def _tree_sha256(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Tree SHA-256 input is not a directory: {resolved}")
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    digest.update(b"witwin-maxwell-tree-v1\0")
    total_bytes = 0
    for path in files:
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        file_metadata = _sha256_file(path)
        size = int(file_metadata["bytes"])
        total_bytes += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(str(file_metadata["sha256"])))
    return {
        "resolved_root": str(resolved),
        "algorithm": "sha256",
        "format": "witwin-maxwell-tree-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    }


def _median_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def _variance_aware_gate(ratios: list[float], *, target_ratio: float) -> dict[str, Any]:
    """Return the S2.3 variance-aware verdict for paired-round ratios.

    Imported lazily so this module keeps its stdlib-only import-time surface
    (``compare_no_feature_op_stream.py`` loads it by path without a package
    context). Fewer than two rounds cannot support a confidence interval, so the
    verdict records that the gate was not applicable rather than fabricating a
    bound.
    """

    if len(ratios) < 2:
        return {
            "applicable": False,
            "reason": "fewer than two paired rounds",
            "target_ratio": float(target_ratio),
            "rounds": len(ratios),
        }
    support_dir = str(Path(__file__).resolve().parent)
    if support_dir not in sys.path:
        sys.path.insert(0, support_dir)
    from perf_variance_gate import evaluate_regression_gate

    result = evaluate_regression_gate(ratios, target_ratio=target_ratio)
    return {
        "applicable": True,
        "criterion": "95% CI upper bound of mean paired-round ratio < target",
        "target_ratio": result.target_ratio,
        "target_regression_pct": result.target_regression_pct,
        "rounds": result.rounds,
        "mean_ratio": result.mean_ratio,
        "median_ratio": result.median_ratio,
        "mad_ratio": result.mad_ratio,
        "ci95_upper_ratio": result.ci95_upper_ratio,
        "ci95_upper_regression_pct": result.ci95_upper_regression_pct,
        "passed": result.passed,
    }


def _insert_checkout_on_path(root: Path) -> None:
    root_text = str(root)
    try:
        sys.path.remove(root_text)
    except ValueError:
        pass
    sys.path.insert(0, root_text)


def _build_scene(mw: Any, *, grid_cells: int) -> Any:
    if grid_cells < 12:
        raise ValueError("grid_cells must be at least 12 so CPML leaves an interior region.")
    domain_size = 0.6
    cell_size = domain_size / grid_cells
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-0.5 * domain_size, 0.5 * domain_size),
                (-0.5 * domain_size, 0.5 * domain_size),
                (-0.5 * domain_size, 0.5 * domain_size),
            )
        ),
        grid=mw.GridSpec.uniform(cell_size),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=1.5 * cell_size,
            source_time=mw.GaussianPulse(
                frequency=FREQUENCY_HZ,
                fwidth=0.5 * FREQUENCY_HZ,
                amplitude=1.0,
            ),
        )
    )
    scene.add_monitor(
        mw.PointMonitor(
            name="probe",
            position=(0.1, 0.0, 0.0),
            fields=("Ez",),
        )
    )
    if getattr(scene, "ports", ()) or getattr(scene, "lumped_elements", ()):
        raise AssertionError("The no-feature benchmark scene must contain no RF state.")
    if len(scene.monitors) != 1 or not isinstance(scene.monitors[0], mw.PointMonitor):
        raise AssertionError("The benchmark may use only its generic field probe.")
    return scene


def _run_sample(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    _insert_checkout_on_path(root)

    import torch

    import witwin.maxwell as mw

    if not torch.cuda.is_available():
        raise RuntimeError("The RF no-feature benchmark requires one CUDA device.")
    if args.warmup < 1:
        raise ValueError("warmup must be at least 1.")
    if args.repeats < 3:
        raise ValueError("repeats must be at least 3 for a useful median.")
    if args.steps < 1:
        raise ValueError("steps must be positive.")

    scene = _build_scene(mw, grid_cells=args.grid_cells)
    samples_ms: list[float] = []
    total_runs = args.warmup + args.repeats
    for run_index in range(total_runs):
        simulation = mw.Simulation.fdtd(
            scene,
            frequency=FREQUENCY_HZ,
            run_time=mw.TimeConfig(time_steps=args.steps),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
            cuda_graph=False,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        result = simulation.run()
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        if run_index >= args.warmup:
            samples_ms.append(elapsed_ms)
        del result, simulation, start, end

    median_ms = float(statistics.median(samples_ms))
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rf_no_feature_sample",
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(root),
        "configuration": {
            "device": "cuda",
            "frequency_hz": FREQUENCY_HZ,
            "grid_cells_per_axis": args.grid_cells,
            "time_steps": args.steps,
            "warmup_runs": args.warmup,
            "timed_runs": args.repeats,
            "full_field_dft": False,
            "cuda_graph": False,
            "ports": 0,
            "lumped_elements": 0,
            "generic_field_monitors": 1,
            "rf_monitors": 0,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "fdtd_backend": os.environ.get("WITWIN_MAXWELL_FDTD_BACKEND", "cuda"),
            "cuda_extension": os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "0"),
        },
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_ms_per_step": median_ms / args.steps,
        "mad_ms": float(_median_absolute_deviation(samples_ms)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("kind") == "rf_no_feature_sample":
            return payload
    raise RuntimeError(f"Sampler did not emit a result JSON object. Output:\n{stdout}")


def _invoke_sampler(
    *,
    python: Path,
    runner: Path,
    root: Path,
    label: str,
    warmup: int,
    repeats: int,
    steps: int,
    grid_cells: int,
) -> dict[str, Any]:
    command = [
        str(python),
        str(runner),
        "sample",
        "--label",
        label,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--steps",
        str(steps),
        "--grid-cells",
        str(grid_cells),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Sampler failed for {label!r} in {root} with exit code "
            f"{completed.returncode}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    sample = _parse_last_json_line(completed.stdout)
    sample_root = Path(sample["git"]["resolved_root"])
    if sample_root != root:
        raise RuntimeError(
            f"Sampler root provenance mismatch: expected {root}, reported {sample_root}."
        )
    sample["execution"] = {
        "resolved_root": str(root),
        "command": command,
    }
    return sample


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()
    python = args.python.resolve()
    runner = Path(__file__).resolve()
    if baseline_root == candidate_root:
        raise ValueError("baseline_root and candidate_root must be different checkouts.")
    if not baseline_root.is_dir() or not candidate_root.is_dir():
        raise ValueError("baseline_root and candidate_root must both be existing directories.")
    if args.rounds < 1:
        raise ValueError("rounds must be positive.")

    baseline_commit = _git_output(
        candidate_root,
        "rev-parse",
        f"{args.baseline_commit}^{{commit}}",
    )
    if baseline_commit is None:
        raise ValueError(
            f"baseline_commit {args.baseline_commit!r} is not available from candidate_root."
        )
    baseline_git_tree = _git_output(candidate_root, "rev-parse", f"{baseline_commit}^{{tree}}")
    if baseline_git_tree is None:
        raise ValueError(f"Could not resolve the Git tree for baseline commit {baseline_commit}.")
    baseline_git = _git_metadata(baseline_root)
    if baseline_git["is_checkout_root"] and (
        baseline_git["commit"] != baseline_commit or baseline_git["dirty_tracked_files"]
    ):
        raise ValueError(
            "A Git-checkout baseline must be clean and exactly match baseline_commit."
        )
    candidate_git_before = _worktree_content_metadata(candidate_root)
    baseline_binding_before = _verify_baseline_snapshot(
        candidate_root,
        baseline_root,
        baseline_commit,
        args.baseline_archive,
    )
    baseline_tree_before = _tree_sha256(baseline_root)
    baseline_archive = (
        None if args.baseline_archive is None else _sha256_file(args.baseline_archive)
    )

    blocks: list[dict[str, Any]] = []
    round_summaries: list[dict[str, float]] = []
    # Every round is ABBA.  Reversing the next round makes the complete sequence
    # symmetric as well and limits bias from temperature or boost-clock drift.
    for round_index in range(args.rounds):
        round_blocks: list[dict[str, Any]] = []
        order = ("baseline", "candidate", "candidate", "baseline")
        if round_index % 2:
            order = ("candidate", "baseline", "baseline", "candidate")
        for label in order:
            root = baseline_root if label == "baseline" else candidate_root
            sample = _invoke_sampler(
                python=python,
                runner=runner,
                root=root,
                label=label,
                warmup=args.warmup,
                repeats=args.repeats,
                steps=args.steps,
                grid_cells=args.grid_cells,
            )
            blocks.append(sample)
            round_blocks.append(sample)
            print(
                f"{label} block median: {sample['median_ms']:.3f} ms "
                f"(MAD {sample['mad_ms']:.3f} ms)",
                file=sys.stderr,
                flush=True,
            )
        baseline_round_ms = float(
            statistics.median(
                block["median_ms"] for block in round_blocks if block["label"] == "baseline"
            )
        )
        candidate_round_ms = float(
            statistics.median(
                block["median_ms"] for block in round_blocks if block["label"] == "candidate"
            )
        )
        round_summaries.append(
            {
                "baseline_median_ms": baseline_round_ms,
                "candidate_median_ms": candidate_round_ms,
                "ratio": candidate_round_ms / baseline_round_ms,
            }
        )

    pooled: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for block in blocks:
        pooled[str(block["label"])].extend(float(value) for value in block["samples_ms"])
    baseline_median = float(statistics.median(pooled["baseline"]))
    candidate_median = float(statistics.median(pooled["candidate"]))
    baseline_tree_after = _tree_sha256(baseline_root)
    if baseline_tree_after != baseline_tree_before:
        raise RuntimeError("The immutable baseline tree changed during the ABBA comparison.")
    baseline_binding_after = _verify_baseline_snapshot(
        candidate_root,
        baseline_root,
        baseline_commit,
        args.baseline_archive,
    )
    if baseline_binding_after != baseline_binding_before:
        raise RuntimeError("The baseline Git-tree binding changed during the ABBA comparison.")
    candidate_git_after = _worktree_content_metadata(candidate_root)
    if candidate_git_after != candidate_git_before:
        raise RuntimeError("candidate_root HEAD or tracked dirty state changed during comparison.")

    paired_ratio = float(statistics.median(round_["ratio"] for round_ in round_summaries))
    regression_pct = 100.0 * (paired_ratio - 1.0)
    # Variance-aware verdict (audit step S2.3): the single-point median ratio
    # above is retained unchanged, but a regression is only certified when the
    # 95% CI upper bound of the paired-round ratio also clears the target. This
    # is additive -- it never relaxes the existing ``passed`` field.
    variance_gate = _variance_aware_gate(
        [float(round_["ratio"]) for round_ in round_summaries],
        target_ratio=1.0 + args.max_regression_pct / 100.0,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rf_no_feature_comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "CUDA event timing, untimed warmup, alternating ABBA/BAAB blocks, "
            "median paired-round ratio"
        ),
        "provenance": {
            "orchestrator_command": [
                str(Path(sys.executable).resolve()),
                str(runner),
                *sys.argv[1:],
            ],
            "baseline": {
                "resolved_root": str(baseline_root),
                "expected_commit": args.baseline_commit,
                "resolved_commit": baseline_commit,
                "git_tree": baseline_git_tree,
                "checkout": baseline_git,
                "git_tree_binding": baseline_binding_before,
                "tree_digest": baseline_tree_before,
                "archive": baseline_archive,
            },
            "candidate": candidate_git_before,
            "python": str(python),
            "runner": str(runner),
        },
        "configuration": {
            "rounds": args.rounds,
            "warmup_runs_per_block": args.warmup,
            "timed_runs_per_block": args.repeats,
            "grid_cells_per_axis": args.grid_cells,
            "time_steps": args.steps,
            "maximum_regression_pct": args.max_regression_pct,
        },
        "baseline": {
            "git": next(block["git"] for block in blocks if block["label"] == "baseline"),
            "environment": next(
                block["environment"] for block in blocks if block["label"] == "baseline"
            ),
            "samples_ms": pooled["baseline"],
            "median_ms": baseline_median,
            "mad_ms": float(_median_absolute_deviation(pooled["baseline"])),
        },
        "candidate": {
            "git": next(block["git"] for block in blocks if block["label"] == "candidate"),
            "environment": next(
                block["environment"] for block in blocks if block["label"] == "candidate"
            ),
            "samples_ms": pooled["candidate"],
            "median_ms": candidate_median,
            "mad_ms": float(_median_absolute_deviation(pooled["candidate"])),
        },
        "regression_pct": regression_pct,
        "passed": regression_pct < args.max_regression_pct,
        "variance_gate": variance_gate,
        "rounds": round_summaries,
        "blocks": blocks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Time the checkout in the current directory.")
    sample.add_argument("--label", default="candidate")
    sample.add_argument("--warmup", type=int, default=1)
    sample.add_argument("--repeats", type=int, default=7)
    sample.add_argument("--steps", type=int, default=2000)
    sample.add_argument("--grid-cells", type=int, default=24)
    sample.add_argument("--output", type=Path)

    compare = subparsers.add_parser("compare", help="Compare two checkout roots in ABBA order.")
    compare.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Exact non-Git extraction of --baseline-archive; extra files are rejected.",
    )
    compare.add_argument("--candidate-root", type=Path, required=True)
    compare.add_argument("--baseline-commit", required=True)
    compare.add_argument(
        "--baseline-archive",
        type=Path,
        required=True,
        help="Git archive whose paths, modes, and raw blobs must match --baseline-commit.",
    )
    compare.add_argument("--python", type=Path, default=Path(sys.executable))
    compare.add_argument("--rounds", type=int, default=2)
    compare.add_argument("--warmup", type=int, default=1)
    compare.add_argument("--repeats", type=int, default=5)
    compare.add_argument("--steps", type=int, default=2000)
    compare.add_argument("--grid-cells", type=int, default=24)
    compare.add_argument("--max-regression-pct", type=float, default=2.0)
    compare.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = _run_sample(args) if args.command == "sample" else _run_compare(args)
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
