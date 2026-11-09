"""Small shared utilities for the default-material GPU requalification studies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("WITWIN_EVIDENCE_ROOT", "evidence")) / "coverage-default-requalification"
POLICY = {"samples": [4, 4, 4], "averaging": "kottke_edge", "pec": "staircase"}


def parser_for(study, tiers):
    parser = argparse.ArgumentParser(description=study)
    parser.add_argument("--tiers", nargs="+", choices=list(tiers), default=list(tiers),
                        help="Select tiers; use one per invocation for isolated GPU memory.")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT / study)
    parser.add_argument("--summarize-only", action="store_true", help="Read JSON without CUDA.")
    parser.add_argument("--allocator-cap-gib", type=float, default=4.0,
                        help="PyTorch allocator cap for this process; recorded per tier, excluded "
                             "from the resume configuration because it does not change results.")
    return parser


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def source_hash(script, fixture):
    """Bind resumed data to the study, fixture and solver sources (including dirty edits)."""
    paths = {Path(script), Path(fixture), Path(__file__)}
    paths.update(p for p in (ROOT / "witwin/maxwell").rglob("*")
                 if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"})
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_records(out, configs):
    records = []
    for tier, config in configs.items():
        path = out / f"{tier}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("config") != config:
                raise ValueError(f"{path}: configuration/source mismatch; use a new --out directory")
            if record.get("status") != "complete":
                raise ValueError(f"{path}: incomplete tier record")
            records.append(record)
    return records


def run_study(args, configs, measure, summarize):
    records = load_records(args.out, configs)
    complete = {record["config"]["tier"] for record in records}
    pending = [tier for tier in dict.fromkeys(args.tiers) if tier not in complete]
    if pending and not args.summarize_only:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Production tiers require CUDA; no CPU fallback. Use --summarize-only.")
        torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES selects the physical GPU.
        properties = torch.cuda.get_device_properties(0)
        cap_bytes = int(args.allocator_cap_gib * 1024**3)
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_bytes / properties.total_memory), 0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        for tier in pending:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.no_grad():
                metrics = measure(tier)
            torch.cuda.synchronize()
            record = {
                "status": "complete", "config": configs[tier], **metrics,
                "wall_time_s": time.perf_counter() - started,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocator_cap_bytes": min(cap_bytes, properties.total_memory),
                "gpu": properties.name, "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            write_json(args.out / f"{tier}.json", record)
            print(json.dumps(record, sort_keys=True), flush=True)
            records.append(record)
            # Save a useful partial summary even if a later tier fails or is interrupted.
            write_json(args.out / "summary.json", summarize(records))
    summary = summarize(records)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def adjacent_changes(h, values):
    return [{"coarse_dx_m": a, "fine_dx_m": b,
             "relative_change": abs(u - v) / abs(v) if v else None}
            for a, b, u, v in zip(h[:-1], h[1:], values[:-1], values[1:])]


def richardson_triple(h, values):
    """Solve y(h)=y_inf+C*h**p on three possibly unequal grid ratios."""
    h = np.asarray(h, dtype=float)
    y = np.asarray(values, dtype=float)
    if (h.shape != (3,) or y.shape != (3,) or not np.isfinite(h).all()
            or not np.isfinite(y).all() or not (h[0] > h[1] > h[2] > 0)):
        raise ValueError("Richardson needs three finite values at decreasing positive spacings")
    result = {"dx_m": h.tolist(), "p": None, "extrapolated": None}
    d01, d12 = y[0] - y[1], y[1] - y[2]
    if d01 * d12 <= 0:
        return dict(result, status="flat_or_nonmonotone")
    target = d01 / d12
    log01, log12 = math.log(h[0] / h[1]), math.log(h[1] / h[2])

    def residual(p):
        return math.expm1(p * log01) / -math.expm1(-p * log12) - target

    # A diagnostic search range, not a qualification threshold.
    if residual(1e-6) * residual(20.0) > 0:
        return dict(result, status="no_positive_order_in_search_range_1e-6_to_20")
    p = brentq(residual, 1e-6, 20.0)
    q = (h[2] / h[1]) ** p
    return dict(result, status="fit", p=p, extrapolated=float(y[2] - d12 * q / (1.0 - q)))


def power_fit(h, values, p, target_h):
    """Fixed-order least squares with residual degrees of freedom retained."""
    h = np.asarray(h, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(h) < 3 or len(h) != len(y) or not np.isfinite(y).all():
        raise ValueError("A residual-based fit requires at least three finite observations")
    scale = float(np.max(h))
    design = np.column_stack((np.ones_like(h), (h / scale) ** p))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    residual_std = float(np.linalg.norm(residuals) / math.sqrt(len(y) - 2))
    target = np.array([1.0, (target_h / scale) ** p])
    leverage = float(target @ np.linalg.pinv(design.T @ design) @ target)
    return {
        "p": p, "limit": float(coefficients[0]),
        "coefficient_scaled": float(coefficients[1]), "h_scale_m": scale,
        "predicted_target": float(target @ coefficients),
        "residuals": residuals.tolist(), "residual_std": residual_std,
        "residual_dof": len(y) - 2,
        "target_prediction_se": residual_std * math.sqrt(1.0 + leverage),
    }
