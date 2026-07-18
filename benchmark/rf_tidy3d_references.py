"""Tidy3D cross-reference generation for the RF wave-level scenes.

Reference-solver policy (audit 2026-07-18, section 3): for the RF port families
Tidy3D covers (Terminal/Lumped/Wave port N-port S; coax; microstrip; rectangular
waveguide), the external cross-reference must be generated through the *existing*
benchmark adapter (`witwin.maxwell.adapters`, `benchmark/`) with a pinned Tidy3D
version and generation parameters, then cached. The analytic transmission-line /
waveguide reference remains the binding first-line gate regardless.

This script is the generation path. When the external service is unavailable
(offline session, or ``WITWIN_BENCHMARK_NO_CLOUD=1``), it does NOT fabricate a
cache: it stamps a ``reference: pending-generation`` marker per scene so the gap
is explicit and the analytic gate keeps binding. Run it in an environment with
Tidy3D + cloud access to populate the caches.

Coverage per the audit's per-plan mapping:
* coax_thru, rectangular_waveguide, microstrip_two_port, lumped_open_short_match,
  differential_pair -> Tidy3D-coverable -> primary cross-reference Tidy3D (+ xFdtd future);
* series_parallel_rlc -> lumped-circuit resonance, analytic first-line, no Tidy3D cross-ref.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from benchmark.paths import CACHE_DIR

TIDY3D_COVERABLE = (
    "rf/coax_thru",
    "rf/rectangular_waveguide",
    "rf/microstrip_two_port",
    "rf/lumped_open_short_match",
    "rf/differential_pair",
)

MARKER_DIR = CACHE_DIR / "rf"


def _tidy3d_available() -> bool:
    if os.environ.get("WITWIN_BENCHMARK_NO_CLOUD") == "1":
        return False
    try:
        import tidy3d  # noqa: F401
    except Exception:
        return False
    # Cloud credentials are required to actually run a reference; treat their
    # absence as unavailable rather than erroring.
    return bool(os.environ.get("TIDY3D_API_KEY") or os.environ.get("SIMCLOUD_APIKEY"))


def _write_pending_markers() -> list[Path]:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for scene in TIDY3D_COVERABLE:
        slug = scene.replace("/", "__")
        path = MARKER_DIR / f"{slug}.pending.json"
        path.write_text(
            json.dumps(
                {
                    "scene": scene,
                    "reference": "pending-generation",
                    "solver": "tidy3d",
                    "policy": "audit-2026-07-18 section 3 (Tidy3D primary cross-reference)",
                    "note": (
                        "External Tidy3D reference not generated in this session; analytic "
                        "transmission-line/waveguide reference binds. Run with Tidy3D + cloud "
                        "credentials to populate the cache."
                    ),
                    "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(path)
    return written


def _generate_references() -> None:  # pragma: no cover - requires external service
    raise NotImplementedError(
        "Tidy3D RF S-parameter reference generation must be wired through the "
        "existing benchmark adapter with a pinned Tidy3D version. This runner "
        "currently only stamps pending-generation markers; populate this function "
        "in an environment with Tidy3D + cloud access."
    )


def main() -> None:
    if _tidy3d_available():
        _generate_references()
        return
    written = _write_pending_markers()
    print(
        "Tidy3D unavailable (offline / no credentials / NO_CLOUD). Stamped "
        f"{len(written)} pending-generation markers under {MARKER_DIR}:"
    )
    for path in written:
        print(f"  {path}")
    print("Analytic transmission-line / waveguide references remain the binding gate.")


if __name__ == "__main__":
    main()
