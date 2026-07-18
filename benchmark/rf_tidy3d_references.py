"""Tidy3D cross-reference *marker* stamping for the RF wave-level scenes.

Reference-solver policy (audit 2026-07-18, section 3): for the RF port families an
external reference solver covers, the cross-reference should be generated through
the *existing* benchmark adapter (`witwin.maxwell.adapters`, `benchmark/`) with a
pinned external-solver version and cached. The analytic transmission-line /
waveguide reference remains the binding first-line gate regardless.

HONEST SCOPE (M3): real adapter-driven generation is NOT yet implemented. This
CLI therefore only stamps ``reference: pending-generation`` markers per scene so
the gap is explicit and the analytic gate keeps binding -- it never fabricates a
cache and never claims to have run the external solver. Wiring the adapter
generation is future work; until then this script's sole effect is the markers.

Coverage per the audit's per-plan mapping:
* coax_thru, rectangular_waveguide, microstrip_two_port, lumped_open_short_match,
  differential_pair -> external-solver-coverable (marker stamped pending);
* series_parallel_rlc -> lumped-circuit resonance, analytic first-line, no external
  cross-ref.
"""

from __future__ import annotations

import json
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
                    "solver": "external-reference-solver",
                    "policy": "audit-2026-07-18 section 3 (external reference solver, primary cross-reference)",
                    "note": (
                        "External reference-solver cross-reference not generated: adapter-driven "
                        "generation is not yet wired (M3). The analytic transmission-line/waveguide "
                        "reference binds. This marker never stands in for a numerical comparison."
                    ),
                    "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> None:
    # Real adapter-driven generation is not implemented (M3): stamp pending
    # markers only, honestly, regardless of whether the external service or its
    # credentials are present. This never fabricates a numerical cross-reference.
    written = _write_pending_markers()
    print(
        f"Stamped {len(written)} reference: pending-generation markers under {MARKER_DIR} "
        "(real external-solver generation is not yet wired; see module docstring):"
    )
    for path in written:
        print(f"  {path}")
    print("Analytic transmission-line / waveguide references remain the binding gate.")


if __name__ == "__main__":
    main()
