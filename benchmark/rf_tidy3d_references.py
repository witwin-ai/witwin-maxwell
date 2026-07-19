"""External-reference-solver cross-reference generation for the RF / antenna scenes (M3).

Reference-solver policy (audit 2026-07-18, section 3): for the RF port and antenna
families an external reference solver covers, the cross-reference is generated
through the *existing* benchmark adapter (``witwin.maxwell.adapters``, exported via
``Scene.to_tidy3d``) and cached under ``benchmark/cache/`` alongside the standard
field-vs-reference scenes. The analytic transmission-line / waveguide / dipole
reference remains the binding first-line gate regardless.

M3 wiring (this module): the generation path is now REAL, not a pending-marker
stub. For each target scene it:

1. builds the ``Scene`` and exports it through the adapter (``Scene.to_tidy3d``),
   capturing any adapter ``NotImplementedError`` (unsupported construct);
2. gates the export on being *physically runnable* -- a reference simulation with
   zero sources cannot produce a scattering matrix or a radiation pattern, so a
   source-less export is refused BEFORE any cloud cost is incurred;
3. if runnable, estimates the cloud cost, enforces the per-scene FlexCredit
   budget, runs one cloud job, extracts the monitors, and writes the ``.h5``
   cache plus a ``.generated.json`` record with the task id and cost;
4. if NOT runnable (or the cloud run fails), writes a ``reference:
   pending-generation`` marker carrying the CONCRETE reason (the captured
   exception or the source-count gate) so the gap is explicit and the analytic
   gate keeps binding. This never fabricates a numerical cross-reference.

Adapter capability status (measured, EXECUTED 2026-07-19): the four authorized
target scenes all export with ``sources == 0``. Their excitation is port-driven
-- a ``WavePort`` TEM launch under ``PortSweep`` / ``PortExcitation`` (coax_thru,
lumped_open_short_match) or a ``LumpedPort`` wire-gap / probe feed (antenna
dipole, patch) -- and the adapter's ``_convert_source`` has no mapping for
port/lumped excitation (only field sources: PointDipole, PlaneWave, GaussianBeam,
ModeSource, UniformCurrentSource, CustomField/CurrentSource). A ``ClosedSurfaceMonitor``
also has no adapter monitor mapping. The generation therefore fail-closes at the
runnable gate for all four scenes with ``sources=0`` recorded, and NO cloud
credits are spent. Mapping port/lumped excitation to the reference solver is a
separate adapter feature (deferred); until then these references stay
``pending-generation`` with the reason recorded here and in
``benchmark/RESULTS.md``.

Invoke with ``python -m benchmark.rf_tidy3d_references [scene ...]``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from benchmark.cache import cache_path, save_tidy3d_result
from benchmark.paths import CACHE_DIR, RESULTS_MD

# Per-scene cloud-cost ceiling (FlexCredits). Mirrors the standard runner budget;
# a reference whose estimate exceeds this is refused rather than run.
MAX_REFERENCE_COST = 2.0

# Set to "1" to forbid any cloud interaction (offline / CI). The generation then
# records pending-generation with the offline reason instead of estimating cost.
NO_CLOUD_ENV_VAR = "WITWIN_BENCHMARK_NO_CLOUD"


# --------------------------------------------------------------------------- #
# Target-scene registry (the four owner-authorized references).               #
# --------------------------------------------------------------------------- #
def _coax_thru():
    from benchmark.scenes.rf.coax_thru import coax_thru_scene

    scene = coax_thru_scene(dx=0.01, device="cpu")
    freqs = (0.6e9, 1.0e9, 1.6e9)
    return scene, freqs


def _lumped_open_short_match():
    from benchmark.scenes.rf.lumped_open_short_match import (
        coax_sol_scene,
        default_frequencies,
    )

    scene = coax_sol_scene("matched", dx=0.01, device="cpu")
    return scene, default_frequencies()


def _half_wave_dipole():
    from benchmark.scenes.antenna.half_wave_dipole import (
        default_frequencies,
        half_wave_dipole_scene,
    )

    design = 3.0e9
    scene = half_wave_dipole_scene(design_frequency=design, device="cpu")
    return scene, default_frequencies(design)


def _patch():
    from benchmark.scenes.antenna.patch import patch_antenna_scene

    freqs = tuple(f * 1e9 for f in (4.4, 4.8, 5.2, 5.6, 6.0))
    scene = patch_antenna_scene(frequencies=freqs, device="cpu")
    return scene, freqs


# name -> builder returning (scene, frequencies). Order is the run order.
REFERENCE_TARGETS = {
    "rf/coax_thru": _coax_thru,
    "rf/lumped_open_short_match": _lumped_open_short_match,
    "antenna/half_wave_dipole": _half_wave_dipole,
    "antenna/patch": _patch,
}

PENDING = "pending-generation"
GENERATED = "generated"


@dataclass
class ReferenceRecord:
    scene: str
    status: str                # generated | pending-generation
    solver: str = "external-reference-solver"
    exported_sources: int | None = None
    exported_monitors: int | None = None
    runnable: bool | None = None
    reason: str = ""
    task_id: str | None = None
    cost_flexcredits: float | None = None
    cache: str | None = None
    stamped_at: str = ""
    policy: str = (
        "audit-2026-07-18 section 3 (external reference solver, primary cross-reference); "
        "analytic first-line reference binds regardless"
    )
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Adapter export + runnable gate.                                             #
# --------------------------------------------------------------------------- #
def _export_simulation(scene, frequencies):
    """Export a Scene through the adapter. Raises on an unsupported construct."""
    domain_size = max(hi - lo for lo, hi in scene.domain.bounds)
    run_time = 20.0 * domain_size / 299_792_458.0
    return scene.to_tidy3d(frequencies=tuple(frequencies), run_time=run_time)


def _runnable_reason(td_sim) -> tuple[bool, str]:
    """A reference simulation must carry at least one source to radiate.

    A source-less export cannot produce an S-matrix or a radiation pattern
    (running it yields only numerical noise), so it is refused before any cloud
    cost. This is the adapter-capability gate for the port/lumped-driven scenes.
    """
    n_sources = len(td_sim.sources)
    if n_sources == 0:
        return False, (
            "adapter exported the scene with sources=0: the port/lumped-port "
            "excitation has no external-reference-solver source mapping "
            "(adapter _convert_source maps only field sources), so the exported "
            "reference simulation has nothing to drive. Refused before any cloud "
            "cost. Mapping port/lumped excitation to the reference solver is a "
            "deferred adapter feature."
        )
    return True, ""


def _run_cloud_reference(name: str, td_sim, frequencies) -> tuple[dict, str, float]:
    """Estimate cost, enforce the budget, run one cloud job, extract monitors.

    Returns (monitors, task_id, cost). Raises on a budget breach or a cloud
    failure -- the caller records the failure fail-closed; it never fabricates.
    """
    import tidy3d.web as web

    from benchmark.runner import _extract_tidy3d_monitors

    slug = name.replace("/", "_")
    job = web.Job(simulation=td_sim, task_name=f"maxwell_reference_{slug}", verbose=False)
    cost = float(job.estimate_cost(verbose=False))
    if cost > MAX_REFERENCE_COST:
        raise RuntimeError(
            f"estimated cost {cost:.4f} FlexCredits exceeds the per-scene budget "
            f"{MAX_REFERENCE_COST:.4f}; reference was not run."
        )
    data = job.run()
    monitors = _extract_tidy3d_monitors(data, td_sim, None)
    task_id = str(getattr(job, "task_id", "") or "")
    return monitors, task_id, cost


def attempt_reference(name: str, *, run_cloud: bool = True) -> ReferenceRecord:
    """Attempt to generate one external-reference cache; fail closed honestly."""
    record = ReferenceRecord(
        scene=name,
        status=PENDING,
        stamped_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    build = REFERENCE_TARGETS.get(name)
    if build is None:
        record.reason = f"unknown reference target '{name}'"
        return record

    try:
        scene, frequencies = build()
    except Exception as exc:  # noqa: BLE001 - record honestly
        record.reason = f"scene build failed: {type(exc).__name__}: {exc}"
        return record

    try:
        td_sim = _export_simulation(scene, frequencies)
    except NotImplementedError as exc:
        record.reason = f"adapter export unsupported: {exc}"
        return record
    except Exception as exc:  # noqa: BLE001 - record honestly
        record.reason = f"adapter export failed: {type(exc).__name__}: {exc}"
        return record

    record.exported_sources = len(td_sim.sources)
    record.exported_monitors = len(td_sim.monitors)
    runnable, reason = _runnable_reason(td_sim)
    record.runnable = runnable
    if not runnable:
        record.reason = reason
        return record

    if not run_cloud or os.environ.get(NO_CLOUD_ENV_VAR) == "1":
        record.reason = (
            "export is runnable but cloud generation was suppressed "
            f"({NO_CLOUD_ENV_VAR}=1 or run_cloud=False); rerun without the offline "
            "flag to generate the cache."
        )
        return record

    try:
        monitors, task_id, cost = _run_cloud_reference(name, td_sim, frequencies)
    except Exception as exc:  # noqa: BLE001 - record fail-closed, never fabricate
        record.reason = f"cloud generation failed: {type(exc).__name__}: {exc}"
        return record

    path = save_tidy3d_result(name, frequencies=tuple(frequencies), monitors=monitors)
    record.status = GENERATED
    record.task_id = task_id
    record.cost_flexcredits = cost
    record.cache = str(path)
    record.reason = ""
    return record


# --------------------------------------------------------------------------- #
# Marker + RESULTS.md persistence.                                            #
# --------------------------------------------------------------------------- #
def write_marker(record: ReferenceRecord) -> Path:
    slug = record.scene.replace("/", "__")
    suffix = "generated" if record.status == GENERATED else "pending"
    directory = CACHE_DIR.joinpath(*record.scene.split("/")[:-1])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.{suffix}.json"
    # A fresh success supersedes a stale pending marker and vice versa.
    other = directory / f"{slug}.{'pending' if suffix == 'generated' else 'generated'}.json"
    if other.exists():
        other.unlink()
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return path


_SECTION_HEADER = "## RF / antenna external reference generation"


def _results_section(records: list[ReferenceRecord]) -> str:
    lines = [
        _SECTION_HEADER,
        "",
        "External reference-solver cross-reference generation for the RF / antenna "
        "scenes (M3, 2026-07-19). Each row is the honest outcome of an adapter-driven "
        "generation attempt (`python -m benchmark.rf_tidy3d_references`): the scene is "
        "exported through `Scene.to_tidy3d`, gated on being physically runnable (>= 1 "
        "source), and only then cost-estimated and cloud-run. A `pending-generation` "
        "status is NOT a fabricated comparison -- it records that no numerical "
        "cross-reference exists yet and names the concrete reason; the analytic "
        "transmission-line / waveguide / dipole references remain the binding gate. The "
        "four scenes below currently export with `sources=0` (their port / lumped-port "
        "excitation has no adapter source mapping), so generation fail-closes at the "
        "runnable gate BEFORE any cloud cost; mapping port/lumped excitation to the "
        "reference solver is a deferred adapter feature.",
        "",
        "| Scene | Exported sources | Exported monitors | Runnable | Reference | Task id | Cost (FlexCredits) | Reason |",
        "| --- | ---: | ---: | :---: | --- | --- | ---: | --- |",
    ]

    def esc(value) -> str:
        return str(value).replace("|", r"\|")

    for record in records:
        reference = "generated" if record.status == GENERATED else PENDING
        lines.append(
            "| {scene} | {src} | {mon} | {run} | {ref} | {task} | {cost} | {reason} |".format(
                scene=record.scene,
                src="-" if record.exported_sources is None else record.exported_sources,
                mon="-" if record.exported_monitors is None else record.exported_monitors,
                run="-" if record.runnable is None else ("yes" if record.runnable else "no"),
                ref=reference,
                task=esc(record.task_id or "-"),
                cost="-" if record.cost_flexcredits is None else f"{record.cost_flexcredits:.4f}",
                reason=esc(record.reason or "-"),
            )
        )
    lines.append("")
    lines.append(f"_RF/antenna reference section regenerated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    return "\n".join(lines)


def _update_results_md(records: list[ReferenceRecord]) -> None:
    section = _results_section(records)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(section, encoding="utf-8")
        return
    text = RESULTS_MD.read_text(encoding="utf-8")
    if _SECTION_HEADER in text:
        head, _, tail = text.partition(_SECTION_HEADER)
        rest = tail.split("\n", 1)[1] if "\n" in tail else ""
        next_idx = rest.find("\n## ")
        remainder = rest[next_idx + 1 :] if next_idx != -1 else ""
        RESULTS_MD.write_text(head + section + remainder, encoding="utf-8")
    else:
        RESULTS_MD.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def generate(selected: list[str] | None = None, *, run_cloud: bool = True) -> list[ReferenceRecord]:
    names = selected or list(REFERENCE_TARGETS)
    records: list[ReferenceRecord] = []
    for name in names:
        print(f"[rf-references] attempting {name} ...", flush=True)
        record = attempt_reference(name, run_cloud=run_cloud)
        marker = write_marker(record)
        detail = (
            f"generated (task={record.task_id}, cost={record.cost_flexcredits})"
            if record.status == GENERATED
            else f"pending-generation ({record.reason})"
        )
        print(f"[rf-references]   {name}: {detail}", flush=True)
        print(f"[rf-references]   marker={marker}", flush=True)
        records.append(record)
    _update_results_md(records)
    return records


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate external-reference-solver caches for the RF / antenna scenes."
    )
    parser.add_argument("scenes", nargs="*", help="Scene names (default: all authorized targets).")
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="Do not touch the cloud; record the runnable gate only.",
    )
    args = parser.parse_args(argv)
    selected = None
    if args.scenes:
        unknown = [s for s in args.scenes if s not in REFERENCE_TARGETS]
        if unknown:
            raise SystemExit(
                f"Unknown reference targets: {unknown}. Available: {list(REFERENCE_TARGETS)}"
            )
        selected = args.scenes
    records = generate(selected, run_cloud=not args.no_cloud)
    generated = sum(1 for r in records if r.status == GENERATED)
    print(
        f"\n{generated}/{len(records)} references generated; "
        f"{len(records) - generated} pending-generation (reason recorded per scene)."
    )


if __name__ == "__main__":
    main()
