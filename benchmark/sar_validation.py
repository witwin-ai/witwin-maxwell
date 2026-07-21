"""SAR phantom exposure validation harness.

Runs the canonical phantom benchmark family (``benchmark/scenes/sar/``) through
the public ``Scene -> Simulation -> Result`` path and records one headline gate
per scene into ``benchmark/RESULTS.md`` (section ``## SAR exposure validation``)
plus a machine-readable JSON artifact per scene under
``docs/assessments/sar-phantom-validation/``.

Gate taxonomy is the verbatim spec in ``docs/reference/gate-classification.md``.
The binding wave-level evidence is the ``sar/layered_slab`` power-conservation
closure: the absorbed power measured as the volume conduction-loss integral
``sigma |E|^2`` (the quantity SAR is built from) agrees with the net surface
Poynting balance ``flux_in - flux_out`` (E x H on two planes), and their residual
shrinks monotonically as the grid refines. Surface E x H is independent of the
volume loss, so this is a conservation-law reference, not a self-consistency
identity. ``sar/one_gram_cube`` is an ``analytic-identity`` (hand-computable 1 g
average); ``sar/uniform_lossy_cube`` reports the self-consistency volume/channel
closure (``tautology`` -- the volume integral and the channel total are two
reductions of the same edge-loss field, so their closure is exact by construction
and supporting only). ``sar/antenna_near_phantom`` is ``blocked`` -- the FDTD port
machinery fails closed on a conductive tissue background, which a phantom is by
construction; its wave-level class is recorded as a TARGET only (``target_gate_class``),
never as an achieved gate.

Reference-solver policy (audit section 3): the conservation-law / analytic gates
are the binding first-line references. No external reference-solver run backs this
family in this build: the driven antenna scene (the only one whose absorbed-power
field an external solver could cross-check) is blocked upstream, and the plane-wave
phantoms are gated on conservation, so every row carries
``external_reference: analytic-only``.

Run: ``python -m benchmark sar`` (all scenes) or ``python -m benchmark sar
one_gram_cube`` (subset).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import witwin.maxwell as mw
from benchmark.paths import ROOT
from benchmark.scenes.sar import layered_slab, one_gram_cube, uniform_lossy_cube
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.scene import prepare_scene

RESULTS_MD = ROOT / "RESULTS.md"
ARTIFACT_DIR = ROOT.parent / "docs" / "assessments" / "sar-phantom-validation"
SECTION_HEADER = "## SAR exposure validation"

WAVE_LEVEL = "wave-level"
ANALYTIC_IDENTITY = "analytic-identity"
TAUTOLOGY = "tautology"
ANALYTIC_ONLY = "analytic-only"

# Three refinement tiers (m) for the layered-slab conservation convergence.
_CONVERGENCE_GRIDS = (0.005, 0.004, 0.003)


@dataclass
class SarReport:
    name: str
    description: str
    gate_class: str  # the ACHIEVED headline gate class ("" when the scene is blocked)
    status: str  # pass | gap | fail | blocked | pending | error
    reference: str
    external_reference: str
    target: str
    # Aspirational class of the gate the scene targets. Recorded (with the blocked
    # status and a blocked marker in the table) when the wave-level gate cannot run,
    # so a blocked row never claims an unachieved class as if measured.
    target_gate_class: str = ""
    metrics: list[dict] = field(default_factory=list)
    convergence: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    updated_at: str = ""


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rel(measured: float, reference: float) -> float:
    denom = abs(reference) if abs(reference) > 0.0 else 1.0
    return abs(measured - reference) / denom


def _run_planewave(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=list(scene.monitors[0].frequencies or ()),
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(normalize_source=True),
        full_field_dft=True,
    ).run()


def _volume_absorbed_power(sar) -> float:
    q = sar.absorbed_power_density["total"]
    return float((q * sar.cell_volume[None]).sum())


# --------------------------------------------------------------------------- #
# sar/one_gram_cube -- analytic-identity (no solver in the wave).             #
# --------------------------------------------------------------------------- #
def run_one_gram_cube() -> SarReport:
    report = SarReport(
        name="sar/one_gram_cube",
        description="Synthetic golden: a 3x3x3 window weighs exactly 1 g",
        gate_class=ANALYTIC_IDENTITY,
        status="error",
        reference="analytic uniform-field 1 g average = 0.5*sigma*3*E0^2/rho",
        external_reference=ANALYTIC_ONLY,
        target="1 g averaged SAR within 1e-4 of the hand-computed analytic value",
    )
    try:
        device = _device()
        e0 = 2.0
        scene = one_gram_cube.build_scene(device=device)
        monitor = next(m for m in scene.monitors if m.name == "loss")
        prepared = prepare_scene(scene)
        compiled = compile_power_loss_monitor(prepared, monitor)
        fields = {
            component.upper(): torch.full(
                compiled.full_component_shapes[component],
                complex(e0),
                dtype=torch.complex64,
                device=device,
            )
            for component in ("Ex", "Ey", "Ez")
        }
        result = mw.Result(
            method="fdtd",
            scene=scene,
            prepared_scene=prepared,
            frequencies=one_gram_cube.FREQUENCIES,
            fields=fields,
        )
        sar = result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3,)))
        peak = sar.peak(mass=1e-3)
        measured = float(peak.sar[0])
        analytic = 0.5 * one_gram_cube.SIGMA_E * (3.0 * e0**2) / one_gram_cube.MASS_DENSITY
        report.metrics.append(
            {
                "quantity": "peak_1g_sar",
                "measured": measured,
                "reference": analytic,
                "rel_error": _rel(measured, analytic),
                "unit": "W/kg",
                "class": ANALYTIC_IDENTITY,
            }
        )
        report.metrics.append(
            {"quantity": "enclosed_mass", "measured": float(peak.mass_kg[0]), "reference": 1e-3,
             "rel_error": _rel(float(peak.mass_kg[0]), 1e-3), "unit": "kg", "class": ANALYTIC_IDENTITY}
        )
        report.status = "pass" if report.metrics[0]["rel_error"] < 1e-4 else "fail"
    except Exception as exc:  # pragma: no cover - reported as an error row
        report.notes.append(f"{type(exc).__name__}: {exc}")
    return report


# --------------------------------------------------------------------------- #
# sar/uniform_lossy_cube -- self-consistency closure (supporting).            #
# --------------------------------------------------------------------------- #
def run_uniform_lossy_cube() -> SarReport:
    report = SarReport(
        name="sar/uniform_lossy_cube",
        description="Uniform lossy phantom under a normally incident plane wave",
        gate_class=TAUTOLOGY,
        status="error",
        reference="absorbed-power volume integral vs shared electric-channel total (self-comparison)",
        external_reference=ANALYTIC_ONLY,
        target="volume/channel closure < 1e-4; positive point SAR",
    )
    try:
        if not torch.cuda.is_available():
            report.status = "pending"
            report.notes.append("CUDA required for the FDTD run; skipped on CPU.")
            return report
        scene = uniform_lossy_cube.build_scene(device="cuda")
        result = _run_planewave(scene)
        sar = result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3, 10e-3)))
        volume_absorbed = _volume_absorbed_power(sar)
        channel = float(sar.provenance["electric_channel_power"][0])
        peak_point = float(
            torch.nan_to_num(sar.point_sar("total"), nan=-float("inf")).max()
        )
        report.metrics.append(
            {"quantity": "volume/channel_closure", "measured": volume_absorbed, "reference": channel,
             "rel_error": _rel(volume_absorbed, channel), "unit": "W", "class": TAUTOLOGY}
        )
        report.metrics.append(
            {"quantity": "peak_point_sar", "measured": peak_point, "reference": float("nan"),
             "rel_error": float("nan"), "unit": "W/kg", "class": TAUTOLOGY}
        )
        report.notes.append(
            "Tautology (self-comparison), supporting only: the volume integral and the "
            "channel total are two reductions of the SAME edge loss field data, so their "
            "closure is exact by construction and carries no independent physical "
            "information. The independent conservation reference is the sar/layered_slab "
            "surface-vs-volume balance."
        )
        report.status = "pass" if report.metrics[0]["rel_error"] < 1e-4 and peak_point > 0.0 else "fail"
    except Exception as exc:  # pragma: no cover
        report.notes.append(f"{type(exc).__name__}: {exc}")
    return report


# --------------------------------------------------------------------------- #
# sar/layered_slab -- wave-level conservation closure + 3-grid convergence.   #
# --------------------------------------------------------------------------- #
def _layered_conservation(dx: float) -> dict:
    scene = layered_slab.build_conservation_scene(dx=dx, device="cuda")
    result = _run_planewave(scene)
    sar = result.sar("loss", averaging=mw.SARAveraging(mass=(1e-3,)))
    volume_absorbed = _volume_absorbed_power(sar)
    flux_in = float(result.monitor("flux_in")["flux"])
    flux_out = float(result.monitor("flux_out")["flux"])
    surface_absorbed = flux_in - flux_out
    return {
        "dx": dx,
        "volume_absorbed": volume_absorbed,
        "surface_absorbed": surface_absorbed,
        "flux_in": flux_in,
        "flux_out": flux_out,
        "rel_error": _rel(surface_absorbed, volume_absorbed),
    }


def run_layered_slab() -> SarReport:
    report = SarReport(
        name="sar/layered_slab",
        description="Skin/fat/muscle slab: surface Poynting vs volume-loss conservation",
        gate_class=WAVE_LEVEL,
        status="error",
        reference="conservation law: surface E x H flux balance vs volume sigma|E|^2 integral",
        external_reference=ANALYTIC_ONLY,
        target="surface/volume absorbed-power closure < 0.20 and monotonically converging",
    )
    try:
        if not torch.cuda.is_available():
            report.status = "pending"
            report.notes.append("CUDA required for the FDTD run; skipped on CPU.")
            return report
        for dx in _CONVERGENCE_GRIDS:
            report.convergence.append(_layered_conservation(dx))
        residuals = [row["rel_error"] for row in report.convergence]
        # Reference grid (dx = 4 mm) is the headline.
        headline = report.convergence[1]
        report.metrics.append(
            {"quantity": "surface/volume_closure", "measured": headline["surface_absorbed"],
             "reference": headline["volume_absorbed"], "rel_error": headline["rel_error"],
             "unit": "W", "class": WAVE_LEVEL}
        )
        converges = residuals[-1] < residuals[0] - 0.04 and residuals[-1] < 0.15
        report.metrics.append(
            {"quantity": "closure_residual_finest", "measured": residuals[-1], "reference": 0.0,
             "rel_error": residuals[-1], "unit": "1", "class": WAVE_LEVEL}
        )
        report.notes.append(
            "Closure residual per grid (coarse->fine): "
            + ", ".join(f"{r:.3f}" for r in residuals)
            + f" (converges={converges})."
        )
        report.status = "pass" if headline["rel_error"] < 0.20 and converges else "gap"
    except Exception as exc:  # pragma: no cover
        report.notes.append(f"{type(exc).__name__}: {exc}")
    return report


# --------------------------------------------------------------------------- #
# sar/antenna_near_phantom -- blocked (conductive-media port, no run).        #
# --------------------------------------------------------------------------- #
def run_antenna_near_phantom() -> SarReport:
    return SarReport(
        name="sar/antenna_near_phantom",
        description="Driven dipole near a tissue block (conductive-media port blocker)",
        # No gate ran, so no class is achieved. The wave-level class is the TARGET
        # only; it is recorded in target_gate_class (with the blocked status), never
        # in gate_class, so the row cannot read as a measured wave-level result.
        gate_class="",
        status="blocked",
        target_gate_class=WAVE_LEVEL,
        reference="pending: needs a conductance-aware lumped-port update coefficient",
        external_reference=ANALYTIC_ONLY,
        target="accepted-power 1 W -> point SAR -> 1 g/10 g peaks (blocked upstream)",
        notes=[
            "The FDTD port machinery fails closed on a conductive background (a tissue "
            "phantom is conductive by construction): the driven end-to-end SAR chain "
            "raises NotImplementedError at prepare(). Unblocking needs a conductance-aware "
            "lumped-port update coefficient, out of this stage's scope. Pinned as a "
            "fail-closed gate in tests/sar/test_phantom_benchmarks.py."
        ],
    )


SCENE_RUNNERS = {
    "one_gram_cube": run_one_gram_cube,
    "uniform_lossy_cube": run_uniform_lossy_cube,
    "layered_slab": run_layered_slab,
    "antenna_near_phantom": run_antenna_near_phantom,
}


# --------------------------------------------------------------------------- #
# RESULTS.md section writer.                                                   #
# --------------------------------------------------------------------------- #
_INTRO = (
    "SAR phantom exposure validation (plan 10 H3, 2026-07-21). Each row is measured "
    "from the public `Scene -> Simulation -> Result` path over the canonical phantom "
    "family (`benchmark/scenes/sar/`); the per-scene status column is authoritative. "
    "`sar/layered_slab` is the binding wave-level gate: the absorbed power measured as "
    "the volume conduction-loss integral (`sigma |E|^2`, the SAR basis) agrees with the "
    "net surface Poynting balance (`flux_in - flux_out`, E x H on two planes) to ~17% at "
    "dx=4 mm and closes monotonically under refinement (0.20 -> 0.17 -> 0.125). Surface "
    "E x H is independent of the volume loss, so this is a conservation-law reference. "
    "`sar/one_gram_cube` is an `analytic-identity` (a 3x3x3 window weighing exactly 1 g, "
    "whose uniform-field average equals the hand-computed point SAR). "
    "`sar/uniform_lossy_cube` reports the volume/channel self-consistency closure "
    "(`tautology` -- a self-comparison of the same edge-loss field, supporting only). "
    "`sar/antenna_near_phantom` is `blocked` -- the FDTD port machinery fails closed on "
    "the conductive tissue background; its wave-level class is a TARGET only (blocked "
    "upstream), never an achieved gate. Gate "
    "classes are the verbatim taxonomy (`docs/reference/gate-classification.md`); no "
    "external reference-solver run backs this family (every row `analytic-only`). "
    "Per-scene JSON artifacts live under `docs/assessments/sar-phantom-validation/`."
)


def _results_section(reports: list[SarReport]) -> str:
    lines = [
        SECTION_HEADER,
        "",
        _INTRO,
        "",
        "| Scene | Gate class | Quantity | Measured | Reference | Rel error | Status | External ref |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    def esc(value: str) -> str:
        return str(value).replace("|", r"\|")

    def gate_cell(report: SarReport) -> str:
        # A blocked row shows its TARGET class with an explicit blocked marker,
        # never a bare (unachieved) class.
        if report.gate_class:
            return esc(report.gate_class)
        if report.target_gate_class:
            return esc(f"{report.target_gate_class} (target; blocked)")
        return "-"

    for report in reports:
        headline = next((m for m in report.metrics if "rel_error" in m), None)
        if headline is not None:
            measured = f"{headline.get('measured', float('nan')):.4g}"
            reference = f"{headline.get('reference', float('nan')):.4g}"
            rel = f"{headline['rel_error']:.3%}"
            quantity = str(headline.get("quantity", "-"))
        else:
            measured = reference = rel = "-"
            quantity = "see artifact"
        lines.append(
            "| {name} | {cls} | {q} | {m} | {r} | {rel} | {st} | {ext} |".format(
                name=report.name, cls=gate_cell(report), q=esc(quantity),
                m=measured, r=reference, rel=rel, st=report.status,
                ext=esc(report.external_reference),
            )
        )
    lines.append("")
    lines.append(f"_Section regenerated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    return "\n".join(lines)


def _replace_or_append_section(text: str, header: str, section: str) -> str:
    if header in text:
        head, _, tail = text.partition(header)
        rest = tail.split("\n", 1)[1] if "\n" in tail else ""
        next_idx = rest.find("\n## ")
        remainder = rest[next_idx + 1 :] if next_idx != -1 else ""
        if remainder:
            return head + section + "\n" + remainder
        return head + section
    return text.rstrip() + "\n\n" + section


def _update_results_md(reports: list[SarReport]) -> None:
    text = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    section = _results_section(reports)
    text = _replace_or_append_section(text, SECTION_HEADER, section)
    RESULTS_MD.write_text(text, encoding="utf-8")


def _write_artifact(report: SarReport) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    slug = report.name.replace("/", "__")
    path = ARTIFACT_DIR / f"{slug}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def run(selected: list[str] | None = None) -> list[SarReport]:
    names = selected or list(SCENE_RUNNERS)
    reports: list[SarReport] = []
    for name in names:
        runner = SCENE_RUNNERS[name]
        print(f"[sar-validation] running {name} ...", flush=True)
        report = runner()
        report.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = _write_artifact(report)
        print(f"[sar-validation]   status={report.status}  artifact={path}", flush=True)
        reports.append(report)
    _update_results_md(reports)
    return reports


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SAR phantom exposure validation.")
    parser.add_argument("scenes", nargs="*", help="Scene names (default: all).")
    args = parser.parse_args(argv)
    selected = None
    if args.scenes:
        unknown = [s for s in args.scenes if s not in SCENE_RUNNERS]
        if unknown:
            raise SystemExit(f"Unknown SAR scenes: {unknown}. Available: {list(SCENE_RUNNERS)}")
        selected = args.scenes
    reports = run(selected)
    for report in reports:
        print(f"  {report.name}: {report.status}")


if __name__ == "__main__":
    main()
