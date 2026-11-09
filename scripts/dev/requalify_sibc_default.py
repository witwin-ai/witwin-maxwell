"""GPU cylinder absorption convergence under the shipped material policy."""

from __future__ import annotations

from pathlib import Path
import sys

from _default_policy_study import (
    POLICY, ROOT, adjacent_changes, parser_for, richardson_triple, run_study, source_hash,
)

sys.path.insert(0, str(ROOT))

TIERS = {**{f"resolved-{dx:g}": ("resolved", dx * 1e-3)
            for dx in (0.6, 0.5, 0.42, 0.35, 0.30)},
         **{f"sibc-{dx:g}": ("sibc", dx * 1e-3) for dx in (1.2, 0.8)}}
FIXTURE = ROOT / "tests/validation/physics/test_sibc_cylinder_convergence.py"


def build_parser():
    return parser_for("sibc", TIERS)


def summarize(records):
    groups = {}
    for kind in ("resolved", "sibc"):
        rows = sorted((r for r in records if r["config"]["kind"] == kind),
                      key=lambda r: -r["config"]["dx_m"])
        h = [r["config"]["dx_m"] for r in rows]
        values = [r["absorbed_power_W"] for r in rows]
        groups[kind] = {"tiers": [r["config"]["tier"] for r in rows],
                        "adjacent_changes": adjacent_changes(h, values)}
        if kind == "resolved":
            triples = [richardson_triple(h[i:i+3], values[i:i+3])
                       for i in range(len(h) - 2)]
            groups[kind]["richardson_triples"] = triples
    candidates = [pair for pair in groups["resolved"]["adjacent_changes"]
                  if pair["relative_change"] is not None and pair["relative_change"] < 0.06]
    triples = groups["resolved"]["richardson_triples"]
    limit = triples[-1]["extrapolated"] if triples else None
    valid_limit = limit is not None and limit > 0
    return {
        "study": "sibc", "policy": POLICY, "qualification": "pending GPU data review",
        "completed_tiers": len(records), "expected_tiers": len(TIERS), **groups,
        "coarsest_resolved_pair_below_6_percent": candidates[0] if candidates else None,
        "P_inf_W": limit,
        "reference_fit": "finest consecutive available resolved triple; inspect all orders",
        "sibc_vs_extrapolated_reference": [
            {"tier": r["config"]["tier"],
             "relative_gap": abs(r["absorbed_power_W"] - limit) / limit if valid_limit else None}
            for r in records if r["config"]["kind"] == "sibc"],
        "missing_tiers": [tier for tier in TIERS
                          if tier not in {r["config"]["tier"] for r in records}],
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    digest = source_hash(Path(__file__), FIXTURE)
    configs = {tier: {"study": "sibc", "tier": tier, "kind": kind, "dx_m": dx,
                      "policy": POLICY, "device": "cuda", "backend": "native",
                      "seed": 0, "source_sha256": digest}
               for tier, (kind, dx) in TIERS.items()}

    def measure(tier):
        import witwin.maxwell as mw
        from tests.validation.physics.test_sibc_cylinder_convergence import _SIGMA, _absorbed_power, _skin_depth

        kind, dx = TIERS[tier]
        material = (mw.Material(eps_r=1.0, sigma_e=_SIGMA) if kind == "resolved"
                    else mw.LossyMetalMedium(conductivity=_SIGMA))
        power = _absorbed_power(
            dx, material, subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_edge"),
            device="cuda",
        )
        if not power > 0:
            raise ValueError(f"{tier}: nonpositive absorbed power {power}")
        return {"dx_m": dx, "cells_per_skin_depth": _skin_depth() / dx,
                "absorbed_power_W": power}

    run_study(args, configs, measure, summarize)


if __name__ == "__main__":
    main()
