"""GPU fixed-sphere TFSF Rayleigh refinement under the shipped material policy."""

from __future__ import annotations

from pathlib import Path
import sys

from _default_policy_study import (
    POLICY, ROOT, parser_for, power_fit, richardson_triple, run_study, source_hash,
)

sys.path.insert(0, str(ROOT))

TIERS = {str(cells): 0.01 / cells for cells in (1, 2, 4)}
FIXTURE = ROOT / "tests/validation/physics/test_postprocess_end_to_end_validation.py"


def build_parser():
    return parser_for("rayleigh", TIERS)


def summarize(records):
    rows = sorted(records, key=lambda r: -r["config"]["dx_m"])
    h = [r["config"]["dx_m"] for r in rows]
    ratios = [r["sigma_max_ratio"] for r in rows]
    summary = {
        "study": "rayleigh", "policy": POLICY, "qualification": "pending GPU data review",
        "resolutions": [{"tier": r["config"]["tier"], "dx_m": r["config"]["dx_m"],
                         **{k: r[k] for k in ("sigma_max_ratio", "rel_l2", "phi0_rel", "phi90_rel")}}
                        for r in rows],
        "missing_tiers": [tier for tier in TIERS if tier not in {r["config"]["tier"] for r in rows}],
        "fits": None, "one_cell_derived_band": None, "trend_toward_analytic": None,
    }
    if len(rows) < 3:
        return summary
    fits = [power_fit(h, ratios, p, 0.01) for p in (1, 2)]
    primary, alternative = fits[1], fits[0]
    center = primary["predicted_target"]
    residual_margin = 2.0 * primary["target_prediction_se"]
    model_margin = abs(center - alternative["predicted_target"])
    margin = residual_margin + model_margin
    errors = [abs(ratio - 1.0) for ratio in ratios]
    decreasing = all(a > b for a, b in zip(errors[:-1], errors[1:]))
    limits_closer = all(abs(fit["limit"] - 1.0) < errors[-1] for fit in fits)
    summary.update(
        fits=fits, observed_order=richardson_triple(h, ratios),
        one_cell_derived_band={
            "center": center, "margin": margin, "lower": max(0.0, center - margin),
            "upper": center + margin, "residual_margin": residual_margin,
            "model_margin": model_margin,
            "method": "p=2 center; 2 residual prediction SE plus p=1 versus p=2 center spread",
            "status": "provisional calibration envelope, not a confidence interval or release gate",
        },
        trend_toward_analytic={
            "absolute_errors_from_one": errors, "errors_strictly_decrease": decreasing,
            "both_fit_limits_closer_than_finest_tier": limits_closer,
            "approaches_one_in_this_study": decreasing and limits_closer,
            "note": "Three tiers cannot establish an exact limit; finite-ka Rayleigh model error remains.",
        },
    )
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    digest = source_hash(Path(__file__), FIXTURE)
    configs = {tier: {"study": "rayleigh", "tier": tier, "radius_cells": int(tier),
                      "dx_m": dx, "policy": POLICY, "device": "cuda", "backend": "native",
                      "seed": 0, "source_sha256": digest, "nf2ff_batch_size": 128}
               for tier, dx in TIERS.items()}

    def measure(tier):
        import witwin.maxwell as mw
        from tests.validation.physics.test_postprocess_end_to_end_validation import _run_tfsf_rayleigh_rcs_validation

        return {"dx_m": TIERS[tier], "radius_cells": int(tier),
                **_run_tfsf_rayleigh_rcs_validation(
                    TIERS[tier], subpixel_samples=mw.SubpixelSpec(samples=4, averaging="kottke_edge"),
                    device="cuda", batch_size=128,
                )}

    run_study(args, configs, measure, summarize)


if __name__ == "__main__":
    main()
