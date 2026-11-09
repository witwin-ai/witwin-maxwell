"""CPU checks for study fits, resumability and shared physical fixtures."""

import importlib
from pathlib import Path

import numpy as np
import pytest

import witwin.maxwell as mw


@pytest.fixture
def studies(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/dev"))
    return tuple(importlib.import_module(name) for name in (
        "_default_policy_study", "requalify_sibc_default", "requalify_rayleigh_default",
    ))


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.2])
def test_irregular_richardson_recovers_order_and_limit(studies, p):
    common, _, _ = studies
    h = np.array([0.60, 0.50, 0.42]) * 1e-3
    values = 7.0 + 2.0 * (h / h[0]) ** p
    fit = common.richardson_triple(h, values)
    assert fit["status"] == "fit"
    assert fit["p"] == pytest.approx(p, rel=1e-8)
    assert fit["extrapolated"] == pytest.approx(7.0, rel=1e-8)


@pytest.mark.parametrize("values", [[1.0, 1.0, 1.0], [1.0, 2.0, 1.5], [1.0, 2.0, 4.0]])
def test_richardson_does_not_invent_a_limit(studies, values):
    fit = studies[0].richardson_triple([1.0, 0.5, 0.25], values)
    assert fit["extrapolated"] is None
    assert fit["p"] is None


def test_fixed_order_fit_and_rayleigh_band(studies):
    common, _, rayleigh = studies
    h = np.array([0.01, 0.005, 0.0025])
    ratios = 1.0 + 0.6 * (h / h[0]) ** 2
    fit = common.power_fit(h, ratios, 2, 0.01)
    assert fit["limit"] == pytest.approx(1.0)
    assert fit["predicted_target"] == pytest.approx(1.6)
    assert fit["residual_std"] < 1e-14
    rows = [{"config": {"tier": str(c), "dx_m": dx}, "sigma_max_ratio": ratio,
             "rel_l2": 0.04, "phi0_rel": 0.03, "phi90_rel": 0.02}
            for c, dx, ratio in zip((1, 2, 4), h, ratios)]
    summary = rayleigh.summarize(rows)
    band = summary["one_cell_derived_band"]
    assert band["center"] == pytest.approx(1.6)
    assert band["margin"] > 0  # The alternative-order model retains uncertainty.
    assert band["lower"] <= 1.6 <= band["upper"]
    assert summary["trend_toward_analytic"]["errors_strictly_decrease"]
    assert summary["fits"][1]["limit"] == pytest.approx(1.0)
    # A wrong fixed-order model can disagree even on exact second-order data.
    assert not summary["trend_toward_analytic"]["approaches_one_in_this_study"]
    for row, ratio in zip(rows, [1.6, 1.3, 1.2]):
        row["sigma_max_ratio"] = ratio
    assert rayleigh.summarize(rows)["trend_toward_analytic"]["approaches_one_in_this_study"]
    rows[-1]["sigma_max_ratio"] = 1.8
    assert not rayleigh.summarize(rows)["trend_toward_analytic"]["approaches_one_in_this_study"]
    assert rayleigh.summarize(rows[:2])["one_cell_derived_band"] is None


def test_fit_residual_has_one_degree_of_freedom(studies):
    fit = studies[0].power_fit([1, 0.5, 0.25], [1.6, 1.18, 1.03], 2, 1)
    assert fit["residual_dof"] == 1
    assert fit["residual_std"] == pytest.approx(np.linalg.norm(fit["residuals"]))
    assert fit["target_prediction_se"] > fit["residual_std"] > 0


def test_sibc_summary_uses_finest_triple_and_fine_denominator(studies):
    _, sibc, _ = studies
    rows = [{"config": {"tier": f"resolved-{dx}", "kind": "resolved", "dx_m": dx},
             "absorbed_power_W": 7.0 + dx**2} for dx in (0.6, 0.5, 0.42, 0.35)]
    rows.append({"config": {"tier": "sibc-0.8", "kind": "sibc", "dx_m": 0.8},
                 "absorbed_power_W": 8.0})
    summary = sibc.summarize(rows)
    assert summary["P_inf_W"] == pytest.approx(7.0)
    pair = summary["coarsest_resolved_pair_below_6_percent"]
    assert pair["coarse_dx_m"] == 0.6
    assert pair["relative_change"] == pytest.approx((7.36 - 7.25) / 7.25)
    assert summary["sibc_vs_extrapolated_reference"][0]["relative_gap"] == pytest.approx(1 / 7)
    assert sibc.summarize([])["P_inf_W"] is None


def test_cli_and_resume_reject_mixed_sources(studies, tmp_path):
    common, sibc, rayleigh = studies
    assert sibc.build_parser().parse_args(["--tiers", "resolved-0.42"]).tiers == ["resolved-0.42"]
    assert rayleigh.build_parser().parse_args(["--tiers", "1", "4"]).tiers == ["1", "4"]
    with pytest.raises(SystemExit):
        sibc.build_parser().parse_args(["--tiers", "unknown"])
    config = {"tier": "1", "policy": common.POLICY, "source_sha256": "abc"}
    common.write_json(tmp_path / "1.json", {"config": config, "status": "complete"})
    assert len(common.load_records(tmp_path, {"1": config})) == 1
    with pytest.raises(ValueError, match="mismatch"):
        common.load_records(tmp_path, {"1": dict(config, source_sha256="def")})
    args = rayleigh.build_parser().parse_args(["--tiers", "1", "--out", str(tmp_path)])
    called = []
    common.run_study(args, {"1": config}, lambda _: called.append(True), lambda rows: {"count": len(rows)})
    assert called == []  # Completed tiers resume without even requiring CUDA.
    assert (tmp_path / "summary.json").exists()


def test_shared_builders_keep_default_policy_and_geometry():
    from tests.validation.physics.test_sibc_cylinder_convergence import _cylinder_scene
    from tests.validation.physics.test_postprocess_end_to_end_validation import _rayleigh_scenes

    policy = mw.SubpixelSpec()
    assert policy.samples == (4, 4, 4) and policy.averaging == "kottke_edge"
    scene = _cylinder_scene(0.003, mw.Material(sigma_e=30), subpixel_samples=policy, device="cpu")
    assert scene.subpixel == policy
    pairs = [_rayleigh_scenes(dx, subpixel_samples=policy, device="cpu")
             for dx in (0.01, 0.005, 0.0025)]
    for scattering, reference in pairs:
        assert scattering.subpixel == reference.subpixel == policy
        assert scattering.domain == pairs[0][0].domain
        assert scattering.structures[0].geometry.radius == 0.01


def test_tiny_cpu_reference_cylinder_measurement():
    from tests.validation.physics.test_sibc_cylinder_convergence import _absorbed_power

    # Two steps only check compilation, execution and the shared flux reader, not physics.
    power = _absorbed_power(
        0.003, mw.Material(eps_r=1.0, sigma_e=30), subpixel_samples=mw.SubpixelSpec(),
        device="cpu", backend="cpu_reference",
        run_time=mw.TimeConfig(time_steps=2, transient_cycles=0, steady_cycles=0),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )
    assert np.isfinite(power)


def test_cpu_reference_rayleigh_rejects_unsupported_tfsf_kernel():
    from tests.validation.physics.test_postprocess_end_to_end_validation import _run_tfsf_rayleigh_rcs_validation

    with pytest.raises(NotImplementedError, match="addBatchedReferenceSourcePatches3D"):
        _run_tfsf_rayleigh_rcs_validation(
            0.01, subpixel_samples=mw.SubpixelSpec(), device="cpu", backend="cpu_reference",
            run_time=mw.TimeConfig(time_steps=2, transient_cycles=0, steady_cycles=0),
            angular_shape=(3, 5), batch_size=4,
        )
