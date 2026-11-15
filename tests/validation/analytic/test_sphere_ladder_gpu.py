"""Small GPU rung of the Mie size ladder (benchmark/analytic/sphere_ladder.py)."""
import pytest
import torch

from benchmark.analytic import sphere_ladder as ladder

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_incident_field_and_empty_box_power():
    output = ladder.run_case(ladder.ladder_case("vacuum", 1, 15))
    record = output["record"]
    assert record["near_rel_l2"] < BOUNDS["vacuum"]["near_rel_l2"]
    # No scatterer: the scattered-field box sees no power, the total-field box no net flux.
    assert abs(record["sigma_sca"]) < 1e-3
    assert abs(record["sigma_abs"]) < 1e-3


@pytest.mark.parametrize("kind", ["pec", "eps"])
def test_one_wavelength_sphere_matches_mie(kind):
    output = ladder.run_case(ladder.ladder_case(kind, 1, 15, eps=4.0))
    record = output["record"]
    for key, bound in BOUNDS[kind].items():
        assert record[key] < bound, (key, record[key], bound)


# Bounds: about twice the errors measured at D = 1 lambda, 15 cells per wavelength
# in the densest medium, on the CUDA core path (A6000, 2026-11-11): vacuum near
# 0.023; PEC (staircase) extinction 0.023, RCS 0.17, near 0.12; eps = 4 extinction
# 0.031, RCS 0.037, near 0.027 (evidence ladder CSVs).
BOUNDS = {
    "vacuum": {"near_rel_l2": 0.05},
    "pec": {"ext_rel_err": 0.05, "rcs_rel_l2": 0.35, "near_rel_l2": 0.25},
    "eps": {"ext_rel_err": 0.065, "rcs_rel_l2": 0.075, "near_rel_l2": 0.055},
}
