import json
import sys
from types import SimpleNamespace

import pytest

from benchmark import runner


class _Job:
    def __init__(self, name, estimate):
        self.task_id, self._estimate = f"task-{name}", estimate

    def estimate_cost(self, verbose=False):
        return self._estimate


@pytest.fixture
def priced(monkeypatch, tmp_path):
    estimates, ran = {}, []
    monkeypatch.delenv(runner.NO_CLOUD_ENV_VAR, raising=False)
    monkeypatch.setattr(runner, "_reference_job",
                        lambda name, *a, **k: (_Job(name, estimates[name]), None, None))
    monkeypatch.setattr(runner, "_run_reference_job", lambda name, *a: ran.append(name))
    monkeypatch.setattr(runner, "load_tidy3d_result", lambda *a, **k: (_ for _ in ()).throw(ValueError("stale")))
    monkeypatch.setitem(sys.modules, "tidy3d.web", SimpleNamespace(real_cost=lambda task_id, verbose=False: 0.025))
    ledger = tmp_path / "ledger.json"

    def call(rows, **caps):
        estimates.clear()
        estimates.update(rows)
        runner.generate_priced_references(list(rows), ledger_path=ledger, **caps)
        return json.loads(ledger.read_text())

    return call, ran, ledger


def test_priced_references_run_only_when_every_estimate_fits(priced):
    call, ran, ledger = priced
    caps = dict(max_cost_per_row=0.025, max_total_cost=0.07, row_caps={"full_tensor_slab": 0.06})
    with pytest.raises(SystemExit, match="over their cap"):
        call({"dipole_vacuum": 0.025, "dipole_ey": 0.026}, **caps)
    assert ran == [] and json.loads(ledger.read_text())["approved_to_run"] is False
    with pytest.raises(SystemExit, match="estimated total"):
        call({"dipole_vacuum": 0.025, "full_tensor_slab": 0.05}, **caps)
    assert ran == []
    result = call({"dipole_vacuum": 0.025, "full_tensor_slab": 0.04}, **caps)
    assert ran == ["dipole_vacuum", "full_tensor_slab"]
    assert result["billed_total"] == pytest.approx(0.05)
    assert result["rows"]["full_tensor_slab"] == {"task_id": "task-full_tensor_slab", "estimate": 0.04,
                                                  "budgeted": 0.04, "cap": 0.06, "billed": 0.025}


def test_priced_references_budget_the_billing_floor_and_price_only_submits_nothing(priced):
    call, ran, ledger = priced
    # A 0.0 estimate is billed at the floor: two such rows do not fit a 0.04 total.
    with pytest.raises(SystemExit, match="estimated total 0.0500"):
        call({"dipole_vacuum": 0.0, "dipole_ey": 0.0}, max_cost_per_row=0.025, max_total_cost=0.04, row_caps={})
    assert ran == []
    result = call({"dipole_vacuum": 0.0, "dipole_ey": 0.01}, max_cost_per_row=0.025, max_total_cost=0.05,
                  row_caps={}, price_only=True)
    assert ran == [] and result["price_only"] is True and result["approved_to_run"] is True
    assert result["estimated_total"] == pytest.approx(2 * runner.REFERENCE_BILLING_FLOOR)
    assert result["rows"]["dipole_vacuum"]["budgeted"] == runner.REFERENCE_BILLING_FLOOR
    assert "billed" not in result["rows"]["dipole_ey"]


def test_priced_references_refuse_valid_caches_and_no_cloud(priced, monkeypatch):
    call, ran, _ = priced
    monkeypatch.setattr(runner, "load_tidy3d_result", lambda *a, **k: {})
    with pytest.raises(SystemExit, match="already has a valid reference cache"):
        call({"dipole_vacuum": 0.025}, max_cost_per_row=0.025, max_total_cost=1.0, row_caps={})
    call({"dipole_vacuum": 0.025}, max_cost_per_row=0.025, max_total_cost=1.0, row_caps={}, replace_valid=True)
    assert ran == ["dipole_vacuum"]
    ran.clear()
    monkeypatch.setenv(runner.NO_CLOUD_ENV_VAR, "1")
    with pytest.raises(SystemExit, match="forbids"):
        call({"dipole_vacuum": 0.025}, max_cost_per_row=0.025, max_total_cost=1.0, row_caps={})
    assert ran == []
