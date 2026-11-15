"""D-P1 numerical lock records of the committed layered evidence (host only).

The R6, R6b, R4b and A4 (F-D, F-E, F-F) evidence values are locked by value: scripts/dev/rfic/evidence_lock.py
re-acquires every producer and judges each committed numeric field with the tiers of the
numerical-equivalence standard, and a renewal writes the moved values together with a record.
These tests check that every committed file carries a well-formed record whose last renewal has
no unaccepted T2, that each renewed or accepted field names an existing committed value, and
that no source-hash pin is reintroduced.
"""
import json
from pathlib import Path
import re

import pytest

DATA = Path(__file__).with_name("data")
FILES = ("fixture_r6_native_evidence.json", "fixture_r6b_shunt_evidence.json", "fixture_r4b_closed_comparison.json",
         "fixture_fd_mim_evidence.json", "fixture_fe_lines_evidence.json", "fixture_ff_octagon_evidence.json")
SOURCE_PINS = re.compile(r"(^|_)source_sha256$|^unchanged_historical_sha256$|^producer_fixture_sha256$")


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _leaf(data, name, field):
    prefix, path = field.split(":", 1)
    assert prefix == name, field
    for key in path.split("/"):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


@pytest.mark.parametrize("name", FILES)
def test_evidence_has_no_source_hash_pins(name):
    data = json.loads((DATA / name).read_text())
    assert not [key for key in _keys(data) if SOURCE_PINS.search(key)]


@pytest.mark.parametrize("name", FILES)
def test_numerical_lock_record(name):
    data = json.loads((DATA / name).read_text())
    lock = data["numerical_lock"]
    assert lock["rule"].startswith("D-P1 ")
    assert Path(__file__).resolve().parents[3].joinpath("scripts/dev/rfic/evidence_lock.py").is_file()
    assert lock["renewals"], "the D-P1 conversion requires one renewal from a full re-acquisition"
    for renewal in lock["renewals"]:
        assert re.fullmatch(r"[0-9a-f]{40}", renewal["acquisition_commit"])
        assert renewal["acquisition"].startswith("<evidence-root>/")
        assert renewal["tier_table"] == renewal["acquisition"] + "/tier_table.json"
        assert re.fullmatch(r"[0-9a-f]{64}", renewal["tier_table_sha256"])
        summary = renewal["summary"]
        assert set(summary) == {"T0", "T1", "accepted_T2", "T2"}
        assert summary["T2"] == 0
        assert summary["accepted_T2"] == len(renewal["accepted_t2"])
        assert len(renewal["renewed_fields"]) == summary["T1"] + summary["accepted_T2"]
        for field in renewal["renewed_fields"]:
            value = _leaf(data, name, field)
            assert isinstance(value, (float, list)) and not isinstance(value, bool), field
        for entry in renewal["accepted_t2"]:
            assert entry["field"] in renewal["renewed_fields"]
            assert entry["approval"].strip() and entry["analysis"].strip()
