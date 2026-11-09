"""CPU coverage for staged and local pinned baseline resolution."""

import json

import pytest

from tests.support.pinned_baselines import baseline_tree


@pytest.fixture
def staged_tree(tmp_path, monkeypatch):
    root = tmp_path / "staged"
    sha = "a" * 40
    package = root / "witwin-maxwell" / sha / "witwin"
    package.mkdir(parents=True)
    (package / "x.py").write_text("x = 1\n")
    manifest = [{
        "project": "witwin-maxwell", "rev": "pinned", "sha": sha,
        "files": 1, "bytes": 6, "sha256": "b" * 64,
    }]
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("WITWIN_BASELINE_TREES", str(root))
    monkeypatch.delenv("WITWIN_REQUIRE_BASELINES", raising=False)
    return root


def test_staged_tree_is_copied(staged_tree, tmp_path):
    tree = baseline_tree("pinned", tmp_path)
    source = staged_tree / "witwin-maxwell" / ("a" * 40) / "witwin" / "x.py"
    copied = tree / "witwin" / "x.py"
    assert tree == tmp_path / "baseline"
    assert copied.read_text() == source.read_text() == "x = 1\n"
    copied.write_text("x = 2\n")
    assert source.read_text() == "x = 1\n"


def test_unknown_staged_revision_fails_without_require(staged_tree, tmp_path):
    with pytest.raises(pytest.fail.Exception, match="not available"):
        baseline_tree("unknown", tmp_path)


def test_ambiguous_staged_prefix_fails(staged_tree, tmp_path):
    path = staged_tree / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.append({**manifest[0], "sha": "a" * 39 + "b"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(pytest.fail.Exception, match="ambiguous"):
        baseline_tree("aaaa", tmp_path)


def test_unknown_local_revision_skips_or_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("WITWIN_BASELINE_TREES", raising=False)
    monkeypatch.delenv("WITWIN_REQUIRE_BASELINES", raising=False)
    rev = "0000000000000000000000000000000000000bad"
    with pytest.raises(pytest.skip.Exception, match="not available"):
        baseline_tree(rev, tmp_path)
    monkeypatch.setenv("WITWIN_REQUIRE_BASELINES", "1")
    with pytest.raises(pytest.fail.Exception, match="not available"):
        baseline_tree(rev, tmp_path)
