"""Record original pytest phases and identities without changing outcomes."""
from pathlib import Path
from collections import Counter
import json
import os
import time
import pytest
from shard_common import object_sha, runtime_identity, write_json
from module_environment import load_policy, module_key, collection_scope, artifact_environment

REQUEST = json.loads(Path(os.environ["PCB_SHARD_REQUEST"]).read_text())
DIRECTORY = Path(REQUEST["attempt_dir"])
STREAM = (DIRECTORY / "events.jsonl").open("x", buffering=1)
COUNTS = Counter()
POLICY = load_policy(REQUEST["module_environment_file"], REQUEST["source_root"]) if REQUEST.get("module_environment_file") else None
SCOPED_MODULES = set()


def event(kind, **values):
    STREAM.write(json.dumps({"event": kind, "time_unix_s": time.time(),
                             "monotonic_s": time.monotonic(), **values}, default=str) + "\n")
    STREAM.flush()
    os.fsync(STREAM.fileno())


def pytest_sessionstart(session):
    event("session_start", arguments=list(session.config.invocation_params.args), pid=os.getpid())
    before = runtime_identity(REQUEST)
    if before["gsp_environment"] != REQUEST["expected_gsp_environment"]:
        event("identity_failure", expected_gsp=REQUEST["expected_gsp_environment"], actual_gsp=before["gsp_environment"])
        pytest.exit("Actual module artifact environment differs before collection/execution", returncode=4)
    write_json(DIRECTORY / "identity-before.json", before)
    expected = REQUEST.get("expected_binding_sha256")
    if expected and before["binding_sha256"] != expected:
        event("identity_failure", expected=expected, actual=before["binding_sha256"])
        pytest.exit("Frozen source/core/control identity changed before test execution", returncode=4)
    expected_versions = REQUEST.get("expected_versions")
    if expected_versions and before["versions"] != expected_versions:
        event("identity_failure", expected_versions=expected_versions, actual_versions=before["versions"])
        pytest.exit("Runtime versions changed before test execution", returncode=4)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    outcome = yield
    collector = outcome.get_result()
    if collector is None or POLICY is None:
        return
    key = module_key(module_path, REQUEST["source_root"])
    original = collector.collect

    def collect_with_artifacts():
        with collection_scope(POLICY, key) as expected:
            SCOPED_MODULES.add(key)
            event("module_collection_start", module=key, gsp_environment=expected)
            try:
                result = list(original())
                if artifact_environment(os.environ) != expected:
                    raise RuntimeError("Test-module import/collection changed its artifact environment")
                return result
            finally:
                event("module_collection_end", module=key, actual_gsp_environment=artifact_environment(os.environ))
    collector.collect = collect_with_artifacts


def pytest_collection_finish(session):
    actual = [item.nodeid for item in session.items]
    if POLICY is not None:
        expected_modules = {module_key(item.path, REQUEST["source_root"]) for item in session.items} & set(POLICY["modules"])
        if not expected_modules <= SCOPED_MODULES:
            event("collection_failure", missing_module_scopes=sorted(expected_modules - SCOPED_MODULES))
            pytest.exit("A configured module collector bypassed artifact scoping", returncode=4)
        if artifact_environment(os.environ) != REQUEST["expected_gsp_environment"]:
            event("identity_failure", actual_gsp_environment=artifact_environment(os.environ))
            pytest.exit("Module collection leaked artifact environment", returncode=4)
    event("collection_finish", count=len(actual), nodeids=actual,
          absolute_selectors={item.nodeid: str(item.path.resolve()) + "::" + item.nodeid.split("::", 1)[1]
                              for item in session.items})
    expected = REQUEST.get("expected_nodeids")
    if expected is not None and actual != expected:
        event("selection_failure", expected=expected, actual=actual)
        pytest.exit("Actual shard selection/order differs from frozen collected nodes", returncode=4)


def pytest_collectreport(report):
    if report.failed:
        event("collection_failure", nodeid=report.nodeid, original_detail=str(report.longrepr))


def pytest_runtest_logstart(nodeid, location):
    event("node_start", nodeid=nodeid, location=location)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    event("phase_start", nodeid=item.nodeid, phase="setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    event("phase_start", nodeid=item.nodeid, phase="call")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    event("phase_start", nodeid=item.nodeid, phase="teardown")


def pytest_runtest_logreport(report):
    values = {"nodeid": report.nodeid, "phase": report.when, "outcome": report.outcome,
              "duration_s": report.duration, "wasxfail": getattr(report, "wasxfail", None),
              "user_properties": report.user_properties}
    if report.failed or report.skipped:
        values["original_detail"] = str(report.longrepr)
    COUNTS[(report.when, report.outcome)] += 1
    event("test_report", **values)


def pytest_runtest_logfinish(nodeid, location):
    event("node_end", nodeid=nodeid)


def pytest_sessionfinish(session, exitstatus):
    event("session_finish", exitstatus=int(exitstatus),
          phase_counts={f"{phase}:{outcome}": n for (phase, outcome), n in COUNTS.items()})


def pytest_unconfigure(config):
    STREAM.flush()
    STREAM.close()
