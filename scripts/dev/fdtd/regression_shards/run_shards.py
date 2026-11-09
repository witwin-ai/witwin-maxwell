"""Durable, source-bound pytest shards. No test outcome is rewritten."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import fcntl
import json
import os
import signal
import shlex
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from shard_common import (SCHEMA, digest, environment_binding, file_binding, object_sha,
                          observed_live, process_fingerprint, read_events, write_json)

from module_environment import load_policy, for_module, artifact_environment, module_key

CONTROL = Path(__file__).resolve().parent
STOP_SIGNAL = None


def request_stop(signum, frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def group_members(record):
    if record["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return []
    leader = process_fingerprint(record["pid"])
    if leader and leader["start_ticks"] != record["start_ticks"]:
        raise RuntimeError("PID reused; refusing to signal any process group")
    if record["pgid"] != record["pid"] or record["sid"] != record["pid"]:
        raise RuntimeError("Worker did not own its new session/process group")
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        item = process_fingerprint(int(path.name))
        if (item and item["pgid"] == record["pgid"] and item["sid"] == record["sid"]
                and item["start_ticks"] >= record["start_ticks"] and item["state"] not in ("Z", "X")):
            members.append(item)
    return members


def terminate_owned_group(record, grace_seconds):
    actions = []
    for signum, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2.0)):
        members = group_members(record)
        if not members:
            break
        actions.append({"signal": int(signum), "time_unix_s": time.time(), "verified_members": members})
        try:
            os.killpg(record["pgid"], signum)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and group_members(record):
            time.sleep(0.05)
    if group_members(record):
        raise RuntimeError("Owned worker group remains live after SIGKILL; manual review required")
    return actions


def current_node(events):
    active = None
    for event in events:
        if event["event"] == "node_start":
            active = event
        elif event["event"] == "node_end" and active and event["nodeid"] == active["nodeid"]:
            active = None
    return active


def supervise(directory, config, process=None):
    """Re-adopt a worker after controller death without claiming a fake exit code."""
    record = json.loads((directory / "worker.json").read_text())
    started = record["launched_monotonic_s"]
    supervision = {"worker": record, "controller": process_fingerprint(os.getpid()),
                   "re_adopted": process is None, "signal_actions": []}
    last_write = 0.0
    reason, target = None, None
    while observed_live(record["process"]):
        events = read_events(directory / "events.jsonl")
        active = current_node(events)
        now = time.time()
        if STOP_SIGNAL is not None:
            reason, target = "controller_interrupted", active["nodeid"] if active else None
        elif time.monotonic() - started > config["shard_timeout"]:
            reason, target = "shard_timeout", active["nodeid"] if active else None
        elif active and time.monotonic() - active["monotonic_s"] > config["node_timeout"]:
            reason, target = "node_timeout", active["nodeid"]
        if reason:
            supervision["signal_actions"] = terminate_owned_group(record["process"], config["termination_grace"])
            break
        if now - last_write >= 2:
            supervision.update(heartbeat_unix_s=now, observed_worker=process_fingerprint(record["process"]["pid"]),
                               current_node=active)
            write_json(directory / "supervision.json", supervision)
            last_write = now
        time.sleep(config["poll_seconds"])
    # An exited leader can leave a hung grandchild retaining CUDA. Clean only its
    # verified session; no global pkill, process-name matching or other job kill.
    if group_members(record["process"]):
        supervision["signal_actions"].extend(terminate_owned_group(record["process"], config["termination_grace"]))
    actual_returncode = process.wait() if process is not None else None
    marker_path = directory / "process-result.json"
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
    if actual_returncode is None and marker:
        returncode = marker["pytest_exit_code"] if marker["identities_unchanged"] else 4
        exit_evidence = "durable child marker; not an observed waitpid status"
    else:
        returncode = actual_returncode
        exit_evidence = "observed Popen.wait status" if process is not None else "unavailable after orphaned abrupt exit"
    supervision.update(finished_unix_s=time.time(), termination_reason=reason, timeout_or_interrupt_target=target,
                       returncode=returncode, exit_evidence=exit_evidence, process_result=marker,
                       final_worker_fingerprint=process_fingerprint(record["process"]["pid"]))
    write_json(directory / "supervision.json", supervision)
    return supervision


def make_environment(source, policy=None):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(CONTROL) + os.pathsep + str(source) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["PWD"] = str(source)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return for_module(environment, policy, None)


def new_attempt(output, config, environment, selection, *, collection=False, number=0, contract=None):
    directory = output / ("collection" if collection else f"attempt-{number:05d}")
    directory.mkdir(exist_ok=False)
    selectors = config["test_paths"] if collection else [contract["node_selectors"][node] for node in selection]
    args = [*selectors, "--rootdir=" + config["pytest_rootdir"], "-p", "shard_progress", "--tb=short", "-q",
            "--junitxml=" + str(directory / "junit.xml"), "--basetemp=" + str(directory / "pytest-temporary"),
            "-o", "junit_family=xunit1", "-o", "cache_dir=" + str(directory / "pytest-cache")]
    if collection:
        args.append("--collect-only")
    if config["gpu"]:
        args.append("--gpu")
    args.extend(config["pytest_args"])
    policy = config["module_environment"]
    selected_module = None if collection else module_key(
        contract["node_selectors"][selection[0]].split("::", 1)[0], config["source_root"])
    child_environment = for_module(environment, policy, selected_module)
    request = {"module_environment_file": config["module_environment_file"],
               "module_key": selected_module,
               "expected_gsp_environment": artifact_environment(child_environment), "schema": SCHEMA, "attempt_dir": str(directory), "source_root": config["source_root"],
               "control_root": str(CONTROL), "external_paths": config["external_paths"], "pytest_args": args}
    if contract:
        request.update(expected_binding_sha256=contract["binding_sha256"], expected_versions=contract["versions"],
                       expected_nodeids=selection)
    write_json(directory / "request.json", request)
    with (directory / "pytest.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, str(CONTROL / "pytest_child.py"), str(directory / "request.json")],
                                   cwd=config["source_root"], env=child_environment, start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    fingerprint = process_fingerprint(process.pid)
    if fingerprint is None:
        raise RuntimeError("Worker disappeared before its process identity could be recorded")
    write_json(directory / "worker.json", {"process": fingerprint, "launched_unix_s": time.time(),
               "launched_monotonic_s": time.monotonic(),
               "argv": [sys.executable, str(CONTROL / "pytest_child.py"), str(directory / "request.json")],
               "request_sha256": digest(directory / "request.json"), "environment_sha256": environment_binding(child_environment)["sha256"],
               "module_key": selected_module, "gsp_environment": artifact_environment(child_environment)})
    return directory, process


def xml_evidence(path):
    if not path.exists():
        return {"exists": False}
    try:
        root = ET.parse(path).getroot()
        return {"exists": True, "sha256": digest(path), "testcases": len(list(root.iter("testcase"))),
                "failures": len(list(root.iter("failure"))), "errors": len(list(root.iter("error"))),
                "skips": len(list(root.iter("skipped")))}
    except ET.ParseError as error:
        return {"exists": True, "sha256": digest(path), "parse_error": repr(error)}


def validate_current(config, environment, contract):
    actual = file_binding(config["source_root"], CONTROL, config["external_paths"], contract["core_root"])
    if actual != contract["binding"]:
        raise RuntimeError("Frozen source/core/helper/control files changed; resume/execution refused")
    if environment_binding(environment) != contract["environment"]:
        raise RuntimeError("Frozen numerical/test/toolchain environment changed; resume refused")
    return actual


def finalise_attempt(directory, config, environment, contract, supervision):
    # Even an os._exit worker cannot hide a concurrent source change: the
    # controller rehashes all bound files independently after worker termination.
    after = validate_current(config, environment, contract)
    events = read_events(directory / "events.jsonl")
    before_path = directory / "identity-before.json"
    child_before = json.loads(before_path.read_text()) if before_path.exists() else None
    expected_gsp = json.loads((directory / "request.json").read_text())["expected_gsp_environment"]
    identity_ok = bool(child_before and child_before.get("gsp_environment") == expected_gsp
                       and child_before["binding"] == contract["binding"]
                       and child_before["versions"] == contract["versions"])
    child_after_path = directory / "identity-after.json"
    child_after = json.loads(child_after_path.read_text()) if child_after_path.exists() else None
    if child_after:
        identity_ok = identity_ok and child_after["binding"] == after and child_after["versions"] == contract["versions"] and child_after.get("gsp_environment") == expected_gsp
    selection = json.loads((directory / "request.json").read_text())["expected_nodeids"]
    reports = defaultdict(list)
    ended = set()
    started = set()
    infrastructure = []
    for event in events:
        if event["event"] == "test_report":
            reports[event["nodeid"]].append(event)
        elif event["event"] == "node_end":
            ended.add(event["nodeid"])
        elif event["event"] == "node_start":
            started.add(event["nodeid"])
        elif event["event"] in ("collection_failure", "selection_failure", "identity_failure"):
            infrastructure.append(event)
    if not identity_ok:
        raise RuntimeError(f"Missing/mismatched actual child import identity: {directory}")
    decisions = {}
    for node in selection:
        phases = reports[node]
        if node in ended and any(e["phase"] == "teardown" for e in phases):
            if any(e["outcome"] == "failed" for e in phases):
                outcome = "failed"
            elif any(e["outcome"] == "skipped" and e["wasxfail"] for e in phases):
                outcome = "xfailed"
            elif any(e["outcome"] == "skipped" for e in phases):
                outcome = "skipped"
            elif any(e["wasxfail"] for e in phases):
                outcome = "xpassed"
            elif {e["phase"] for e in phases} == {"setup", "call", "teardown"} and all(e["outcome"] == "passed" for e in phases):
                outcome = "passed"
            else:
                outcome = "incomplete_phase_reports"
            decisions[node] = {"outcome": outcome, "terminal": True, "reports": phases}
        elif node in started:
            reason = supervision["termination_reason"]
            outcome = ("interrupted" if reason == "controller_interrupted" else
                       "timeout" if reason in ("node_timeout", "shard_timeout") else "process_crashed")
            decisions[node] = {"outcome": outcome, "terminal": outcome != "interrupted", "reports": phases,
                               "termination_reason": reason, "returncode": supervision["returncode"],
                               "timeout_or_interrupt_target": supervision["timeout_or_interrupt_target"]}
    session_finish = next((e for e in reversed(events) if e["event"] == "session_finish"), None)
    all_done = all(n in decisions and decisions[n]["terminal"] for n in selection)
    if supervision["termination_reason"] in ("node_timeout", "shard_timeout"):
        infrastructure.append({"event": "supervisor_timeout", "reason": supervision["termination_reason"],
                               "target": supervision["timeout_or_interrupt_target"],
                               "returncode": supervision["returncode"],
                               "detail": "A process timeout remains nonpass even if every node phase already finished."})
    elif supervision["termination_reason"] == "controller_interrupted" and all_done:
        infrastructure.append({"event": "interrupted_after_node_completion", "returncode": supervision["returncode"]})
    if supervision["termination_reason"] is None and supervision["returncode"] not in (0, 1):
        infrastructure.append({"event": "abnormal_process_exit", "returncode": supervision["returncode"],
                               "exit_evidence": supervision["exit_evidence"]})
    if all_done and not session_finish:
        infrastructure.append({"event": "missing_session_finish", "detail": "Node reports survive, but shard did not finish normally."})
    result = {"schema": SCHEMA, "selection": selection, "worker": supervision["worker"],
              "supervision": supervision, "identities_unchanged": identity_ok,
              "binding_before_sha256": contract["binding_sha256"], "binding_after_sha256": object_sha(after),
              "child_after_identity_available": child_after is not None, "decisions": decisions,
              "session_finish": session_finish, "infrastructure_events": infrastructure,
              "xml": xml_evidence(directory / "junit.xml"), "events_sha256": digest(directory / "events.jsonl")
              if (directory / "events.jsonl").exists() else None, "finalized_unix_s": time.time()}
    result["evidence_sha256"] = {path.name: digest(path) for path in directory.iterdir()
                                 if path.is_file() and path.name != "attempt-result.json"}
    write_json(directory / "attempt-result.json", result)
    return result


def load_allowed(config):
    if not config["allowed_skips"]:
        return set()
    record = json.loads(Path(config["allowed_skips"]).read_text())
    return {(item["nodeid"], item["original_detail"]) for item in record["allowed_skips"]}


def rebuild_ledger(output, selection, allowed):
    ledger = {node: {"nodeid": node, "outcome": "pending", "terminal": False, "attempts": []} for node in selection}
    infrastructure = []
    singleton_modules = set()
    for directory in sorted(output.glob("attempt-*")):
        path = directory / "attempt-result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        assert result["identities_unchanged"]
        if any(not (directory / name).is_file() or digest(directory / name) != expected
               for name, expected in result["evidence_sha256"].items()):
            raise RuntimeError(f"Finalized shard evidence changed: {directory}")
        infrastructure.extend({"attempt": directory.name, **event} for event in result["infrastructure_events"])
        for node, decision in result["decisions"].items():
            if ledger[node]["terminal"]:
                raise RuntimeError(f"Completed node repeated in another finalized shard: {node}")
            ledger[node]["attempts"].append({"attempt": directory.name, **decision})
            ledger[node].update(decision, completed_attempt=directory.name)
        unfinished = [n for n in result["selection"] if not ledger[n]["terminal"]]
        no_started = not result["decisions"] and result["supervision"]["termination_reason"] != "controller_interrupted"
        if no_started:
            singleton_modules.update(n.split("::")[0] for n in unfinished)
            if len(result["selection"]) == 1:
                node = result["selection"][0]
                ledger[node].update(outcome="not_executed_startup_failure", terminal=True,
                                    completed_attempt=directory.name, reports=[],
                                    explanation="Single-node process failed before node_start; no test result is invented.")
    for node, record in ledger.items():
        record["skip_reviews"] = [{"phase": e["phase"], "original_detail": e["original_detail"],
                                   "wasxfail": e["wasxfail"], "listed": (node, e["original_detail"]) in allowed}
                                  for e in record.get("reports", []) if e["outcome"] == "skipped"]
    return ledger, infrastructure, singleton_modules


def publish(output, ledger, infrastructure, *, state, error=None):
    counts = dict(Counter(item["outcome"] for item in ledger.values()))
    actual_skips = [dict(nodeid=node, **e) for node, item in ledger.items() for e in item.get("skip_reviews", [])]
    summary = {"schema": SCHEMA, "state": state, "updated_unix_s": time.time(),
               "controller": process_fingerprint(os.getpid()), "counts": counts,
               "collected": len(ledger), "terminal_nodes": sum(item["terminal"] for item in ledger.values()),
               "all_nodes_terminal": bool(ledger) and all(item["terminal"] for item in ledger.values()),
               "all_nodes_passed": bool(ledger) and all(item["outcome"] == "passed" for item in ledger.values()),
               "unlisted_skips": [e for e in actual_skips if not e["listed"]], "all_skip_records": actual_skips,
               "final_node_phase_outcomes": dict(Counter(e["phase"] + ":" + e["outcome"]
                    for item in ledger.values() for e in item.get("reports", []))),
               "all_attempt_phase_outcomes": dict(Counter(e["phase"] + ":" + e["outcome"]
                    for item in ledger.values() for attempt in item["attempts"] for e in attempt["reports"])),
               "infrastructure_events": infrastructure, "error": error,
               "interpretation": "Original outcomes only. A terminal timeout/crash/skip is not PASS; private or historical runs do not close release gates."}
    write_json(output / "node-outcomes.json", ledger)
    write_json(output / "summary.json", summary)
    return summary


def configure(args):
    source = args.source_root.resolve()
    paths = [str(Path(path).resolve()) for path in args.test_path] if args.test_path else [str(source / "tests")]
    if args.nodes_file:
        # Explicit node selectors can be absolute or relative to source_root.
        paths = [line for line in args.nodes_file.read_text().splitlines() if line.strip()]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("Node selector file must contain a nonempty unique selection")
    external = []
    selected_roots = [str(source)]
    for item in paths:
        path = Path(item.split("::")[0])
        path = path if path.is_absolute() else source / path
        if not path.exists():
            raise ValueError(f"Missing selected path {path}")
        if not path.resolve().is_relative_to(source):
            external.append(str(path.resolve().parent if path.is_file() else path.resolve()))
            selected_roots.append(str(path.resolve().parent if path.is_file() else path.resolve()))
    for value in (args.nodes_file, args.allowed_skips):
        if value:
            external.append(str(value.resolve()))
    for path in args.pin_path:
        if not path.exists():
            raise ValueError(f"Missing explicit immutable input: {path}")
        external.append(str(path.resolve()))
    policy = load_policy(args.module_environment, source) if args.module_environment else None
    if policy is not None:
        external += [str(args.module_environment.resolve()), *policy["files"]]
    return {"module_environment_file": str(args.module_environment.resolve()) if args.module_environment else None,
            "module_environment": policy, "source_root": str(source), "pytest_rootdir": os.path.commonpath(selected_roots),
            "test_paths": paths, "external_paths": sorted(set(external)),
            "gpu": args.gpu, "nodes_file": str(args.nodes_file.resolve()) if args.nodes_file else None,
            "allowed_skips": str(args.allowed_skips.resolve()) if args.allowed_skips else None,
            "pin_paths": sorted(str(path.resolve()) for path in args.pin_path),
            "nodes_per_shard": args.nodes_per_shard, "node_timeout": args.node_timeout,
            "shard_timeout": args.shard_timeout, "termination_grace": args.termination_grace,
            "poll_seconds": args.poll_seconds, "pytest_args": args.pytest_arg}


def collect(output, config, environment):
    directory, process = new_attempt(output, config, environment, [], collection=True)
    supervision = supervise(directory, config, process)
    events = read_events(directory / "events.jsonl")
    finishes = [e for e in events if e["event"] == "collection_finish"]
    if (supervision["returncode"] != 0 or supervision["termination_reason"] or len(finishes) != 1
            or any(e["event"] == "collection_failure" for e in events)):
        raise RuntimeError(f"Complete collection did not succeed; see {directory}")
    before = json.loads((directory / "identity-before.json").read_text())
    after = json.loads((directory / "identity-after.json").read_text())
    if before["binding"] != after["binding"] or before["versions"] != after["versions"] or before.get("gsp_environment") != after.get("gsp_environment"):
        raise RuntimeError("Source/runtime identity changed during complete collection")
    nodes = finishes[0]["nodeids"]
    if not nodes or len(nodes) != len(set(nodes)):
        raise RuntimeError("Empty or duplicate full collection; no nodes are silently deduplicated")
    contract = {"schema": SCHEMA, "config": config, "selection": nodes,
                "node_selectors": finishes[0]["absolute_selectors"], "selection_sha256": object_sha(nodes),
                "binding": before["binding"], "binding_sha256": before["binding_sha256"],
                "versions": before["versions"], "core_root": before["binding"]["core_root"],
                "actual_imports_at_collection": before["actual_module_files"],
                "environment": environment_binding(environment), "created_unix_s": time.time()}
    validate_current(config, environment, contract)
    write_json(directory / "collection-result.json", {"supervision": supervision,
               "identities_unchanged": True, "count": len(nodes), "selection_sha256": object_sha(nodes),
               "xml": xml_evidence(directory / "junit.xml")})
    write_json(output / "contract.json", contract)
    (output / "contract.sha256").write_text(digest(output / "contract.json") + "\n")
    write_json(output / "selection.json", nodes)
    return contract


def run(args):
    global STOP_SIGNAL
    STOP_SIGNAL = None
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = configure(args)
    output = args.output.resolve()
    if output.is_relative_to(CONTROL) or output.is_relative_to(Path(config["source_root"])):
        raise ValueError("Output must be outside frozen source_root and control_root")
    if any(Path(path).is_dir() and output.is_relative_to(Path(path)) for path in config["external_paths"]):
        raise ValueError("Output must be outside every immutable selected/pinned input directory")
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Cannot resume an output directory that does not exist")
    lock = (output / "controller.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another controller holds the output lock; refusing concurrent/resumed execution")
    environment = make_environment(config["source_root"], config["module_environment"])
    controller = {"process": process_fingerprint(os.getpid()), "argv": sys.argv,
                  "resuming": args.resume, "started_unix_s": time.time(), "control_file_sha256": digest(__file__)}
    history = output / "controllers"
    history.mkdir(exist_ok=True)
    write_json(history / f"controller-{time.time_ns()}.json", controller)
    write_json(output / "controller.json", controller)
    ledger, infrastructure = {}, []
    try:
        if args.resume:
            if not (output / "contract.json").exists():
                raise RuntimeError("Incomplete original collection has no frozen contract; use a new output")
            if digest(output / "contract.json") != (output / "contract.sha256").read_text().strip():
                raise RuntimeError("Frozen contract content changed")
            contract = json.loads((output / "contract.json").read_text())
            if config != contract["config"] or json.loads((output / "selection.json").read_text()) != contract["selection"]:
                raise RuntimeError("Resume config or selection differs from frozen contract")
            validate_current(config, environment, contract)
        else:
            contract = collect(output, config, environment)
        allowed = load_allowed(config)
        if args.action == "collect":
            ledger, infrastructure, _ = rebuild_ledger(output, contract["selection"], allowed)
            publish(output, ledger, infrastructure, state="collection_complete_execution_pending")
            return 0
        # Resume an existing owned worker before scheduling any new nodes. A
        # missing controller exit does not make an incomplete shard successful.
        for directory in sorted(output.glob("attempt-*")):
            if not (directory / "attempt-result.json").exists():
                if not (directory / "worker.json").exists():
                    raise RuntimeError(f"Unrecorded worker launch; cannot safely resume {directory}")
                supervision = supervise(directory, config)
                finalise_attempt(directory, config, environment, contract, supervision)
        while True:
            ledger, infrastructure, singletons = rebuild_ledger(output, contract["selection"], allowed)
            pending = [node for node in contract["selection"] if not ledger[node]["terminal"]]
            if not pending:
                state = "completed_all_passed" if all(r["outcome"] == "passed" for r in ledger.values()) and not infrastructure else "completed_with_nonpass_or_infrastructure"
                summary = publish(output, ledger, infrastructure, state=state)
                return 0 if state == "completed_all_passed" else 1
            if STOP_SIGNAL is not None:
                publish(output, ledger, infrastructure, state="interrupted_resumable")
                return 128 + STOP_SIGNAL
            validate_current(config, environment, contract)
            module = pending[0].split("::")[0]
            limit = 1 if module in singletons else config["nodes_per_shard"]
            selection = [node for node in pending if node.split("::")[0] == module][:limit]
            attempt_number = len(list(output.glob("attempt-*"))) + 1
            publish(output, ledger, infrastructure, state="running")
            directory, process = new_attempt(output, config, environment, selection, number=attempt_number, contract=contract)
            supervision = supervise(directory, config, process)
            finalise_attempt(directory, config, environment, contract, supervision)
    except BaseException as error:
        # Do not hide a missing identity or damaged ledger behind a success file.
        detail = {"exception": repr(error), "traceback": traceback.format_exc()}
        write_json(output / f"controller-error-{time.time_ns()}.json", detail)
        if not ledger and (output / "summary.json").exists():
            previous = json.loads((output / "summary.json").read_text())
            previous.update(previous_state=previous["state"], state="controller_error_not_qualification",
                            error=detail, updated_unix_s=time.time())
            write_json(output / "summary.json", previous)
        else:
            publish(output, ledger, infrastructure, state="controller_error_not_qualification", error=detail)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("run", "launch", "collect", "status"))
    result.add_argument("--source-root", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--test-path", action="append", default=[])
    result.add_argument("--nodes-file", type=Path)
    result.add_argument("--gpu", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--nodes-per-shard", type=int, default=128)
    result.add_argument("--node-timeout", type=float, default=1800)
    result.add_argument("--shard-timeout", type=float, default=7200)
    result.add_argument("--termination-grace", type=float, default=2)
    result.add_argument("--poll-seconds", type=float, default=.25)
    result.add_argument("--allowed-skips", type=Path)
    result.add_argument("--module-environment", type=Path)
    result.add_argument("--pin-path", action="append", default=[], type=Path)
    result.add_argument("--pytest-arg", action="append", default=[])
    return result


def main():
    args = parser().parse_args()
    if args.action == "status":
        for name in ("summary.json", "controller.json"):
            path = args.output / name
            if path.exists():
                print(json.dumps({name: json.loads(path.read_text())}, indent=2))
        active = [p for p in sorted(args.output.glob("attempt-*")) if not (p / "attempt-result.json").exists()]
        for directory in active:
            events = read_events(directory / "events.jsonl")
            worker = json.loads((directory / "worker.json").read_text()) if (directory / "worker.json").exists() else None
            print(json.dumps({"active_attempt": str(directory), "worker": worker,
                  "worker_observed_live": observed_live(worker["process"]) if worker else None,
                  "current_node": current_node(events), "last_event": events[-1] if events else None,
                  "partial_phase_outcomes": dict(Counter(e["phase"] + ":" + e["outcome"] for e in events
                                                         if e["event"] == "test_report")),
                  "interpretation": "Live partial records are not a finalized shard result."}, indent=2))
        return 0
    if args.source_root is None:
        raise ValueError("--source-root is required for run/launch")
    if not 1 <= args.nodes_per_shard <= 256:
        raise ValueError("nodes-per-shard must be in1..256")
    if min(args.node_timeout, args.shard_timeout, args.poll_seconds, args.termination_grace) <= 0:
        raise ValueError("Timeout/poll/grace values must be positive")
    unsafe = ("--maxfail", "--last-failed", "--lf", "--failed-first", "--ff", "--deselect", "--ignore",
              "--dist", "--numprocesses", "--collect-only", "--co", "--stepwise", "--sw")
    for token in [*args.pytest_arg, *shlex.split(os.environ.get("PYTEST_ADDOPTS", ""))]:
        if token in ("-x", "-k", "-m", "-n") or token.startswith(unsafe):
            raise ValueError(f"Selection/early-stop/parallel pytest option is not allowed: {token}")
    if args.action == "launch":
        # The controller owns a new session and log, so the caller exits promptly.
        # Each actual pytest worker owns a different new session of its own.
        parent = args.output.resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(Path(__file__).resolve()), "run", *sys.argv[2:]]
        launch_path = parent / (args.output.name + f".launch-{time.time_ns()}.json")
        log_path = launch_path.with_suffix(".log")
        with log_path.open("xb") as log:
            child = subprocess.Popen(argv, cwd=args.source_root, start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        record = {"schema": SCHEMA, "controller": process_fingerprint(child.pid), "argv": argv,
                  "stdout_stderr_log": str(log_path), "output": str(args.output.resolve()), "launched_unix_s": time.time()}
        write_json(launch_path, record)
        print(json.dumps(record))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
