"""Shared-probe bisection of several failing pytest node sets over first-parent merges.

    python scripts/dev/integration_multibisect.py BASE HEAD NODES.json OUT [--timeout S] [--gpu I]
        [--lock PATH] [--min-free-mib M] [--launch-blocking]

NODES.json maps a label (usually the test file) to its node ids. Every probe runs one label in its own
process in a dedicated worktree (.worktrees/ib-multibisect), optionally with CUDA_LAUNCH_BLOCKING=1, and a per-commit
extension build directory under OUT/builds. A probe has three outcomes:

    PASS          pytest exit 0 and a pytest summary line
    FAIL          pytest exit 1 and a pytest summary line
    INCONCLUSIVE  anything else: timeout (exit 124), no summary line (truncated log), exit 2-5

An inconclusive probe (except exit 2/5, which is deterministic) is retried once with twice the timeout;
the timeout starts after the optional lock is acquired. If it stays inconclusive the label's search
stops and is reported as UNRESOLVED with its remaining interval; inconclusive is never counted as bad.
Labels whose BASE does not PASS or whose HEAD does not FAIL are reported NOT BISECTABLE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = re.compile(r"^=+ .*\b(passed|failed|errors?|no tests ran)\b.* in [0-9.]+s|^\d+ (passed|failed)\b.* in [0-9.]+s", re.M)


def classify(returncode, text):
    if returncode in (0, 1) and SUMMARY.search(text):
        return "PASS" if returncode == 0 else "FAIL"
    return "INCONCLUSIVE"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base"); parser.add_argument("head"); parser.add_argument("nodes", type=Path); parser.add_argument("out", type=Path)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lock", help="flock path held for each probe")
    parser.add_argument("--min-free-mib", type=int, default=0, help="wait for this much free memory on --gpu before each probe")
    parser.add_argument("--launch-blocking", action="store_true",
                        help="set CUDA_LAUNCH_BLOCKING=1 to localise device asserts (much slower for long tests)")
    args = parser.parse_args(argv)
    nodes = json.loads(args.nodes.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    tree = ROOT / ".worktrees" / "ib-multibisect"
    if not tree.exists():
        git("worktree", "add", "--detach", str(tree), args.base)
    commits = git("rev-list", "--first-parent", "--reverse", f"{args.base}..{args.head}").split()

    def run_once(commit, label, limit):
        subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--detach", commit], check=True)
        while args.min_free_mib and int(subprocess.check_output(
                ["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)) < args.min_free_mib:
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(tree), **({"CUDA_LAUNCH_BLOCKING": "1"} if args.launch_blocking else {}),
                   WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR=str(args.out / "builds" / commit[:8]))
        # Lock first, then time: waiting for a contended lock must not consume the probe's timeout.
        command = [*(["flock", args.lock] if args.lock else []), "timeout", str(limit), sys.executable, "-m", "pytest",
                   "-p", "no:cacheprovider", "-q", "--gpu", *nodes[label]]
        log = args.out / f"{commit[:8]}-{Path(label).stem}-t{limit}.log"
        with log.open("w") as handle:
            returncode = subprocess.run(command, cwd=tree, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        return classify(returncode, log.read_text(errors="replace")), returncode

    def probe(commit, label):
        outcome, returncode = run_once(commit, label, args.timeout)
        # Collection/usage errors (2) and "no tests ran" (5) are deterministic; only retry other inconclusive probes.
        if outcome == "INCONCLUSIVE" and returncode not in (2, 5):
            outcome, _ = run_once(commit, label, 2 * args.timeout)
        print(f"{commit[:8]} {label}: {outcome}", flush=True)
        return outcome

    low = {label: -1 for label in nodes}
    high = {label: len(commits) - 1 for label in nodes}
    state = {label: "open" for label in nodes}
    for label in nodes:
        base, head = probe(args.base, label), probe(args.head, label)
        if (base, head) != ("PASS", "FAIL"):
            state[label] = f"NOT BISECTABLE (base {base}, head {head})"
    while open_labels := [label for label in nodes if state[label] == "open" and high[label] - low[label] > 1]:
        widest = max(open_labels, key=lambda label: high[label] - low[label])
        middle = (low[widest] + high[widest]) // 2
        for label in [label for label in open_labels if low[label] < middle < high[label]]:
            outcome = probe(commits[middle], label)
            if outcome == "PASS":
                low[label] = middle
            elif outcome == "FAIL":
                high[label] = middle
            else:
                good = commits[low[label]][:8] if low[label] >= 0 else args.base
                state[label] = f"UNRESOLVED between {good} (good) and {commits[high[label]][:8]} (bad): inconclusive at {commits[middle][:8]}"
    summary = {label: state[label] if state[label] != "open" else
               "FIRST BAD " + commits[high[label]][:8] + " " + git("log", "-1", "--format=%s", commits[high[label]])[:90]
               for label in nodes}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\nSUMMARY (unresolved steps are listed explicitly):")
    for label, verdict in summary.items():
        print(f"  {label}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
