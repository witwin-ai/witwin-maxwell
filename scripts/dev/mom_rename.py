"""Deterministic SIE -> MoM rename of the witwin-maxwell tree.

The rename is a pure function of (path, bytes) for every tracked file, so it can be
re-run on any base commit and re-derived by a reviewer:

    python scripts/dev/mom_rename.py --plan     # report what would change
    python scripts/dev/mom_rename.py --apply    # move/rewrite files and write the receipt
    python scripts/dev/mom_rename.py --verify   # check a renamed tree against the receipt

Classes of tracked files:

* frozen: ``docs/assessments/**``, ``docs/archive/**`` and the explicit ``FROZEN_FILES``
  (historical epoch harnesses whose bytes are bound by archives or by other harnesses,
  the consumer-metadata bridge, the frozen benchmark cache). They keep path and bytes.
* evidence: JSON files under ``tests/**/data/`` or ``tests/**/fixtures/``. Only their
  ``"<repo path>": "<sha256>"`` pins are remapped (path through the path map, digest
  through the byte map); no other byte changes.
* binary: not UTF-8 or containing NUL; moved, never rewritten.
* text: everything else; moved and rewritten with ``transform_text``.

A pin (a sha256 literal of a tracked file) is renewed only when the pinned file is
rewritten by this tool and the pinned digest equals sha256 of the old bytes (current
proof) or of a historical blob of the same path (historical proof); the new digest is
sha256(transform(old bytes)). Every renewal is recorded in the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

RECEIPT = "scripts/dev/mom_rename_receipt.json"
RELEASE_DOCS = ("CHANGELOG.md", "docs/reference/release-notes-0.6.0.md")
TOOL = "scripts/dev/mom_rename.py"

FROZEN_PREFIXES = ("docs/assessments/", "docs/archive/", "benchmark/cache/sie/",
                   "scripts/dev/fdtd/consumer_metadata/")
# Historical provider-epoch harnesses: they bind frozen provider trees and each other by
# path and sha256 (see docs/assessments/combined-direct-full-v*), so they keep old names.
FROZEN_FILES = frozenset({
    "benchmark/sie_combined_full_acceptance_v1.py",
    "benchmark/sie_combined_full_acceptance_v2.py",
    "benchmark/sie_combined_full_acceptance_v3.py",
    "benchmark/sie_combined_full_acceptance_v3_costs.json",
    "benchmark/sie_combined_full_acceptance_v3_inputs.json",
    "benchmark/sie_single_buffer_equivalence.py",
    "benchmark/sie_single_buffer_equivalence_v2.py",
    "benchmark/sie_single_buffer_equivalence_v3.py",
    "benchmark/sie_successor_preflight.py",
    "benchmark/test_sie_combined_full_acceptance_v1.py",
    # Independent Mie oracle: Maxwell-Bench reads it at this path and pins its bytes.
    "benchmark/sie_reference.py",
    # Source inventory of the combined-direct v1/v2 epoch harnesses.
    "docs/reference/sie_direct_storage.md",
    # Consumer-metadata bridge: re-receipted by a separate task.
    "tests/api/public/test_consumer_metadata.py",
    # Release documents edited by the release coordinator; --suggest writes the patch.
    *("CHANGELOG.md", "docs/reference/release-notes-0.6.0.md"),
    # Asserts that the old public names are gone; added after the rename.
    "tests/api/public/test_mom_naming.py",
    TOOL, RECEIPT,
})
# Per-file spans that refer to frozen trees or frozen harness internals and must keep
# their historical spelling. Each entry: path -> list of regexes; a match is copied as is.
FILE_PROTECT: dict[str, tuple[str, ...]] = {
    # Synthetic git histories for the frozen e95 single-buffer harness, whose reviewed
    # dependency check is spelled against the historical 'sie/operator.py' layout.
    "tests/sie/test_single_buffer_assembly.py": (r"(?:baseline|candidate|root)/'witwin/maxwell/sie[^'\n]*'",),
    # Mutation of a checkout of the frozen provider-epoch-v3 commit.
    "tests/sie/test_provider_epoch_v3.py": (r"'witwin/maxwell/sie/physics\.py'",),
}

_KEPT_STEMS = sorted({Path(p).stem for p in FROZEN_FILES if "sie" in Path(p).name}, key=len, reverse=True)

_PROTECT = [
    r"(?:/home|/mnt)/[^\s'\"`)\]>,;]*",                     # external absolute paths
    r"\.worktrees/[^\s'\"`)\]>,;]*",                        # existing worktree names
    r"(?:docs/|\.\./|(?<![\w/.\-]))(?:assessments|archive)/[^\s'\"`)\]>,;]*",  # frozen docs
    r"benchmark/cache/sie/(?:witwin-maxwell-local-acceptance-20260912\.zip|bench-compact-frequency-20260912)",
    r"(?<![\w/.\-])sie/[a-z0-9]+(?:-[a-z0-9]+)+(?![\w/])",  # git branch names sie/<x-y>
    # Formulation names (surface integral equation), also persisted as formulation and
    # payload identities that frozen Bench archives and readers match.
    r"SIE[-/]VIE",
    r"(?<![A-Za-z0-9])sie_vie(?![A-Za-z0-9])",
    r"dgf_sie(?![A-Za-z0-9])",
    r"pmchwt_scalar_sie(?![A-Za-z0-9])",
    *(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9_])" for s in _KEPT_STEMS),
]
_REPLACE = [
    ("method", r"SimulationMethod\.SIE\b"),
    ("member", r"(?m:^)(?P<indent>[ \t]+)SIE = (?P<q>[\"'])sie(?P=q)"),
    ("rfic", r"(?<![A-Za-z0-9])sie_rfic_mom(?![A-Za-z0-9])"),  # avoid mom_rfic_mom
    ("upper", r"(?<![A-Za-z0-9])SIE(?![a-z])"),
    ("lower", r"(?<![A-Za-z0-9])sie(?![A-Za-z0-9])"),
]


def _scanner(extra=()):
    parts = [f"(?P<p{i}>{p})" for i, p in enumerate((*extra, *_PROTECT))]
    parts += [f"(?P<{name}>{p})" for name, p in _REPLACE]
    return re.compile("|".join(parts))


_SCANNERS: dict[tuple, re.Pattern] = {}


def _upper(text, start, end):
    """SIE inside an all-caps identifier becomes MOM, otherwise MoM."""
    nxt = text[end:end + 2]
    prv = text[max(0, start - 2):start]
    if (len(nxt) == 2 and nxt[0] == "_" and (nxt[1].isupper() or nxt[1].isdigit())) or \
            (len(prv) == 2 and prv[1] == "_" and (prv[0].isupper() or prv[0].isdigit())):
        return "MOM"
    return "MoM"


def transform_text(text: str, path: str = "") -> str:
    extra = FILE_PROTECT.get(path, ())
    scanner = _SCANNERS.get(extra) or _SCANNERS.setdefault(extra, _scanner(extra))

    def repl(m):
        kind = m.lastgroup
        if kind.startswith("p"):
            return m.group(0)
        if kind == "method":
            return "SimulationMethod.MOM"
        if kind == "member":
            q = m.group("q")
            return f"{m.group('indent')}MOM = {q}mom{q}"
        if kind == "rfic":
            return "rfic_mom"
        if kind == "upper":
            return _upper(m.string, m.start(), m.end())
        return "mom"
    return scanner.sub(repl, text)


def _toml_text(text: str) -> str:
    """integration_suites.toml: historical [[merge_declarations]] records stay verbatim."""
    out, block, frozen = [], [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("["):
            out.append("".join(block) if frozen else transform_text("".join(block)))
            block, frozen = [], line.strip() == "[[merge_declarations]]"
        block.append(line)
    out.append("".join(block) if frozen else transform_text("".join(block)))
    return "".join(out)


def map_path(path: str) -> str:
    if is_frozen(path):
        return path
    return transform_text(path)


def is_frozen(path: str) -> bool:
    return path in FROZEN_FILES or path.startswith(FROZEN_PREFIXES)


def is_evidence(path: str) -> bool:
    return path.startswith("tests/") and path.endswith(".json") and ("/data/" in path or "/fixtures/" in path)


def _decode(data: bytes):
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform_bytes(path: str, data: bytes) -> bytes:
    """New bytes of a non-evidence file (identity for frozen and binary files)."""
    if is_frozen(path):
        return data
    text = _decode(data)
    if text is None:
        return data
    if path == "scripts/dev/integration_suites.toml":
        return _toml_text(text).encode()
    return transform_text(text, path).encode()


class Tree:
    def __init__(self, root: Path):
        self.root = root
        out = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        self.files = sorted(p for p in out.decode().split("\0") if p)
        self.paths = set(self.files)
        self.data: dict[str, bytes] = {}
        self._history: dict[str, dict[str, bytes]] = {}

    def read(self, path: str) -> bytes:
        if path not in self.data:
            self.data[path] = (self.root / path).read_bytes()
        return self.data[path]

    def history(self, path: str) -> dict[str, bytes]:
        """sha256 -> bytes of every historical blob of ``path`` reachable from HEAD."""
        if path not in self._history:
            revs = subprocess.run(["git", "-C", str(self.root), "log", "--format=%H", "HEAD", "--", path],
                                  capture_output=True, text=True).stdout.split()
            blobs = {}
            for rev in revs:
                r = subprocess.run(["git", "-C", str(self.root), "show", f"{rev}:{path}"], capture_output=True)
                if r.returncode == 0:
                    blobs.setdefault(sha(r.stdout), (rev, r.stdout))
            self._history[path] = blobs
        return self._history[path]


_PIN = re.compile(r'"(?P<path>[^"\n]+)"(?P<sep>\s*:\s*)"(?P<digest>[0-9a-f]{64})"')
_HEX = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def plan(root: Path) -> dict:
    tree = Tree(root)
    moves, rewrites = {}, {}
    for path in tree.files:
        if is_evidence(path):
            continue
        new_path = map_path(path)
        data = tree.read(path)
        new = transform_bytes(path, data)
        if new_path != path:
            moves[path] = new_path
        if new != data:
            rewrites[path] = new
    guard_map = _guard_identities(tree, moves, rewrites)
    old_sha = {p: sha(tree.read(p)) for p in tree.files if p in moves or p in rewrites}
    pins, problems = [], []

    def renew(path, digest, where):
        """New digest of a pin on ``path`` or None when the pinned file is untouched."""
        if path not in old_sha:
            return None
        if digest == old_sha[path]:
            new_digest, proof = sha(rewrites.get(path, tree.read(path))), "current"
        else:
            hit = tree.history(path).get(digest)
            if hit is None:
                problems.append(dict(where=where, path=path, digest=digest, reason="pin matches no blob of path"))
                return None
            rev, blob = hit
            new_digest, proof = sha(transform_bytes(path, blob)), f"historical:{rev}"
        pins.append(dict(file=where, old_path=path, old_sha256=digest, new_path=moves.get(path, path),
                         new_sha256=new_digest, proof=proof))
        return new_digest

    # Evidence files may pin each other: resolve in dependency order.
    evidence = [p for p in tree.files if is_evidence(p)]
    pending = set(evidence)
    while pending:
        progressed = False
        for path in sorted(pending):
            text = _decode(tree.read(path))
            deps = {m.group("path") for m in _PIN.finditer(text or "")} & pending - {path}
            if deps:
                continue
            new_path = map_path(path)
            if text is not None:
                def sub(m, where=path):
                    p = m.group("path")
                    if p not in tree.paths:
                        return m.group(0)
                    digest = renew(p, m.group("digest"), where)
                    np = moves.get(p, p)
                    return f'"{np}"{m.group("sep")}"{digest or m.group("digest")}"'
                new_text = _PIN.sub(sub, text)
                if new_text != text:
                    _check_evidence(text, new_text, moves)
                    rewrites[path] = new_text.encode()
                    old_sha[path] = sha(tree.read(path))
            if new_path != path:
                moves[path] = new_path
                old_sha.setdefault(path, sha(tree.read(path)))
            pending.discard(path)
            progressed = True
        if not progressed:
            raise RuntimeError(f"cyclic evidence pins: {sorted(pending)}")
    # Other 64-hex literals of the old bytes of a rewritten file, in live text files.
    # Code and tests are renewed by the current proof; prose documents keep the digest
    # as a historical record of what ran and are listed in the receipt.
    changed = {old_sha[p]: p for p in rewrites if not is_evidence(p)}
    notes = []
    for path in sorted(p for p in tree.files if not is_frozen(p) and not is_evidence(p)):
        text = _decode(rewrites.get(path, tree.read(path)))
        if text is None or not any(m.group(0) in changed for m in _HEX.finditer(text)):
            continue
        if path.startswith("docs/") or path.endswith(".md"):
            for m in _HEX.finditer(text):
                if m.group(0) in changed:
                    notes.append(dict(file=path, old_path=changed[m.group(0)], old_sha256=m.group(0),
                                      reason="historical record in prose; digest kept"))
            continue
        def sub(m, where=path):
            digest = m.group(0)
            return renew(changed[digest], digest, where) if digest in changed else digest
        rewrites[path] = _HEX.sub(sub, text).encode()
        old_sha.setdefault(path, sha(tree.read(path)))
    targets = {}
    for old, new in moves.items():
        if new in tree.paths and new not in moves:
            problems.append(dict(path=old, reason=f"target {new} already tracked"))
        if new in targets:
            problems.append(dict(path=old, reason=f"target {new} also produced by {targets[new]}"))
        targets[new] = old
    return dict(tree=tree, moves=moves, rewrites=rewrites, old_sha=old_sha, pins=pins, problems=problems,
                notes=notes, guard_map=guard_map)


GUARD_MANIFEST = "docs/reference/fdtd-capability-guards.json"


def _guard_identities(tree: Tree, moves: dict, rewrites: dict) -> dict:
    """Pair capability-guard identities of every rewritten provider module by line.

    The guard identity of scripts/dev/fdtd/guard_census.py hashes (path, scope, raise
    expression, enclosing conditions), so moving or renaming a raise site changes it.
    The rewrite never adds or removes lines, so old and new guards pair by line; the
    resulting map renews the ids in the reviewed guard manifest and lets the census
    translate its frozen pre-rename baseline.
    """
    import ast
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mom_guard_census", tree.root / "scripts/dev/fdtd/guard_census.py")
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)

    def guards(path, text):
        visitor = census._Guards(path)
        visitor.visit(ast.parse(text))
        seen, out = {}, []
        for guard in sorted(visitor.guards, key=lambda value: value["line"]):
            digest = guard["id"]
            guard["id"] = f"{digest}:{seen.get(digest, 0)}"
            seen[digest] = seen.get(digest, 0) + 1
            out.append(guard)
        return out
    mapping = {}
    for path in tree.files:
        if not (path.startswith("witwin/") and path.endswith(".py")) or (path not in moves and path not in rewrites):
            continue
        old_text = tree.read(path).decode("utf-8-sig")
        new_text = rewrites.get(path, tree.read(path)).decode("utf-8-sig")
        if old_text.count("\n") != new_text.count("\n"):
            raise RuntimeError(f"rewrite changed the line structure of {path}")
        old, new = guards(path, old_text), guards(moves.get(path, path), new_text)
        if [g["line"] for g in old] != [g["line"] for g in new]:
            raise RuntimeError(f"guard sites moved in {path}")
        for a, b in zip(old, new):
            if a["id"] != b["id"]:
                mapping[a["id"]] = {key: b[key] for key in GUARD_FIELDS}
    mapping = dict(sorted(mapping.items()))
    if GUARD_MANIFEST in rewrites:
        rewrites[GUARD_MANIFEST] = renew_guard_manifest(rewrites[GUARD_MANIFEST], mapping)
    return mapping


GUARD_FIELDS = ("id", "path", "scope", "expression", "conditions", "message")


def renew_guard_manifest(data: bytes, mapping: dict) -> bytes:
    """Replace the identity fields of moved raise sites with the paired new guard."""
    text = data.decode()
    manifest = json.loads(text)
    if json.dumps(manifest, indent=2) + "\n" != text:
        raise RuntimeError("guard manifest is not in canonical indent=2 form")
    for entry in (*manifest["active"], *manifest["retired"]):
        if entry["id"] in mapping:
            entry.update(mapping[entry["id"]])
    return (json.dumps(manifest, indent=2) + "\n").encode()


def _check_evidence(old: str, new: str, moves: dict):
    """Evidence JSON may differ only in remapped path keys and renewed digests."""
    a, b = json.loads(old), json.loads(new)

    def same(x, y):
        if isinstance(x, dict):
            return isinstance(y, dict) and len(x) == len(y) and all(
                moves.get(k, k) in y and same(v, y[moves.get(k, k)]) for k, v in x.items())
        if isinstance(x, list):
            return isinstance(y, list) and len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
        if isinstance(x, str) and isinstance(y, str) and x != y:
            return bool(re.fullmatch(r"[0-9a-f]{64}", x) and re.fullmatch(r"[0-9a-f]{64}", y))
        return type(x) is type(y) and x == y
    if not same(a, b):
        raise RuntimeError("evidence rewrite changed more than pins")


def receipt(result: dict, base: str) -> dict:
    tree, moves, rewrites, old_sha = result["tree"], result["moves"], result["rewrites"], result["old_sha"]
    files = []
    for path in sorted(set(moves) | set(rewrites)):
        new = rewrites.get(path, tree.read(path))
        plain = transform_bytes(path, tree.read(path))
        mode = ("evidence" if is_evidence(path) else "move" if path not in rewrites else
                "text" if new == plain else "text+renewed")
        files.append(dict(old_path=path, old_sha256=old_sha[path], new_path=moves.get(path, path),
                          new_sha256=sha(new), mode=mode))
    return dict(schema="mom_rename_receipt_v1", base_commit=base,
                tool_sha256=sha((tree.root / TOOL).read_bytes()),
                frozen_prefixes=list(FROZEN_PREFIXES), frozen_files=sorted(FROZEN_FILES),
                files=files, pins=result["pins"], kept_digest_records=result["notes"],
                guard_id_map=result["guard_map"])


def apply(root: Path, result: dict, base: str):
    tree, moves, rewrites = result["tree"], result["moves"], result["rewrites"]
    for path in sorted(set(moves) | set(rewrites)):
        data = rewrites.get(path, tree.read(path))
        target = root / moves.get(path, path)
        if path in moves:
            subprocess.check_call(["git", "-C", str(root), "rm", "-q", "--cached", "--", path])
            (root / path).unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Moved packages must not survive as empty (namespace-importable) directories.
    for directory in sorted({(root / p).parent for p in moves}, key=lambda d: len(d.parts), reverse=True):
        for cache in directory.glob("__pycache__"):
            for item in cache.iterdir():
                item.unlink()
            cache.rmdir()
        while directory != root and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent
    for path in sorted({moves.get(p, p) for p in set(moves) | set(rewrites)}):
        subprocess.check_call(["git", "-C", str(root), "add", "-f", "--", path])
    (root / RECEIPT).write_text(json.dumps(receipt(result, base), indent=1, sort_keys=True) + "\n")
    subprocess.check_call(["git", "-C", str(root), "add", "-f", "--", RECEIPT])


def verify(root: Path) -> list[str]:
    """Re-derive every recorded file from its base blob and compare with the tree."""
    rec = json.loads((root / RECEIPT).read_text())
    moved = {row["old_path"]: row["new_path"] for row in rec["files"]}
    errors = []
    for row in rec["files"]:
        old = subprocess.run(["git", "-C", str(root), "show", f"{rec['base_commit']}:{row['old_path']}"],
                             capture_output=True).stdout
        if sha(old) != row["old_sha256"]:
            errors.append(f"base blob mismatch: {row['old_path']}")
        current = (root / row["new_path"]).read_bytes()
        if sha(current) != row["new_sha256"]:
            errors.append(f"tree differs from receipt: {row['new_path']}")
        expected = transform_bytes(row["old_path"], old)
        if row["mode"] in ("text+renewed", "evidence"):
            # Renewed pins and guard identities are the only further replacements.
            pins = {(pin["old_path"], pin["old_sha256"]): pin for pin in rec["pins"] if pin["file"] == row["old_path"]}
            if row["mode"] == "evidence":
                # Same site rule as plan(): only "<tracked path>": "<sha256>" pins change.
                def sub(m):
                    pin = pins.get((m.group("path"), m.group("digest")))
                    if pin is None:
                        return f'"{moved.get(m.group("path"), m.group("path"))}"{m.group("sep")}"{m.group("digest")}"'
                    return f'"{pin["new_path"]}"{m.group("sep")}"{pin["new_sha256"]}"'
                text = _PIN.sub(sub, old.decode())
            else:
                text = expected.decode()
                for (_, digest), pin in pins.items():
                    text = text.replace(digest, pin["new_sha256"])
            expected = text.encode()
            if row["old_path"] == GUARD_MANIFEST:
                expected = renew_guard_manifest(expected, rec["guard_id_map"])
        if expected != current:
            errors.append(f"tool(old) != new: {row['new_path']}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--apply", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--suggest", type=Path, metavar="PATCH",
                       help="write the text rewrite of the excluded release documents as a patch")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.suggest:
        import difflib
        chunks = []
        for path in RELEASE_DOCS:
            if (root / path).is_file():
                old = (root / path).read_text()
                new = transform_text(old, path)
                chunks += difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                               f"a/{path}", f"b/{path}")
        args.suggest.write_text("".join(chunks))
        print(f"wrote {args.suggest} ({sum(1 for c in chunks if c.startswith('@@'))} hunks)")
        return 0
    if args.verify:
        errors = verify(root)
        print("\n".join(errors) or "receipt verified")
        return 1 if errors else 0
    base = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    result = plan(root)
    print(f"moves={len(result['moves'])} rewrites={len(result['rewrites'])} pins={len(result['pins'])} "
          f"problems={len(result['problems'])}")
    for problem in result["problems"]:
        print("PROBLEM", json.dumps(problem))
    if args.apply:
        if result["problems"]:
            print("refusing to apply with problems")
            return 2
        apply(root, result, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
