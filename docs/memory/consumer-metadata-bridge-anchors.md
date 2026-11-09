# Consumer-metadata bridge: /tmp anchors and pinned locations

`tests/api/public/test_consumer_metadata.py` re-derives the consumer bridge from absolute,
SHA-pinned inputs (`inputs_sha256` in the archived provenance under
`docs/assessments/fdtd-consumer-metadata-bridge-2026-10-31/` and `.../2026-11-01/`). The pins
are absolute by design, and they include the bridge control script itself, so the files must
exist at exactly those paths.

## /tmp anchors (copied 2026-11-01)

468 of the pins, plus the frozen consumer source roots, live under `/tmp`; they are the
2026-10-27 anchors and later bootstrap sources. A full copy, with the `/tmp` layout preserved,
is at:

    <evidence-root>/pcb-consumer-anchors-1027/tmp/...

All 468 pinned files were verified there against their pinned SHA256 (0 mismatches); about 2 GB
across 26 top-level entries. The originals stay in place.

**Recovery after a /tmp wipe:** `cp -a <evidence-root>/pcb-consumer-anchors-1027/tmp/. /tmp/`
(then the pins verify unchanged; no regeneration).

## Pinned repository locations (do not remove)

The pins also reference files inside these git worktrees, which must not be removed until
the archives record repo-relative pins:
- `.worktrees/consumer-core045`: a detached read-only checkout of d0de9534, restored 2026-11-01;
- `.worktrees/consumer-metadata-v3`: branch pcb/consumer-metadata-v3.

The portable re-bootstrap (repo-relative pins plus a resolver in derive_consumer_bridge.py) is
planned with the MoM v5 consumer rebaseline.
