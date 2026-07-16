# Network embedding implementation record

Status: active
Plan: `next-functional-2026-07/03-touchstone-network-embedding.md`
Branch: `codex/network-embedding`

## Dependency audit

- The RF `NetworkData` schema and power-wave convention required from plan 01 are present at the implementation baseline.
- `LumpedPort` and `TerminalPort` lower to the same compiled sparse Yee geometry and are available for single-device FDTD coupling.
- `WavePort` does not yet expose a time-domain terminal injection operator, so embedded-network connections must reject it until that separate contract exists.
- Spatial FDTD currently rejects RF ports. The required public port reference-point ownership field and a hot-path scalar reduce/scatter transport are not present. Phase 4 spatial multi-device ownership remains blocked on those plan 01/02 contracts; the implementation must not guess ownership from sparse edge order.

## Phase 0 evidence

Implemented:

- Strict Touchstone 1.x/2.0 parsing with line-numbered errors.
- RI, MA, and DB pairs; Hz, kHz, MHz, and GHz frequencies.
- S, Z, and Y input with the version 1 normalization rules.
- Full, Lower, and Upper matrices, both explicit 2-port orders, and per-port version 2 reference impedances.
- Preserved source format metadata, comments, parser warnings, port order, and reference impedances.
- N-port S/Z/Y writing and `NetworkData.from_touchstone(...)` interoperability.

Acceptance evidence:

- Static one-, two-, and four-port corpus fixtures.
- Float64 RI/MA/DB import/export error below `1e-10`.
- Independent 2-port ordering assertions.
- Bad-token, incomplete-data, NaN, frequency-order, impedance, and suffix-mismatch diagnostics assert source line numbers.
- Targeted parser/writer and `NetworkData` contract suite: 63 passed before the final name-boundary regression.
- Full RF network plus `NetworkData` contract regression: 102 passed after all review fixes.
- Static analysis of all changed Python modules and tests: passed.

Independent review findings resolved before acceptance:

- Named port comments now survive export/import.
- Touchstone 1.x rows with more than four pairs wrap conformingly.
- DC samples are accepted while negative frequencies remain rejected.
- Version-specific data-line layout, strict ASCII, and canonical keyword syntax are validated.
- Exact zero DB output uses a finite floor instead of a non-standard infinity token.
- Duplicate resolved port names report the second offending comment line.
- Public network port names reject leading/trailing whitespace so named-port text round-trips are lossless.
