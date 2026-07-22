# Surface-impedance Phase 0 — P09 fixer pass (2026-07-17)

Durable record for the P09 verdict-closure pass on the surface-impedance Phase 0
slice (`feat(surface-impedance): freeze Phase 0 contract, rational model, fitter,
funnel`, HEAD 0473c3e). Records the fixes applied and the deferrals that are real
but out of scope, so they are tracked in a trusted doc rather than only in the
untrusted dev report.

## Test-count accuracy

The three new Phase 0 CPU suites collect **28** tests, not the 34 the original
dev report claimed:

- `tests/fdtd/surface_impedance/test_surface_impedance_reference.py` — 11
- `tests/materials/surface_impedance/test_rational_surface_impedance.py` — 12
- `tests/materials/sheet/test_surface_impedance_funnel.py` — 5

All 28 pass on CPU (`CUDA_VISIBLE_DEVICES=`). This pass adds 3 more to the
interoperability-adapter suite (`tests/api/adapters/tidy3d/`) for the new
adapter fail-close, giving 31 surface-impedance-related new tests.

## CUDA-deferred gates (real, out of scope for Phase 0 CPU work)

Two acceptance items in the frozen `SurfaceAcceptanceBudget` are exercised only on
GPU and are NOT covered by the CPU suites:

1. The incumbent narrowband good-conductor solver run
   (`tests/validation/physics/test_lossy_metal_sibc.py`) that produces the 5%
   `narrowband_reproduction_relative_error` evidence.
2. The end-to-end prepared-runtime surface-impedance path (the generalized rational
   stepping kernel is unwired; the compiler funnel fails closed until Phase 1).

Both are deferred to the GPU validation window / Phase 1 runtime work.

## Fixes applied in this pass

- **Adapter fail-close (MAJOR):** `adapters/tidy3d.py::_convert_material` now
  raises `NotImplementedError` for a generic `SurfaceImpedanceMedium` instead of
  falling through to `td.Medium(permittivity=1.0)` and silently exporting as
  vacuum. Census reconciled: External interoperability adapter 18 -> 19, total
  134 -> 135, budget bumped to 135.
- **Representation contradiction:** `RationalSurfaceImpedance.__init__` now raises
  when a passed `RationalModel`'s representation contradicts an explicit
  `representation` kwarg (previously the kwarg was silently ignored).
- **Out-of-band extrapolation:** documented that passivity/accuracy are certified
  only over `frequency_range`; out-of-band queries are inspection-only.
- **Coincidence tolerance:** `compiler/materials.py::_geometries_coincide` scales
  its tolerance to the geometry extents (sizes and positions) instead of clamping
  at a 1-metre absolute floor, so micron-scale scenes are not falsely merged.
- **Partial-overlap deferral:** documented in `_geometries_coincide` that a PEC /
  2D sheet partially overlapping the metal face is not detected in Phase 0.
- **Frozen budget field rename:** `no_sibc_runtime_regression` ->
  `no_surface_impedance_runtime_regression`, updating the frozen test in the same
  change (done before the freeze hardens to avoid a change-record).

## Known pre-existing flaky (NOT a P09 regression)

`tests/fdtd/thin_wire/test_wire_finite_conductivity.py::test_fit_is_passive_and_discretization_is_stable`
is order-dependent flaky (RNG-sensitive vector-fit pole initialization in the
shared fitter). The P09 branch touches neither `rational.py` nor the wire code, so
this is out of scope; the shared fitter should seed pole initialization
deterministically in a separate fix.
