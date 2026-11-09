# Exact periodic dyadic regular configuration restoration

The internal checked regular adapter needs a reproducible identity before it can
be integrated into complete periodic solver snapshots. Its versioned JSON
configuration now records the full material/lattice binding, saved precision,
normal, spectral pole policy, execution/cache budgets, regular-part budgets and
effective primary integration settings.

Restoration binds existing validated CUDA `TensorMedium` and `LatticeSpec`
objects. It checks exact schema and binding, constructs a fresh empty mode cache,
and rejects any disagreement between saved and derived settings (including the
mandatory primary absolute budget and zero relative tolerance). It runs no
Green evaluation and restores no historical convergence or qualification claim.
Subsequent evaluation must execute all configured admission gates.

This is an internal persistence prerequisite. Public Scene background/source
compilation and complete Result/system snapshots remain open. Serialization of
fixed settings does not preserve an autograd graph or confer physical admission.

## Targeted validation

Physical GPU 1 checks cover eight distinct nodes across two preserved runs.
Coverage includes JSON
round trips in 1D/2D/3D, complex64/complex128, reciprocal/nonreciprocal tensors,
exact 3D modal values, independently rebound object identities, normal ownership,
malformed schemas, wrong bindings, derived budget drift and finite-value guards.
An actual scalar 2D regular evaluation checks equal values/receipts after restore
and a deliberately inadequate restored budget must fail admission.

- `/tmp/gsp-periodic-kernel-restore-v1`: 7 passed, 1 failed, 190.282 s.
  The sole failure expected `RuntimeError` for a mutated material, while the
  existing contract correctly raises `ValueError`. The failure is retained.
- `/tmp/gsp-periodic-kernel-restore-v2`: the corrected failure node passed in
  4.002 s. The seven covered passing nodes were not rerun. Production code did
  not change between runs; this is not a newly green full-suite run.
- Both runners bind unchanged source/test inventories during execution. The
  frozen periodic merge regression predates this stage and does not cover it.

Independent terminal review found no remaining confirmed P1/P2. The reviewer
verified both receipts, runners, XML/logs and 602 before/after source/test hashes
per run; the current inventory matches v2. Production code is identical across
the two runs. Reversing the failed assertion correction and unused helper cleanup
reconstructs the v1 test-file hash, retaining coverage of the seven passing nodes.

Receipt SHA256:
- v1: `17b8f3cc6331a5759151e5d11e2d1e13c5eb41d9a2feadab72450df483df9a73`
- v2: `f7c8e34854aa5b10421d9e83c275655104438ad69c2a630eaab45852436248a1`
