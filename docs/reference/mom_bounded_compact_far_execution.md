# Experimental bounded compact FAR execution

Use `mw.FMMConfig(table_storage="compact_shared_displacement_v1", far_action_batch=32, order=6, kernel_error_budget=...)` in the existing `Scene -> Simulation.mom(...).prepare().run() -> Result` flow. `table_storage="compact"` is also supported. This explicit free-space scalar option accepts order0..12 and native complex128 currents; fixed geometry/materials and original kernel/storage/solve admission remain required. Default `far_action_batch=1` preserves the original strict payload and action.

The new `cartesian_fmm_batched_compact_config_v1` includes the batch limit and all original fields. Physical Result/Bench exact candidate state persistence binds this complete identity. Restore never constructs an executable compressed operator. Changed batches are rejected under the producer identity.

Execution batches at most32 logical pairs,4 RHS and16 target rows using original coefficients, reversal parity and scalar electromagnetic formulas. Ordered per-pair target accumulation and fixed-current first/higher gradients remain. The128MiB visible-workspace preflight is a tensor estimate, excluding backend/allocator peaks and whole solver resources. Read the phase assessment for native controls, preserved fixture failures and open full physics/complexity/resource acceptance. No production qualification is granted.
