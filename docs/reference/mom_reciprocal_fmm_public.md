# Experimental reciprocal scalar FMM

Select `FMMConfig(near_storage="scalar_reciprocal_dk_v1")` through `MoMConfig(operator_backend="fmm", ...)` and the public Scene/Simulation/Result path. Native complex128 and free-space scalar media are required by this version. All numerical settings are bound by `cartesian_fmm_reciprocal_dk_config_v1`. Existing physical schema11 persistence and Bench write/read lifted-state APIs retain exact currents and identities. Declare an independent Scene/configuration before reading a completed producer.

The packed near policy stores one off-diagonal D/K direction and full diagonals. Retained admission is distinct from simultaneous whole GPU peaks. These experimental controls do not confer physics, periodic, tensor, nonlinear, whole-resource or population qualification. See the corresponding assessment.
