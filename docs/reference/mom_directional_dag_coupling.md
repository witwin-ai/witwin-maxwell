# Prescribed material-bound DAG coupling

Use `DirectionalFFTElectromagneticM2L.from_dag_contexts(test, test_context, source, source_context, medium, normalization_length, ...)` with explicit kernel/ideal-RWG budgets and a common balancing impedance. Both owners must be current `DirectionalRWGDAGProjection` instances with matching actual medium wave number, selected carrier, polynomial order and CUDA device. Context IDs are distinct from physical tree nodes; support radii are derived from complete physical plus/minus supports.

`DirectionalDAGFarActions` composes admitted couplings in full electric/magnetic physical coordinates. Rectangles must be disjoint. Its kernels are the original shared upward/FFT/downward and full Hermitian operations; neighboring contexts share child maps, and incoming parent locals sum before descent. The existing fixed-phase sparse factory does not admit DAG owners. Empty and bounded-action ownership follows the common FAR component contract.

The ideal bound includes maximum context-to-leaf phase products before continuous Green/gradient and material weighting. It excludes phase derivative, quadrature and native arithmetic certification. An aggregate FAR receipt is not a complete regional/system certificate. The tested phase covers two selected actual-material rectangles, not automatic global cones or a complete DAG solver. Geometry/material/configuration tensors remain fixed and guarded, including backward from saved forward/H outputs.

See `docs/assessments/sie_directional_dag_coupling.md` for preserved failures, unchanged gates and precise component inventory/exclusions. Public/periodic integration, whole resources/crossover/scaling and production qualification remain open.
