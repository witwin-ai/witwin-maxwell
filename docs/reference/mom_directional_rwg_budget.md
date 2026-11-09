# Prescribed directional RWG ideal budget contract

DirectionalFFTElectromagneticM2L.from_projections accepts two currently valid
DirectionalRWGProjection maps, sampled scalar medium, explicit normalization,
KernelErrorBudget, positive finite ideal_rwg_error_budget and balancing_impedance.
Device, polynomial order, passive wave number and unit carrier match exactly.
Selections remain nonempty and unique. Centers are the owned projection centers.

Before geometry gathers or integral-statistic allocations, visible temporary
allowance is directional_bound_workspace_bytes(1,1)+4096 plus each projection's
4096*full_basis_faces+512*full_basis_edges+1024*selected_edges. This includes the
full-basis rwg_integral_logs work, not just selected edges. Caller projection
arrays are input ownership. It is a conservative visible tensor preflight, not a
measured peak or whole-system resource certificate. All triangle corners of the
selected edges' plus/minus faces determine each radius, with a small native safety
factor. Convexity encloses each full triangle. Native nonfinite/zero scales reject;
no formal outward floating-point enclosure is claimed.

Use the continuous carrier-restored Green/physical-gradient component bounds.
For each selected edge, U bounds integral |f| and Q bounds integral |div f|.
Aggregate selected statistics log(sum U²),log(sum Q²),log(sum UQ). The existing
PMCHWT balanced Frobenius formula uses D_ij<=G(U_i U_j+Q_i Q_j/|k|²),
K_ij<=sqrt(3)*gradient_component*U_i U_j. With eta>0 and e=i*omega*epsilon,
m=-i*omega*mu, squared balanced block bound is the sum over selected edge pairs of
2*eta²*K_ij²+(eta⁴*|e|²+|m|²)*D_ij². Existing block_error_log/_receipt implement
this in CUDA log space. Admit both explicit kernel and balanced absolute budgets
before allocating scalar translation coefficients. Kernel and absolute RWG budgets
are different from relative sampled physical gates.

The existing EM scalar action/Hermitian/current/higher-current math is reused.
External projection identities and current tensor/geometry guards, both budgets,
balancing impedance and cached flat receipt values bind owner action, receipt and
saved backward. New owned storage is only the configured scalar translation;
projection maps remain caller-owned. Default direct EM state/receipts/actions and
DirectionalRWGProjection remain unchanged. Factory receipt
experimental_directional_fft_prescribed_rwg_budget_v1 includes verified complete
selected support and ideal RWG admission for composition with these exact maps.
A scalar translation's own receipt continues to disclaim automatic support proof.

No bound on arbitrary unrelated moment inputs, finite carrier quadrature, native
roundoff, nested changing-carrier paths, global aggregation, physical/resources/
scaling or production qualification is implied. The global/final requirements
and all original failure evidence remain open.
