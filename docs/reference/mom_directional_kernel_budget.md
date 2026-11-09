# Internal continuous directional kernel admission

The primary fmm_error scalar/batch bound evaluates a reviewed bivariate Cauchy
estimate for real displacement R, D=norm(R), real unit carrier u, passive finite
k and nonnegative declared source/target radii. For 0<rho<sqrt(2)-1, set
alpha=sqrt(1-2rho-rho*rho). The analytic positive-real branch satisfies
Re(s),abs(s)>=alpha*D, and

    abs(s-D-u.z) <= D*(rho*norm(R/D-u)+rho*rho/(alpha+1-rho)).

Carrier-removed Green majorant is
exp(-Im(k)*D+abs(k)*D*residual)/(4*pi*alpha*D). Each physical-gradient
component, exp(-ik*u.z)*partial_j G, has multiplier
(abs(k)+1/(alpha*D))*(1+rho)/alpha. This is not the derivative of the
carrier-removed Green alone. Separate source/target degree p Cauchy truncation
uses q=(r_target+r_source)/(rho*D)<1 and the omitted-union tail
q**(p+1)*(2-q**(p+1))/(1-q)**2. One zero-radius side uses the one-variable
tail; both zero radii give exact zero. Restoring the real-support carrier adds
exp(Im(k)*norm(u)*(r_target+r_source)); center decay is combined in log space.

Eight fixed rho=.05,.10,...,.40 and nine support-adaptive candidates at radius
ratio times (1.25,1.5,2,3,4,8,12,16,32) form independent valid minima. Only
0<rho<=.4 and rho>radius ratio qualify. Invalid intermediate candidates use a
safe contour and are excluded by the mask. This does not relax any budget.
CUDA float64 batches accept at most32 centers and16 orders, all0..12. Workspace
bytes are batch*order_count*17*8*32+batch*17*8*32+4096. Native unrepresentable
scales reject, positive underflow is conservatively floored, and unavailable
entries require their false availability mask to be observed before any admission.
The native evaluation is not interval arithmetic or outward-rounding certification.

CartesianDirectionalFFTTranslation accepts optional test_radius, source_radius
and KernelErrorBudget together. Bound admission precedes coefficient preparation;
create_many admits all rows in bounded chunks before constructing any translation.
Configured objects own and count an independent16-byte radii tensor. Semantic
budget values, cached ideal receipt and radii versions are guarded at action,
receipt and saved backward. Without these arguments, existing defaults, receipt,
inventory and coefficient/physical gradient action remain unchanged. The EM
wrapper forwards this policy while retaining the original PMCHWT algebra.

Receipt experimental_directional_coefficient_fft_kernel_budget_v1 records
ideal_truncation_admission and complete_support_radius_verified=False. Bounds
cover an ideal continuous kernel polynomial, each gradient component separately.
They do not certify input supports, native coefficient/FFT precision, quadrature,
phase transfers, weighted/global RWG operators, physical or production qualification.
