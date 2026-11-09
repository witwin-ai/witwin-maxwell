# Analytic incident-trace lifting: exploratory design

Status: continuous equations independently reviewed; CUDA far-field prototype
running outside the production tree. This draft does not amend the frozen v1
contract or claim GSP-01 completion.

The subsequent [representation and migration decision](mom_lifting_representation_contract_v2.md)
defines the candidate's explicit archive-versus-new-solve contract and the
required enabling amendment. Main runtime v1 remains unchanged.

## Motivation and mathematical scope

The total-trace RWG baseline has a nonzero scattered field at identical interior
and exterior media. Fixed-mesh assembly and decoder order increases leave that
residual essentially unchanged. Direct linear solves have residuals near 1e-14.
Subtracting a numerical background solve from the decoder would change source
semantics and conceal the original discrepancy. Instead, investigate an
explicitly enriched trial space containing the exact analytic incident trace.

For exp(-i omega t), write physical total trace c = c_inc + y, where
J_inc = n cross H_inc, M_inc = -n cross E_inc and y is represented by RWG.
The continuous background identity A0 c_inc = b and A = A_ext + A_int imply

    A y = -(A_int - A_background,int) c_inc.

The source divergences are +i omega epsilon_ext n dot E_inc and
+i omega mu_ext n dot H_inc. Evaluate these analytic functions directly at
source quadrature nodes, without projecting c_inc into RWG first.
For each row/source panel pair, integrate the difference kernels at identical
nodes. Material-weighted differences retain weak singularities and still need
touching and adaptive near quadrature. Implement stable expm1-based differences
before complex64 or very small contrast qualification. The same-panel magnetic
principal-value term vanishes by planarity and should be set exactly to zero.

Exterior scattered fields are Phi_ext y. Interior total fields are

    F_inc - Phi_int y - (Phi_int - Phi_ext) c_inc,

which equals -Phi_int(c_inc + y) inside and satisfies the interior Maxwell
system. The background incident field is not itself an interior-medium solution.
Tangential material-independent half jumps cancel in kernel differences;
normal jumps weighted by inverse epsilon or mu do not generally cancel.

## Required explicit production contract if adopted

- Version formulation and current-kind identities. y is scattered-trace
  coefficients; it must never be exposed as complete physical J/M coefficients.
- Carry each RHS's analytic incident source identity in the result, modal
  queries, snapshots, exact persistence and cache keys. Batch permutations must
  permute carrier and coefficients together.
- Expose total physical currents as analytic carrier plus RWG current, with
  appropriate trace and power integration. Any coefficient-only view must
  declare its meaning and representability limits.
- Re-derive query traces, continuity, power and interior decoders. Exterior
  y-Jacobian remains Phi_ext and interior y-Jacobian remains -Phi_int; source
  derivatives also include the carrier and RHS and must retain PyTorch graphs.
- Do not silently substitute this representation into legacy snapshots or
  arbitrary algebraic RHS. Source data supports verification and a separately
  identified new solve; it does not make old total-RWG to lifted coefficients
  exactly convertible. Preserve unsupported historical results as archives.
- Replace the primary method only after independent nonzero weak contrast,
  resonance, near/far field, full-angle power and convergence tests pass.
  No hidden baseline correction or zero-material special-case is acceptable.

## Prototype limitations

/tmp/gsp01_analytic_lifting_probe.py uses common-node direct kernel subtraction,
fixed order 6 and only exterior far-field samples. It does not implement stable
small-contrast kernels, adaptive near estimates, interior fields, persistence,
trace/power integration or production metadata. Its qualification flag is always
false. Zero contrast is an encoded identity, so its success alone is not an
independent physical validation. The independent Mie oracle has roundoff-level
nonzero output at exact contrast zero; absolute error is the meaningful metric.

## Public current semantics: independent design review

Prefer a single authoritative scattered coefficient tensor y plus the exact
analytic carrier. Do not make projected total coefficients P(c_inc)+y the
canonical stored state: the analytic trace is generally outside RWG, and two
authoritative tensors would create consistency and cancellation risks.

If adopted, expose `scattered_coefficients`, `J_scattered`, `M_scattered`, and
an explicit `evaluate_total_currents(points, face_indices=...)`. The latter
adds the analytic trace to the RWG scattered trace on an explicitly identified
face. Edge/vertex normals are not inferred through hidden nearest-face choices.
The new representation must reject ambiguous legacy `.coefficients`, `.J` and
`.M` access with an actionable explanation; silently returning scattered labels
under old total-current names is not acceptable. Any projected total-current
export is a named derived diagnostic with a projection/error identity.

Keep the underlying equivalent-current potential linear in its RWG tensor.
Result-level queries compose that linear action with the carrier; this preserves
an unambiguous operator/adjoint interface without pretending that a physical
total-field decoder at fixed illumination is homogeneous in y.

Surface payload v2 must bind exact y, mesh, coefficient order/current kind,
formulation/kernel/decoder/RHS-integration versions and every RHS's complete
carrier (source, complex amplitude/phase and frequency). Reject missing labels,
carrier mismatches and unconverted v1 artifacts. There is no automatic numerical
conversion by subtracting projected incident coefficients. Bench datasets must
bind these identities and must not merge old total-RWG labels with new scattered
labels or share unqualified normalization/residual targets.

## Stable small-separation difference gradient

With z=i*(ki-ke)*r and d=expm1(z), use

    grad(Gi-Ge) = exp(i*ke*r)/(4*pi*r^3)
                 * [i*ke*r*d + R(z)] * displacement,
    R(z) = z*exp(z)-expm1(z)
         = sum_{n=2}^infinity (n-1)/n! * z^n.

A bounded Horner expansion through n=14 for |z|<0.25 avoids cancellation
between first-order terms. The exploratory CUDA scalar check against an
independent 100-digit oracle passed all 40 combinations of complex64/128,
separations 1e-12 through 1 m and contrast zero through 0.1. This checks only
kernel arithmetic at the specified representable inputs; it does not establish
surface quadrature, source parameter conditioning or solver qualification.
