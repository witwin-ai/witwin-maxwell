# 3D checked periodic regular composition

Status: implementation and focused ownership/routing validation passed independent audit; physical qualification remains open. Depends on 3D source-plane commit afd8fe8 (integrated as a3a86c4). The actual nonreciprocal 3D off-plane job completed and passed independent audit before main integration. All 600 integrated source/test hashes exactly match the audited final adapter tree.

The existing checked regular adapter accepts three-dimensional lattices. Off-plane queries use the checked stacked spectrum minus the checked free primary. Source-plane queries in the principal layer use the existing checked limit. For r = u + m a3 on a translated layer, the regular value is exp(i m beta.a3) [R(u) + G_free(u)] - G_free(r). Nonprimary lattice singularities are rejected before wrapping.

Translated-layer estimates allocate A/2 to the regular limit, A/8 each to the two free integrations, and A/8 to the floating-point resolution guard. Bloch vectors are real under the lattice contract. Principal-layer and 1D/2D budgets are preserved. These estimators are not certified global bounds.

Targeted controls cover both handednesses and positive/negative translated layers, original-primary ownership, absolute budgets, nonprimary singularities beyond cutoff, precision rejection, empty and mixed plane/off-plane queries. Actual 3D scalar and full-tensor physical comparisons, material gradients through plane limits, public background material API, production assembly convergence and final acceptance remain separate open obligations.

## Focused evidence

- `/tmp/gsp-stacked-regular-gates-v1`: 10 passed, no failures/errors/skips, 8.753 s; receipt SHA256 `73e9bd290241d9fe75e4982c5ef6f7d76cd4659eadc9f87e72fe6e537abd1dc2`.
- `/tmp/gsp-stacked-regular-routing-v1`: one new 3D matrix/field routing node passed, 9.028 s; receipt SHA256 `226fd5dae4c8729764fd7cd64959a52d29e0cb91d07b245351a488251edb9d8f`.
- Each run binds 600 source/test hashes before and after. Production is identical between the two runs. Only the routing test was subsequently parameterized; exact reconstruction verified that all earlier ten nodes are unchanged.
- Independent final review found no P1/P2. The routing test substitutes adapter evaluation and checks primary ownership, matrix/field dispatch and current gradients. It does not prove actual 3D numerical assembly convergence. No full regression was repeated.

## Actual nonreciprocal off-plane adapter control

`/tmp/gsp-stacked-regular-actual-v1` passed in55.151seconds on main6f2114e.
The adapter recomputed both its checked3D spectrum and free primary. Its two
balanced absolute differences from accepted nonprimary spatial images were
3.920049712266874e-5 and4.006081732801563e-5, each below.002. Source-plane
count is zero; the primary reached128x32 with absolute.002/8 budget and the
subtraction-resolution estimates were5.5373e-13 and1.45752e-12. No source-plane
or translated-plane numerical acceptance is inferred from this control.

Only the independent-reference value was reused from the accepted40-call
`gsp-stacked-tensor-actual-v1` image comparison. Producer receipt, runner and
reference hashes are pinned, with exact queries, material, frequency,
reciprocity, lattice and Bloch vector checks. All other production source hashes
match the producer; the image class and reference-support AST in the changed
periodic module also match. Producer artifact hashes are checked before and
after. The candidate values are freshly evaluated. The image representation
shares the underlying free Maxwell kernel and is not an independent physical
solver. All600 source/test hashes remained unchanged.

Receipt SHA256de40934f87dce3f45d9cfd146ee9b1f1e96d41a8fd6cfd43a71eb48a96b6b6f0;
result SHA256d1f330172ad23bb3ca506382acd0714417a4a53e25af23f7a458e42af7b1a8a4.
The saved result binds value, component receipts, comparison, reference value,
queries, material, frequency, lattice, Bloch vector and integration config.
General domains, source-plane composition, assembly/scattering accuracy and
production qualification remain open.

## Actual translated-plane scalar control

`/tmp/gsp-stacked-translated-actual-v1` completed successfully in1654.272seconds
on015f3b9. Queries are(.1,.07,0)+m*a3 for m=1,-2, with scalar epsilon=mu=1+2i,
f300MHz, lattice(.7,0,0),(0,.8,0),(.05,.1,.9), and Bloch(.2,.1,0).
The actual adapter recomputes the local regular limit and both primary values.
An independent scalar Ewald Hessian, minus the analytic free Helmholtz Hessian
at each ORIGINAL physical query, provides the reference. Separate real-space,
reciprocal-space, quadrature and combined refinement checks all pass.0025;
maximum reference difference is8.12e-13. This reference does not call the free
dyadic integration implementation used by the candidate.

The translated local limit uses A/2=.01, h=.12 through.0075, shells8 with
maximum1024 and four halvings. The first Richardson comparison correctly fails
.005 (.00619385); subsequent differences.000412515 and.0000262285 both pass.
The translation-resolution estimates are5.70e-11<.0025. Final balanced absolute
errors[1.7557902576641824e-6,1.755762919173101e-6] pass.02 for both layers.
Seven checked free evaluation artifacts plus full reference and candidate
records are saved. These scalar free evaluations use the exact analytic
specialization, not numerical quadrature. Independent final audit found no
P1/P2. All600 source/test hashes and incremental artifact hashes
remain unchanged.

Receipt SHA256c07cbc5ab998f286392c15489351ef5c50983c965ccad182ba9aa9aa6b769af2;
reference SHA25607e9a11610b172ad2327075aeab33691b7be154bbd21518ba21acaac85d71d21;
result SHA25615e0d561bf4b71b757a8be0ea690481199538681f79d5c77d709ece3affeb21a.
This closes a selected scalar translated-source-plane value control. Full-tensor
source-plane physics, actual assembly/scattering convergence, general material
domains, material gradients and production qualification remain open.
