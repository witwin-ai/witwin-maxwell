# Finite field convergence admission

Separate robustness increment from base 547e1c4. The scalar and tensor field
surface estimators and periodic surface/Ewald estimators previously compared
error <= threshold without requiring every metric finite. Finite field values
can overflow norm accumulation; finite tolerances can overflow thresholds. Such
values must not establish convergence. Every affected acceptance gate now checks
error, reference norm and threshold finiteness. Periodic Ewald additionally
checks both complete-field and correction norms before their minimum is used.
The exhaustive Ewald matrix refinement already has these guards and is unchanged.

No tolerance, physical gate or normal finite arithmetic is loosened. Seventeen
CUDA synthetic regression nodes exercise threshold overflow, finite-input norm
overflow, and normal finite acceptance across free scalar/tensor, periodic
scalar/tensor, and independent Ewald field refinement. They do not construct or
solve a physical operator and do not establish physical accuracy. Syntax and
diff checks pass; independent static audit requested. Numerical verification
will queue after the existing frozen merge regression and 1D dyadic checks,
without modifying or restarting either predecessor. Stage commit is pending.


Independent static audit confirmed the guards preserve finite-case behavior and
requested explicit one-sided Ewald overflow coverage. Two additional controls
keep the image difference zero and minimum-based threshold finite while either
the complete-field norm or correction norm alone overflows. Both must reject;
the correction-only case uses exact primary/image cancellation. These tests
specifically protect against minimum hiding invalid reference metrics. No
confirmed production P1/P2 remains in static review. CUDA acceptance is pending.


CUDA acceptance completed on physical GPU1 after the queued line-kernel checks.
`/tmp/gsp-field-admission-targeted-v1` records 27 passed, zero failures/errors/
skips (pytest 5.70 seconds), with all 590 numerical/test hashes unchanged.
Receipt SHA256: 3bb93546a528184f129012a8c9b2f987dfa839873b86e0dc72a4011b5c2ceac5.
The scope is 17 synthetic finiteness controls, four periodic tensor field-policy
cases and six fused-sheet cases. Finite controls and explicit overflow rejection
pass; this does not close actual adaptive tensor-field physical failures or
change their tolerances. Earlier pending entries record the pre-run state.
