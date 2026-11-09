# Primary-subtracted two-dimensional tensor source-plane limit

The existing outgoing tensor Rayleigh representation requires nonzero normal
separation. A new explicit limit estimator subtracts the full free-space
primary at positive/negative offsets, then approaches the source plane. The
result is the nonprimary regular correction, including at the primary origin;
it must never be mistaken for the full dyadic distribution or a contact term.
Nonprimary lattice singularities remain rejected even beyond the mode budget.

At each height, reciprocal shells double until both signs independently satisfy
an absolute balanced-tensor difference budget of one eighth of the requested
absolute error. Free-space spatial quadrature also uses a purely absolute
one-eighth budget. These gates run before signed averaging, preventing odd
spectral errors or large primary norms from hiding unresolved spectral tails.
The symmetric mean has even height powers; Richardson extrapolation removes the leading quadratic term. Two consecutive extrapolated differences must satisfy half the total absolute budget.
All differences are estimators, not certified total error bounds. No artificial
loss or implicit spectral-pole bypass is introduced.

The first CUDA controls cover cancellation, independent budget exhaustion,
invalid input/nonprimary poles, scalar-limit comparison against independent
nonprimary images, and preservation of actual tensor primary-quadrature failure.
Evidence: /tmp/gsp-periodic-plane-limit-v1, validation pending. No success is
claimed before its terminal receipt. General-tensor success, strict error
certification, lossless/Wood limits, the 1D source-plane counterpart and public
integration remain required subsequent work. This is not a replacement for
those original requirements, nor an Ewald implementation.


Preserved first controls: v1 five pass/one scalar-limit failure (36.70s);
v2 six pass/one scalar-limit failure (38.99s). The latter adds an independently
audited catastrophic-primary-cancellation rejection using 64*eps times the sum
of full periodic/primary balanced norms, compared with the component budget.
At height .02, shell128-to256 difference at the primary origin was .0752125
against .0025. The earlier .08-to.04 height difference also exceeded .01.
No tolerance was increased. Version3 uses symmetric Richardson differences,
initial height .16, three halvings and max512 shells under the same total .02
absolute tolerance. Validation is running; do not claim a successful limit yet.


Version3 finished 6 passed/1 failed (115.70s), unchanged sources. Its reciprocal
gates passed, but the first extrapolated difference at one point was .01726945
against .01; only the second difference passed, so the limit correctly rejected.
Version4 retained the material, queries, absolute tolerance .02, max512 shells
and three halvings, using initial height .12 to resolve the asymptotic regime.
Only the failed actual case and a new quadratic manufacturing control ran:
2 passed, zero failures/errors/skips in120.40s. All592 source hashes unchanged.
The new remainder2+z+100z² requires Richardson cancellation to satisfy the
1e-7 assertion; the unextrapolated final mean misses by .015625. Independent
static audit confirmed this coverage and closure of the earlier precision P2.
Across v3/v4 all eight current nodes have targeted passing evidence. Preserve
v1-v3 failures; do not rewrite them into a green full run. No full suite rerun.
The successful actual case is scalar-limit only; true tensor primary budget
failure propagates correctly, but general-tensor success is not established.


A subsequent genuine full-tensor forward control has now passed; it does not
close the full material domain. Frequency300MHz, lattice(.7,0,0)/(0,.8,0),
Bloch(.2,.1,0), queries(0,0,0) and(.1,.07,0). Both tensors add2jI to real parts
epsilon=[[1,.1,.05],[.1,1.3,.08],[.05,.08,1.6]],
mu=[[1.1,.04,0],[.04,1.2,.06],[0,.06,1.4]].

/tmp/gsp-plane-tensor-primary-v1 records four purely absolute primary checks
at heights.12/.06/.03/.015, final128x64 quadrature,113.388s total. The actual
periodic control /tmp/gsp-plane-tensor-actual-v1 reuses their exact retained
results (four hits), passes in2003.358s, then restores the original primary
evaluator before independent image summation. Image shells1->2 tail errors
2.8437e-5/7.4484e-5 satisfy.0025; periodic-image balanced differences are
2.328258966e-5/5.652429064e-5 versus.02. Overall2086.926s. All365 production
hashes and retained producer inputs remain unchanged. Receipt SHA256:
84ae1a596adf086ddc4511cdff317cc7fb11946b3f3428e28e12c2ea5906409a.

Independent evidence review checked saved tensors/receipts/artifact hashes and
a supplementary primary-binding-audit.json against declared material, query
and quadrature configuration, with no unresolved confirmedP1/P2. The consumer
script itself did not enforce all those checks; the supplement closes this
actual run only. Artifact hash binding begins at consumer entry, not producer
completion. Import paths were reproduced in a separate interpreter with the
same PYTHONPATH, not introspected in the live process. Do not claim a stronger
provenance chain. This selected strong-loss forward success proves neither
gradients, the full material domain nor public scene integration or production
qualification. No repeated full regression was run for this evidence update.
