# Exact polynomial remainder and physical curl terms

### Factored evanescent tensor plane remainder primitive completed

One native CUDA complex128 internal primitive in the primary dyadic Green
module constructs regular evanescent plane blocks with explicit inverse-length
polynomial factors. It preserves actual fixed real symmetric SPD epsilon/mu,
including genuinely nonproportional and noncommuting coefficients; no material
projection, symmetrization, removed imaginary response, SPD tolerance weakening,
artificial damping, finite spectral cutoff or CPU production fallback.
Original17 top-level functions and all395 other production modules remain exact.
The primitive is NOT wired into spatial integration, public solvers or persistence.

The transverse-aligned tangential similarity S=diag(s,1,s,1) balances the
Maxwell generator as A=A0+x*A1+x^2*A2, x=1/s. Embedding and forcing have
explicit small polynomial corrections. Coupled Newton matrix-sign iterations
retain projector differences through the inverse difference identity. A bounded
outgoing generator removes incoming exponential growth; an upper triangular
matrix exponential retains the propagator difference. EE/HH subtract their
Coulomb baseline through the three-term product difference. EH/HE retain full
physical curl terms; normalized x=0 cross terms can survive unbalancing and
must not be subtracted with Coulomb. Singular/nonconverged/nonfinite/unresolved
projectors and unsupported domains refuse before returning a spectrum.

The original three FAILED plane points reuse closed exact MP100 arrays without
recomputing reference/native full spectra, older twelve complete-plane controls,
three successful regular points or expensive original spatial covariance/HB.
First numerical candidate CLOSED1FAIL/XML4.307s:
six EE/HH blocks passed1.7e-14..2.0e-13, six EH/HE failed.7045..7058. The failure
and original full primary/test source remain preserved. Original prelaunch
review incorrectly treated the normalized baseline as pure diagonal Coulomb;
that report is retained and explicitly corrected by the incremental audit.

Terminal-only correction inserts exactly one electric/magnetic channel mask:
EE/HH select their unchanged factored difference; EH/HE select full propagation.
An exact source inverse and channel-preserving continuation proof bind the
closed six diagonal PASS without rerunning them or relabelling the old snapshot.
Only six FAILED cross blocks rerun, now PASS5.3e-16..5.8e-15 unchanged.005 gates,
XML3.997s. One genuinely NEW full noncommuting SPD
medium at two NEW q vectors is compared against independent full6x6/Coulomb
MP100 subtraction; all eight selected blocks pass unchanged.005. Four NEW
physical-domain checks reject1e-20 imaginary loss,1e-16 roundoff asymmetry,
indefinite coefficients and material gradients without modifying actual input.
General collector5PASS/XML4.375s. Saved arrays
support subsequent reuse. No previously covered numerical PASS is replayed.

Original sessions26462/42965 and the general session are drained; execution-known
runner identities absent outside sandbox, unobserved pytest identities honestly
null. Frozen closure415/429/411 inputs; GPU1/installed Core. Reference artifact
path/SHA pairing is explicit; absence of both skips ordinary repository tests,
partial/missing/mismatched input refuses. Original prelaunch test-contract P2,
whole original test and exact inverse retained; no prelaunch numeric replay.

This closes selected internal spectral comparisons only. Spatial quadrature,
original20h covariance failure, tensor cell/public AC/HB and general complex,
trainable/indefinite tensor response remain OPEN/unqualified. No original matrix,
solve, producer/archive, capacity/resource or full regression replay. LIVE f30,
all-point scaling, magnetic DAG and tensor providers remain immutable. Independent
PHASE audit precedes human-owned commit. GSP01/GSP05/GSP06/GSP09OPEN; Goal ACTIVE;
original scope and all acceptance thresholds unchanged. Current four-repository
full acceptance and at least two independent FINAL Goal audits remain required;
no production qualification.
