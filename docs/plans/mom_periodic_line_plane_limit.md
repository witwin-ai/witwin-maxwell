# One-dimensional tensor primary-excluded plane limits

The 1D reciprocal tensor line kernel now composes the shared two-sided primary
subtraction, subtraction-resolution guard and Richardson limit estimator.
The supplied unit normal must be perpendicular to the lattice line. Each height
checks quadrature, transverse range, reciprocal range, refined quadrature and
combined spectral differences separately at absolute budget A/8 before sign
averaging. Free-space primary integration uses the same component budget.
Two consecutive Richardson differences must satisfy A/2. Insufficient budgets
raise structured failures; these differences are not certified error bounds.
The line estimator returns its original requested-budget result. The existing
2D entry uses the shared engine without changing its numerical sequence.

An explicit transverse_map='sinh' option uses q=k0*sinh(t), with
 t=(2u-1)*asinh(Q/k0) and exact integration Jacobian. The fixed vacuum scale is
independent of constitutive parameters. The finite interval remains [-Q,Q].
The default linear map retains its original version; the sinh map is separately
identified as outgoing_line_sinh_residues_v1. Finite mapping guards reject
unrepresentable quadrature nodes or weights.

## Targeted validation

- /tmp/gsp-line-plane-gates-v1: 14 composition/shared guard nodes pass.
- /tmp/gsp-line-sinh-targeted-v1: 23 affected nodes pass, no failures/errors/skips,
  31.49 seconds. All 594 numerical/test source hashes remain unchanged.
  Receipt SHA256: 8f7e28f42d97fef06e57a7868bb9ef312da511fb5851c7859801a7feb15e9d1f.
- Mapping controls cover the constant integration measure, an independent scalar
  Ewald/Hessian comparison and actual general tensor material finite differences.
  Synthetic controls cover limit composition, independent budget failures,
  precision loss and invalid inputs. These are not full material-domain tests.

## Actual 1D source-plane comparison

Both runs use epsilon=mu=(1+2j)I, frequency 300 MHz, lattice (.7,0,0),
Bloch vector (.2,0,0), plane points (0,0,0) and (.1,.07,0), normal (0,0,1),
initial height .12, three halvings, transverse cutoff 2048, quadrature order
256, sinh mapping, and absolute acceptance .02 in the impedance-balanced norm.

The first run /tmp/gsp-line-plane-actual-v1 failed after 328.744 seconds at
h=.015: reciprocal shells 128->256 differed by 2.471614439 > .0025.
Its failed receipt and 19 finite spectra are preserved. Receipt SHA256:
e3b45e6d3a11230992b7df4fe3d95dd58b9fd42d87076522c4fe40c77d756527.

The refined /tmp/gsp-line-plane-actual-v2 run uses shells 256->512 with the
same remaining settings and thresholds. It passed with exit code zero after
2507.443 seconds. Twenty spectra were retained; four exactly matching finite
spectra were reused after checking source identity, artifact hashes, queries,
materials, frequency, lattice, Bloch vector and all evaluation arguments.
All estimator gates were rerun. This is an experiment-specific cache; kernel
policies are unchanged between these two runners. Timing is not isolated
performance evidence because another process shared physical GPU 1.

Final reciprocal-tail difference at h=.015 is 2.16513e-7 < .0025.
The consecutive Richardson differences are [.00155977436,.00589265540] and
[.000102628072,.000399741165], both below .01. Independent primary-excluded
image shells 6->12 differ by less than 1e-6. Final image-comparison errors are
[1.4484653473491231e-5,2.7219848553292378e-5], both below .02.
All 594 numerical/test hashes remain unchanged; actual imported module paths
and runner SHA are recorded. Receipt SHA256:
ec500dd46657c07a999e7a4fb5aae20a06725624efdf0f60e9a4a69e1ae74fe1.
Terminal evidence inspection is retained in review.json alongside the receipt.

## Remaining scope

This actual plane-limit success is a strong-loss scalar-limit control.
General reciprocal and nonreciprocal tensor plane limits, broader loss regimes,
full material gradients, public assembly, physical accuracy and complexity
qualification remain open. No production qualification follows from this stage.
GSP-01 and all unrelated preserved failures remain open; the full goal and
final acceptance thresholds are unchanged.

A general reciprocal-tensor follow-up runner is prepared at
/tmp/gsp-line-plane-tensor-v1.py but has not run. It reuses four accepted
free-space primary values at exactly matching points and materials, with saved
configuration, producer receipt, dtype, shape and finiteness checks. Reviewed
producer script identity binds the frequency absent from the old artifacts.
Independent preflight audit closed a missing saved-configuration binding; both
primary and spectral instrumentation restore before the independent image
oracle. This preparation is not numerical evidence.


## Selected reciprocal full-tensor source-plane control

The frozen Maxwell `838e69e421161af2968899a83db8c0eb9fa03113` control
`/tmp/gsp-line-plane-tensor-v2` completed in 29540.620 seconds on physical GPU1.
All 594 source/test hashes were unchanged; runner and all reused producer hashes
were verified independently. Terminal receipt SHA256:
`877793959205c57487203e052a161b05fe2fe483a0dade1e77844b1c321ecf8f`.
Runner SHA256:
`64a7463c98c33b7ee5fd27a37fa87b7d08e0be1b5bbcf1c2c5867d12110f57c1`.

Frequency is 300 MHz, lattice (.7,0,0) m, Bloch (.2,0,0) rad/m, and normal z.
Relative epsilon real rows are (1,.1,.05), (.1,1.3,.08), (.05,.08,1.6);
relative mu real rows are (1.1,.04,0), (.04,1.2,.06), (0,.06,1.4).
Both add 2i times the identity, with reciprocal transpose symmetry.
Queries are the origin and (.1,.07,0) m; absolute balanced tolerance is .02.

Four heights .12/.06/.03/.015 m each pass the five independent transverse
quadrature/tail, reciprocal-tail and combined checks: Q2048 to4096,
n256/512/1024/2048 and shells256 to512. Every component threshold is .0025.
At the last height the largest transverse-tail difference is 2.96316e-5 and
reciprocal difference is 2.08551e-6; maximum primary-subtraction resolution
estimate is 6.23829e-8. Consecutive Richardson maxima .00515964 and .000348359
are both below .01. All original lower-resolution checks remain in the receipt.

Seventeen new spectra plus three exact spectra from the interrupted v1 were
retained, with finite complex128 [4,6,6] values and exact query/material/frequency/
settings bindings. Four independently produced primary heights were reused;
the old artifacts lack a frequency field, which is bound by the pinned producer
runner at 300 MHz. The v1 interruption record remains unchanged and is not
relabeled as a completed run.

The nonprimary image reference compares shells1 to2; tail differences
[2.76985e-5,7.26754e-5] are below .0025. Its three free-space integration groups
converge with absolute budget .02/128. Final balanced absolute discrepancies
[4.446368518065514e-5,5.369688342078379e-5] are below [.02,.02].
Periodic artifact SHA256:
`8dd45e11bbf9863a3c8fc1e7d3ba461a698745acf8481a8e91b047a8f633aacb`;
reference artifact SHA256:
`07f2eda640057918119e03418954c9ad001aa25adeec1539ff05361b7d1a370f`.
The final reference artifact's full physical binding relies on the hashed
runner/receipt. Independent read-only evidence audit found no confirmed P1/P2.

This is a selected strongly lossy reciprocal tensor representation control.
The reference shares the free-space Maxwell implementation. It does not prove
general lossless/nonreciprocal plane limits, surface scattering, global error,
performance, or production qualification, and does not cover later public API
work. GSP-01 and every previously recorded failure remain open.
