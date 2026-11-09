# Original optimizer comparison plan

User request: finish commits and compare numerical results and performance with PNGF.

1. Verify the completed native GPU implementation is fully committed.
2. Build an external-header C++ adapter against upstream revision
   `a3864e2841919cad9fcea1dc5478779111ff9de1`. Do not vendor upstream code.
3. Generate physical nonuniform Yee-grid responses on GPU 1, explicitly permute
   between Maxwell's sorted indices and upstream's x-fast planar ordering.
4. Run original PEC candidate/accept methods and public Maxwell trial/accept on
   identical responses, masks, incident fields and three observations. Preserve
   feed gaps; include shared tile edges, both flip directions and rejected trials.
5. Check projected results against each other and a fresh CUDA reduced LU;
   measure synchronized GPU API calls and CPU calls separately, including the
   accepted-update costs. Record CPU threads, precision, revision and input hashes.
6. Commit the reproducible harness, rerun at that clean revision, retain raw
   measurements, review numerical and timing scopes, and commit the assessment.

Acceptance threshold: maximum per-trial relative projection-vector error <= 1e-7,
finite outputs and matching changed-row ranks. Three sizes, 90 trials each,
60 accepts, CPU 1/4 threads and physical GPU 1. Cold Maxwell preparation is
reported separately; this experiment does not time original MUMPS/APF and cannot
support a claim about paper-scale end-to-end performance.

Completed: measured all three sizes at clean commit 74e41d2. All numerical and
raw-data readback checks pass. Original CPU 4T is faster for the complete timed
update sequence at all three sizes. See `docs/assessments/fdfd-pngf-optimizer.md`.
