# Public screened Ewald FFT FMM

`FMMConfig(table_storage='fft_ewald_v1')` selects FFT translation of the actual
configured scalar Ewald exterior kernel. Use `MoMConfig(operator_backend='fmm',
periodic=MoMPeriodicConfig(...), dtype=torch.complex128,
riesz_storage='local_rwg_v1', ...)` and iterative stopping without a dense
fallback. Coordinates are float64 and the native fixed order lies in 0..12.
The version is `cartesian_fmm_ewald_fft_config_v1`, binding every FMM field and
both shift policies. Public snapshots and Bench preserve this tag; only
internal free-space near-image and body-interior translations use `fft_v1`.
The existing free-space FFT config and all older payload bytes remain unchanged.

The root coefficients come from the existing original screened Ewald kernel,
with primary and all declared nonprimary near-image free-space kernels excluded.
Each excluded near image is restored by one phased free-space hierarchy, with
original exact leaf quadrature. Body interiors are evaluated once in free space.
The root uses physical zero displacement and independent normalization
`sigma_min(lattice)/(2*order+2)`. Its temporary compact coefficient constructor is
setup workspace, released after independent FFT ownership is established.

The retained capacity plan and actual tensor enumeration include the FFT root,
its physical displacement and scale, and each near index and phase exactly
once. For dimension d, each declared near-image metadata entry uses d*8+16
bytes. The planned M2L category is the root FFT table plus all free-space FFT
near/interior tables. Input meshes, shared lattice, solver/Riesz state,
setup/action/backward workspace and allocator metadata remain excluded. Visible
FFT workspace and physical RHS chunking follow the separately reviewed FFT
contract; this is not a full peak-memory guarantee.

A finite-image periodic kernel cannot select this tag. Free-space FFT remains
explicitly rejected by periodic constructors. Automatic periodic order/shell
selection, kernel/RWG error-budget admission and free-space bound substitution
remain unsupported and reject before heavy construction. Native order/precision,
original provider/kernel/lattice/cutoff identity and retained budgets are checked
before projections or root expansions. Existing lattice, source phase and pole
semantics still apply: incident waves must occupy the declared Bloch sector.

Acceptance uses original configured-cutoff weak all-column matrices in 1D, 2D,
3D and a multibody case with nonprimary near images, plus Hermitian dot products,
fixed-current gradients and exact retained inventories. A fresh nonzero-Bloch
public producer and independent Bench declaration test exact current/config
restoration with constructors and solves forbidden. No repeated field-query
acceptance is needed for this storage change.

These comparisons share the declared Ewald cutoffs and quadrature. They do not
certify cutoff convergence, infinite-periodic physics, material gradients,
large-system resources, directional high-frequency behavior, complexity,
production populations or the final generalized solver Goal. Physical,
complexity, full-peak and infinite-periodic qualification flags remain false.
