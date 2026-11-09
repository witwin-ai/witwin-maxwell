# Contiguous surface assembly storage

The existing assembler stores four contiguous blocks and then materializes a
second full matrix when converting them to the public row-major layout. Its
global pair classifier also runs while those blocks are resident. Both lifetimes
obstruct the actual core18 mesh refinement under the unchanged 29e9-byte budget.

The isolated change writes directly to one contiguous final matrix. For local
edge indices r,c and block indices i,j, the destination is (r+iE,c+jE), exactly
the previous block-permute mapping. Category selection, batch order, both media,
quadrature calls, four block contributions and accumulation remain unchanged.
The destination itself is contiguous; strided block views are not scatter
destinations. This matters because the installed accumulating CUDA scatter can
copy a noncontiguous destination.

Pair classification precedes matrix allocation. Global pair numbers and their
row/column arrays are released after their last use. Selected batch identities
recover the same integer rows and columns by division and remainder. There is
no host numerical accumulation, new quadrature rule, streamed operator, solver
fallback, approximation or new physical domain.

For the candidate n24 mesh, E=17280 and N=2E. A complex128 matrix alone occupies
19,110,297,600 bytes. Avoiding its terminal duplicate and separating classifier
temporaries removes two specific overlaps; it does not establish total fit.
The real Gram, Cholesky factor, validation, integration workspaces and eventual
solve still require independent measured peak accounting. A separate actual
workspace query measured 4,942,336 requested bytes for the installed typed LU
backend, but did not factor this matrix or validate a production storage mode.

Required verification before this assembly change is adopted includes exact
source-bound old/new physical matrices across shapes and dtypes, multiple batch
partitions, quadrature metadata, RHS and actions, adjoints, eight-RHS native
solves and true residuals, and the supported fixed-current gradient interface.
Observed allocation changes must be separated from total peak and shared-device
timing. Full affected MoM/public tests and independent review remain required.

This document describes an isolated implementation and its verification scope.
The running production provider remains e95ab35. No n24 mesh solve, formal label,
external-reference qualification or H2/G0 acceptance is established here.
