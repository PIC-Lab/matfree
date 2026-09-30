# Generalized sparse eigenanalysis (PIC-Lab fork)

`matfree.generalized.eigs(A, E, options=EigenOptions(...))` computes selected
finite modes of `A x = lambda E x`. `E=None` means identity. Singular mass
matrices are supported when the shifted pencil is nonsingular. Matrices may be
real or complex NumPy arrays or SciPy sparse matrices. Use float64/complex128.

## Installation and example

Install this fork, not the upstream PyPI release, for this API:

```sh
pip install -e '.[ssa]'
```

```python
import jax
from scipy.sparse import diags
from matfree.generalized import EigenOptions, eigs

jax.config.update("jax_enable_x64", True)
a = diags([-1., -2., -3., -4.])
e = diags([1., 1., 1., 0.])
result = eigs(a, e, options=EigenOptions(k=2, shift=-0.1, backend="scipy"))
print(result.eigenvalues)
print(result.participation)
```

For GPU use `backend="cudss"`, a working CUDA JAX installation and a Spineax
build exposing `cudss.prepare_solve`. The ordinary upstream 0.0.5 wheel may not
provide this extension: requesting cuDSS then raises an explicit error.
`auto` chooses cuDSS when a GPU and that API are available, otherwise SciPy.
An explicitly requested backend never silently falls back.

## Algorithm and convergence

1. Equilibrate both matrices with identical invertible row/coordinate changes.
2. Factor `K=A-shift*E` and its adjoint once each; retain factors throughout the
   analysis. Complex systems use an equivalent real 2-by-2 block embedding.
3. Apply Matfree's fully reorthogonalized Arnoldi to `T=K^-1 E` and `T.H`.
   Recover descriptor left vectors by applying `K^-H` to the left operator's
   Ritz vectors. This avoids directly iterating the poorly scaled left pencil.
4. Select the requested `k` eigenvalues nearest the shift, within optional
   frequency limits. `positive_only` rejects numerically real modes using the
   requested relative tolerance. Recover `lambda=shift+1/mu`.
5. Expand the subspace until left/right normwise backward errors pass for BOTH
   original and equilibrated pencils, and matched eigenvalues agree between
   successive expansions. Full-dimensional extraction skips the stability test.
   Exhaustion raises `EigenConvergenceError`; no dense fallback is hidden.

The backward error is `||Ax-lambda Ex|| / ((||A||F+|lambda| ||E||F)||x||)`.
Tiny errors on a badly scaled original pencil alone are insufficient, hence the
additional equilibrated-pencil gate and eigenvalue-stability gate.

This is expanding Arnoldi, not implicitly restarted Arnoldi. Memory is O(n*m)
for basis dimension m, capped by `max_subspace`. It does not guarantee discovery
of every mode in a frequency interval or every unstable mode. A shift exactly
on an eigenvalue is singular; choose another shift. Near repeated/defective
modes individual participation factors may not be identifiable; such detected
failures raise. Repeated eigenspaces need a future block/subspace treatment.

## Participation and output

Left/right vectors are mapped back to the original coordinates and normalized
so `y.H E x=1`. Reported participation is `abs(x*conj(E.H@y))`, normalized to
sum to one per mode. This reduces to conventional state participation for
`E=I`; zero-mass algebraic coordinates have zero participation when E is diagonal.
Coordinates and units still define the meaning of a participation factor.

Results include right and left vectors, both sets of residuals, actual subspace
sizes, two-factorization count, solve count (including left recovery solves),
and factorization/iteration wall times. These times include JIT compilation.

Host work remains in sparse preparation, equilibration, factor diagnostics,
mode matching and convergence decisions at subspace boundaries. The cuDSS
Krylov loop has no Python callback per matrix-vector application. The small
projected dense eigendecomposition follows the installed JAX backend choice;
GPU-only placement of that operation is not promised. The SciPy reference uses
JAX Arnoldi with a CPU callback into cached SuperLU. This orchestration API is
not differentiable or externally jittable; existing Matfree APIs are unchanged.

## Verification

```sh
JAX_PLATFORMS=cpu python -m pytest -q tests/test_generalized.py tests/test_eig tests/test_decomp
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -q tests/test_generalized.py
```

Tests compare eigenvalues, participation, biorthogonality and residuals against
independent SciPy LAPACK for real/complex pencils, singular masses, complex
shifts, extreme coordinate scaling, frequency selection, failure handling and
resource cleanup. JAX-EMT's `tools/qualify_sparse_ssa.py` additionally compares
two-area, IEEE 13-bus and IEEE 123-bus physical models.

## Measured comparison (2026-09-30)

Single sequential runs on the WSL RTX 4080 workstation, JAX 0.11.1. Sparse
columns include preparation, factorization and Arnoldi compilation; dense
columns measure only the full eigensolve. These are different spectrum scopes.

| Case | Pencil order | Dense full spectrum | Matfree/SciPy | Matfree/cuDSS | Selected modes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-area | 40 | 0.00023 s | 0.533 s | 1.846 s | 3 |
| IEEE 13 | 323 | 0.0784 s | 1.372 s | 4.317 s | 4 |
| IEEE 123 | 1675 | 24.844 s | 3.474 s | 7.161 s | 4 |

Across these cases the largest GPU eigenvalue difference versus the dense
reference was 8.25e-7, and the largest normalized participation difference was
6.56e-9. Both sparse backends reused exactly two factorizations per analysis.
Small systems favor dense SSA; this implementation does not claim a GPU win
against SciPy sparse LU on these selected-mode problems.

See [machine-readable evidence](ssa-validation.json) for source hashes, residuals and convergence histories.
