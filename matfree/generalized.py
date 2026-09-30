"""Selected finite eigenpairs of regular descriptor pencils using shift-invert.

This optional SciPy/JAX layer owns host setup and convergence decisions. Arnoldi
and cuDSS solves run on the GPU without a Python callback per Krylov vector.
SciPy factors use a CPU JAX callback. Matrices and results are host sparse/NumPy
objects. The routine is not differentiable and is not itself jittable.
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import splu

from matfree import decomp


@dataclass(frozen=True)
class EigenOptions:
    """Select k finite modes nearest shift, optionally within a frequency band.

    Frequency is abs(imag(lambda))/(2*pi); positive_only excludes real modes.
    Expansion stops at max_subspace (or the matrix order). Failure raises, never
    silently returns unconverged modes. The selected set is not a certificate
    that all unstable modes or all modes in a band have been found.
    """

    k: int = 6
    shift: complex = 0.0
    backend: str = "auto"
    tolerance: float = 1e-8
    initial_subspace: int = 24
    max_subspace: int = 256
    minimum_frequency_hz: float = 0.0
    maximum_frequency_hz: float = float("inf")
    positive_only: bool = False
    seed: int = 0

    def __post_init__(self):
        """Validate convergence limits and mode selection."""
        for name in ("k", "initial_subspace", "max_subspace"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if self.max_subspace < self.k or self.initial_subspace > self.max_subspace:
            message = "subspace limits must accommodate k and initial_subspace"
            raise ValueError(message)
        if self.backend not in {"auto", "scipy", "cudss"}:
            message = "backend must be auto, scipy, or cudss"
            raise ValueError(message)
        if not np.isfinite(self.shift):
            message = "shift must be finite"
            raise ValueError(message)
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            message = "tolerance must lie in (0, 1)"
            raise ValueError(message)
        if not np.isfinite(self.minimum_frequency_hz) or self.minimum_frequency_hz < 0:
            message = "minimum frequency must be finite and nonnegative"
            raise ValueError(message)
        if (
            np.isnan(self.maximum_frequency_hz)
            or self.maximum_frequency_hz < self.minimum_frequency_hz
        ):
            message = "invalid frequency interval"
            raise ValueError(message)


@dataclass(frozen=True)
class GeneralizedEigenResult:
    """Converged finite modes; columns are eigenvectors.

    Left vectors satisfy y.H E x = 1 for each matched simple mode. Participation
    is abs(x * conj(E.H y)), normalized per mode. Algebraic coordinates have zero
    participation for diagonal descriptor masses with zero algebraic entries.
    Repeated/defective modes do not have unique individual participation factors;
    ill-conditioned left/right overlaps raise rather than inventing factors.
    """

    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray
    participation: np.ndarray
    right_residuals: np.ndarray
    left_residuals: np.ndarray
    balanced_right_residuals: np.ndarray
    balanced_left_residuals: np.ndarray
    backend: str
    subspace_sizes: tuple
    factorization_count: int
    solve_count: int
    timings: dict


class EigenConvergenceError(RuntimeError):
    """The requested number of validated finite modes was not obtained."""


def _matrix(value, name):
    result = sparse.csc_matrix(value, dtype=np.complex128)
    result.sum_duplicates()
    result.eliminate_zeros()
    if result.shape[0] != result.shape[1] or result.shape[0] < 1:
        message = f"{name} must be a nonempty square matrix"
        raise ValueError(message)
    if not np.all(np.isfinite(result.data)):
        message = f"{name} must be finite"
        raise ValueError(message)
    return result


def _backend(requested):
    if requested != "auto":
        return requested
    try:
        from spineax import cudss

        if jax.devices("gpu") and callable(getattr(cudss, "prepare_solve", None)):
            return "cudss"
    except (ImportError, RuntimeError):
        pass
    return "scipy"


class _Factor:
    def __init__(self, matrix, backend):
        self.token = None
        self.backend = backend
        # Real embedding permits complex shifts with real-only prepared APIs.
        block = sparse.bmat(
            [[matrix.real, -matrix.imag], [matrix.imag, matrix.real]], format="csr"
        )
        block.eliminate_zeros()
        self.n = matrix.shape[0]
        row = np.asarray(abs(block).max(axis=1).toarray()).ravel()
        if np.any(row == 0):
            message = "singular shifted pencil; choose a different shift"
            raise ValueError(message)
        row = 1 / row
        block = sparse.diags(row) @ block
        col = np.asarray(abs(block).max(axis=0).toarray()).ravel()
        if np.any(col == 0):
            message = "singular shifted pencil; choose a different shift"
            raise ValueError(message)
        col = 1 / col
        block = (block @ sparse.diags(col)).tocsr()
        block.sort_indices()
        self.device = jax.devices("cpu" if backend == "scipy" else "gpu")[0]
        with jax.default_device(self.device):
            self.row, self.col = jnp.asarray(row), jnp.asarray(col)
        if backend == "scipy":
            try:
                self.lu = splu(block.tocsc())
            except RuntimeError as exc:
                message = "singular shifted pencil; choose a different shift"
                raise ValueError(message) from exc
        else:
            from spineax import cudss

            self.cudss = cudss
            if not callable(getattr(cudss, "prepare_solve", None)):
                message = "cuDSS backend requires Spineax prepare_solve"
                raise RuntimeError(message)
            try:
                with jax.default_device(self.device):
                    self.token = cudss.analyze(
                        jnp.asarray(block.data),
                        jnp.asarray(block.indptr, dtype=jnp.int32),
                        jnp.asarray(block.indices, dtype=jnp.int32),
                        mtype_id="general",
                        mview_id="full",
                        reordering="btf_colamd",
                    )
                    self.token = cudss.factorize(self.token, self.token.values)
                    info = cudss.query(self.token)
                    diag = np.asarray(info["diag"])
                    if (
                        np.any(np.asarray(info["npivots"]) != 0)
                        or not np.all(np.isfinite(diag))
                        or np.any(diag == 0)
                    ):
                        message = "cuDSS regularized a singular shifted pencil"
                        raise ValueError(message)  # noqa: TRY301 - cleanup must release the token
                    self.prepared = cudss.prepare_solve(self.token)
            except BaseException:
                self.close()
                raise

    def solve(self, rhs):
        b = jnp.concatenate((rhs.real, rhs.imag)) * self.row
        if self.backend == "scipy":
            x = jax.pure_callback(
                lambda b: self.lu.solve(np.asarray(b)),
                jax.ShapeDtypeStruct(b.shape, b.dtype),
                b,
            )
        else:
            x = self.prepared(b)
        x = self.col * x
        return x[: self.n] + 1j * x[self.n :]

    def close(self):
        if self.token is not None:
            self.cudss.release(self.token)
            self.token = None


def _operator(mass, factor, *, adjoint=False):
    from jax.experimental.sparse import BCOO

    with jax.default_device(factor.device):
        device_mass = BCOO.from_scipy_sparse(mass)
    if adjoint:
        return lambda v: device_mass @ factor.solve(v)
    return lambda v: factor.solve(device_mass @ v)


def _ritz(operator, start, size, shift):
    # Use Matfree's Arnoldi; solve only the small projected dense problem.
    arnoldi = decomp.hessenberg(size, reortho="full", custom_vjp=False)

    @jax.jit
    def run(v):
        basis, h, *_ = arnoldi(operator, v)
        mu, small = jnp.linalg.eig(h)
        return mu, basis.T @ small

    with jax.default_device(start.device):
        mu, vectors = map(np.asarray, run(start))
    mu_scale = np.max(np.abs(mu[np.isfinite(mu)]), initial=0.0)
    finite = np.isfinite(mu) & (np.abs(mu) > np.finfo(float).eps * mu_scale)
    values = shift + 1 / mu[finite]
    vectors = vectors[:, finite]
    norms = np.linalg.norm(vectors, axis=0)
    valid = np.isfinite(values) & np.isfinite(norms) & (norms > 0)
    return values[valid], vectors[:, valid] / norms[valid]


def _residuals(a, e, values, vectors):
    av, ev = a @ vectors, e @ vectors
    scale = (
        sparse.linalg.norm(a) + np.abs(values) * sparse.linalg.norm(e)
    ) * np.linalg.norm(vectors, axis=0)
    return np.linalg.norm(av - ev * values, axis=0) / np.maximum(
        scale, np.finfo(float).tiny
    )


def _balance_pencil(a, e):
    # Equivalent row/coordinate changes, never E inverse or constraint removal.
    # Balance A and E jointly so the Krylov metric does not mix extreme units.
    rows, cols = np.ones(a.shape[0]), np.ones(a.shape[0])
    for _ in range(12):
        magnitude = abs(a).maximum(abs(e))
        if np.any(np.asarray(magnitude.max(axis=1).toarray()) == 0) or np.any(
            np.asarray(magnitude.max(axis=0).toarray()) == 0
        ):
            message = "singular descriptor pencil contains an empty row or column"
            raise ValueError(message)
        r = np.asarray(magnitude.max(axis=1).toarray()).ravel()
        r = 1 / np.sqrt(np.maximum(r, np.finfo(float).tiny))
        a, e = sparse.diags(r) @ a, sparse.diags(r) @ e
        rows *= r
        magnitude = abs(a).maximum(abs(e))
        c = np.asarray(magnitude.max(axis=0).toarray()).ravel()
        c = 1 / np.sqrt(np.maximum(c, np.finfo(float).tiny))
        a, e = a @ sparse.diags(c), e @ sparse.diags(c)
        cols *= c
    return a.tocsc(), e.tocsc(), rows, cols


_DEFAULT_OPTIONS = EigenOptions()


def eigs(a, e=None, *, options=_DEFAULT_OPTIONS):
    """Solve a regular, possibly singular-mass pencil near one target shift.

    Two factorizations (right and adjoint) are reused across all expansions.
    Left Arnoldi applies T.H = E.H (A-sigma E)^-H, then recovers descriptor
    left vectors with an additional adjoint solve per selected candidate.
    Normwise backward errors ||Av-lambda Ev||/((||A||F+|lambda| ||E||F)||v||)
    are checked for both original and equilibrated pencils in both directions; a
    second subspace must also reproduce the selected eigenvalues, unless the
    full space was used. Exhaustion raises EigenConvergenceError. This is an
    adaptive expanding Arnoldi method, not an implicitly restarted algorithm.
    """
    started = time.perf_counter()
    if not jax.config.x64_enabled:
        message = "generalized eigs requires jax_enable_x64=True"
        raise ValueError(message)
    a = _matrix(a, "A")
    n = a.shape[0]
    e = sparse.eye(n, format="csc", dtype=complex) if e is None else _matrix(e, "E")
    if e.shape != a.shape or options.k > n:
        message = "incompatible pencil dimensions or k exceeds matrix order"
        raise ValueError(message)
    original_a, original_e = a, e
    a, e, row_scale, col_scale = _balance_pencil(a, e)
    backend = _backend(options.backend)
    before = time.perf_counter()
    factors = []
    try:
        for matrix in (
            a - options.shift * e,
            a.conj().T - np.conj(options.shift) * e.conj().T,
        ):
            factors.append(_Factor(matrix, backend))
        setup = time.perf_counter() - before
        operators = (
            _operator(e, factors[0]),
            _operator(e.conj().T, factors[1], adjoint=True),
        )
        rng = np.random.default_rng(options.seed)
        with jax.default_device(factors[0].device):
            starts = [
                jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
                for _ in range(2)
            ]
        limit = min(n, options.max_subspace)
        size = min(limit, max(options.initial_subspace, options.k + 2))
        sizes, previous = [], None
        recovery_solves = 0
        best = 0
        while True:
            sizes.append(size)
            rr = lr = balanced_rr = balanced_lr = np.array([])
            values, right = _ritz(operators[0], starts[0], size, options.shift)
            lv, left = _ritz(operators[1], starts[1], size, np.conj(options.shift))
            freq = np.abs(values.imag) / (2 * np.pi)
            eligible = (freq >= options.minimum_frequency_hz) & (
                freq <= options.maximum_frequency_hz
            )
            if options.positive_only:
                eligible &= values.imag > options.tolerance * np.maximum(1, abs(values))
            indices = np.flatnonzero(eligible)
            indices = indices[np.argsort(np.abs(values[indices] - options.shift))][
                : options.k
            ]
            values, right = values[indices], right[:, indices]
            if len(values) and len(lv) >= len(values):
                rows, cols = linear_sum_assignment(
                    abs(values[:, None] - lv.conj()[None, :])
                )
                left = left[:, cols[np.argsort(rows)]]
                with jax.default_device(factors[1].device):
                    left = np.asarray(
                        jax.jit(lambda w: jax.lax.map(factors[1].solve, w))(
                            jnp.asarray(left.T)
                        )
                    ).T
                recovery_solves += len(values)
                balanced_rr = _residuals(a, e, values, right)
                balanced_lr = _residuals(a.conj().T, e.conj().T, values.conj(), left)
                right = col_scale[:, None] * right
                left = row_scale[:, None] * left
                right /= np.linalg.norm(right, axis=0)
                left /= np.linalg.norm(left, axis=0)
                rr = _residuals(original_a, original_e, values, right)
                lr = _residuals(
                    original_a.conj().T, original_e.conj().T, values.conj(), left
                )
                valid = (
                    (rr <= options.tolerance)
                    & (lr <= options.tolerance)
                    & (balanced_rr <= options.tolerance)
                    & (balanced_lr <= options.tolerance)
                )
                best = int(np.sum(valid))
                stable = size == n
                if previous is not None and len(previous) == len(values):
                    ri, ci = linear_sum_assignment(
                        abs(values[:, None] - previous[None, :])
                    )
                    stable = stable or bool(
                        np.all(
                            abs(values[ri] - previous[ci])
                            <= options.tolerance * np.maximum(1, abs(values[ri]))
                        )
                    )
                if len(values) == options.k and np.all(valid) and stable:
                    gaps = abs(values[:, None] - values[None, :])
                    np.fill_diagonal(gaps, np.inf)
                    if np.any(
                        gaps < 100 * np.finfo(float).eps * np.maximum(1, abs(values))
                    ):
                        message = "repeated modes require invariant-subspace participation, not individual factors"
                        raise EigenConvergenceError(message)
                    overlaps = np.sum(left.conj() * (original_e @ right), axis=0)
                    condition_scale = np.linalg.norm(left, axis=0) * np.linalg.norm(
                        original_e @ right, axis=0
                    )
                    if np.any(
                        abs(overlaps) <= 100 * np.finfo(float).eps * condition_scale
                    ):
                        message = "ill-conditioned or defective modes: participation is not identifiable"
                        raise EigenConvergenceError(message)
                    left = left / overlaps.conj()
                    participation = abs(right * (original_e.conj().T @ left).conj())
                    participation /= participation.sum(axis=0)
                    return GeneralizedEigenResult(
                        values,
                        right,
                        left,
                        participation,
                        rr,
                        lr,
                        balanced_rr,
                        balanced_lr,
                        backend,
                        tuple(sizes),
                        2,
                        2 * sum(sizes) + recovery_solves,
                        {
                            "preparation_seconds": before - started,
                            "factorization_seconds": setup,
                            "iteration_seconds": time.perf_counter() - before - setup,
                            "total_seconds": time.perf_counter() - started,
                        },
                    )
            previous = values
            if size == limit:
                message = f"only {best}/{options.k} residual-qualified modes at subspace {size}, right={rr}, left={lr}, balanced_right={balanced_rr}, balanced_left={balanced_lr}, values={values}; increase max_subspace, adjust shift/band, or inspect pencil conditioning"
                raise EigenConvergenceError(message)
            size = min(limit, max(size + 8, int(np.ceil(size * 1.5))))
    finally:
        for factor in factors:
            factor.close()
