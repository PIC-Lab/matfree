"""Generalized eigenpairs against independent LAPACK reference."""

import jax
import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment

from matfree.generalized import EigenConvergenceError, EigenOptions, eigs

jax.config.update("jax_enable_x64", True)


def pencil():
    rng = np.random.default_rng(14)
    blocks = [
        np.array([[-0.1 - i * 0.03, -(i + 1) * 2.0], [(i + 1) * 2.0, -0.1 - i * 0.03]])
        for i in range(15)
    ]
    f = linalg.block_diag(*blocks)
    q, _ = np.linalg.qr(rng.normal(size=f.shape))
    f = q @ f @ q.T
    gx = rng.normal(size=(6, 30)) * 0.03
    fz = rng.normal(size=(30, 6)) * 0.04
    a = np.block([[f + fz @ gx, fz], [gx, np.eye(6)]])
    e = np.diag(np.r_[np.ones(30), np.zeros(6)])
    return sparse.csr_matrix(a), sparse.csr_matrix(e)


def compare(backend):
    a, e = pencil()
    opts = EigenOptions(
        k=4, shift=-0.2 + 5j, backend=backend, initial_subspace=12, max_subspace=36
    )
    result = eigs(a, e, options=opts)
    val, left, right = linalg.eig(a.toarray(), e.toarray(), left=True, right=True)
    valid = np.isfinite(val)
    val, left, right = val[valid], left[:, valid], right[:, valid]
    ix = np.argsort(abs(val - opts.shift))[: opts.k]
    val, left, right = val[ix], left[:, ix], right[:, ix]
    _, order = linear_sum_assignment(abs(result.eigenvalues[:, None] - val[None, :]))
    np.testing.assert_allclose(result.eigenvalues, val[order], atol=1e-7)
    p = abs(right * (e.T @ left).conj())
    p /= p.sum(axis=0)
    np.testing.assert_allclose(result.participation, p[:, order], atol=1e-6)
    np.testing.assert_allclose(
        np.diag(result.left_eigenvectors.conj().T @ e @ result.right_eigenvectors),
        1,
        atol=1e-10,
    )
    assert np.max(result.right_residuals) < opts.tolerance
    assert np.max(result.left_residuals) < opts.tolerance
    assert result.factorization_count == 2
    assert len(result.subspace_sizes) > 1
    assert np.all(result.participation[30:] == 0)
    return result


def test_scipy_descriptor_complex_shift():
    compare("scipy")


@pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="GPU required"
)
def test_cudss_descriptor_complex_shift():
    compare("cudss")


def test_failure_is_explicit():
    a, e = pencil()
    with pytest.raises(EigenConvergenceError):
        eigs(
            a,
            e,
            options=EigenOptions(
                k=4,
                backend="scipy",
                initial_subspace=6,
                max_subspace=6,
                tolerance=1e-12,
            ),
        )


def test_band_and_positive_selection():
    a, e = pencil()
    r = eigs(
        a,
        e,
        options=EigenOptions(
            k=2,
            shift=-0.2 + 5j,
            backend="scipy",
            initial_subspace=20,
            max_subspace=36,
            minimum_frequency_hz=0.5,
            maximum_frequency_hz=1.1,
            positive_only=True,
        ),
    )
    np.testing.assert_allclose(np.sort(r.eigenvalues.imag), [4, 6], atol=1e-7)


def test_singular_shift():
    with pytest.raises(ValueError, match="singular"):
        eigs(np.diag([0.0, 1.0, 2.0]), options=EigenOptions(k=1, backend="scipy"))


def test_small_full_space():
    r = eigs(
        np.array([[-1.0, -2.0], [2.0, -1.0]]),
        options=EigenOptions(k=2, backend="scipy"),
    )
    np.testing.assert_allclose(np.sort(r.eigenvalues.imag), [-2, 2], atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"tolerance": 0},
        {"shift": float("nan")},
        {"backend": "dense"},
        {"initial_subspace": 30, "max_subspace": 20},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError, match=r".+"):
        EigenOptions(**kwargs)


def test_no_modes_in_band_fails_cleanly():
    with pytest.raises(EigenConvergenceError):
        eigs(
            np.diag([-1.0, -2.0, -3.0]),
            options=EigenOptions(k=1, backend="scipy", positive_only=True),
        )


def test_empty_pencil_row_is_singular():
    with pytest.raises(ValueError, match="singular"):
        eigs(
            np.diag([-1.0, 0.0]),
            np.diag([1.0, 0.0]),
            options=EigenOptions(k=1, backend="scipy"),
        )


def test_ill_scaled_descriptor_matches_known_finite_modes():
    a, e = pencil()
    rows = np.geomspace(1e-7, 1e7, a.shape[0])
    cols = np.geomspace(1e6, 1e-6, a.shape[0])
    aa = sparse.diags(rows) @ a @ sparse.diags(cols)
    ee = sparse.diags(rows) @ e @ sparse.diags(cols)
    opts = EigenOptions(
        k=4, shift=-0.2 + 5j, backend="scipy", initial_subspace=12, max_subspace=36
    )
    ref = eigs(a, e, options=opts)
    result = eigs(aa, ee, options=opts)
    _, order = linear_sum_assignment(
        abs(result.eigenvalues[:, None] - ref.eigenvalues[None, :])
    )
    np.testing.assert_allclose(result.eigenvalues, ref.eigenvalues[order], atol=1e-7)
    np.testing.assert_allclose(
        result.participation, ref.participation[:, order], atol=1e-6
    )


def test_complex_general_mass_matrix_and_adjoint_modes():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    e = np.eye(12) + 0.05 * (rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    opts = EigenOptions(
        k=3, shift=-0.4 + 2j, backend="scipy", initial_subspace=8, max_subspace=12
    )
    result = eigs(a, e, options=opts)
    val, left, right = linalg.eig(a, e, left=True, right=True)
    ix = np.argsort(abs(val - opts.shift))[: opts.k]
    _, order = linear_sum_assignment(
        abs(result.eigenvalues[:, None] - val[ix][None, :])
    )
    np.testing.assert_allclose(result.eigenvalues, val[ix][order], atol=1e-10)
    p = abs(right * (e.conj().T @ left).conj())
    p /= p.sum(axis=0)
    np.testing.assert_allclose(result.participation, p[:, ix][:, order], atol=1e-9)
    np.testing.assert_allclose(
        result.left_eigenvectors.conj().T @ e @ result.right_eigenvectors,
        np.eye(opts.k),
        atol=1e-9,
    )


def test_factors_release_on_convergence_failure(monkeypatch):
    from matfree import generalized

    original = generalized._Factor.close
    closed = []

    def close(self):
        closed.append(self)
        original(self)

    monkeypatch.setattr(generalized._Factor, "close", close)
    a, e = pencil()
    with pytest.raises(EigenConvergenceError):
        eigs(
            a,
            e,
            options=EigenOptions(
                k=4, backend="scipy", initial_subspace=6, max_subspace=6
            ),
        )
    assert len(closed) == 2
