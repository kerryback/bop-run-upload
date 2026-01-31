"""
Integration test: Compare sparse vs dense storage for full SDF computation.

This test:
1. Creates small simulated arrays mimicking BGN/KP14 structure
2. Saves them as both dense (v1) and sparse (v2) format
3. Runs the SDF compute logic with both
4. Verifies numerical equivalence
"""

import sys
import os
import numpy as np
import pickle
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sparse_3d import save_sparse_3d, load_sparse_3d, Sparse3D
from scipy.sparse import csr_matrix, kron


def create_mock_bgn_arrays(T, N, seed=42):
    """Create mock BGN-style arrays for testing."""
    np.random.seed(seed)

    # chi: (T, T, N) - lower triangular with sparsity
    chi = np.zeros((T, T, N))
    for t in range(T):
        for s in range(t + 1):
            # Projects arrive, some die
            arrive = np.random.random(N) < 0.3
            survive = np.random.random(N) < 0.95
            chi[t, s, arrive & survive] = 1.0

    # Other arrays (2D)
    r = np.random.random(T + 1) * 0.05 + 0.02
    sigmaj = np.random.random((T, N)) * 0.2 + 0.1
    beta = np.random.random((T, N)) * 0.1
    corr_zj = np.random.random((T, N)) * 0.5
    corr_zr = 0.3
    exp_beta = np.exp(-beta)
    exp_corr_zr = np.exp(-0.5 * sigmaj**2 * corr_zj**2 * corr_zr**2)

    return {
        'chi': chi,
        'r': r,
        'sigmaj': sigmaj,
        'beta': beta,
        'corr_zj': corr_zj,
        'corr_zr': corr_zr,
        'exp_beta': exp_beta,
        'exp_corr_zr': exp_corr_zr,
    }


def create_mock_kp14_arrays(T, N, seed=43):
    """Create mock KP14-style arrays for testing."""
    np.random.seed(seed)

    # chi, K, uj: (T, T, N) - projects that exist at time t from arrival s
    chi = np.zeros((T, T, N))
    K = np.zeros((T, T, N))
    uj = np.zeros((T, T, N))

    for s in range(T):
        # Projects arrive at time s
        arrive = np.random.random(N) < 0.2

        for t in range(s, T):
            # Some die each period
            if t == s:
                alive = arrive.copy()
            else:
                alive = alive & (np.random.random(N) < 0.98)

            chi[s, t, alive] = 1.0
            K[s, t, alive] = np.random.random(alive.sum()) * 100
            uj[s, t, alive] = np.random.random(alive.sum()) * 0.5 + 0.5

    # Other arrays
    EtA = np.random.random((T, N)) + 0.5
    eps = np.random.random((T, N)) * 0.2 + 0.9

    return {
        'chi': chi,
        'K': K,
        'uj': uj,
        'EtA': EtA,
        'eps': eps,
        'alpha': 0.33,
    }


def bgn_sdf_step(t, chi, exp_beta, sigmaj, corr_zj, corr_zr, exp_corr_zr, N):
    """
    Simulate one step of BGN sdf_loop.
    Returns intermediate results for comparison.
    """
    # Handle both Sparse3D and dense
    if hasattr(chi, 'get_row_slice'):
        chisp = chi.get_row_slice(t, t + 1)
        chi_slice = chisp.toarray()
    else:
        chisp = csr_matrix(chi[t, :t+1, :])
        chi_slice = chi[t, :t+1, :]

    col1 = chisp.multiply(exp_beta[:t+1, :])
    col2 = chisp.multiply(sigmaj[:t+1, :] * corr_zj[:t+1, :])
    col3 = chisp.multiply(exp_corr_zr[:t+1, :])

    # Kron operation
    result3 = kron(col1, col1)
    term3_sum = result3.sum(axis=0).A.reshape(N, N)

    # Diagonal correction term
    diag3 = (chi_slice * exp_beta[:t+1, :] ** 2).sum(axis=0)

    return {
        'col1_sum': np.array(col1.sum(axis=0)).flatten(),
        'col2_sum': np.array(col2.sum(axis=0)).flatten(),
        'col3_sum': np.array(col3.sum(axis=0)).flatten(),
        'term3_sum': term3_sum,
        'diag3': diag3,
    }


def kp14_sdf_step(t, K, chi, uj, EtA, alpha, N):
    """
    Simulate one step of KP14 sdf_loop.
    Returns intermediate results for comparison.
    """
    # Handle both Sparse3D and dense
    if hasattr(K, 'get_col_slice_dense'):
        K_slice = K.get_col_slice_dense(t, t + 1)
        chi_slice = chi.get_col_slice_dense(t, t + 1)
        uj_slice = uj.get_col_slice_dense(t, t + 1)
    else:
        K_slice = K[:t+1, t, :]
        chi_slice = chi[:t+1, t, :]
        uj_slice = uj[:t+1, t, :]

    Ktalpha = K_slice ** alpha
    Ktalpha_sp = csr_matrix(Ktalpha)
    col = Ktalpha_sp.multiply(csr_matrix(EtA[:t+1, :]))

    result = kron(col, col)
    term1_sum = result.sum(axis=0).A.reshape(N, N)

    part2 = np.sum(chi_slice * EtA[:t+1, :] * Ktalpha, axis=0)

    # uj-based calculation
    uj_sum = np.sum(uj_slice * Ktalpha, axis=0)

    return {
        'Ktalpha_sum': Ktalpha.sum(axis=0),
        'col_sum': np.array(col.sum(axis=0)).flatten(),
        'term1_sum': term1_sum,
        'part2': part2,
        'uj_sum': uj_sum,
    }


def test_bgn_dense_vs_sparse():
    """Test BGN: dense storage vs sparse storage produce identical results."""
    print("Test BGN: Dense vs Sparse storage...")

    T, N = 30, 50
    arrays = create_mock_bgn_arrays(T, N)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save chi as sparse
        sparse_dir = os.path.join(tmpdir, 'chi_sparse')
        meta = save_sparse_3d(arrays['chi'], sparse_dir)
        sparse_list = load_sparse_3d(sparse_dir, meta['n_slices'])
        chi_sparse = Sparse3D(sparse_list, meta['shape'])

        print(f"  chi sparsity: {meta['sparsity']:.1%}")

        max_diff = 0.0
        for t in [5, 10, 20, T-2]:
            # Dense path
            result_dense = bgn_sdf_step(
                t, arrays['chi'],
                arrays['exp_beta'], arrays['sigmaj'],
                arrays['corr_zj'], arrays['corr_zr'],
                arrays['exp_corr_zr'], N
            )

            # Sparse path
            result_sparse = bgn_sdf_step(
                t, chi_sparse,
                arrays['exp_beta'], arrays['sigmaj'],
                arrays['corr_zj'], arrays['corr_zr'],
                arrays['exp_corr_zr'], N
            )

            # Compare all outputs
            for key in result_dense:
                diff = np.abs(result_dense[key] - result_sparse[key]).max()
                max_diff = max(max_diff, diff)

        assert max_diff < 1e-10, f"BGN mismatch: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff:.2e})")


def test_kp14_dense_vs_sparse():
    """Test KP14: dense storage vs sparse storage produce identical results."""
    print("Test KP14: Dense vs Sparse storage...")

    T, N = 30, 50
    arrays = create_mock_kp14_arrays(T, N)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save K, chi, uj as sparse
        K_meta = save_sparse_3d(arrays['K'], os.path.join(tmpdir, 'K'))
        chi_meta = save_sparse_3d(arrays['chi'], os.path.join(tmpdir, 'chi'))
        uj_meta = save_sparse_3d(arrays['uj'], os.path.join(tmpdir, 'uj'))

        K_sparse = Sparse3D(
            load_sparse_3d(os.path.join(tmpdir, 'K'), K_meta['n_slices']),
            K_meta['shape']
        )
        chi_sparse = Sparse3D(
            load_sparse_3d(os.path.join(tmpdir, 'chi'), chi_meta['n_slices']),
            chi_meta['shape']
        )
        uj_sparse = Sparse3D(
            load_sparse_3d(os.path.join(tmpdir, 'uj'), uj_meta['n_slices']),
            uj_meta['shape']
        )

        print(f"  K sparsity: {K_meta['sparsity']:.1%}")
        print(f"  chi sparsity: {chi_meta['sparsity']:.1%}")
        print(f"  uj sparsity: {uj_meta['sparsity']:.1%}")

        max_diff = 0.0
        for t in [5, 10, 20, T-2]:
            # Dense path
            result_dense = kp14_sdf_step(
                t, arrays['K'], arrays['chi'], arrays['uj'],
                arrays['EtA'], arrays['alpha'], N
            )

            # Sparse path
            result_sparse = kp14_sdf_step(
                t, K_sparse, chi_sparse, uj_sparse,
                arrays['EtA'], arrays['alpha'], N
            )

            # Compare all outputs
            for key in result_dense:
                diff = np.abs(result_dense[key] - result_sparse[key]).max()
                max_diff = max(max_diff, diff)

        assert max_diff < 1e-10, f"KP14 mismatch: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff:.2e})")


def test_metadata_versioning():
    """Test that metadata correctly tracks version and sparse info."""
    print("Test metadata versioning...")

    T, N = 10, 20
    np.random.seed(48)
    arr = np.random.random((T, T, N))
    arr[arr < 0.8] = 0  # Make 80% sparse

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = os.path.join(tmpdir, 'test')
        meta = save_sparse_3d(arr, sparse_dir)

        # Simulate metadata.pkl structure
        metadata = {
            'n_arrays': 5,
            'N': N,
            'T': T,
            'model': 'test',
            'burnin': 100,
            'sparse_info': {
                0: {'is_sparse': True, 'name': 'chi', **meta},
                1: {'is_sparse': False},
            },
            'version': 2
        }

        # Verify structure
        assert metadata['version'] == 2
        assert metadata['sparse_info'][0]['is_sparse'] == True
        assert metadata['sparse_info'][0]['sparsity'] > 0.7
        assert metadata['sparse_info'][1]['is_sparse'] == False

        print("  PASSED")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("SPARSE 3D INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_bgn_dense_vs_sparse,
        test_kp14_dense_vs_sparse,
        test_metadata_versioning,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
