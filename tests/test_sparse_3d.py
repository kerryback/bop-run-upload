"""
Tests for sparse 3D matrix storage and usage.

Verifies that:
1. save_sparse_3d/load_sparse_3d roundtrip preserves data
2. Sparse3D slicing matches dense numpy slicing
3. SDF compute functions produce identical results with dense vs sparse input
"""

import sys
import os
import numpy as np
import tempfile
import shutil
from scipy.sparse import csr_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sparse_3d import save_sparse_3d, load_sparse_3d, Sparse3D


def test_roundtrip_basic():
    """Test that save/load roundtrip preserves data exactly."""
    print("Test 1: Basic roundtrip...")

    # Create test array with known sparsity pattern
    T, N = 50, 100
    arr = np.zeros((T, T, N))

    # Fill with sparse pattern (like chi - lower triangular with random deaths)
    np.random.seed(42)
    for t in range(T):
        for s in range(t + 1):
            # ~10% of entries are non-zero
            mask = np.random.random(N) < 0.1
            arr[t, s, mask] = np.random.random(mask.sum())

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = os.path.join(tmpdir, 'test_sparse')
        meta = save_sparse_3d(arr, sparse_dir)

        print(f"  Shape: {meta['shape']}, Sparsity: {meta['sparsity']:.1%}")

        sparse_list = load_sparse_3d(sparse_dir, meta['n_slices'])
        sparse3d = Sparse3D(sparse_list, meta['shape'])

        # Verify each slice matches
        max_diff = 0.0
        for t in range(T):
            reconstructed = sparse3d.sparse_list[t].toarray()
            original = arr[t, :, :]
            diff = np.abs(reconstructed - original).max()
            max_diff = max(max_diff, diff)

        assert max_diff < 1e-10, f"Roundtrip error: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff})")


def test_get_row_slice():
    """Test get_row_slice matches dense indexing (BGN pattern)."""
    print("Test 2: get_row_slice (BGN pattern: chi[t, :t+1, :])...")

    T, N = 50, 100
    np.random.seed(43)

    # Create sparse array
    arr = np.zeros((T, T, N))
    for t in range(T):
        for s in range(t + 1):
            mask = np.random.random(N) < 0.15
            arr[t, s, mask] = np.random.random(mask.sum())

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = os.path.join(tmpdir, 'test_sparse')
        meta = save_sparse_3d(arr, sparse_dir)
        sparse_list = load_sparse_3d(sparse_dir, meta['n_slices'])
        sparse3d = Sparse3D(sparse_list, meta['shape'])

        # Test get_row_slice for various t values
        max_diff = 0.0
        for t in [0, 10, 25, T-1]:
            # Dense slice
            dense_slice = arr[t, :t+1, :]

            # Sparse slice
            sparse_slice = sparse3d.get_row_slice(t, t + 1)
            reconstructed = sparse_slice.toarray()

            diff = np.abs(reconstructed - dense_slice).max()
            max_diff = max(max_diff, diff)

            # Also verify it's a CSR matrix
            assert isinstance(sparse_slice, csr_matrix), "get_row_slice should return CSR"

        assert max_diff < 1e-10, f"get_row_slice error: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff})")


def test_get_col_slice_dense():
    """Test get_col_slice_dense matches dense indexing (KP14 pattern)."""
    print("Test 3: get_col_slice_dense (KP14 pattern: K[:t+1, t, :])...")

    T, N = 50, 100
    np.random.seed(44)

    # Create sparse array (like K in KP14)
    arr = np.zeros((T, T, N))
    for s in range(T):
        for t in range(s, T):  # Upper triangular for column access
            mask = np.random.random(N) < 0.12
            arr[s, t, mask] = np.random.random(mask.sum()) * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = os.path.join(tmpdir, 'test_sparse')
        meta = save_sparse_3d(arr, sparse_dir)
        sparse_list = load_sparse_3d(sparse_dir, meta['n_slices'])
        sparse3d = Sparse3D(sparse_list, meta['shape'])

        # Test get_col_slice_dense for various t values
        max_diff = 0.0
        for t in [1, 10, 25, T-1]:
            # Dense slice: arr[:t+1, t, :]
            dense_slice = arr[:t+1, t, :]

            # Sparse3D slice
            reconstructed = sparse3d.get_col_slice_dense(t, t + 1)

            diff = np.abs(reconstructed - dense_slice).max()
            max_diff = max(max_diff, diff)

            # Verify shape
            assert reconstructed.shape == dense_slice.shape, \
                f"Shape mismatch: {reconstructed.shape} vs {dense_slice.shape}"

        assert max_diff < 1e-10, f"get_col_slice_dense error: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff})")


def test_bgn_sdf_operations():
    """Test BGN-style sparse operations produce same results."""
    print("Test 4: BGN SDF sparse operations...")

    T, N = 30, 50
    np.random.seed(45)

    # Create chi-like array (lower triangular, sparse)
    chi_dense = np.zeros((T, T, N))
    for t in range(T):
        for s in range(t + 1):
            mask = np.random.random(N) < 0.2
            chi_dense[t, s, mask] = 1.0

    # Create other arrays
    exp_beta = np.random.random((T, N)) * 0.5 + 0.5
    sigmaj = np.random.random((T, N)) * 0.1
    corr_zj = np.random.random((T, N))

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = os.path.join(tmpdir, 'chi_sparse')
        meta = save_sparse_3d(chi_dense, sparse_dir)
        sparse_list = load_sparse_3d(sparse_dir, meta['n_slices'])
        chi_sparse = Sparse3D(sparse_list, meta['shape'])

        max_diff = 0.0
        for t in [5, 15, T-2]:
            # --- Dense path (old code) ---
            chisp_dense = csr_matrix(chi_dense[t, :t+1, :])
            col1_dense = chisp_dense.multiply(exp_beta[:t+1, :])
            col2_dense = chisp_dense.multiply(sigmaj[:t+1, :] * corr_zj[:t+1, :])

            # --- Sparse3D path (new code) ---
            chisp_sparse = chi_sparse.get_row_slice(t, t + 1)
            col1_sparse = chisp_sparse.multiply(exp_beta[:t+1, :])
            col2_sparse = chisp_sparse.multiply(sigmaj[:t+1, :] * corr_zj[:t+1, :])

            # Compare results
            diff1 = np.abs(col1_dense.toarray() - col1_sparse.toarray()).max()
            diff2 = np.abs(col2_dense.toarray() - col2_sparse.toarray()).max()
            max_diff = max(max_diff, diff1, diff2)

            # Test kron operation (expensive but important)
            from scipy.sparse import kron
            kron_dense = kron(col1_dense, col1_dense)
            kron_sparse = kron(col1_sparse, col1_sparse)

            # Compare sum (main usage)
            sum_dense = kron_dense.sum(axis=0).A.reshape(N, N)
            sum_sparse = kron_sparse.sum(axis=0).A.reshape(N, N)
            diff_kron = np.abs(sum_dense - sum_sparse).max()
            max_diff = max(max_diff, diff_kron)

        assert max_diff < 1e-10, f"BGN operations error: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff})")


def test_kp14_sdf_operations():
    """Test KP14-style sparse operations produce same results."""
    print("Test 5: KP14 SDF sparse operations...")

    T, N = 30, 50
    alpha = 0.33
    np.random.seed(46)

    # Create K-like array
    K_dense = np.zeros((T, T, N))
    chi_dense = np.zeros((T, T, N))
    uj_dense = np.zeros((T, T, N))

    for s in range(T):
        for t in range(s, T):
            mask = np.random.random(N) < 0.15
            K_dense[s, t, mask] = np.random.random(mask.sum()) * 100
            chi_dense[s, t, mask] = 1.0
            uj_dense[s, t, mask] = np.random.random(mask.sum())

    EtA = np.random.random((T, N)) + 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save all sparse arrays
        K_meta = save_sparse_3d(K_dense, os.path.join(tmpdir, 'K'))
        chi_meta = save_sparse_3d(chi_dense, os.path.join(tmpdir, 'chi'))
        uj_meta = save_sparse_3d(uj_dense, os.path.join(tmpdir, 'uj'))

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

        max_diff = 0.0
        for t in [5, 15, T-2]:
            # --- Dense path (old code) ---
            K_slice_dense = K_dense[:t+1, t, :]
            chi_slice_dense = chi_dense[:t+1, t, :]
            uj_slice_dense = uj_dense[:t+1, t, :]

            Ktalpha_dense = K_slice_dense ** alpha
            Ktalpha_sp_dense = csr_matrix(Ktalpha_dense)
            col_dense = Ktalpha_sp_dense.multiply(csr_matrix(EtA[:t+1, :]))

            part2_dense = np.sum(chi_slice_dense * EtA[:t+1, :] * Ktalpha_dense, axis=0)

            # --- Sparse3D path (new code) ---
            K_slice_sparse = K_sparse.get_col_slice_dense(t, t + 1)
            chi_slice_sparse = chi_sparse.get_col_slice_dense(t, t + 1)
            uj_slice_sparse = uj_sparse.get_col_slice_dense(t, t + 1)

            Ktalpha_sparse = K_slice_sparse ** alpha
            Ktalpha_sp_sparse = csr_matrix(Ktalpha_sparse)
            col_sparse = Ktalpha_sp_sparse.multiply(csr_matrix(EtA[:t+1, :]))

            part2_sparse = np.sum(chi_slice_sparse * EtA[:t+1, :] * Ktalpha_sparse, axis=0)

            # Compare
            diff_K = np.abs(K_slice_dense - K_slice_sparse).max()
            diff_chi = np.abs(chi_slice_dense - chi_slice_sparse).max()
            diff_uj = np.abs(uj_slice_dense - uj_slice_sparse).max()
            diff_Ktalpha = np.abs(Ktalpha_dense - Ktalpha_sparse).max()
            diff_col = np.abs(col_dense.toarray() - col_sparse.toarray()).max()
            diff_part2 = np.abs(part2_dense - part2_sparse).max()

            max_diff = max(max_diff, diff_K, diff_chi, diff_uj,
                          diff_Ktalpha, diff_col, diff_part2)

        assert max_diff < 1e-10, f"KP14 operations error: max diff = {max_diff}"
        print(f"  PASSED (max diff = {max_diff})")


def test_sparsity_calculation():
    """Test that sparsity is calculated correctly."""
    print("Test 6: Sparsity calculation...")

    T, N = 20, 30

    # Create array with known sparsity
    arr = np.zeros((T, T, N))
    # Fill exactly 10% of entries
    total_elements = T * T * N
    n_nonzero = total_elements // 10

    np.random.seed(47)
    flat_indices = np.random.choice(total_elements, n_nonzero, replace=False)
    arr.flat[flat_indices] = np.random.random(n_nonzero)

    expected_sparsity = 1.0 - (n_nonzero / total_elements)

    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_dir = os.path.join(tmpdir, 'test')
        meta = save_sparse_3d(arr, sparse_dir)

        # Allow small tolerance due to rounding
        assert abs(meta['sparsity'] - expected_sparsity) < 0.01, \
            f"Sparsity mismatch: {meta['sparsity']:.3f} vs expected {expected_sparsity:.3f}"

        print(f"  PASSED (sparsity = {meta['sparsity']:.1%}, expected ~{expected_sparsity:.1%})")


def test_edge_cases():
    """Test edge cases: empty arrays, all zeros, single element."""
    print("Test 7: Edge cases...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # All zeros
        arr_zeros = np.zeros((10, 10, 5))
        meta = save_sparse_3d(arr_zeros, os.path.join(tmpdir, 'zeros'))
        assert meta['sparsity'] == 1.0, "All zeros should be 100% sparse"
        assert meta['nnz_total'] == 0, "All zeros should have 0 nnz"

        # Single non-zero
        arr_one = np.zeros((10, 10, 5))
        arr_one[5, 3, 2] = 1.0
        meta = save_sparse_3d(arr_one, os.path.join(tmpdir, 'one'))
        assert meta['nnz_total'] == 1, "Should have exactly 1 nnz"

        # Load and verify
        sparse_list = load_sparse_3d(os.path.join(tmpdir, 'one'), meta['n_slices'])
        sparse3d = Sparse3D(sparse_list, meta['shape'])

        # Check the value is preserved
        slice_5 = sparse3d.get_row_slice(5, 10).toarray()
        assert slice_5[3, 2] == 1.0, "Value not preserved"

        print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SPARSE 3D MATRIX TESTS")
    print("=" * 60)

    tests = [
        test_roundtrip_basic,
        test_get_row_slice,
        test_get_col_slice_dense,
        test_bgn_sdf_operations,
        test_kp14_sdf_operations,
        test_sparsity_calculation,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
