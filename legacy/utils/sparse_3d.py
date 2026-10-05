"""
Sparse storage utilities for TxTxN matrices.

This module provides functions to save and load 3D arrays in sparse format,
optimized for the access patterns used in SDF computation:
- BGN: chi[t, :t+1, :] - row slice
- KP14: K[:t+1, t, :], chi[:t+1, t, :], uj[:t+1, t, :] - column slice
"""

import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz


def save_sparse_3d(arr, base_path):
    """
    Save a 3D array as a collection of 2D sparse matrices.

    Slices along axis 0: arr[i, :, :] saved as {i}.npz

    Args:
        arr: numpy array of shape (D0, D1, D2)
        base_path: Directory path to save sparse files

    Returns:
        dict with shape info and sparsity stats
    """
    os.makedirs(base_path, exist_ok=True)

    shape = arr.shape
    n_slices = shape[0]
    nnz_total = 0

    for i in range(n_slices):
        slice_2d = arr[i, :, :]
        sparse_mat = csr_matrix(slice_2d)
        nnz_total += sparse_mat.nnz
        save_npz(os.path.join(base_path, f'{i}.npz'), sparse_mat)

    dense_elements = int(np.prod(shape))
    sparsity = 1.0 - (nnz_total / dense_elements) if dense_elements > 0 else 0.0

    return {
        'shape': shape,
        'n_slices': n_slices,
        'nnz_total': nnz_total,
        'sparsity': sparsity
    }


def load_sparse_3d(base_path, n_slices):
    """
    Load a collection of 2D sparse matrices.

    Args:
        base_path: Directory containing sparse .npz files
        n_slices: Number of slices to load

    Returns:
        List of sparse CSR matrices
    """
    sparse_list = []
    for i in range(n_slices):
        sparse_mat = load_npz(os.path.join(base_path, f'{i}.npz'))
        sparse_list.append(sparse_mat)
    return sparse_list


class Sparse3D:
    """
    Wrapper class for accessing 3D sparse arrays with the specific
    access patterns used in SDF computation.

    Stored as list of 2D CSR matrices along axis 0:
        sparse_list[i] = original_arr[i, :, :]

    Supports:
    - get_row_slice(t, end): returns arr[t, :end, :] as sparse (BGN pattern)
    - get_col_slice(col, end): returns arr[:end, col, :] as sparse (KP14 pattern)
    """

    def __init__(self, sparse_list, shape):
        """
        Args:
            sparse_list: List of 2D CSR matrices
            shape: Original 3D shape tuple (D0, D1, D2)
        """
        self.sparse_list = sparse_list
        self.shape = shape

    def get_row_slice(self, row_idx, col_end=None):
        """
        Get arr[row_idx, :col_end, :] as sparse matrix.

        Used by BGN: chi[t, :t+1, :]

        Args:
            row_idx: Row index (first dimension)
            col_end: Number of columns to include (default: all)

        Returns:
            CSR sparse matrix of shape (col_end, D2)
        """
        mat = self.sparse_list[row_idx]
        if col_end is not None and col_end < mat.shape[0]:
            return mat[:col_end, :]
        return mat

    def get_col_slice(self, col_idx, row_end=None):
        """
        Get arr[:row_end, col_idx, :] as sparse matrix.

        Used by KP14: K[:t+1, t, :], chi[:t+1, t, :], uj[:t+1, t, :]

        Args:
            col_idx: Column index (second dimension)
            row_end: Number of rows to include (default: all)

        Returns:
            CSR sparse matrix of shape (row_end, D2)
        """
        if row_end is None:
            row_end = len(self.sparse_list)

        # Each sparse_list[s] is arr[s, :, :], we need arr[s, col_idx, :]
        rows = []
        for s in range(row_end):
            row = self.sparse_list[s].getrow(col_idx)
            rows.append(row)

        from scipy.sparse import vstack
        return vstack(rows, format='csr')

    def get_col_slice_dense(self, col_idx, row_end=None):
        """
        Get arr[:row_end, col_idx, :] as dense array.

        Useful when operations like **alpha require dense computation.

        Args:
            col_idx: Column index (second dimension)
            row_end: Number of rows to include (default: all)

        Returns:
            Dense numpy array of shape (row_end, D2)
        """
        if row_end is None:
            row_end = len(self.sparse_list)

        D2 = self.shape[2]
        result = np.zeros((row_end, D2))

        for s in range(row_end):
            result[s, :] = self.sparse_list[s].getrow(col_idx).toarray().ravel()

        return result
