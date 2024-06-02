from typing import Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


class Decomposition:
    def __init__(self, u, s, vh):
        self.u = u
        self.s = s
        self.vh = vh


class Utils:
    @staticmethod
    def validate_matrix_shape(
            matrix: sparse.spmatrix,
            shape: Tuple[int, int],
            name: str
    ):
        """
        Raises if ``matrix`` does not have shape ``shape``. The error message will contain ``name``.
        """
        if matrix.shape != shape:
            raise ValueError(f'Dimensions of `{name}` {matrix.shape} are inconsistent. Expected {shape}.')

    @staticmethod
    def eigenvalue_decomposition(
            matrix: sparse.spmatrix
    ) -> Decomposition:
        """
        Calculate eigenvalue decomposition of a sparse ``matrix`` as a ``Decomposition``.
        Using singular value decomposition suitable for sparse matrices.
        """
        u, eigenvalues, vh = svds(matrix, k=min(matrix.shape)-1)  # k is the number of singular values to compute
        eigenvalues_mat = sparse.diags(eigenvalues)
        return Decomposition(u, eigenvalues_mat, vh)

    @staticmethod
    def reduce_decomposition(
            decomposition: Decomposition,
            rank: int
    ) -> Decomposition:
        """
        Reduce an eigenvalue decomposition ``decomposition`` such that only ``rank`` number of biggest eigenvalues
        remain. Returns another ``Decomposition`` adapted for sparse matrix operations.
        """
        u, s, vh = decomposition.u, decomposition.s, decomposition.vh
        return Decomposition(
            u[:, :rank],
            s[:rank, :rank],
            vh[:rank, :]
        )
    @staticmethod
    def block_hankel_matrix(
            matrix: np.ndarray,
            num_block_rows: int
    ) -> sparse.csr_matrix:
        """
        Calculate a block Hankel matrix based on input matrix ``matrix`` with ``num_block_rows`` block rows,
        using sparse matrix operations to handle large data efficiently. The shape of ``matrix`` is interpreted
        in row-order, like the structure of a ``pd.DataFrame``: the rows are measurements and the columns are data sources.
        The returned block Hankel matrix has a columnar structure where each column of the returned matrix consists
        of ``num_block_rows`` block rows (measurements).
        """
        hankel_rows_dim = num_block_rows * matrix.shape[1]
        hankel_cols_dim = matrix.shape[0] - num_block_rows + 1
        hankel = sparse.lil_matrix((hankel_rows_dim, hankel_cols_dim))

        for block_row_index in range(hankel_cols_dim):
            if block_row_index % 100 == 0:
                print(f'Processing block row {block_row_index} of {hankel_cols_dim}', end='\r')
            flattened_block_rows = matrix[block_row_index:block_row_index+num_block_rows, :].flatten()
            hankel[:, block_row_index] = flattened_block_rows.flatten()

        return hankel.tocsr()

    @staticmethod
    def vectorize(matrix: np.ndarray) -> np.ndarray:
        """
        Given a matrix ``matrix`` of shape ``(a, b)``, return a vector of shape ``(a*b, 1)`` with all columns of
        ``matrix`` stacked on top of each other.
        """
        return matrix.flatten(order='F').reshape(-1, 1)

    @staticmethod
    def unvectorize(vector: np.ndarray, num_rows: int) -> np.ndarray:
        """
        Given a vector ``vector`` of shape ``(num_rows*b, 1)``, return a matrix of shape ``(num_rows, b)`` such that
        the stacked columns of the returned matrix equal ``vector``.
        """
        if vector.shape[0] % num_rows != 0 or vector.shape[1] != 1:
            raise ValueError(f'Vector shape {vector.shape} and `num_rows`={num_rows} are incompatible')
        return vector.reshape((num_rows, -1), order='F')


    @staticmethod
    def vectorize(
            matrix: np.ndarray
    ) -> sparse.csr_matrix:
        """
        Given a matrix ``matrix`` of shape ``(a, b)``, return a vector of shape ``(a*b, 1)`` with all columns of
        ``matrix`` stacked on top of each other using sparse matrix operations.
        """
        return sparse.csr_matrix(matrix.flatten(order='F').reshape(-1, 1))

    @staticmethod
    def unvectorize(
            vector: sparse.csr_matrix,
            num_rows: int
    ) -> sparse.csr_matrix:
        """
        Given a vector ``vector`` of shape ``(num_rows*b, 1)``, return a matrix of shape ``(num_rows, b)`` such that
        the stacked columns of the returned matrix equal ``vector``, using sparse matrix operations.
        """
        if vector.shape[0] % num_rows != 0 or vector.shape[1] != 1:
            raise ValueError(f'Vector shape {vector.shape} and `num_rows`={num_rows} are incompatible')
        b = vector.shape[0] // num_rows
        return sparse.csr_matrix(vector.toarray().reshape((num_rows, b), order='F'))