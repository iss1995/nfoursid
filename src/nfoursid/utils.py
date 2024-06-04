from typing import Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import sparse
from scipy.sparse.linalg import svds
import torch


class Decomposition:
    def __init__(self, u, s, vh):
        self.u = u
        self.s = s
        self.vh = vh

    def to_array(self):
        self.u, self.s, self.vh = self.u.toarray(), self.s.toarray(), self.vh.toarray()


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
            matrix: sparse.spmatrix,
            k_max: int = 5000
    ) -> Decomposition:
        """
        Calculate eigenvalue decomposition of a sparse ``matrix`` as a ``Decomposition``.
        Using singular value decomposition suitable for sparse matrices.
        """
        u, eigenvalues, vh = svds(matrix, return_singular_vectors=True, k=min(min(matrix.shape)-1, k_max), which='LM')
        # eigenvalues = svds(matrix, return_singular_vectors=False , k=min(matrix.shape)-1, which='LM')
        # u, vh = None, None

        # Sort the eigenvalues in descending order
        idxs = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idxs]
        u, vh = u[:, idxs], vh[idxs, :]
        eigenvalues_mat = sparse.lil_matrix((u.shape[1], vh.shape[0]))
        # eigenvalues_mat = sparse.lil_matrix(matrix.shape)
        for i, val in enumerate(eigenvalues):
            eigenvalues_mat[i, i] = val

        eigenvalues_mat = eigenvalues_mat.tocsr()
        u = None

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
        u, s, vh = decomposition.u, decomposition.s.toarray(), decomposition.vh
        s = sparse.csr_matrix(s[:rank, :rank])
        if u is not None:
            u = u[:, :rank]
        if vh is not None:
            vh = vh[:rank, :]
        return Decomposition(
            u=u,
            s=s,
            vh=vh
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
    def fill_hankel_slice(args):
        matrix, start, end, num_block_rows = args
        hankel_slice = sparse.lil_matrix((num_block_rows * matrix.shape[1], end - start))
        for block_row_index in range(end - start):
            flattened_block_rows = matrix[block_row_index+start:block_row_index+start+num_block_rows, :].flatten()
            hankel_slice[:, block_row_index] = flattened_block_rows.flatten()
        return hankel_slice

    @staticmethod
    def block_hankel_matrix_parallel(matrix: np.ndarray, num_block_rows: int) -> sparse.csr_matrix:
        hankel_cols_dim = matrix.shape[0] - num_block_rows + 1

        # Split the task into slices
        num_slices = cpu_count() - 1
        slice_size = hankel_cols_dim // num_slices
        slices = [(matrix, i*slice_size, (i+1)*slice_size, num_block_rows) for i in range(num_slices)]
        # Make sure the last slice goes to the end
        slices[-1] = (matrix, (num_slices-1)*slice_size, hankel_cols_dim, num_block_rows)

        # Create a multiprocessing Pool and fill each slice in parallel
        with Pool() as p:
            hankel_slices = p.map(Utils.fill_hankel_slice, slices)

        # Concatenate the slices to form the full Hankel matrix
        hankel = sparse.hstack(hankel_slices).tocsr()

        return hankel

    @staticmethod
    def split_and_apply_block_hankel_matrix(matrix: np.ndarray, num_groups: int, num_block_rows: int):

        # Split the matrix into groups
        matrix_data = DataMatrix(matrix, chunks=num_groups, hankel_block_rows=num_block_rows)
        groups = matrix_data.split()
        matrix_data.reset_data(np.asarray([])[None, None])

        # Create a multiprocessing Pool and apply the function to each group in parallel
        with Pool(processes=num_groups) as p:
            results = p.map(matrix_data, groups)

        # Concatenate the results into a single matrix
        results = sparse.hstack(results)

        return results

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

    @staticmethod
    def sparse_qr(matrix: sparse.spmatrix, mode='NATURAL'):
        """
        Perform QR decomposition on a sparse matrix using direct sparse operations.

        Parameters:
        - matrix (sparse.spmatrix): A sparse matrix.

        Returns:
        - Q (sparse.spmatrix): Orthogonal matrix in sparse format.
        - R (sparse.spmatrix): Upper triangular matrix in sparse format.
        """
        if not sparse.isspmatrix_csc(matrix):
            matrix = matrix.tocsc()

        # Direct sparse QR decomposition using SuiteSparseQR, if available
        try:
            from sksparse.cholmod import cholesky
            factorization = cholesky(matrix.T.dot(matrix))
            R = factorization.L().T
            Q = sparse.linalg.spsolve(R, matrix.T, permc_spec=mode).T

        except ImportError:
            raise ImportError("The required library scikit-sparse is not installed. "
                              "Please install it to use this QR decomposition.")

        Q = sparse.csr_matrix(Q)
        R = sparse.csr_matrix(R)

        return Q, R

    @staticmethod
    def qr_torch(matrix: sparse.spmatrix, mode='complete'):
        """
        Perform QR decomposition on a sparse matrix using PyTorch.

        Parameters:
        - matrix (sparse.spmatrix): A sparse matrix.

        Returns:
        - Q (torch.Tensor): Orthogonal matrix.
        - R (torch.Tensor): Upper triangular matrix.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert the sparse matrix to a dense numpy array
        matrix_np = matrix.toarray()

        # Convert the numpy array to a PyTorch tensor
        matrix_torch = torch.from_numpy(matrix_np).to(device)

        # Perform the QR decomposition
        Q, R = torch.linalg.qr(matrix_torch, mode=mode)

        # Convert the PyTorch tensors to sparse matrices
        Q = sparse.csr_matrix(Q.cpu().numpy())
        R = sparse.csr_matrix(R.cpu().numpy())

        return Q, R

    @staticmethod
    def sparse_pseudo_inverse(matrix, k=None):
        """
        Compute the pseudo-inverse of a sparse matrix using sparse SVD.

        Parameters:
        - matrix (scipy.sparse.spmatrix): Sparse matrix whose pseudo-inverse is needed.
        - k (int): Number of singular values and vectors to compute. If None, use a heuristic based on matrix size.

        Returns:
        - pinv (scipy.sparse.spmatrix): Pseudo-inverse of the input matrix.
        """
        # Determine k if not provided
        if k is None:
            k = min(matrix.shape) - 1

        # Compute the truncated SVD
        u, s, vt = svds(matrix, k=k)

        # Invert the non-zero singular values
        s_inv = 1.0 / s

        # Compute the pseudo-inverse
        pinv = vt.T @ sparse.diags(s_inv) @ u.T

        return sparse.csr_matrix(pinv)


class DataMatrix:

    def __init__(self, data: np.ndarray, chunks=1, hankel_block_rows=1):
        self.data = data
        self.chunks = chunks
        self.num_rows = data.shape[0]
        self.num_cols = data.shape[1]
        self.hankel_block_rows = hankel_block_rows

    def __call__(self, data) -> sparse.csr_matrix:
        return Utils.block_hankel_matrix(data, self.hankel_block_rows)

    def __len__(self):
        return self.num_rows

    def split(self):
        chunk_size = self.num_rows // self.chunks
        return [self.data[i*chunk_size:(i+1)*chunk_size, :] for i in range(self.chunks)]

    def reset_data(self, data):
        self.data = data
        self.num_rows = data.shape[0]
        self.num_cols = data.shape[1]
