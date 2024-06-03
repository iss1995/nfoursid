from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from .state_space import StateSpace
from .utils import Utils, Decomposition


class NFourSID:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            output_columns: List[str],
            input_columns: List[str] = None,
            num_block_rows: int = 2
    ):
        self.u_columns = input_columns or []
        self.y_columns = output_columns
        self.num_block_rows = num_block_rows

        self._set_input_output_data(dataframe)
        self._initialize_instance_variables()

    def _initialize_instance_variables(self):
        self.R22, self.R32 = None, None
        self.R32_decomposition = None
        self.x_dim = None

    def _set_input_output_data(self, dataframe: pd.DataFrame):
        u_frame = dataframe[self.u_columns]
        if u_frame.isnull().any().any():
            raise ValueError('Input data cannot contain nulls')
        y_frame = dataframe[self.y_columns]
        if y_frame.isnull().any().any():
            raise ValueError('Output data cannot contain nulls')
        self.u_array = u_frame.to_numpy()
        self.y_array = y_frame.to_numpy()
        self.u_dim = self.u_array.shape[1]
        self.y_dim = self.y_array.shape[1]

    def subspace_identification(self):
        u_hankel = Utils.block_hankel_matrix_parallel(self.u_array, self.num_block_rows).tocsr()
        y_hankel = Utils.block_hankel_matrix_parallel(self.y_array, self.num_block_rows).tocsr()

        u_past, u_future = u_hankel[:, :-self.num_block_rows], u_hankel[:, self.num_block_rows:]
        y_past, y_future = y_hankel[:, :-self.num_block_rows], y_hankel[:, self.num_block_rows:]
        u_instrumental_y = sparse.vstack([u_future, u_past, y_past, y_future])

        # q, r = map(lambda matrix: matrix.T, np.linalg.qr(u_instrumental_y.toarray().T, mode='reduced'))
        q, r = map(lambda matrix: matrix.T, Utils.sparse_qr(u_instrumental_y.T))

        y_rows, u_rows = self.y_dim * self.num_block_rows, self.u_dim * self.num_block_rows
        self.R32 = r[-y_rows:, u_rows:-y_rows]
        self.R22 = r[u_rows:-y_rows, u_rows:-y_rows]
        self.R32_decomposition = Utils.eigenvalue_decomposition(self.R32)

    def system_identification(self, rank: int = None) -> Tuple[StateSpace, np.ndarray]:
        if self.R32_decomposition is None:
            raise Exception('Perform subspace identification first.')
        if rank is None:
            rank = self.y_dim * self.num_block_rows
        self.x_dim = rank

        observability_decomposition = self._get_observability_matrix_decomposition()

        return self._identify_state_space(observability_decomposition)

    def apply_n4sid(self, rank: int = None) -> Tuple[StateSpace, np.ndarray]:
        if rank is None:
            rank = self.y_dim * self.num_block_rows

        self.x_dim = rank

        observability_decomposition = self._apply_observability_decomposition()

        covariance_matrix, abcd = self._calculate_covariance_matrix(observability_decomposition)

        q = covariance_matrix[:self.x_dim, :self.x_dim]
        r = covariance_matrix[self.x_dim:, self.x_dim:]
        s = covariance_matrix[:self.x_dim, self.x_dim:]
        state_space_covariance_matrix = np.block([
            [r, s.T],
            [s, q]
        ])
        return (
            StateSpace(
                abcd[:self.x_dim, :self.x_dim],
                abcd[:self.x_dim, self.x_dim:],
                abcd[self.x_dim:, :self.x_dim],
                abcd[self.x_dim:, self.x_dim:],
            ),
            (state_space_covariance_matrix + state_space_covariance_matrix.T) / 2
        )

    def _apply_observability_decomposition(self):
        u_hankel = Utils.block_hankel_matrix_parallel(self.u_array, self.num_block_rows).tocsr()
        y_hankel = Utils.block_hankel_matrix_parallel(self.y_array, self.num_block_rows).tocsr()

        u_past, u_future = u_hankel[:, :-self.num_block_rows], u_hankel[:, self.num_block_rows:]
        y_past, y_future = y_hankel[:, :-self.num_block_rows], y_hankel[:, self.num_block_rows:]
        # u_instrumental_y = sparse.hstack([u_future, u_past, y_past, y_future])

        # approximate QR with Cholesky
        _, r = Utils.sparse_qr(sparse.hstack([u_future, u_past, y_past, y_future]).T, mode='NATURAL')

        y_rows, u_rows = self.y_dim * self.num_block_rows, self.u_dim * self.num_block_rows

        R22 = r[u_rows:-y_rows, u_rows:-y_rows]
        R32 = r[-y_rows:, u_rows:-y_rows]

        u_and_y = sparse.hstack([u_hankel, y_hankel])
        observability = R32 @ sparse.linalg.pinv(R22 @ u_and_y)
        observability_decomposition = Utils.reduce_decomposition(
            Utils.eigenvalue_decomposition(observability),
            self.x_dim
        )

        return observability_decomposition

    def _calculate_covariance_matrix(self, observability_decomposition: Decomposition) -> np.ndarray:
        x = (np.sqrt(observability_decomposition.s.diagonal()) @ observability_decomposition.vh)[:, :-1]
        last_y, last_u = self.y_array[self.num_block_rows:, :].T, self.u_array[self.num_block_rows:, :].T
        x_and_y = np.concatenate([x[:, 1:], last_y[:, :-1]])
        x_and_u = np.concatenate([x[:, :-1], last_u[:, :-1]])
        abcd = np.linalg.pinv(x_and_u @ x_and_u.T) @ x_and_u @ x_and_y.T
        abcd = abcd.T
        residuals = x_and_y - abcd @ x_and_u
        covariance_matrix = residuals @ residuals.T / residuals.shape[1]
        return covariance_matrix, abcd

    def _identify_state_space(self, observability_decomposition: Decomposition) -> Tuple[StateSpace, np.ndarray]:
        x = (np.sqrt(observability_decomposition.s) @ observability_decomposition.vh)[:, :-1]
        last_y, last_u = self.y_array[self.num_block_rows:, :].T, self.u_array[self.num_block_rows:, :].T
        x_and_y = np.concatenate([x[:, 1:], last_y[:, :-1]])
        x_and_u = np.concatenate([x[:, :-1], last_u[:, :-1]])
        abcd = np.linalg.pinv(x_and_u @ x_and_u.T) @ x_and_u @ x_and_y.T
        abcd = abcd.T
        residuals = x_and_y - abcd @ x_and_u
        covariance_matrix = residuals @ residuals.T / residuals.shape[1]
        q = covariance_matrix[:self.x_dim, :self.x_dim]
        r = covariance_matrix[self.x_dim:, self.x_dim:]
        s = covariance_matrix[:self.x_dim, self.x_dim:]
        state_space_covariance_matrix = np.block([
            [r, s.T],
            [s, q]
        ])
        return (
            StateSpace(
                abcd[:self.x_dim, :self.x_dim],
                abcd[:self.x_dim, self.x_dim:],
                abcd[self.x_dim:, :self.x_dim],
                abcd[self.x_dim:, self.x_dim:],
            ),
            (state_space_covariance_matrix + state_space_covariance_matrix.T) / 2
        )

    def _get_observability_matrix_decomposition(self) -> Decomposition:
        u_hankel = Utils.block_hankel_matrix_parallel(self.u_array, self.num_block_rows).tocsr()
        y_hankel = Utils.block_hankel_matrix_parallel(self.y_array, self.num_block_rows).tocsr()
        u_and_y = sparse.vstack([u_hankel, y_hankel])
        observability = self.R32 @ np.linalg.pinv(self.R22.toarray()) 
        observability = sparse.csr_matrix(observability) @ u_and_y
        observability_decomposition = Utils.reduce_decomposition(
            Utils.eigenvalue_decomposition(observability),
            self.x_dim
        )
        return observability_decomposition

    def plot_eigenvalues(self, ax: plt.Axes):
        if self.R32_decomposition is None:
            raise Exception('Perform subspace identification first.')

        ax.semilogy(np.diag(self.R32_decomposition.s.toarray()), 'x')
        ax.set_title('Estimated observability matrix decomposition')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.grid(True)
