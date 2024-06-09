# %% Importing libraries and defining functions
from datetime import datetime
from re import A
from typing import Callable, List, Tuple
from pandas import DataFrame
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from nfoursid.nfoursid import NFourSID as NFourSID_n

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.nfoursid.state_space import StateSpace
from src.nfoursid.nfoursid import NFourSID
from src.nfoursid.kalman import Kalman
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-colorblind")

np.random.seed(0)  # reproducable results


def discrete_state_space(ss, dt, cutoff=1e-4):
    """
    Converts a continuous state-space system to a discrete one with a given sampling time dt.

    Parameters:
        ss (StateSpace): The continuous state-space system.
        dt (float): Sampling period.
    Returns:
        StateSpace: The discrete state-space system.
    """
    # Compute the matrix exponential of A * dt for Ad
    Ad = expm(ss.a * dt)

    # Compute Bd using numerical integration
    def integrand(tau, _):
        return (expm(ss.a * (dt - tau)) @ ss.b).ravel()

    tau_span = [0, dt]
    initial_state = np.zeros((ss.a.shape[0], ss.b.shape[1])).ravel()
    sol = solve_ivp(integrand, tau_span, initial_state.ravel(), method="RK45", t_eval=[dt])

    Bd = sol.y[:, -1].reshape(ss.a.shape[0], ss.b.shape[1])
    Bd[ np.abs(Bd) < cutoff ] = 0
    Ad[ np.abs(Ad) < cutoff ] = 0
    return StateSpace(Ad, Bd, ss.c, ss.d)


def construct_stable_state_space(
    n: int, Ms: List[int], Ks: List[int], Cs: List[int], L: int = 1, conditioning_coeff: float = 1.1
) -> Tuple[StateSpace, np.ndarray]:
    """
    Construct the dynamic state-space model of a multiple mass-spring-damper system. Force can only be applied to the
    topmost mass. The state of the system is the position and velocity of each mass, x = [x1, v1, x2, v2, ..., xn, vn].
    The system is conditioned to be stable, meaning that the given parameters may change to ensure stability.
    The input is the force applied to the topmost mass, and the output is the position of each mass.
    params:
        n: number of masses
        Ms: List of masses of the system
        Ks: List of spring constants
        Cs: List of damping coefficients
        L: length of each spring
        conditioning_coeff: coefficient to adjust the eigenvalues of the system
    returns:
        ss: StateSpace object representing the system
        F: force vector
    """
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, n))
    C = np.zeros((n, 2 * n))
    D = np.zeros((n, n))
    F = np.zeros((n, 1))

    # initialize the matrices
    B[1::2, :] = np.eye(n)
    C[:, ::2] = np.eye(n)

    # Construct the A matrix
    if n == 1:
        m_1 = Ms[0]
        k_1 = Ks[0]
        c_1 = Cs[0]
        A[0, 1] = 1
        A[1, 0] = -k_1 / m_1
        A[1, 1] = -c_1 / m_1
        F[0] = k_1 * L / m_1
    else:
        A[0, 1] = 1
        A[1, 0] = -(Ks[0] - Ks[1]) / Ms[0]
        A[1, 1] = -Cs[0] / Ms[0]
        A[1, 2] = -Ks[1] / Ms[0]
        F[0] = (Ks[0] + Ks[1]) * L / Ms[0]

    # Construct the A matrix
    for i in range(1, n - 1):
        j = 2 * i
        # velocity
        A[j, j + 1] = 1

        # acceleration
        A[j + 1, j - 2] = Ks[i] / Ms[i]
        A[j + 1, j] = -(Ks[i] - Ks[i + 1]) / Ms[i]
        A[j + 1, j + 1] = -Cs[i] / Ms[i]
        A[j + 1, j + 2] = -Ks[i + 1] / Ms[i]
        F[i] = (Ks[i] + Ks[i + 1]) * L / Ms[i]

    # last mass
    i = n - 1
    j = 2 * i
    A[j, j + 1] = 1

    A[j + 1, j - 2] = Ks[i] / Ms[i]
    A[j + 1, j] = -Ks[i] / Ms[i]
    A[j + 1, j + 1] = -Cs[i] / Ms[i]
    F[i] = Ks[i] * L / Ms[i]

    # check stability
    eigvals, eigvectors = np.linalg.eig(A)
    if np.any(eigvals.real > 0):
        while np.any(eigvals.real > 0):
            print("System is unstable. Adjusting the parameters to ensure stability.")
            # adjust the parameters to ensure stability
            maxeigval = np.max(eigvals.real)
            eigvals -= maxeigval * conditioning_coeff
            A_candidate = eigvectors @ np.diag(eigvals) @ np.linalg.inv(eigvectors)
            eigvals, eigvectors = np.linalg.eig(A_candidate)
        A = A_candidate

        A.imag = np.zeros_like(A.real)

    return StateSpace(A, B, C, D), F


def euler_forward(ss: StateSpace, x0: np.ndarray, u: np.ndarray, noise_model: Callable) -> np.ndarray:
    """
    Simulate the system using the Euler forward method.
    params:
        ss: StateSpace object representing the system
        x0: initial state of the system
        u: input to the system
        noise_model: function that returns the noise vector
    returns:
        y: output of the system at the current time step
        x: state of the system after one time step
    """
    x = ss.a @ x0 + ss.b @ u
    y = ss.c @ x0 + ss.d @ u + noise_model()
    return y, x


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    params:
        x: input to the function
    returns:
        y: output of the function
    """
    return 2 / (1 + np.exp(-x)) - 1 / 2


def prbs(N, Amplitude=1):
    """
    Generate a Pseudo Random Binary Sequence (PRBS) signal of length N.

    Parameters:
    - N (int): Length of the PRBS signal.

    Returns:
    - np.ndarray: The generated PRBS signal consisting of -1 and 1.
    """
    # Seed the random number generator for reproducibility
    np.random.seed(0)

    # Generate a random sequence of 0s and 1s
    prbs_signal = np.random.choice([-Amplitude, Amplitude], size=N)

    return prbs_signal


def introduce_initial_conditions(ss: StateSpace, x0: np.ndarray) -> StateSpace:
    """
    Introduce initial conditions to the state-space model.
    params:
        ss: StateSpace object representing the system
        x0: initial state of the system
    returns:
        ss: StateSpace object with initial conditions
    """
    ss_new = StateSpace(ss.a.copy(), ss.b.copy(), ss.c.copy(), ss.d.copy(), x_init=x0)
    return ss_new


def random_system(n: int):
    """
    Construct a random system with n masses.
    params:
        n: number of masses
    returns:
        ss: StateSpace object representing the system
        F: force vector
    """
    Ms = [3 + random.randint(-2, 3) for _ in range(n)]
    Ks = [20 + random.randint(-10, 10) for _ in range(n)]
    Cs = [100 + random.randint(-10, 10) for _ in range(n)]
    L = 1
    ss, F = construct_stable_state_space(n, Ms, Ks, Cs, L, conditioning_coeff=1.05)
    return ss, F


def deterministic_system(n: int):
    """
    Construct a deterministic system with n masses.
    params:
        n: number of masses
    returns:
        ss: StateSpace object representing the system
        F: force vector
    """
    Ms = [3 for _ in range(n)]
    Ks = [20 for _ in range(n)]
    Cs = [100 for _ in range(n)]
    L = 1
    ss, F = construct_stable_state_space(n, Ms, Ks, Cs, L, conditioning_coeff=1.05)
    return ss, F


if __name__ == "__main__":
    print()
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Construct the state-space model of the system
    n = 3
    L = 1
    DT = 01e-2
    PERIODS = 15
    PERIOD_LENGTH = 1000
    BURN_IN_PERIODS = 1 * (PERIODS // 4) + 1
    EXTRAPOLATION_LENGTH = PERIODS * PERIOD_LENGTH
    BURN_IN_LENGTH = BURN_IN_PERIODS * PERIOD_LENGTH
    # FORCING = np.ones((EXTRAPOLATION_LENGTH, 1))
    phase = np.linspace(0, PERIODS * 2 * np.pi, EXTRAPOLATION_LENGTH)
    # FORCING = (5 * np.sin(10 * phase) + 5 * np.cos(phase)) * \
    #            np.tanh(np.linspace(0, EXTRAPOLATION_LENGTH*DT, EXTRAPOLATION_LENGTH))
    _prbs = prbs(EXTRAPOLATION_LENGTH // PERIODS, Amplitude=50)
    FORCING = np.tile(_prbs, PERIODS)
    FORCING_TEST = (5 * np.sin(10 * phase) + 5 * np.cos(phase)) * np.tanh(
        np.linspace(0, EXTRAPOLATION_LENGTH * DT, EXTRAPOLATION_LENGTH)
    )
    # FORCING = np.ones((EXTRAPOLATION_LENGTH, 1))

    SAVE_FOLDER = os.path.join(os.path.dirname(__file__), "../results")
    name_tag = f"n{n}_L{L}_P{PERIODS}_PL{PERIOD_LENGTH}_BI{BURN_IN_PERIODS}_DT{DT}"
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIGURES_FOLDER = os.path.join(SAVE_FOLDER, f"figures/{name_tag}/{date_tag}")
    MODELS_FOLDER = os.path.join(SAVE_FOLDER, f"models/{name_tag}/{date_tag}")
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    ss, F = deterministic_system(n)
    ss = discrete_state_space(ss, DT)

    # check stability
    eigvals = np.linalg.eigvals(ss.a)
    unstable = (eigvals.real**2 + eigvals.imag**2) > 1
    # warn if the system is unstable
    if np.any(unstable):
        print("The system is unstable.")
    # define cmap for the scatter plot
    colors = ["blue" if not unst else "red" for unst in unstable]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(eigvals.real, eigvals.imag, c=colors, cmap="coolwarm")
    ax.set_title("Eigenvalues of the system")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")

    unitary_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
    ax.plot(unitary_circle.real, unitary_circle.imag, "--", color="black")
    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    fig.tight_layout()
    ax.axis("equal")

    plt.savefig(os.path.join(FIGURES_FOLDER, "eigenvalues.png"), bbox_inches="tight", dpi=300)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # find stationary point
    x0_p = np.linalg.solve(ss.a, ss.b @ (-F))
    print(f"Stationary point: {x0_p.T}")

    # Simulate the system with forward Euler integration
    y = np.zeros((EXTRAPOLATION_LENGTH, n))
    x = np.zeros((EXTRAPOLATION_LENGTH, 2 * n))
    u = np.zeros((EXTRAPOLATION_LENGTH, n))
    x0 = x0_p.copy()
    for i in range(EXTRAPOLATION_LENGTH):
        _u = F.copy()
        _u[1] += FORCING_TEST[i]
        _y, x0 = euler_forward(ss, x0, _u, lambda: np.random.standard_normal((n, 1)) * 1e-3)
        y[i, :] = _y.flatten()
        x[i, :] = x0.flatten()
        u[i, :] = _u.flatten()

    # plot the results
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for i in range(n):
        axs[0].plot(y[:, i], label=f"mass {i+1}", color=f"C{i}")
        axs[1].plot(x[:, 2 * i], label=f"position mass {i+1}", color=f"C{i}")
        axs[1].plot(x[:, 2 * i + 1], label=f"velocity mass {i+1}", linestyle="--", color=f"C{i}")
    axs[2].plot(FORCING_TEST, label="forcing")
    axs[2].plot(
        np.tanh(np.linspace(0, EXTRAPOLATION_LENGTH * DT, EXTRAPOLATION_LENGTH)), label="activation", linestyle="--"
    )

    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    axs[0].set_title("Response of the system to the forcing")
    axs[2].set_xlabel("time [s]")
    axs[0].set_ylabel("position [m]")
    axs[1].set_ylabel("state")
    axs[2].set_ylabel("force [N]")
    fig.tight_layout()

    plt.savefig(os.path.join(FIGURES_FOLDER, "Test_data.png"), bbox_inches="tight", dpi=300)
    DataFrame(y).to_csv(os.path.join(MODELS_FOLDER, "test_target.csv"))
    DataFrame(FORCING_TEST).to_csv(os.path.join(MODELS_FOLDER, "test_forcing.csv"))
    DataFrame(u).to_csv(os.path.join(MODELS_FOLDER, "test_input.csv"))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # generate data
    u = np.zeros((EXTRAPOLATION_LENGTH, n))
    for i in range(EXTRAPOLATION_LENGTH):
        _u = F.copy() + 0.1 * np.random.standard_normal((n, 1)) * np.max(FORCING)
        _u[1] += FORCING[i]
        noise = np.random.standard_normal((n, 1)) * 1e-3

        ss.step(_u, noise)
        u[i, :] = _u.flatten()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    ss.plot_input_output(fig)
    fig.tight_layout()

    plt.savefig(os.path.join(FIGURES_FOLDER, "Training_data.png"), bbox_inches="tight", dpi=300)
    DataFrame(np.asarray(ss.ys).squeeze()).to_csv(os.path.join(MODELS_FOLDER, "training_target.csv"))
    DataFrame(FORCING).to_csv(os.path.join(MODELS_FOLDER, "training_forcing.csv"))
    DataFrame(u).to_csv(os.path.join(MODELS_FOLDER, "training_input.csv"))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sys ID with the original package
    nfoursid_n = NFourSID_n(
        ss.to_dataframe().iloc[
            BURN_IN_LENGTH:, :
        ],  # the state-space model can summarize inputs and outputs as a dataframe
        output_columns=ss.y_column_names,
        input_columns=ss.u_column_names,
        num_block_rows=4 * n,
    )
    nfoursid_n.subspace_identification()

    fig, ax = plt.subplots(figsize=(8, 8))
    nfoursid_n.plot_eigenvalues(ax)  # estimated observability matrix eigenvalues
    fig.tight_layout()  # <- number of eigenvalues that stand out is your state

    plt.savefig(os.path.join(FIGURES_FOLDER, "eigenvalues_n4sid.png"), bbox_inches="tight", dpi=300)

    ORDER_OF_MODEL_TO_FIT = 2 * n
    ss_n, covariance_matrix_1 = nfoursid_n.system_identification(rank=ORDER_OF_MODEL_TO_FIT)

    DataFrame(ss_n.a).to_csv(os.path.join(MODELS_FOLDER, "A_n.csv"))
    DataFrame(ss_n.b).to_csv(os.path.join(MODELS_FOLDER, "B_n.csv"))
    DataFrame(ss_n.c).to_csv(os.path.join(MODELS_FOLDER, "C_n.csv"))
    DataFrame(ss_n.d).to_csv(os.path.join(MODELS_FOLDER, "D_n.csv"))
    DataFrame(covariance_matrix_1).to_csv(os.path.join(MODELS_FOLDER, "covariance_matrix_n.csv"))
    DataFrame(x0_p).to_csv(os.path.join(MODELS_FOLDER, "x0_n.csv"))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # train the model
    nfoursid = NFourSID(
        ss.to_dataframe().iloc[BURN_IN_LENGTH:, :],
        output_columns=ss.y_column_names,
        input_columns=ss.u_column_names,
        num_block_rows=4 * n,
    )

    ss_identified, covariance_matrix = nfoursid.apply_n4sid(rank=2 * n)

    DataFrame(ss_identified.a).to_csv(os.path.join(MODELS_FOLDER, "A.csv"))
    DataFrame(ss_identified.b).to_csv(os.path.join(MODELS_FOLDER, "B.csv"))
    DataFrame(ss_identified.c).to_csv(os.path.join(MODELS_FOLDER, "C.csv"))
    DataFrame(ss_identified.d).to_csv(os.path.join(MODELS_FOLDER, "D.csv"))
    DataFrame(covariance_matrix).to_csv(os.path.join(MODELS_FOLDER, "covariance_matrix.csv"))
    DataFrame(x0_p).to_csv(os.path.join(MODELS_FOLDER, "x0.csv"))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # compare the identified and the original system
    y_pred = np.zeros((EXTRAPOLATION_LENGTH, n))
    x_pred = np.zeros((EXTRAPOLATION_LENGTH, 2 * n))
    x_pred_nominal = np.zeros((EXTRAPOLATION_LENGTH, 2 * n))
    y_pred_nominal = np.zeros((EXTRAPOLATION_LENGTH, n))
    x0 = x0_p.copy()
    x0_nominal = x0_p.copy()
    for i in range(EXTRAPOLATION_LENGTH):
        u = F.copy()
        u[1] += FORCING_TEST[i]
        _y, x0 = euler_forward(ss_identified, x0, u, lambda: np.random.standard_normal((n, 1)) * 1e-3)
        _y_nominal, x0_nominal = euler_forward(ss_n, x0_nominal, u, lambda: np.random.standard_normal((n, 1)) * 1e-3)
        y_pred[i, :] = _y.flatten()
        x_pred[i, :] = x0.flatten()
        y_pred_nominal[i, :] = _y_nominal.flatten()
        x_pred_nominal[i, :] = x0_nominal.flatten()

    # calculate the error with transients
    error = np.mean(np.abs((y - y_pred)) / np.abs(y)) * 100
    error_n = np.mean(np.abs((y - y_pred_nominal)) / np.abs(y)) * 100
    print(f"Mean relative error with transients:\n\tNominal {error_n:0.4f}% \tIdentified {error:0.4f}%")

    # calculate the error without transients
    error = np.mean((np.abs((y - y_pred)) / np.abs(y))[BURN_IN_LENGTH:]) * 100
    error_n = np.mean((np.abs((y - y_pred_nominal)) / np.abs(y))[BURN_IN_LENGTH:]) * 100
    print(f"Mean relative error without transients:\n\tNominal {error_n:0.4f}% \tIdentified {error:0.4f}%")

    # plot the results
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    _time = np.arange(BURN_IN_LENGTH, EXTRAPOLATION_LENGTH) * DT
    for i in range(n):
        axs[0].plot(_time, y[BURN_IN_LENGTH:, i], label=f"mass {i+1}", color=f"C{i}", alpha=0.3)
        axs[0].plot(
            _time, y_pred[BURN_IN_LENGTH:, i], label=f"mass {i+1} identified", linestyle="--", color=f"C{i}", alpha=0.5
        )
        axs[0].plot(
            _time,
            y_pred_nominal[BURN_IN_LENGTH:, i],
            label=f"mass {i+1} nominal",
            linestyle="-.",
            color=f"C{i}",
            alpha=0.8,
        )
    axs[1].plot(_time, ((y - y_pred) / y)[BURN_IN_LENGTH:] * 100, label="error identified")
    axs[1].plot(_time, ((y - y_pred_nominal) / y)[BURN_IN_LENGTH:] * 100, label="error nominal")
    axs[2].plot(_time[:PERIOD_LENGTH], FORCING[:PERIOD_LENGTH], label="forcing")
    # axs[2].plot(_time,
    #     np.tanh(np.linspace(0, EXTRAPOLATION_LENGTH * DT, EXTRAPOLATION_LENGTH))[BURN_IN_LENGTH:], label="activation", linestyle="--"
    # )

    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    axs[0].set_title("Response of the system to the forcing")
    axs[2].set_xlabel("time [s]")
    axs[0].set_ylabel("position [m]")
    axs[1].set_ylabel("error")
    axs[2].set_ylabel("force [N]")

    axs[1].set_ylim([-200, 200])
    fig.tight_layout()

    plt.savefig(os.path.join(FIGURES_FOLDER, "Comparison.png"), bbox_inches="tight", dpi=300)

    DataFrame(100 * np.abs((y - y_pred)) / np.abs(y)).to_csv(os.path.join(MODELS_FOLDER, "error.csv"))
    DataFrame(100 * np.abs((y - y_pred_nominal)) / np.abs(y)).to_csv(os.path.join(MODELS_FOLDER, "error_n.csv"))
# %%
