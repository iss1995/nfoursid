# %% Importing libraries and defining functions
from typing import Callable, List, Tuple
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import os
import sys

from sympy import li


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.nfoursid.state_space import StateSpace
from src.nfoursid.nfoursid import NFourSID
from src.nfoursid.kalman import Kalman

np.random.seed(0)  # reproducable results


def discrete_state_space(ss, dt):
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
    sol = solve_ivp(integrand, tau_span, initial_state.ravel(), method='RK45', t_eval=[dt])

    Bd = sol.y[:, -1].reshape(ss.a.shape[0], ss.b.shape[1])

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
        A[j + 1, j] = - (Ks[i] - Ks[i + 1]) / Ms[i]
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
            eigvals -= maxeigval*conditioning_coeff
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
    return 2 / (1 + np.exp(-x)) - 1/2


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Construct the state-space model of the system
n = 10
Ms = [3 for _ in range(n)]
Ks = [10 for _ in range(n)]
Cs = [100 for _ in range(n)]
L = 1
DT = 01e-2
EXTRAPOLATION_LENGTH = 30000
# FORCING = np.ones((EXTRAPOLATION_LENGTH, 1))
phase = np.linspace(0, 5 * np.pi, EXTRAPOLATION_LENGTH)
# FORCING = (50 * np.sin(10 * phase) + 50 * np.cos(phase)) * \
#            np.tanh(np.linspace(0, EXTRAPOLATION_LENGTH*DT, EXTRAPOLATION_LENGTH))
FORCING = np.ones((EXTRAPOLATION_LENGTH, 1))

ss, F = construct_stable_state_space(n, Ms, Ks, Cs, L, conditioning_coeff=1.05)
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Simulate the system
for i in range(EXTRAPOLATION_LENGTH):
    input_state = F.copy()
    input_state[-1] += FORCING[i]
    noise = np.random.standard_normal((n, 1)) * 1e-3

    ss.step(input_state, noise)

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
ss.plot_input_output(fig)
fig.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# find stationary point
x0 = np.linalg.solve(ss.a, ss.b @ (-F))
print(f"Stationary point: {x0.T}")

# Simulate the system with forward Euler integration
y = np.zeros((EXTRAPOLATION_LENGTH, n))
x = np.zeros((EXTRAPOLATION_LENGTH, 2 * n))
for i in range(EXTRAPOLATION_LENGTH):
    u = F.copy()
    u[0] += FORCING[i]
    _y, x0 = euler_forward(ss, x0, u, lambda: np.random.standard_normal((n, 1)) * 1e-3)
    y[i, :] = _y.flatten()
    x[i, :] = x0.flatten()

# plot the results
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
for i in range(n):
    axs[0].plot(y[:, i], label=f"mass {i+1}", color=f"C{i}")
    axs[1].plot(x[:, 2 * i], label=f"position mass {i+1}", color=f"C{i}")
    axs[1].plot(x[:, 2 * i + 1], label=f"velocity mass {i+1}", linestyle="--", color=f"C{i}")
axs[2].plot(FORCING, label="forcing")
axs[2].plot(np.tanh(np.linspace(0, EXTRAPOLATION_LENGTH*DT, EXTRAPOLATION_LENGTH)), label= "activation", linestyle="--")

# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
axs[0].set_title("Response of the system to the forcing")
axs[2].set_xlabel("time [s]")
axs[0].set_ylabel("position [m]")
axs[1].set_ylabel("state")
axs[2].set_ylabel("force [N]")
fig.tight_layout()




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
