import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


class StateEstimator:
    """
    Implements the Luenberger observer for state estimation when only ż is measurable
    """

    def __init__(self, A, B, x0_estimate):
        self.A = A  # System matrix (2x2)
        self.B = B  # Input matrix (2x1)
        self.C = np.array([[0, 1]])  # Measurement matrix (1x2): only measuring ż
        self.L = np.array(
            [[-99], [20]]
        )  # Observer gains from characteristic polynomial (s+1)²
        self.x_hat = x0_estimate.reshape(2, 1)  # Initial state estimate (2x1)

    def update(self, dt, y, u):
        """
        Update state estimate using observer dynamics
        ẋ̂ = (A-LC)x̂ + Bu + Ly
        """
        # Reshape inputs to ensure correct dimensions
        y = np.array([[y]])  # Measurement (1x1)
        u = np.array([[u]])  # Control input (1x1)

        # Observer dynamics
        x_hat_dot = (
            (self.A - self.L @ self.C) @ self.x_hat  # (A-LC)x̂
            + self.B @ u  # Bu
            + self.L @ y  # Ly
        )

        # Euler integration
        self.x_hat = self.x_hat + x_hat_dot * dt
        return self.x_hat


def compute_cost(t, states, controls, L, Q):
    """
    Compute the cost J = ∫(x'Lx + u'u)dt + x(T)'Qx(T)
    """
    # Compute running cost at each time point
    running_cost = np.zeros_like(t)
    for i in range(len(t)):
        x = states[:, i].reshape(2, 1)
        u = controls[i]
        running_cost[i] = float(x.T @ L @ x + u * u)

    # Compute integral cost and terminal cost
    integral_cost = np.trapz(running_cost, t)
    x_T = states[:, -1].reshape(2, 1)
    terminal_cost = float(x_T.T @ Q @ x_T)

    return integral_cost + terminal_cost


def simulate_infinite_horizon(T_sim, x0_true, x0_estimate):
    """
    Simulate the system with infinite horizon LQR controller and state estimation
    """
    # System matrices as defined in the problem
    A = np.array([[0, 1], [-1, 0]])  # System dynamics
    B = np.array([[0], [1]])  # Input matrix
    L = np.array([[2, 0], [0, 4]])  # State cost matrix
    R = np.array([[1]])  # Control cost matrix
    Q = np.array([[1, 0], [0, 10]])  # Terminal cost matrix

    # Solve for infinite horizon optimal gains
    K_inf = solve_continuous_are(A, B, L, R)

    # Simulation parameters
    dt = 0.001  # Time step
    t_points = np.arange(0, T_sim + dt, dt)

    # Initialize true state and estimator
    x = x0_true.reshape(2, 1)  # True initial state
    estimator = StateEstimator(A, B, x0_estimate)  # Initialize estimator

    # Storage arrays
    states = np.zeros((2, len(t_points)))  # True states
    estimated_states = np.zeros((2, len(t_points)))  # Estimated states
    controls = np.zeros(len(t_points))  # Control inputs

    for i, t in enumerate(t_points):
        # Get measurement: y = Cx (only measuring ż)
        y = float((estimator.C @ x)[0, 0])

        # Compute control input: u = -Kx̂
        u = float(-(B.T @ K_inf @ estimator.x_hat)[0, 0])
        # u = 0 # Uncomment to use the control input

        # Store current values
        states[:, i] = x.flatten()
        estimated_states[:, i] = estimator.x_hat.flatten()
        controls[i] = u

        # Update true system: ẋ = Ax + Bu
        x_dot = A @ x + B * u
        x = x + x_dot * dt

        # Update state estimate
        estimator.update(dt, y, u)

    # Compute cost
    cost = compute_cost(t_points, states, controls, L, Q)

    return t_points, states, estimated_states, controls, K_inf, cost


def plot_results(t, states, est_states, controls, T_sim):
    """
    Plot the simulation results
    """
    plt.figure(figsize=(15, 10))

    # Plot states and estimated states
    plt.subplot(311)
    plt.plot(t, states[0], "b-", label="z(t) actual", linewidth=2)
    plt.plot(t, states[1], "g-", label="ż(t) actual", linewidth=2)
    plt.plot(t, est_states[0], "b--", label="z(t) estimated", linewidth=1)
    plt.plot(t, est_states[1], "g--", label="ż(t) estimated", linewidth=1)
    plt.title(f"State Trajectories (T = {T_sim})")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.grid(True)

    # Plot control input
    plt.subplot(312)
    plt.plot(t, controls, "r-", label="u(t)", linewidth=2)
    plt.title("Control Input")
    plt.xlabel("Time")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid(True)

    # Plot estimation errors
    plt.subplot(313)
    error_z = states[0] - est_states[0]
    error_zdot = states[1] - est_states[1]
    plt.plot(t, error_z, "b-", label="z error", linewidth=2)
    plt.plot(t, error_zdot, "g-", label="ż error", linewidth=2)
    plt.title("Estimation Error")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    return plt.gcf()


# Main simulation
if __name__ == "__main__":
    # Time horizons to simulate
    T_values = [1, 10, 50]

    # Initial conditions
    x0_true = np.array([[10.0], [0.0]])  # True initial state
    x0_estimate = np.array(
        [[0.0], [0.0]]
    )  # Initial state estimate (different from true)

    # Run simulations for each time horizon
    for T_sim in T_values:
        print(f"\nSimulating with T = {T_sim}")

        # Run simulation
        t, states, est_states, controls, K_inf, cost = simulate_infinite_horizon(
            T_sim, x0_true, x0_estimate
        )

        # Plot results
        fig = plot_results(t, states, est_states, controls, T_sim)
        plt.savefig(f"infinite_horizon_T_{T_sim}.png")
        plt.close()

        print(f"Cost for T = {T_sim}: {cost:.2f}")

    # Print the infinite horizon gain matrix
    print("\nInfinite horizon gain matrix K_inf:")
    print(K_inf)
