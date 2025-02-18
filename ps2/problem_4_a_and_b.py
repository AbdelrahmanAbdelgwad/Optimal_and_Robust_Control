import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


def riccati_rhs(t, k_flat, A, B, L, Q, T):
    """
    Right-hand side of the Riccati differential equation.
    k_flat is the flattened 2x2 matrix K
    Returns the flattened version of dK/dt
    """
    # Reshape the flattened k into 2x2 matrix
    K = k_flat.reshape(2, 2)

    # Compute dK/dt = -A^T K - KA + KBB^T K - L
    dK_dt = -A.T @ K - K @ A + K @ B @ B.T @ K - L

    return dK_dt.flatten()


def solve_finite_horizon_lqr(T):
    """
    Solve finite horizon LQR for time T
    Returns the Riccati solution for all time points
    """
    # System matrices
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    L = np.array([[2, 0], [0, 4]])
    Q = np.array([[1, 0], [0, 10]])

    # Solve Riccati equation backwards in time
    t_span = (T, 0)
    K_T_flat = Q.flatten()

    # Dense time grid for accurate K(t) interpolation
    t_eval = np.linspace(T, 0, 1000)

    # Critical fix: Add dense_output=True to get interpolation capability
    sol = solve_ivp(
        riccati_rhs,
        t_span,
        K_T_flat,
        args=(A, B, L, Q, T),
        t_eval=t_eval,
        method="RK45",
        dense_output=True,  # This is crucial for interpolation
    )

    # Store the solution values at each time point for interpolation
    K_t_values = np.array([k.reshape(2, 2) for k in sol.y.T])
    t_points = sol.t

    # Create interpolation function
    def K_interpolator(t):
        if isinstance(t, (float, int)):
            t = np.array([t])
        K_interp = np.zeros((len(t), 2, 2))
        for i, ti in enumerate(t):
            # Find the closest time point
            idx = np.argmin(np.abs(t_points - ti))
            K_interp[i] = K_t_values[idx]
        return K_interp[0] if len(t) == 1 else K_interp

    return K_interpolator, A, B


def compute_cost(t, states, u, L, Q):
    """
    Compute the cost for a given trajectory
    """
    # Compute running cost integral
    running_cost = np.zeros_like(t)
    for i in range(len(t)):
        x = states[:, i]
        running_cost[i] = x.T @ L @ x + u[i] ** 2

    integral_cost = np.trapz(running_cost, t)

    # Add terminal cost
    terminal_cost = states[:, -1].T @ Q @ states[:, -1]

    return integral_cost + terminal_cost


def simulate_system(T_design, T_sim, K_interpolator, A, B):
    """
    Simulate the system with time-varying K(t)
    Parameters:
        T_design (float): The design horizon for which K(t) was computed
        T_sim (float): The simulation horizon (can be different from T_design)
        K_interpolator (callable): Function that returns K(t) for t ≤ T_design
        A, B (np.ndarray): System matrices
    """

    def system_dynamics(t, x):
        # For t > T_design, use K(T_design)
        if t > T_design:
            K_t = K_interpolator(T_design)
        else:
            K_t = K_interpolator(t)
        # Compute time-varying control input
        u = -B.T @ K_t @ x
        return A @ x + B @ u.flatten()

    # Initial conditions
    x0 = np.array([10, 0])

    # Time points for simulation
    t_eval = np.linspace(0, T_sim, 1000)

    # Simulate system
    sol = solve_ivp(system_dynamics, (0, T_sim), x0, t_eval=t_eval, method="RK45")

    # Compute control inputs for plotting
    u_list = []
    for i, t in enumerate(sol.t):
        if t > T_design:
            K_t = K_interpolator(T_design)
        else:
            K_t = K_interpolator(t)
        u = -B.T @ K_t @ sol.y[:, i]
        u_list.append(u[0])

    return sol.t, sol.y, np.array(u_list)


def simulate_infinite_horizon(T_sim):
    """
    Simulate the system with infinite horizon LQR controller
    """
    # System matrices
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    L = np.array([[2, 0], [0, 4]])
    R = np.array([[1]])

    # Solve algebraic Riccati equation for infinite horizon
    K_inf = solve_continuous_are(A, B, L, R)
    K_inf_gains = B.T @ K_inf
    print(f"Gain K_inf:\n{K_inf_gains}")

    def system_dynamics_inf(t, x):
        u = -B.T @ K_inf @ x

        return A @ x + B @ u.flatten()

    x0 = np.array([10, 0])
    t_eval = np.linspace(0, T_sim, 1000)

    sol = solve_ivp(system_dynamics_inf, (0, T_sim), x0, t_eval=t_eval, method="RK45")

    u_list = []
    for i in range(len(sol.t)):
        u = -B.T @ K_inf @ sol.y[:, i]
        u_list.append(u[0])

    return sol.t, sol.y, np.array(u_list), K_inf


# Main simulation and plotting
T_design_values = [1, 10, 50]  # Design horizons
T_sim = [1, 10, 50]  # Fixed simulation horizon for all cases

# Cost matrices for evaluation
L = np.array([[2, 0], [0, 4]])
Q = np.array([[1, 0], [0, 10]])

counter = 0
for T_design in T_design_values:
    print(f"\nSolving for design horizon T = {T_design}")

    # Get finite horizon solution
    K_interpolator, A, B = solve_finite_horizon_lqr(T_design)
    t_finite, states_finite, u_finite = simulate_system(
        T_design, T_sim[counter], K_interpolator, A, B
    )

    # Get infinite horizon solution
    t_inf, states_inf, u_inf, K_inf = simulate_infinite_horizon(T_sim[counter])

    # Compute costs
    cost_finite = compute_cost(t_finite, states_finite, u_finite, L, Q)
    cost_inf = compute_cost(t_inf, states_inf, u_inf, L, Q)

    print(f"Finite horizon cost: {cost_finite:.2f}")
    print(f"Infinite horizon cost: {cost_inf:.2f}")

    # # Create figure with 6 subplots
    # plt.figure(figsize=(15, 10))

    # Create figure with 2 subplots
    plt.figure(figsize=(10, 5))

    # 1. State trajectories
    # plt.subplot(231)
    # plt.plot(t_finite, states_finite[0], "b-", label="z(t) finite", linewidth=2)
    # plt.plot(t_finite, states_finite[1], "g-", label="ż(t) finite", linewidth=2)
    plt.subplot(121)
    plt.plot(t_inf, states_inf[0], "b--", label="z(t) infinite", linewidth=1)
    plt.plot(t_inf, states_inf[1], "g--", label="ż(t) infinite", linewidth=1)
    plt.title(f"State Trajectories (T = {T_design})")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.grid(True)

    # 2. Control inputs
    plt.subplot(122)
    # plt.plot(t_finite, u_finite, "r-", label="u(t) finite", linewidth=2)
    # plt.subplot(232)
    plt.plot(t_inf, u_inf, "r--", label="u(t) infinite", linewidth=1)
    plt.title(f"Control Input (T = {T_design})")
    plt.xlabel("Time")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid(True)

    # K(t) elements over time with infinite horizon values
    t_K = np.linspace(0, T_design, 100)
    K_t_values = np.array([K_interpolator(t) for t in t_K])

    # # 3. K₁₁(t)
    # plt.subplot(234)
    # plt.plot(t_K, K_t_values[:, 0, 0], "b-", label="Finite", linewidth=2)
    # # plt.axhline(
    # #     y=K_inf[0, 0],
    # #     color="b",
    # #     linestyle="--",
    # #     label=f"Infinite ({K_inf[0,0]:.2f})",
    # #     linewidth=1,
    # # )
    # plt.title("K₁₁(t)")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)

    # # 4. K₁₂(t)
    # plt.subplot(235)
    # plt.plot(t_K, K_t_values[:, 0, 1], "g-", label="Finite", linewidth=2)
    # # plt.axhline(
    # #     y=K_inf[0, 1],
    # #     color="g",
    # #     linestyle="--",
    # #     label=f"Infinite ({K_inf[0,1]:.2f})",
    # #     linewidth=1,
    # # )
    # plt.title("K₁₂(t)")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)

    # # 5. K₂₁(t)
    # plt.subplot(233)
    # plt.plot(t_K, K_t_values[:, 1, 0], "r-", label="Finite", linewidth=2)
    # # plt.axhline(
    # #     y=K_inf[1, 0],
    # #     color="r",
    # #     linestyle="--",
    # #     label=f"Infinite ({K_inf[1,0]:.2f})",
    # #     linewidth=1,
    # # )
    # plt.title("K₂₁(t)")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)

    # # 6. K₂₂(t)
    # plt.subplot(236)
    # plt.plot(t_K, K_t_values[:, 1, 1], "m-", label="Finite", linewidth=2)
    # # plt.axhline(
    # #     y=K_inf[1, 1],
    # #     color="m",
    # #     linestyle="--",
    # #     label=f"Infinite ({K_inf[1,1]:.2f})",
    # #     linewidth=1,
    # # )
    # plt.title("K₂₂(t)")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"ps2_problem_4_T_{T_design}.png")

    print(f"\nInfinite horizon gain matrix K_inf:")
    print(K_inf)
    print(f"\nFinal finite horizon gain matrix K(T_design):")
    print(K_interpolator(T_design))

    counter += 1
