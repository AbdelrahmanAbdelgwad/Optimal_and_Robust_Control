import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import signal
import control as ctrl

# System parameters
r0 = 10  # nominal radius
beta = 10  # constant
w0 = np.sqrt(beta / r0**3)  # nominal angular velocity

# Define state-space matrices
A = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 0, -2 * w0 / r0],
        [0, 0, 0, 1],
        [0, 2 * w0 * r0, w0**2 + 2 * beta / r0**3, 0],
    ]
)

B = np.array([[0, 0], [0, 1 / r0], [0, 0], [1, 0]])

# Controller design
Q = np.diag([4, 4, 4, 4])  # state cost
R = np.diag([0.1, 0.1])  # control cost
K, _, _ = ctrl.lqr(A, B, Q, R)


def controlled_nonlinear_system(state, t):
    """Nonlinear system with state feedback control"""
    theta, theta_dot, r, r_dot = state
    delta_state = np.array(
        [
            theta - w0 * t,
            theta_dot - w0,
            r - r0,
            r_dot,
        ]
    )
    u = -K @ delta_state
    theta_ddot = -2 * r_dot * theta_dot / r + u[1] / r
    r_ddot = r * theta_dot**2 - beta / r**2 + u[0]
    return [theta_dot, theta_ddot, r_dot, r_ddot]


def controlled_linear_system(state, t):
    """Linearized system with state feedback control"""
    u = -K @ state
    return A @ state + B @ u


def simulate_and_plot(ax, r_init, t, plot_title):
    """Simulate and plot both systems for a given initial radius"""
    # Initial conditions
    theta0, r_dot0 = 0, 0
    theta_dot0 = w0
    r0_perturbed = r_init

    # Solve nonlinear system
    state0 = [theta0, theta_dot0, r0_perturbed, r_dot0]
    nl_sol = odeint(controlled_nonlinear_system, state0, t)

    # Solve linear system
    delta_state0 = [0, 0, r0_perturbed - r0, 0]
    lin_sol = odeint(controlled_linear_system, delta_state0, t)
    lin_sol[:, 0] += w0 * t  # Add nominal theta
    lin_sol[:, 2] += r0  # Add nominal radius

    # Create plot
    ax.plot(nl_sol[:, 0], nl_sol[:, 2], "b-", label="Nonlinear")
    ax.plot(lin_sol[:, 0], lin_sol[:, 2], "r--", label="Linear")
    ax.plot(w0 * t, np.full_like(t, r0), "g:", label="Nominal")
    ax.set_title(plot_title)
    ax.legend()


# Time vector
t = np.linspace(0, 200, 1000)

# Figure 1: Original problem conditions (close and far)
plt.figure(figsize=(12, 5))

# Close case (r = 1.01r0)
ax1 = plt.subplot(121, projection="polar")
simulate_and_plot(ax1, 1.01 * r0, t, "Close Case (r = 1.01r0)")

# Far case (r = 1.1r0)
ax2 = plt.subplot(122, projection="polar")
simulate_and_plot(ax2, 1.1 * r0, t, "Far Case (r = 1.1r0)")

plt.tight_layout()
plt.savefig("ps1/ps1_q6_e_1.png")


# Figure 2: Stability analysis with multiple initial conditions
# Create logarithmically spaced initial radii to test system limits
stability_radii = np.logspace(np.log10(1.01), np.log10(1000), 12) * r0

plt.figure(figsize=(15, 12))
rows, cols = 4, 3
for i, r_init in enumerate(stability_radii, 1):
    if i <= rows * cols:  # Ensure we don't exceed subplot grid
        ax = plt.subplot(rows, cols, i, projection="polar")
        simulate_and_plot(ax, r_init, t, f"r = {r_init/r0:.1f}r0")

plt.tight_layout()
plt.savefig("ps1/ps1_q6_e_2.png")
plt.show()

# Print analysis of system behavior
print("\nStability Analysis:")
print("------------------")
for r_init in stability_radii:
    state0 = [0, w0, r_init, 0]
    nl_sol = odeint(controlled_nonlinear_system, state0, t)
    final_radius = nl_sol[-1, 2]
    error = abs(final_radius - r0) / r0 * 100
    status = "Stable" if error < 5 else "Unstable"
    print(f"Initial r/r0 = {r_init/r0:.1f}: {status} (Final error: {error:.1f}%)")
