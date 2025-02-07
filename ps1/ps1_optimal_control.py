import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# System parameters
r0 = 10  # nominal radius
beta = 10  # constant
w0 = np.sqrt(beta / r0**3)  # angular velocity


def nonlinear_system(state, t):
    """
    Nonlinear system dynamics
    state = [theta, theta_dot, r, r_dot]
    """
    theta, theta_dot, r, r_dot = state

    # Equations of motion
    theta_ddot = -2 * r_dot * theta_dot / r
    r_ddot = r * theta_dot**2 - beta / r**2

    return [theta_dot, theta_ddot, r_dot, r_ddot]


def linear_system(state, t):
    """
    Linearized system dynamics around nominal solution
    state = [delta_theta, delta_theta_dot, delta_r, delta_r_dot]
    """
    # System matrix derived from linearization
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, -2 * w0 / r0],
            [0, 0, 0, 1],
            [0, 2 * w0 * r0, w0**2 + 2 * beta / r0**3, 0],
        ]
    )

    return A @ state


# Time vector
t = np.linspace(0, 200, 1000)

# Initial conditions for "close" case
theta0_close = 0
theta_dot0_close = w0
r0_close = 1.01 * r0  # 1.01*r0 as specified
r_dot0_close = 0

# Initial conditions for "far" case
theta0_far = 0
theta_dot0_far = w0
r0_far = 1.1 * r0  # 1.1*r0 as specified
r_dot0_far = 0

# Solve systems for "close" case
state0_close = [theta0_close, theta_dot0_close, r0_close, r_dot0_close]
nl_sol_close = odeint(nonlinear_system, state0_close, t)

# Solve linearized system for "close" case
delta_state0_close = [0, 0, 0.01 * r0, 0]  # Perturbations from nominal
lin_sol_close = odeint(linear_system, delta_state0_close, t)

# Add nominal solution to linearized solution
lin_sol_close[:, 0] += w0 * t  # theta
lin_sol_close[:, 1] += w0  # theta_dot
lin_sol_close[:, 2] += r0  # r
# r_dot remains the same (nominal is 0)

# Solve systems for "far" case
state0_far = [theta0_far, theta_dot0_far, r0_far, r_dot0_far]
nl_sol_far = odeint(nonlinear_system, state0_far, t)

# Solve linearized system for "far" case
delta_state0_far = [0, 0, 0.1 * r0, 0]  # Perturbations from nominal
lin_sol_far = odeint(linear_system, delta_state0_far, t)

# Add nominal solution to linearized solution
lin_sol_far[:, 0] += w0 * t  # theta
lin_sol_far[:, 1] += w0  # theta_dot
lin_sol_far[:, 2] += r0  # r
# r_dot remains the same (nominal is 0)

# Create polar plots
plt.figure(figsize=(15, 6))

# Plot for "close" initial conditions
plt.subplot(121, projection="polar")
plt.plot(nl_sol_close[:, 0], nl_sol_close[:, 2], "b-", label="Nonlinear")
plt.plot(lin_sol_close[:, 0], lin_sol_close[:, 2], "r--", label="Linear")
plt.plot(w0 * t, np.full_like(t, r0), "g:", label="Nominal")
plt.title("Close to Nominal (r0 = 1.01r0)")
plt.legend()

# Plot for "far" initial conditions
plt.subplot(122, projection="polar")
plt.plot(nl_sol_far[:, 0], nl_sol_far[:, 2], "b-", label="Nonlinear")
plt.plot(lin_sol_far[:, 0], lin_sol_far[:, 2], "r--", label="Linear")
plt.plot(w0 * t, np.full_like(t, r0), "g:", label="Nominal")
plt.title("Far from Nominal (r0 = 1.1r0)")
plt.legend()

plt.tight_layout()
plt.savefig("ps1/ps1_q6_d.png")
plt.show()
