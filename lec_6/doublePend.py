import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# === 1. System Parameters ===
m = 1.0  # Mass of each point mass (kg)
l = 1.0  # Length of each massless rod (m)
g = -9.81  # Acceleration due to gravity (m/s^2)
t_max = 20.0  # Final simulation time (seconds)

# Linearized system matrices (from previous derivation)
A = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-2 * g / l, g / l, 0, 0],
        [-2 * g / l, 2 * g / l, 0, 0],
    ]
)

B = np.array([[0], [0], [1 / (m * l**2)], [1 / (m * l**2)]])

# Define cost matrices
beta = 0.01
alpha = 1
Q = np.eye(4) * alpha  # State cost weighting
# # penalize the first state more
# Q[0, 0] = 20
# # penalize the third state more
# Q[2, 2] = 2
R = beta * np.eye(1)  # Control cost weighting

# Solve the Continuous Algebraic Riccati Equation (CARE)
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute the LQR gain
K = np.linalg.inv(R) @ B.T @ P  # Optimal gain matrix


# === 3. Torque Control Function using LQR ===
def torque_control(state):
    """
    Computes the LQR control input given the system state.
    """
    # Extract state variables
    theta1, theta2, dtheta1, dtheta2 = state

    # Define the state deviation from the equilibrium (theta1=pi, theta2=0)
    x = np.array(
        [
            [theta1 - np.pi],  # Small deviations around theta1 = pi
            [theta2 - 0],  # Small deviations around theta2 = 0
            [dtheta1],
            [dtheta2],
        ]
    )

    # Compute the optimal control input
    u = -K @ x  # LQR control law

    # return 0 * float(u)  # Convert to scalar
    return float(u)


# === 3. Equations of Motion ===
def double_pendulum_dynamics(t, x):
    """
    Computes the derivatives for the double pendulum.
    x = [theta1, theta2, dtheta1, dtheta2]
    Returns [dtheta1, dtheta2, ddtheta1, ddtheta2]
    """
    theta1, theta2, dtheta1, dtheta2 = x

    # Compute control torque
    tau = torque_control(x)

    # Trigonometric shortcuts
    sin1, cos1 = np.sin(theta1), np.cos(theta1)
    sin2, cos2 = np.sin(theta2), np.cos(theta2)
    sin12, cos12 = np.sin(theta1 - theta2), np.cos(theta1 - theta2)

    # === Inertia Matrix ===
    D = np.array([[2 * m * l**2, m * l**2 * cos12], [m * l**2 * cos12, m * l**2]])

    # === Coriolis & Centrifugal Terms ===
    C = np.array(
        [
            -m * l**2 * sin12 * (dtheta2**2 + 2 * dtheta1 * dtheta2),
            m * l**2 * sin12 * dtheta1**2,
        ]
    )

    # === Gravity Terms ===
    G = np.array([-2 * m * g * l * sin1, -m * g * l * sin2])

    # === Input Torque ===
    B = np.array([1, 0])  # Actuation at upper joint

    # Solve for accelerations
    RHS = B * tau - C - G
    ddtheta = np.linalg.solve(D, RHS)  # Matrix inversion

    return [dtheta1, dtheta2, ddtheta[0], ddtheta[1]]


# === 4. Solve the ODE ===
x0 = [np.pi, np.pi / 4, 0.0, 0.0]  # Initial conditions
t_span = (0, t_max)
num_samples = 5000
t_eval = np.linspace(0, t_max, num_samples)

sol = solve_ivp(
    double_pendulum_dynamics, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-8
)

theta1_vals = sol.y[0, :]
theta2_vals = sol.y[1, :]

# === 5. Compute Torque Over Time ===
tau_vals = np.zeros_like(sol.t)
for i, t_i in enumerate(sol.t):
    state_i = sol.y[:, i]
    tau_vals[i] = torque_control(state_i)

# === 7. Plot Control Torque vs Time ===
fig2, ax2 = plt.subplots()
ax2.plot(sol.t, tau_vals, label="Torque (tau)")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Torque [N·m]")
ax2.set_title("Control Torque vs. Time")
ax2.legend()

# === 6. Visualization ===
fig, ax = plt.subplots()
ax.set_aspect("equal", "box")
ax.set_xlim(-2 * l, 2 * l)
ax.set_ylim(-2 * l, 2 * l)
ax.set_title("Double Pendulum Simulation (Torque=0)")

(line1,) = ax.plot([], [], "o-", lw=2, color="blue")  # Upper rod
(line2,) = ax.plot([], [], "o-", lw=2, color="red")  # Lower rod


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(i):
    """Update animation for frame i"""
    th1, th2 = theta1_vals[i], theta2_vals[i]

    # Compute mass positions
    x1, y1 = l * np.sin(th1), -l * np.cos(th1)
    x2, y2 = x1 + l * np.sin(th2), y1 - l * np.cos(th2)

    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    return line1, line2


# Set interval to match real-time playback (milliseconds per frame)
real_time_interval = (t_max / num_samples) * 1000  # Convert seconds to milliseconds

# Create animation with real-time playback
anim = FuncAnimation(
    fig,
    animate,
    frames=num_samples,
    interval=real_time_interval,
    blit=True,
    init_func=init,
)


plt.show()
