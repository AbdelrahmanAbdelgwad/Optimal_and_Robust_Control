import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig


class myParams:
    def __init__(self, g, L, m, c, pos):
        self.g = g  # gravitational acceleration in m/s^2
        self.L = L  # length of pendulum in meters
        self.m = m  # mass of pendulum in kg
        self.c = c  # damping coefficient
        self.pos = pos  # linearize around 1: top or -1 bot


# Define simulation parameters
dt = 0.001  # time step
T = 10  # length of time for the sim
params = myParams(g=9.81, L=1, m=1, c=0.1, pos=1)
pi_val = np.pi
eps = 2


# Define the full nonlinear dynamics function
def fullPend_dyn(z, v, params):
    f = np.zeros(zo.shape)

    f[0] = z[1]
    f[1] = (
        -(params.g / (params.L**2)) * np.sin(z[0])
        - (params.c / (params.m * (params.L**2))) * z[1]
        + v
    )

    return f


# Define the linearized system matrices
A = np.array(
    [
        [0, 1],
        [params.pos * params.g / params.L**2, -params.c / (params.m * (params.L**2))],
    ]
)
B = np.array([[0], [1]])


# define initial states
if params.pos == 1:
    zo = np.array([pi_val, 0])
else:
    zo = np.array([0, 0])


xo = np.array([0, 0])

# Initialize storage for results
n_steps = int(T / dt)
time = np.linspace(0, T, n_steps)
z = np.zeros((n_steps, len(zo)))
x = np.zeros((n_steps, len(xo)))
z[0, :] = zo - [eps, 0]
x[0, :] = z[0, :] - zo

K = np.array([[11, 0.9]])
Acl = A - np.dot(B, K)


# do the sim
for i in range(n_steps - 1):
    # calculate the control
    v = -np.dot(K, z[i, :] - zo)

    # first the full dynamics
    z[i + 1, :] = z[i, :] + dt * fullPend_dyn(z[i, :], v, params)

    # now the linearized
    x[i + 1, :] = x[i, :] + dt * (np.dot(Acl, x[i, :]))


# plot results
plt.figure(figsize=(12, 6))
plt.plot(z[:, 0], z[:, 1], "b")
plt.plot(x[:, 0] + zo[0], x[:, 1], "r")
plt.gca().legend(("nonlin", "lin"))
plt.show()
