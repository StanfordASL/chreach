import numpy as np
import jax.numpy as jnp
from jax import jacfwd, vmap
import matplotlib.pyplot as plt
from matplotlib import gridspec
from time import time

np.random.seed(0)

# -----------------------------------------------
# Parameters
n_x = 2
x0 = jnp.array([-1., 0.])
T = 7.5 # final time
N = 200 # number of discretization timesteps of [0, T]
dt = T / N
obs_p = jnp.zeros(2) # obstacle position
goal_p = jnp.array([1., 0.]) # goal position
# -----------------------------------------------

# -----------------------------------------------
# Disturbances
w_ball_radius = 0.1 # ball
def n_W(w):
    # outward-pointing normal vector of 
    # $\partial\mathcal{W}$ at $w$
    normal_vector = w / jnp.linalg.norm(w)
    return normal_vector
def n_W_inverse(n):
    # w\in\partial\mathcal{W}$ such that n_w(w)=n
    w = w_ball_radius * n
    return w
def pmp_disturbance_w_from_q(q):
    return n_W_inverse(q)
def pmp_disturbances_ws_from_qs_vec(qs_vec):
    return vmap(pmp_disturbance_w_from_q)(qs_vec)
def pmp_disturbances_ws_from_qs_mat(qs_mat):
    return vmap(vmap(pmp_disturbance_w_from_q))(qs_mat)
def project(v, u):
    # projection onto the tangent space of the sphere
    # of radius $||v||$.
    u_projected = u - jnp.dot(v, u) * v
    return u_projected
# -----------------------------------------------

# -----------------------------------------------
# Dynamics
def f(x):
    r_goal = goal_p - x
    r_obs = obs_p - x
    F_attraction = r_goal / jnp.linalg.norm(r_goal) 
    F_repulsion = -r_obs / jnp.linalg.norm(r_obs) 
    x_dot = F_attraction + F_repulsion
    return x_dot
def f_dx(x):
    return jacfwd(f)(x)
# -----------------------------------------------

# -----------------------------------------------
# Augmented ODE
def pmp_dyns(x, q):
    # optimal disturbance
    wstar = pmp_disturbance_w_from_q(q)
    # augmented dynamics
    x_dot = f(x) + wstar
    q_dot = -project(q, f_dx(x).T @ q)
    return (x_dot, q_dot)
def next_pmp_state(x, q):
    k1_x, k1_q = pmp_dyns(x, q)
    k2_x, k2_q = pmp_dyns(x + 0.5*dt*k1_x, q + 0.5*dt*k1_q)
    k3_x, k3_q = pmp_dyns(x + 0.5*dt*k2_x, q + 0.5*dt*k2_q)
    k4_x, k4_q = pmp_dyns(x + dt*k3_x, q + dt*k3_q)
    xn = x + (1.0 / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x) * dt
    qn = q + (1.0 / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q) * dt
    return (xn, qn)
def pred_pmp_state_trajectory(x0, q0):
    states = jnp.zeros((N+1, n_x))
    qs = jnp.zeros((N+1, n_x))
    states = states.at[0].set(x0)
    qs = qs.at[0].set(q0)
    for t in range(N):
        xn, qn = next_pmp_state(states[t], qs[t])
        states = states.at[t+1].set(xn)
        qs = qs.at[t+1].set(qn)
    return (states, qs)
def pred_pmp_state_trajectories(x0, q0s):
    M = q0s.shape[0]
    X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)
    trajs_xs, trajs_qs = vmap(pred_pmp_state_trajectory)(X0s, q0s)
    return trajs_xs, trajs_qs
# -----------------------------------------------

# -----------------------------------------------
# Algorithm 1
M = 10000 # number of points
theta_vals = np.linspace(
    1e-3, 2*np.pi-1e-3, M, endpoint=False)
theta_vals = np.append(
    theta_vals, 
    np.linspace(-1e-3, 1e-3, 1001))
x = w_ball_radius * np.cos(theta_vals)
y = w_ball_radius * np.sin(theta_vals)
wstar_0_vec = np.stack((x, y)).T # (M, 2)
n_wstar_0_vec = vmap(n_W)(wstar_0_vec)
# integrate
xs, qs = pred_pmp_state_trajectories(x0, n_wstar_0_vec)
xTs = xs[:, -1, :]
ws = pmp_disturbances_ws_from_qs_mat(qs)
# -----------------------------------------------

# -----------------------------------------------
fig = plt.figure(figsize=[5, 5])
plt.plot(xTs[:, 0], xTs[:, 1],
    color='b')
for t in range(0, N, 20):
    plt.plot(xs[:, t, 0], xs[:, t, 1],
        color='k', alpha=0.25)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])
plt.yticks([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])
plt.xlim([-2.1, 2.1])
plt.ylim([-2.1, 2.1])
plt.tight_layout()
plt.show()
# -----------------------------------------------
