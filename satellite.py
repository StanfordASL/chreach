import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, vstack, hstack, eye
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import osqp

import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, vmap
from jax.lax import while_loop, fori_loop
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rc, rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)

from time import time

from src.fibonacci import fibonacci_lattice
from src.stats import sample_pts_unit_ball, sample_pts_unit_sphere
from src.viz import set_axes_equal, plot_3d_convex_hull

np.random.seed(0)
B_evaluate_bounds = True
B_generate_main_results = True
B_generate_mpc_results = True
B_plot_main_results = True
B_plot_mpc_results = True
B_reachability_comparison = True
B_main_plot = False

# -----------------------------------------
dt = 1.0 # discretization time
S = 10 # number of controls
M = 100 # number of samples
n_x = 7 # (q, omega) - number of state variables
n_u = 3 # number of control variables

num_MPC_runs = 100
MPC_horizon = 60 # horizon for MPC (in steps of dt time)

num_scp_iters_max = 15
OSQP_POLISH = True
OSQP_TOL = 1e-3

if (B_generate_main_results or 
    B_generate_mpc_results or 
    B_reachability_comparison):
    # get constraints padding
    with open('results/satellite_epsilons.npy', 
        'rb') as f:
        epsilons_omega_padding = np.load(f)
        epsilons_control_padding = np.load(f)
        H_bar_one_step_taylor_method = np.load(f)
        print("epsilons_omegas =", epsilons_omega_padding)
        print("epsilons_control =", epsilons_control_padding)

# constants
J_inertia = jnp.diag( # inertia matrix
    jnp.array([5., 2., 1.]))
J_inertia_inverse = jnp.diag( # inertia matrix inverse
    jnp.array([1. / 5., 1. / 2., 1.]))
u_max = 0.1 # max control torque
omega_max = 0.1 # max angular velocity
R = 1 * np.eye(n_u) # control cost matrix
Q = 10 * np.eye(n_x) # state cost matrix
x_ref = jnp.array([0, 0, 0, 1., 0, 0, 0]) # reference state
# initial state
rpy_0 = np.array([180, 45, 45])
q_0 = Rotation.from_euler('xyz', rpy_0, degrees=True).as_quat()
omega_0 = jnp.array([-1, -4.5, 4.5]) * np.pi / 180.
x0 = jnp.concatenate((q_0, omega_0))
# parameters for feedback gain
Q_lqr = 1e1 * np.eye(3)
R_lqr = 1e0 * np.eye(n_u)
# -----------------------------------------

# -----------------------------------------------
# Disturbances
w_ball_radius = 0.01 # ball

@jit
def project(v, u):
    # projection onto the tangent space of the sphere
    # of radius $||v||$.
    u_projected = u - jnp.dot(v, u) * v
    return u_projected

@jit
def n_W(w):
    # outward-pointing normal vector of 
    # $\partial\mathcal{W}$ at $w$
    normal_vector = w / jnp.linalg.norm(w)
    return normal_vector

@jit
def n_W_inverse(n):
    # w\in\partial\mathcal{W}$ such that n_w(w)=n
    w = w_ball_radius * n
    return w

@jit
def pmp_disturbance_w_from_q(q):
    return n_W_inverse(q)

@jit
def pmp_disturbances_ws_from_qs_vec(qs_vec):
    return vmap(pmp_disturbance_w_from_q)(qs_vec)

@jit
def pmp_disturbances_ws_from_qs_mat(qs_mat):
    return vmap(vmap(pmp_disturbance_w_from_q))(qs_mat)
# -----------------------------------------

# -----------------------------------------
class Model:
    def __init__(self, M):
        # initial / final states
        self.x0 = jnp.zeros(n_x)

        # M initial disturbance values w^i(0)
        w0s = fibonacci_lattice(w_ball_radius, M)
        self.initial_disturbances = w0s # (M, 3)

        self.compute_feedback_gains()

    # dynamics
    @partial(jit, static_argnums=(0,))
    def f(self, x, u):
        q, omega = x[:4], x[4:]
        ox, oy, oz = omega
        # matrices 
        Omega = 0.5 * jnp.array([
            [0, -ox, -oy, -oz],
            [ox, 0, oz, -oy],
            [oy, -oz, 0, ox],
            [oz, oy, -ox, 0]])
        omega_cross = jnp.array([
            [0, -oz, oy],
            [oz, 0, -ox],
            [-oy, ox, 0]])
        # dynamics
        q_dot = Omega @ q
        omega_dot = J_inertia_inverse @ (
            u - omega_cross @ J_inertia @ omega)
        x_dot = jnp.concatenate((q_dot, omega_dot), 
            axis=-1)
        return x_dot

    @partial(jit, static_argnums=(0,))
    def f_dx(self, x, u):
        return jacrev(self.f, argnums=(0))(x, u)

    @partial(jit, static_argnums=(0,))
    def f_du(self, x, u):
        return jacrev(self.f, argnums=(1))(x, u)

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_open_loop(self, x, u):
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5*dt*k1, u)
        k3 = self.f(x + 0.5*dt*k2, u)
        k4 = self.f(x + dt*k3, u)
        xn = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4) * dt
        return xn

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_open_loop_dx(self, x, u):
        return jacrev(self.next_nominal_state_open_loop, argnums=(0))(x, u)

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_open_loop_du(self, x, u):
        return jacrev(self.next_nominal_state_open_loop, argnums=(1))(x, u)

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_open_loop_dxu(self, x, u):
        J_dx, J_du = jacrev(self.next_nominal_state_open_loop, argnums=(0, 1))(x, u)
        J = jnp.concatenate((J_dx, J_du), axis=-1)
        return J

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_open_loop_hessian_ddxu(self, x, u):
        H_ddx, H_ddu = jacrev(self.next_nominal_state_open_loop_dxu, argnums=(0, 1))(x, u)
        H = jnp.concatenate((H_ddx, H_ddu), axis=-1)
        return H

    # feedback gains: LQR controller
    def compute_feedback_gains(self):
        A = self.next_nominal_state_open_loop_dx(x_ref+1e-4, np.zeros(n_u)+1e-4)
        B = self.next_nominal_state_open_loop_du(x_ref+1e-4, np.zeros(n_u)+1e-4)
        A = A[4:, 4:]
        B = B[4:, :]
        # feedback only on omega
        K = np.zeros((n_u, 3))
        P = Q_lqr
        for t in range(S-1, -1, -1):
            AtP = A.T @ P
            AtPA = AtP @ A
            AtPB = AtP @ B
            RplusBtPB_inv = np.linalg.inv(R_lqr + B.T @ P @ B)
            P = Q_lqr + AtPA - AtPB @ RplusBtPB_inv @ (AtPB.T)
        K = -RplusBtPB_inv @ (B.T @ P @ A)
        self.feedback_gain = jnp.array(K)
        return self.feedback_gain

    @partial(jit, static_argnums=(0,))
    def closed_loop_control(self, 
        u_bar, omega):
        # u_nom - (n_u,)
        # omega - (3,)
        linear_feedback_term = self.feedback_gain @ omega
        u = u_bar + linear_feedback_term
        return u

    @partial(jit, static_argnums=(0,))
    def closed_loop_control_trajectory(self, 
        us_nom_mat, omegas):
        # us_nom_mat - (S, n_u)
        # omegas - (S+1, 3)
        us_mat = vmap(self.closed_loop_control)(us_nom_mat, omegas[:-1, :])
        return us_mat

    # ---------------------------------
    # Optimization variable z
    #   z = (xs_vec, us_vec)
    # where
    # - xs_vec is of shape (S+1)*n_x
    # ----- nominal state trajectory
    # - us_vec is of shape S*n_u
    # ----- nominal control trajectory
    @partial(jit, static_argnums=(0,))
    def convert_z_to_variables(self, z):
        xs_vec = z[:((S+1)*n_x)]
        us_vec = z[((S+1)*n_x):]
        return xs_vec, us_vec

    @partial(jit, static_argnums=(0,))
    def convert_xs_vec_to_xs_mat(self, xs_vec):
        xs_mat = jnp.reshape(xs_vec, (n_x, S+1), 'F')
        xs_mat = xs_mat.T # (S+1, n_x)
        xs_mat = jnp.array(xs_mat)
        return xs_mat

    @partial(jit, static_argnums=(0,))
    def convert_us_vec_to_us_mat(self, us_vec):
        us_mat = jnp.reshape(us_vec, (n_u, S), 'F')
        us_mat = us_mat.T # (S, n_u)
        us_mat = jnp.array(us_mat)
        return us_mat

    @partial(jit, static_argnums=(0,))
    def convert_us_mat_to_us_vec(self, us_mat):
        us_vec = jnp.reshape(us_mat, (S*n_u), 'C')
        return us_vec

    @partial(jit, static_argnums=(0,))
    def convert_z_to_xs_us_mats(self, z):
        xs_vec, us_vec, _, _ = self.convert_z_to_variables(z)
        xs_mat = self.convert_xs_vec_to_xs_mat(xs_vec)
        us_mat = self.convert_us_vec_to_us_mat(us_vec)
        return xs_mat, us_mat

    def initial_guess(self):
        z = np.concatenate((
            np.tile(x0, S+1) + 1e-6, 
            np.zeros(S*n_u) + 1e-6), axis=-1)
        return jnp.array(z)
    # ---------------------------------

    # ---------------------------------
    # nominal dynamics
    @partial(jit, static_argnums=(0,))
    def f_closed_loop(self, x, u):
        omega = x[4:]
        u_with_feedback = self.closed_loop_control(u, omega)
        x_dot = self.f(x, u_with_feedback)
        return x_dot

    @partial(jit, static_argnums=(0,))
    def f_closed_loop_dx(self, x, u):
        return jacrev(self.f_closed_loop, argnums=(0))(x, u)

    @partial(jit, static_argnums=(0,))
    def f_closed_loop_du(self, x, u):
        return jacrev(self.f_closed_loop, argnums=(1))(x, u)

    @partial(jit, static_argnums=(0,))
    def next_state_closed_loop(self, x, u, w):
        w_term = jnp.concatenate((jnp.zeros(4), J_inertia_inverse @ w))
        k1 = self.f_closed_loop(x, u) + w_term
        k2 = self.f_closed_loop(x + 0.5*dt*k1, u) + w_term
        k3 = self.f_closed_loop(x + 0.5*dt*k2, u) + w_term
        k4 = self.f_closed_loop(x + dt*k3, u) + w_term
        xn = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4) * dt
        return xn

    @partial(jit, static_argnums=(0,))
    def state_closed_loop_trajectory(self, us_mat, x0, ws_mat):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # ws_mat - (S, n_x)
        # returns
        # xs_mat - (S+1, n_x)
        def next_state_closed_loop_fori_loop(t, us_ws_states):
            us_mat, ws_mat, states = us_ws_states
            u_t, w_t, x_t = us_mat[t, :], ws_mat[t, :], states[t, :]
            x_n = self.next_state_closed_loop(
                x_t, u_t, w_t)
            states = states.at[t+1].set(x_n)
            return (us_mat, ws_mat, states)
        states = jnp.zeros((S+1, n_x))
        states = states.at[0, :].set(x0)
        us_ws_states = (us_mat, ws_mat, states)
        _, _, states = fori_loop(0, S, 
            next_state_closed_loop_fori_loop, us_ws_states)
        return states

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_closed_loop(self, x, u):
        k1 = self.f_closed_loop(x, u)
        k2 = self.f_closed_loop(x + 0.5*dt*k1, u)
        k3 = self.f_closed_loop(x + 0.5*dt*k2, u)
        k4 = self.f_closed_loop(x + dt*k3, u)
        xn = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4) * dt
        return xn

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_closed_loop_dx(self, x, u):
        return jacrev(self.next_nominal_state_closed_loop, argnums=(0))(x, u)

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_closed_loop_du(self, x, u):
        return jacrev(self.next_nominal_state_closed_loop, argnums=(1))(x, u)

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_closed_loop_dxu(self, x, u):
        J_dx, J_du = jacrev(self.next_nominal_state_closed_loop, argnums=(0, 1))(x, u)
        J = jnp.concatenate((J_dx, J_du), axis=-1)
        return J

    @partial(jit, static_argnums=(0,))
    def next_nominal_state_closed_loop_hessian_ddxu(self, x, u):
        H_ddx, H_ddu = jacrev(self.next_nominal_state_closed_loop_dxu, argnums=(0, 1))(x, u)
        H = jnp.concatenate((H_ddx, H_ddu), axis=-1)
        return H

    @partial(jit, static_argnums=(0,))
    def nominal_initial_constraints(self, x0):
        Aeq = jnp.zeros((n_x, (S+1)*n_x + S*n_u))
        Aeq = Aeq.at[:, :n_x].set(jnp.eye(n_x))
        leq = x0
        ueq = leq
        return Aeq, leq, ueq

    @partial(jit, static_argnums=(0,))
    def nominal_dynamics_constraints(self, xs_mat, us_mat):
        def nominal_dynamics_constraint(x, u):
            # xn = f_next(x, u)
            # => xn = (f_next(xp, up) + f_next_dx(xp, up) @ (x-xp)
            #                         + f_next_du(xp, up) @ (u-up))) 
            # => An @ xn + A @ x + B @ u = constant
            f_val = self.next_nominal_state_closed_loop(x, u)
            f_dx_val = self.next_nominal_state_closed_loop_dx(x, u)
            f_du_val = self.next_nominal_state_closed_loop_du(x, u)
            A = f_dx_val
            An = -jnp.eye(n_x)
            B = f_du_val
            constant = - (f_val - f_dx_val @ x - f_du_val @ u)
            return A, B, An, constant
        As, Bs, Ans, cs = vmap(nominal_dynamics_constraint)(xs_mat[:-1], us_mat)
        Aeq = jnp.zeros((S*n_x, (S+1)*n_x + S*n_u))
        leq = jnp.zeros(S*n_x)
        for t in range(S):
            idx_con = t*n_x
            idx_x  = t*n_x
            idx_xn = idx_x + n_x
            idx_u  = (S+1)*n_x + t*n_u
            idx_un = idx_u + n_u
            Aeq = Aeq.at[idx_con:(idx_con+n_x), idx_x:idx_xn].set(As[t]) # x_{t}
            Aeq = Aeq.at[idx_con:(idx_con+n_x), idx_u:idx_un].set(Bs[t]) # u_{t}
            Aeq = Aeq.at[idx_con:(idx_con+n_x), idx_xn:(idx_xn+n_x)].set(Ans[t]) # x_{t+1}
            leq = leq.at[idx_con:(idx_con+n_x)].set(cs[t])
        ueq = leq
        return Aeq, leq, ueq
    # ---------------------------------

    # ---------------------------------
    # robust dynamics, i.e., trajectories that result from 
    # integrating the augmented ODE from initial disturbances w(0)
    @partial(jit, static_argnums=(0,))
    def f_omega(self, omega, u):
        x = jnp.concatenate((
            jnp.zeros(4), omega), axis=-1)
        omega_dot = self.f(x, u)[4:]
        return omega_dot

    @partial(jit, static_argnums=(0,))
    def f_omega_domega(self, omega, u):
        return jacrev(self.f_omega, argnums=(0))(omega, u)

    @partial(jit, static_argnums=(0,))
    def augmented_dynamics(self, u_bar, omega, q):
        # control with feedback
        linear_feedback_term = self.feedback_gain @ omega
        u = u_bar + linear_feedback_term
        # optimal disturbance
        wstar = pmp_disturbance_w_from_q(q)
        # f(x) + g(x)w where g(x) = J_inv
        J, Jinv = J_inertia, J_inertia_inverse
        x_dot = self.f_omega(omega, u) + Jinv @ wstar
        # dynamics of additional state q
        q_dot = -project(q, 
            Jinv.T @ self.f_omega_domega(omega, u).T @ J.T @ q)
        return (x_dot, q_dot)

    @partial(jit, static_argnums=(0,))
    def next_augmented_state(self, u_bar, o, q):
        # note: o = omega is of shape (3,)
        k1_o, k1_q = self.augmented_dynamics(u_bar, o, q)
        k2_o, k2_q = self.augmented_dynamics(u_bar, o + 0.5*dt*k1_o, q + 0.5*dt*k1_q)
        k3_o, k3_q = self.augmented_dynamics(u_bar, o + 0.5*dt*k2_o, q + 0.5*dt*k2_q)
        k4_o, k4_q = self.augmented_dynamics(u_bar, o + dt*k3_o, q + dt*k3_q)
        on = o + (1.0 / 6.0) * (k1_o + 2*k2_o + 2*k3_o + k4_o) * dt
        qn = q + (1.0 / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q) * dt
        return (on, qn)

    @partial(jit, static_argnums=(0,))
    def extremal_trajectory(self, us_mat, x0, w0):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0     - (n_x,)
        def next_augmented_state_fori_loop(t, us_omegas_qs):
            us_mat, omegas, qs = us_omegas_qs
            u_t, o_t, q_t = us_mat[t, :], omegas[t, :], qs[t, :]
            o_n, q_n = self.next_augmented_state(
                u_t, o_t, q_t)
            omegas = omegas.at[t+1].set(o_n)
            qs = qs.at[t+1].set(q_n)
            return (us_mat, omegas, qs)
        q0 = n_W(w0)
        omegas = jnp.zeros((S+1, 3))
        omegas = omegas.at[0, :].set(x0[4:])
        qs = jnp.zeros((S+1, 3))
        qs = qs.at[0].set(q0)
        us_omegas_qs = (us_mat, omegas, qs)
        _, omegas, qs = fori_loop(0, S, 
            next_augmented_state_fori_loop, us_omegas_qs)
        return omegas, qs

    @partial(jit, static_argnums=(0,))
    def extremal_omega_trajectory(self, us_mat, x0, w0):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0     - (n_x,)
        omegas, qs = self.extremal_trajectory(us_mat, x0, w0)
        return omegas

    @partial(jit, static_argnums=(0,))
    def extremal_disturbance_trajectory(self, us_mat, x0, w0):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0     - (n_x,)
        omegas, qs = self.extremal_trajectory(us_mat, x0, w0)
        ws = pmp_disturbances_ws_from_qs_vec(qs)
        return ws

    @partial(jit, static_argnums=(0,))
    def extremal_omega_trajectory_dw0(self, 
        us_mat, x0, w0):
        J_dw0 = jacfwd(self.extremal_omega_trajectory,
            argnums=(2))(us_mat, x0, w0)
        return J_dw0

    @partial(jit, static_argnums=(0,))
    def extremal_omega_trajectory_hessian_ddw0(self, 
        us_mat, x0, w0):
        H_dw0 = jacfwd(self.extremal_omega_trajectory_dw0,
            argnums=(2))(us_mat, x0, w0)
        return H_dw0

    @partial(jit, static_argnums=(0,))
    def extremal_omega_trajectories(self, us_mat, x0, w0s):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0s    - (M, 3)
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)
        Omegas = vmap(self.extremal_omega_trajectory)(
            Us, X0s, w0s)
        return Omegas

    def extremal_omega_trajectories_dw0s(self, 
        us_tensor, x0s, w0s):
        # us_tensor - (M, S, n_u)
        # x0s       - (M, n_x)
        # w0s       - (M, 3)
        Omegas_dw0s = vmap(self.extremal_omega_trajectory_dw0)(
            us_tensor, x0s, w0s)
        return Omegas_dw0s

    def extremal_omega_trajectories_hessian_ddw0s(self, 
        us_tensor, x0s, w0s):
        # us_tensor - (M, S, n_u)
        # x0s       - (M, n_x)
        # w0s       - (M, 3)
        Omegas_ddw0s = vmap(self.extremal_omega_trajectory_hessian_ddw0)(
            us_tensor, x0s, w0s)
        return Omegas_ddw0s

    @partial(jit, static_argnums=(0,))
    def extremal_closed_loop_control_trajectory(self,
        us_mat, x0, w0):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0     - (n_x,)
        omegas = self.extremal_omega_trajectory(us_mat, x0, w0)
        us = self.closed_loop_control_trajectory(us_mat, omegas)
        return us

    @partial(jit, static_argnums=(0,))
    def extremal_closed_loop_control_trajectory_dw0(self, 
        us_mat, x0, w0):
        J_dw0 = jacfwd(self.extremal_closed_loop_control_trajectory,
            argnums=(2))(us_mat, x0, w0)
        return J_dw0

    @partial(jit, static_argnums=(0,))
    def extremal_closed_loop_control_trajectory_hessian_ddw0(self, 
        us_mat, x0, w0):
        H_dw0 = jacfwd(self.extremal_closed_loop_control_trajectory_dw0,
            argnums=(2))(us_mat, x0, w0)
        return H_dw0

    @partial(jit, static_argnums=(0,))
    def extremal_closed_loop_control_trajectories(self,
        us_mat, x0, w0s):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0s    - (M, S, n_x)
        X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        Us = vmap(self.extremal_closed_loop_control_trajectory)(
            Us, X0s, w0s)
        return Us

    def extremal_closed_loop_control_trajectories_dw0s(self, 
        us_tensor, x0s, w0s):
        # us_tensor - (M, S, n_u)
        # x0s       - (M, n_x)
        # w0s       - (M, S, n_x)
        controls_dw0s = vmap(self.extremal_closed_loop_control_trajectory_dw0)(
            us_tensor, x0s, w0s)
        return controls_dw0s

    def extremal_closed_loop_control_trajectories_hessian_ddw0s(self, 
        us_tensor, x0s, w0s):
        # us_tensor - (M, S, n_u)
        # x0s       - (n_x,)
        # w0s       - (M, S, n_x)
        X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)
        controls_ddw0s = vmap(self.extremal_closed_loop_control_trajectory_hessian_ddw0)(
            us_tensor, x0s, w0s)
        return controls_ddw0s

    @partial(jit, static_argnums=(0,))
    def extremal_disturbance_trajectories(self, us_mat, x0, w0s):
        # us_mat - (S, n_u)
        # x0     - (n_x,)
        # w0     - (n_x,)
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)
        Ws = vmap(self.extremal_disturbance_trajectory)(
            Us, X0s, w0s)
        return Ws
    # ---------------------------------

    # ----------------------------------------------------
    # OSQP
    @partial(jit, static_argnums=(0,))
    def get_angular_velocity_constraints(self, omegas):
        omegas = jnp.reshape(omegas, 3*(S+1), 'C')
        return omegas

    @partial(jit, static_argnums=(0,))
    def get_control_constraints(self, us_mat, omegas):
        # add feedback
        us_mat = self.closed_loop_control_trajectory(us_mat, omegas)
        us_vec = jnp.reshape(us_mat, S*n_u, 'C')
        return us_vec

    @partial(jit, static_argnums=(0,))
    def get_all_constraints_coeffs(self, 
        us_mat, x0, w0):
        # Returns (A, l, u) corresponding to all constraints
        # such that l <= A uvec <= u.
        def all_constraints(
            us_mat, x0, w0):
            omegas = self.extremal_omega_trajectory(us_mat, x0, w0)
            val_angular = self.get_angular_velocity_constraints(
                omegas)
            val_control = self.get_control_constraints(
                us_mat, omegas)
            return (val_angular, val_control)
        def all_constraints_dus(
            us_mat, x0, w0):
            jac = jacfwd(all_constraints)(
                us_mat, x0, w0)
            return jac

        val_omega, val_control = all_constraints(
            us_mat, x0, w0)
        val_omega_du, val_control_du = all_constraints_dus(
            us_mat, x0, w0)

        # reshape gradient
        val_omega_du = jnp.reshape(val_omega_du, ((S+1)*3, S*n_u), 'C')
        val_control_du = jnp.reshape(val_control_du, (S*n_u, S*n_u), 'C')

        # convert to a vector
        us_vec = self.convert_us_mat_to_us_vec(us_mat)
        # adjust epsilon padding in time to vector
        epsilons_omega = jnp.repeat(epsilons_omega_padding, 3)
        epsilons_control = jnp.repeat(epsilons_control_padding, n_u)

        # angular velocity constraints
        # omg_min <= omega(u) <= omg_max
        # => linearize at up and enforce
        #  omg_min <= omega(up) + grad_u omega(u) (u - up) <= omg_max
        val_omg = val_omega - val_omega_du @ us_vec
        val_omg_low = -omega_max - val_omg + epsilons_omega
        val_omg_up = omega_max - val_omg - epsilons_omega

        # control constraints (same as before)
        val_con = val_control - val_control_du @ us_vec
        val_con_low = -u_max - val_con + epsilons_control
        val_con_up = u_max - val_con - epsilons_control

        return (val_omega_du, val_omg_low, val_omg_up,
            val_control_du, val_con_low, val_con_up)

    @partial(jit, static_argnums=(0,))
    def get_all_constraints_coeffs_all(self, us_mat, x0):
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)

        constraints_vals = vmap(self.get_all_constraints_coeffs)(
            Us, X0s, self.initial_disturbances)
        (val_omg_du, val_omg_low, val_omg_up,
            val_con_du, val_con_low, val_con_up) = constraints_vals

        # stack constraints for each of the $M$ samples
        val_omg_du = jnp.reshape(val_omg_du, (M*(S+1)*3, S*n_u), 'C')
        val_omg_low = jnp.reshape(val_omg_low, (M*(S+1)*3), 'C')
        val_omg_up = jnp.reshape(val_omg_up, (M*(S+1)*3), 'C')
        val_con_du = jnp.reshape(val_con_du, (M*S*n_u, S*n_u), 'C')
        val_con_low = jnp.reshape(val_con_low, (M*S*n_u), 'C')
        val_con_up = jnp.reshape(val_con_up, (M*S*n_u), 'C')

        # combine all constraints
        constraints_du = jnp.vstack([val_omg_du, val_con_du])
        constraints_low = jnp.hstack([val_omg_low, val_con_low])
        constraints_up = jnp.hstack([val_omg_up, val_con_up])
        return constraints_du, constraints_low, constraints_up

    def get_solution_cost(self, us_mat, xs_mat):
        cost = 0.
        for t in range(S):
            xt, ut = xs_mat[t, :], us_mat[t, :]
            cost += dt * (xt - x_ref).T @ Q @ (xt - x_ref)
            cost += dt * ut.T @ R @ ut
        xN = xs_mat[-1, :]
        cost += dt * (xN - x_ref).T @ Q @ (xN - x_ref)
        return cost

    def get_objective_coeffs(self):
        # Returns (P, q) corresponding to objective
        #        min (1/2 z^T P z + q^T z)
        # where z = umat is the optimization variable.

        # (x-x_ref).T @ Q (x-x_ref)
        # => x.T @ Q @ x + (-2 * x_ref.T @ Q) @ x + constant
        #         (P)      (2  *    q       ) 
        # => pack onto z = (x_vec, u_vec)

        # Quadratic Objective
        P = sparse.block_diag([
            sparse.kron(eye(S+1), Q),
            sparse.kron(eye(S), R)], format='csc')
        # Linear Objective
        q = -2. * x_ref.T @ Q
        q = 0.5 * q
        q = np.tile(q, S+1)
        q = np.concatenate((q, np.zeros(S*n_u)), axis=-1)
        return P, q

    def get_constraints_coeffs(self, z, x0, scp_iter):
        xs_vec, us_vec = self.convert_z_to_variables(z)
        xs = self.convert_xs_vec_to_xs_mat(xs_vec)
        us = self.convert_us_vec_to_us_mat(us_vec)
        # Constraints: l <= A z <= u
        # nominal constraints
        A_x0, l_x0, u_x0 = self.nominal_initial_constraints(x0)
        A_dyn, l_dyn, u_dyn = self.nominal_dynamics_constraints(xs, us)
        # robust constraints
        A_rob, l_rob, u_rob = self.get_all_constraints_coeffs_all(us, x0)
        #
        A_rob = np.concatenate((
            np.zeros((A_rob.shape[0], n_x*(S+1))),
            A_rob), axis=-1)
        A_x0, A_dyn, A_rob = csr_matrix(A_x0), csr_matrix(A_dyn), csr_matrix(A_rob)
        A = vstack([A_x0, A_dyn, A_rob], format='csc')
        l = np.hstack([l_x0, l_dyn, l_rob])
        u = np.hstack([u_x0, u_dyn, u_rob])
        return A, l, u

    def define_problem(self, z, x0):
        # objective and constraints
        P, q = self.get_objective_coeffs()
        A, l, u = self.get_constraints_coeffs(
            z, x0, 0)
        # Setup OSQP problem
        self.osqp_prob = osqp.OSQP()
        self.osqp_prob.setup(
            P, q, A, l, u,
            eps_abs=OSQP_TOL, eps_rel=OSQP_TOL,
            # linsys_solver="qdldl",
            # linsys_solver="mkl pardiso",
            warm_start=True, verbose=False,
            polish=OSQP_POLISH)
        return True

    def update_problem(self, z, x0, scp_iter=0):
        A, l, u = self.get_constraints_coeffs(
            z, x0, scp_iter)
        self.osqp_prob.update(l=l, u=u)
        self.osqp_prob.update(Ax=A.data)
        return True

    def solve(self):
        self.res = self.osqp_prob.solve()
        if self.res.info.status != 'solved':
            print("[solve]: Problem infeasible.")
        z = self.res.x
        xs_vec, us_vec = self.convert_z_to_variables(z)
        xs_mat = self.convert_xs_vec_to_xs_mat(xs_vec)
        us_mat = self.convert_us_vec_to_us_mat(us_vec)
        return z, xs_mat, us_mat
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # ellipsoidal uncertainty propagation
    def get_next_ellipsoidal_uncertainty_set_omega(self, 
        mu_k, u_k, Q_k, H_bar):
        # mu_k : (3)
        # u_k  : (n_u)
        # Q_k  : (3, 3)
        # H_bar: (scalar) - hessian bound

        # linear component
        x_k = jnp.concatenate((jnp.zeros(4), mu_k))
        A = self.next_nominal_state_closed_loop_dx(x_k, u_k)[4:, 4:]
        Q_nom = A @ Q_k @ A.T
        # additive disturbance
        delta_w = dt * w_ball_radius
        # linearization error bound via Taylor remainder
        eigs_Q, vecs = jnp.linalg.eigh(Q_k)
        eig_max_Q = jnp.max(eigs_Q)
        delta_x_taylor = H_bar * eig_max_Q**2 / 2.0
        # disturbance + linearization parts
        # Note: it is not sqrt(dimension), there is a small typo in the original paper
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8619572
        # this is corrected in  https://github.com/befelix/safe-exploration/
        omega_dim = 3
        Q_disturbance_and_taylor = omega_dim * jnp.eye(omega_dim) * (delta_x_taylor + delta_w)**2
        # bound sum of ellipsoids
        # 1) nominal + (disturbance + linearization error)
        c  = jnp.sqrt(jnp.trace(Q_nom) / jnp.trace(Q_disturbance_and_taylor))
        Q_n = ((c + 1) / c) * Q_nom + (1 + c) * Q_disturbance_and_taylor
        return Q_n

    def get_ellipsoidal_uncertainty_tube_omega(self, 
        xs_nom, us_nom, H_bar):
        # xs_nom - (S+1, n_x)
        # us_nom - (S, n_u)
        #
        # returns ellipsoidal matrices such that 
        # (x(t)-xs_nom(t)).T @ Q(t)^{-1} @ (x(t)-xs_nom(t)) <= 1
        # for all times t
        # Note: we only propagate the omega uncertainty.
        Qs = jnp.zeros((S+1, 3, 3))
        Qs = Qs.at[1, :, :].set(dt * w_ball_radius**2 * np.eye(3))
        for t in range(1, S):
            omegat_nom, ut_nom, Qt = xs_nom[t, 4:], us_nom[t, :], Qs[t, :, :]
            Qn = self.get_next_ellipsoidal_uncertainty_set_omega(
                omegat_nom, ut_nom, Qt, H_bar)
            Qs = Qs.at[t+1, :, :].set(Qn)
        return Qs

    def get_ellipsoidal_uncertainty_tube_closed_loop_controls(self, 
        xs_nom, us_nom, H_bar):
        # xs_nom - (S+1, n_x)
        # us_nom - (S, n_u)
        us = self.closed_loop_control_trajectory(us_nom, xs_nom[:, 4:])
        Qs = self.get_ellipsoidal_uncertainty_tube_omega(xs_nom, us_nom, H_bar)
        for t in range(S+1):
            Q_omega = Qs[t, :, :]
            Q_control = self.feedback_gain @ Q_omega @ self.feedback_gain.T
            Qs = Qs.at[t, :, :].set(Q_control)
        return us, Qs
    # ---------------------------------

def L2_error_us(us_mat, us_mat_prev):
    # us_mat - (S, n_u)
    # us_mat_prev - (S, n_u)
    error = np.mean(np.linalg.norm(us_mat-us_mat_prev, axis=-1))
    error = error / np.mean(np.linalg.norm(us_mat, axis=-1))
    return error
# -----------------------------------------

model = Model(M)


if B_generate_main_results:
    print("-------------------------------------------")
    print("[spacecraft.py] >>> Generating main results")
    model = Model(M)
    # Initial compilation (JAX)
    z_prev = model.initial_guess()
    model.define_problem(
        z_prev, x0)
    for scp_iter in range(7):
        model.update_problem(
            z_prev, x0, scp_iter)
        z, xs, us = model.solve()
        us_prev = us
        z_prev = z
    # Solve the problem
    start_time = time()
    total_define_time = 0
    total_solve_time = 0
    z_prev = model.initial_guess()
    for scp_iter in range(num_scp_iters_max):
        print("scp_iter =", scp_iter)
        # define
        define_time = time()
        model.update_problem(
            z_prev, x0, scp_iter)
        total_define_time += time()-define_time
        # solve
        solve_time = time()
        z, xs, us = model.solve()
        total_solve_time += time()-solve_time
        # compute error
        L2_error = L2_error_us(us, us_prev)
        us_prev = us
        z_prev = z
        print("L2_error =", L2_error)
    print("Total elapsed = ", time()-start_time)
    print(">> defining: ", total_define_time)
    print(">> solving: ", total_solve_time)
    with open('results/satellite_traj.npy', 
        'wb') as f:
        np.save(f, xs.to_py())
        np.save(f, us.to_py())
    print("-----------------------------------------")



if B_plot_main_results:
    print("-----------------------------------------")
    print("[spacecraft.py] >>> Plotting main results")
    with open('results/satellite_traj.npy', 
        'rb') as f:
        xs = np.load(f)
        us = np.load(f)
    # get extremal angular velocities
    omegas = model.extremal_omega_trajectories(
        us, x0, model.initial_disturbances)
    us_closed_loop = model.extremal_closed_loop_control_trajectories(
        us, x0, model.initial_disturbances)
    ts = dt*np.arange(S+1)

    fig = plt.figure(figsize=[6,3])
    colors = ['r', 'g', 'b', 'm']
    # q-trajectory
    plt.plot(ts, xs[:, 0], color=colors[0], label=r'$q_1$')
    plt.plot(ts, xs[:, 1], color=colors[1], label=r'$q_2$')
    plt.plot(ts, xs[:, 2], color=colors[2], label=r'$q_3$')
    plt.plot(ts, xs[:, 3], color=colors[3], label=r'$q_4$')
    for i in range(4):
        # reference
        plt.plot(
            ts,
            np.ones(S+1) * x_ref[i],
            'k--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$q(t)$', fontsize=24, rotation=0, labelpad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([-0.55, 1.05])
    plt.legend(loc='center',
        fontsize=20, handletextpad=0.5, 
        labelspacing=0.2, handlelength=1)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6,3])
    for j in range(3):
        # omega-trajectory
        plt.plot(
            ts,
            xs[:, 4+j],
            label=str(j), color=colors[j])
        # reference
        plt.plot(
            ts,
            np.ones(S+1) * x_ref[4+j],
            'k--')
        # extremal trajectories
        for i in range(M):
            plt.plot(
                ts,
                omegas[i, :, j],
                color=colors[j], alpha=0.05)
    plt.plot(
        ts,
        -omega_max * np.ones(S+1),
        'r--')
    plt.plot(
        ts,
        omega_max * np.ones(S+1),
        'r--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$\omega(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6,3])
    plt.plot(
        ts,
        -u_max * np.ones(S+1),
        'r--')
    plt.plot(
        ts,
        u_max * np.ones(S+1),
        'r--')
    # extremal closed loop controls
    for j in range(3):
        for i in range(M):
            last_control = (us[-1, :] + 
                model.feedback_gain @ omegas[i, -1, :])
            plt.plot(
                ts,
                np.concatenate((us_closed_loop[i, :, j], last_control[j:j+1])),
                color=colors[j], alpha=0.05)
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$u(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()
    plt.show()
    print("-----------------------------------------")



if B_reachability_comparison:
    print("--------------------------------------------")
    print("[spacecraft.py] >>> Comparing reachable sets")
    model = Model(M)
    with open('results/satellite_traj.npy',
        'rb') as f:
        xs = np.load(f)
        us = np.load(f)
    omegas = model.extremal_omega_trajectories(
        us, x0, model.initial_disturbances)
    us_closed_loop = model.extremal_closed_loop_control_trajectories(
        us, x0, model.initial_disturbances)
    ts = dt*np.arange(S+1)

    # Lipschitz-based method
    Qs_omega_taylor = model.get_ellipsoidal_uncertainty_tube_omega(
        xs, us, H_bar_one_step_taylor_method)
    us_taylor, Qs_control_taylor = model.get_ellipsoidal_uncertainty_tube_closed_loop_controls(
        xs, us, H_bar_one_step_taylor_method)

    # naive sampling-based method 
    X0s = jnp.repeat(x0[jnp.newaxis, :], M, axis=0) # (M, n_x)
    Us = jnp.repeat(us[jnp.newaxis, :, :], M, axis=0)
    Ws = w_ball_radius * sample_pts_unit_ball(3, M * S)
    Ws = np.reshape(Ws, (M, S, 3))
    omegas_naive = vmap(model.state_closed_loop_trajectory)(
        Us, X0s, Ws)[:, :, 4:8]
    us_closed_loop_naive = vmap(model.closed_loop_control_trajectory)(
        Us, omegas_naive)

    colors = ['r', 'g', 'b', 'm']
    fig = plt.figure(figsize=[5, 5])
    for j in range(2, 3):
        # proposed approach (extremal trajectories)
        ys_min = jnp.min(omegas[:, :, j], axis=0)
        ys_max = jnp.max(omegas[:, :, j], axis=0)
        if j == 2:
            plt.fill_between(ts, ys_min, ys_max, 
                facecolor=(0,0,1,0.0), edgecolor=(0,0,1,1.0),
                label=r'$\textrm{Algorithm 1}$')
        else:
            plt.plot(ts, ys_min, colors[j])
            plt.plot(ts, ys_max, colors[j])
        # Lipschitz-based tube
        ys_min = xs[:, 4+j] - np.sqrt(Qs_omega_taylor[:, 4+j, 4+j])
        ys_max = xs[:, 4+j] + np.sqrt(Qs_omega_taylor[:, 4+j, 4+j])
        if j == 2:
            # plt.fill_between(ts, ys_min, ys_max, alpha=0.15, color=colors[j],
            #     label=r'$\textrm{Lipschitz tube}$')
            plt.fill_between(ts, ys_min, ys_max, 
                facecolor=(0,0,1,0.15), edgecolor=(0,0,1,0.5), linestyle='dashed',
                label=r'$\textrm{Lipschitz tube}$')
        else:
            plt.fill_between(ts, ys_min, ys_max, alpha=0.15, color=colors[j])
        # naive sampling
        ys_min = jnp.min(omegas_naive[:, :, j], axis=0)
        ys_max = jnp.max(omegas_naive[:, :, j], axis=0)
        if j == 2:
            plt.fill_between(ts, ys_min, ys_max, 
                alpha=0.4, color=colors[j], hatch="X", edgecolor=colors[j], linewidth=0.5,
                label=r'$\textrm{Naive sampling}$')
        else:
            plt.fill_between(ts, ys_min, ys_max, 
                alpha=0.4, color=colors[j], hatch="X", edgecolor=colors[j], linewidth=0.5)
    plt.plot(ts, -omega_max * np.ones(S+1), 'r--')
    plt.plot(ts, omega_max * np.ones(S+1), 'r--')
    plt.text(6.5, 1.05*omega_max, r'$\omega_{\textrm{max}}$', 
        fontsize=24, color='r')
    plt.xlabel(r'$t$', fontsize=24)
    # plt.ylabel(r'$\omega_i(t)$', fontsize=24, rotation=0, labelpad=24)
    plt.title(r'\textrm{reachable set estimates for }$\omega_3(t)$', 
        fontsize=24, pad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.legend(fontsize=18, loc='lower left')
    plt.ylim((0.04, 0.12))
    plt.tight_layout()

    fig = plt.figure(figsize=[5, 5])
    plt.plot(ts, -u_max * np.ones(S+1), 'r--')
    plt.plot(ts, u_max * np.ones(S+1), 'r--')
    for j in range(2, 3):
        # proposed approach (extremal trajectories)
        ys_min = jnp.min(us_closed_loop[:, :, j], axis=0)
        ys_max = jnp.max(us_closed_loop[:, :, j], axis=0)
        ys_min = np.concatenate((ys_min, ys_min[-1:]))
        ys_max = np.concatenate((ys_max, ys_max[-1:]))
        if j == 2:
            # plt.fill_between(ts, ys_min, ys_max, 
            #     alpha=0., color=colors[j],
            #     label=r'$\textrm{Algorithm 1}$')
            # plt.plot(ts, ys_min, colors[j])
            # plt.plot(ts, ys_max, colors[j])
            plt.fill_between(ts, ys_min, ys_max, 
                facecolor=(0, 0, 1, 0.0), edgecolor=(0,0,1,1.0),
                label=r'$\textrm{Algorithm 1}$')
        # Lipschitz-based tube
        us_bar = np.concatenate((us_taylor, us_taylor[-1:, :]))[:, j]
        ys_min = us_bar - np.sqrt(Qs_control_taylor[:, 4+j, 4+j])
        ys_max = us_bar + np.sqrt(Qs_control_taylor[:, 4+j, 4+j])
        if j == 2:
            plt.fill_between(ts, ys_min, ys_max, 
                facecolor=(0,0,1,0.15), edgecolor=(0,0,1,0.5), linestyle='dashed',
                label=r'$\textrm{Lipschitz tube}$')
        else:
            plt.fill_between(ts, ys_min, ys_max, alpha=0.15, color=colors[j])
        # naive sampling
        ys_min = jnp.min(us_closed_loop_naive[:, :, j], axis=0)
        ys_max = jnp.max(us_closed_loop_naive[:, :, j], axis=0)
        ys_min = np.concatenate((ys_min, ys_min[-1:]))
        ys_max = np.concatenate((ys_max, ys_max[-1:]))
        if j == 2:
            plt.fill_between(ts, ys_min, ys_max, 
                alpha=0.4, color=colors[j], hatch="X", edgecolor=colors[j], linewidth=0.5,
                label=r'$\textrm{Naive sampling}$')
        else:
            plt.fill_between(ts, ys_min, ys_max, 
                alpha=0.4, color=colors[j], hatch="X", edgecolor=colors[j], linewidth=0.5)

    plt.xlabel(r'$t$', fontsize=24)
    # plt.ylabel(r'$u_i(t)$', fontsize=24, rotation=0, labelpad=16)
    plt.title(r'\textrm{reachable set estimates for }$u_3(t)$', 
        fontsize=24, pad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()
    plt.legend(fontsize=18, loc='lower left')
    plt.ylim((-0.06, 0.025))
    plt.show()
    print("--------------------------------------------")



if B_main_plot:
    print("----------------------------------------")
    print("[spacecraft.py] >>> Plotting main figure")
    M = 40
    S = 17
    # S = 30
    model = Model(M)
    with open('results/satellite_traj.npy',
        'rb') as f:
        xs = np.load(f)
        us = np.load(f)
    x0 = np.zeros(n_x)
    thetas1 = np.linspace(0, 2*np.pi, S)
    thetas2 = np.linspace(0, np.pi, S)
    # us = np.zeros((S, n_u))
    # us[:, 0] = 0.4 * u_max * np.cos(thetas1)
    # us[:, 1] = 0.3 * u_max * np.sin(thetas1)
    # us[:, 2] = 0.3 * u_max * np.cos(thetas2 + np.pi)
    us[15:, 2] *= 0.8
    # us[:, 0] = 1 * u_max * np.cos(thetas1)
    omegas = model.extremal_omega_trajectories(
        us, x0, model.initial_disturbances)
    ws = model.extremal_disturbance_trajectories(
        us, x0, model.initial_disturbances)
    ts = dt*np.arange(S+1)

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)

    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(projection='3d')
    ax.set_alpha(0.1)
    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
    cmap = plt.get_cmap('jet') # winter, plasma, rainbow
    colors = [cmap(float(t)/S) for t in range(S+1)]
    ax.scatter(omegas[0, 0, 0], omegas[0, 0, 1], omegas[0, 0, 2], s=30, color='b')
    for i in range(M):
        points = np.array([omegas[i, :, 0], omegas[i, :, 1], omegas[i, :, 2]]).T
        points = points.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for t in range(S):
            segment = segments[t]
            line, = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                color=colors[t], linewidth=2, alpha=0.05)
            line.set_solid_capstyle('round')
    for t in range(S+1):
        if t>0:
            plot_3d_convex_hull(omegas[:, t, :], ax, colors[t])
    plt.title(r'extremal trajectories $x_w(t)$', fontsize=24, pad=-30)
    ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.01))
    ax.w_xaxis.line.set_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_yaxis.line.set_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_zaxis.line.set_color((0.5, 0.5, 0.5, 0.01))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.tick_params(axis='x', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.tick_params(axis='y', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.tick_params(axis='z', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    # ax.set_zlabel(r'$z$', fontsize=30, rotation=0)
    ax.view_init(30, 45)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.tight_layout()

    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(projection='3d')
    ax.set_alpha(0.1)
    cmap = plt.get_cmap('jet') # winter, plasma, rainbow
    colors = [cmap(float(t)/(S-1)) for t in range(S-1)]
    # plot W set
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = w_ball_radius * np.cos(u) * np.sin(v)
    y = w_ball_radius * np.sin(u) * np.sin(v)
    z = w_ball_radius * np.cos(v)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.gray)
    mappable.set_array(z)
    ax.plot_surface(x, y, z, 
        norm=mappable.norm, cmap=mappable.cmap, 
        linewidth=0, antialiased=True, alpha=0.1)
    # compute alphas
    w_reference_alpha = np.array([w_ball_radius, -w_ball_radius, w_ball_radius])
    dists = np.linalg.norm(ws[:, :, :] - w_reference_alpha, axis=-1)
    alphas = 1.0 - (dists-np.min(dists)) / (np.max(dists)-np.min(dists))
    alphas = np.sqrt(np.sqrt(alphas))
    #
    for i in range(M):
        ax.scatter(ws[i, 0, 0], ws[i, 0, 1], ws[i, 0, 2], 
            s=30, color='b', alpha=alphas[i, 0])
        # ax.plot(ws[i, :, 0], ws[i, :, 1], ws[i, :, 2], c=colors)
        points = np.array([ws[i, :, 0], ws[i, :, 1], ws[i, :, 2]]).T
        points = points.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for t in range(S-1):
            segment = segments[t]
            line, = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                color=colors[t], linewidth=3,
                alpha=alphas[i, t])
            line.set_solid_capstyle('butt')
    plt.title(r'extremal disturbances $w(t)$', fontsize=24, pad=-30)
    ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.01))
    ax.w_xaxis.line.set_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_yaxis.line.set_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_zaxis.line.set_color((0.5, 0.5, 0.5, 0.01))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.tick_params(axis='x', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.tick_params(axis='y', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.tick_params(axis='z', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.view_init(30, -45)
    plt.tight_layout()

    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(projection='3d')
    ax.set_alpha(0.1)
    cmap = plt.get_cmap('jet') # winter, plasma, rainbow    
    # plot W set
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = w_ball_radius * np.cos(u) * np.sin(v)
    y = w_ball_radius * np.sin(u) * np.sin(v)
    z = w_ball_radius * np.cos(v)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.gray)
    mappable.set_array(z)
    ax.plot_surface(x, y, z, 
        norm=mappable.norm, cmap=mappable.cmap, 
        linewidth=0, antialiased=True, alpha=0.1)
    ax.scatter(ws[:, 0, 0], ws[:, 0, 1], ws[:, 0, 2], s=50, color='b')
    ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.01))
    ax.w_xaxis.line.set_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_yaxis.line.set_color((0.5, 0.5, 0.5, 0.01)) 
    ax.w_zaxis.line.set_color((0.5, 0.5, 0.5, 0.01))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.tick_params(axis='x', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.tick_params(axis='y', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.tick_params(axis='z', which=u'both',length=3, color=[0.5, 0.5, 0.5, 0.01])
    ax.view_init(30, -45)
    plt.tight_layout()
    plt.show()
    print("-----------------------------------------")



if B_generate_mpc_results:
    print("------------------------------------------")
    print("[spacecraft.py] >>> Generating MPC results")
    model = Model(M)

    comp_time_define_per_iter = np.zeros((num_MPC_runs, MPC_horizon))
    comp_time_solve_per_iter = np.zeros((num_MPC_runs, MPC_horizon))

    ws = w_ball_radius * sample_pts_unit_ball(3, num_MPC_runs * MPC_horizon)
    ws = np.reshape(ws, (num_MPC_runs, MPC_horizon, 3))

    x0s = np.zeros((num_MPC_runs, n_x))
    x0s[:, :4] = sample_pts_unit_sphere(4, num_MPC_runs).T
    x0s[:, 4:] = np.random.uniform(-omega_max, omega_max, (num_MPC_runs, 3))

    us_closed_loop = jnp.zeros((num_MPC_runs, MPC_horizon, n_u))
    xs_closed_loop = jnp.zeros((num_MPC_runs, MPC_horizon + 1, n_x))
    for MPC_run in range(num_MPC_runs):
        print("MPC repeat =", MPC_run)
        xs_closed_loop = xs_closed_loop.at[MPC_run, 0, :].set(
            x0s[MPC_run])
        # Initially, fully solve the problem
        z_prev = model.initial_guess()
        model.define_problem(
            z_prev, x0)
        for scp_iter in range(num_scp_iters_max):
            model.update_problem(
                z_prev, x0, scp_iter)
            z, xs, us = model.solve()
            us_prev = us
            z_prev = z
        for t in range(MPC_horizon):
            xt = xs_closed_loop[MPC_run, t, :]
            # compute control input
            for scp_iter in range(1):
                define_time = time()
                model.update_problem(
                    z_prev, xt, scp_iter)
                define_time = time()-define_time

                solve_time = time()
                z, xs, us = model.solve()
                solve_time = time()-solve_time

                us_prev = us
                z_prev = z
                comp_time_define_per_iter[MPC_run, t] = define_time
                comp_time_solve_per_iter[MPC_run, t] = solve_time

            ut = us[0, :]
            wt = ws[MPC_run, t, :]
            # simulate the system
            us_closed_loop = us_closed_loop.at[MPC_run, t, :].set(
                model.closed_loop_control(ut, xt[4:]))
            xs_closed_loop = xs_closed_loop.at[MPC_run, t+1, :].set(
                model.next_state_closed_loop(xt, ut, wt))
    print("comp_time_define_per_iter =", 
        np.mean(comp_time_define_per_iter))
    print("comp_time_solve_per_iter =", 
        np.mean(comp_time_solve_per_iter))
    print("comp_time_total_per_iter =", 
        np.mean(comp_time_define_per_iter+comp_time_solve_per_iter))
    with open('results/satellite_mpc_traj.npy', 
        'wb') as f:
        np.save(f, xs_closed_loop.to_py())
        np.save(f, us_closed_loop.to_py())
        np.save(f, ws)
        np.save(f, comp_time_define_per_iter)
        np.save(f, comp_time_solve_per_iter)
    print("-----------------------------------------")



if B_plot_mpc_results:
    print("----------------------------------------")
    print("[spacecraft.py] >>> Plotting MPC results")
    with open('results/satellite_mpc_traj.npy', 
        'rb') as f:
        xs = np.load(f)
        us = np.load(f)
        ws = np.load(f)
    ts = dt*np.arange(MPC_horizon+1)

    fig = plt.figure(figsize=[6, 4])
    colors = ['r', 'g', 'b', 'm']
    # q-trajectory
    for MPC_run in range(num_MPC_runs):
        plt.plot(ts, xs[MPC_run, :, 0], color=colors[0], alpha=0.1)
        plt.plot(ts, xs[MPC_run, :, 1], color=colors[1], alpha=0.1)
        plt.plot(ts, xs[MPC_run, :, 2], color=colors[2], alpha=0.1)
        plt.plot(ts, xs[MPC_run, :, 3], color=colors[3], alpha=0.1)
    for i in range(4):
        # reference
        plt.plot(
            ts,
            np.ones(MPC_horizon+1) * x_ref[i],
            'k--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$q(t)$', fontsize=24, rotation=0, labelpad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([-1.05, 1.05])
    # plt.legend(loc='center',
    #     fontsize=20, handletextpad=0.5, 
    #     labelspacing=0.2, handlelength=1)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6, 4])
    for j in range(3):
        # omega-trajectory
        for MPC_run in range(num_MPC_runs):
            plt.plot(
                ts,
                xs[MPC_run, :, 4+j],
                label=str(j), color=colors[j], alpha=0.1)
        # reference
        plt.plot(
            ts,
            np.ones(MPC_horizon+1) * x_ref[4+j],
            'k--')
    plt.plot(
        ts,
        -omega_max * np.ones(MPC_horizon+1),
        'r--')
    plt.plot(
        ts,
        omega_max * np.ones(MPC_horizon+1),
        'r--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$\omega(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6, 4])
    plt.plot(
        ts,
        -u_max * np.ones(MPC_horizon+1),
        'r--')
    plt.plot(
        ts,
        u_max * np.ones(MPC_horizon+1),
        'r--')
        # extremal closed loop controls
    for j in range(3):
        for MPC_run in range(num_MPC_runs):
            plt.plot(
                ts,
                np.concatenate((
                    us[MPC_run, :, j], 
                    us[MPC_run, -1, j:j+1])),
                color=colors[j], alpha=0.1)
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$u(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()
    plt.show()
    print("-----------------------------------------")


# -----------------------------------------
if B_evaluate_bounds:
    print("---------------------------------------------")
    print("[spacecraft.py] >>> Evaluating epsilon bounds")
    # compute delta
    M = 100 # number of points
    r = w_ball_radius
    w0s = fibonacci_lattice(r, M)
    # Compute maximal minimal distance between points
    # as an conservative approximation of delta such 
    # that the M points of the lattice form a 
    # delta-covering.
    # Indeed, points are approximately evenly spread
    # on the surface of the sphere. Thus, the distance
    # between neighbors (the min distance for each
    # point) is approximately always the same. By 
    # taking the maximum such distance for each point,
    # we get an over-approximation of the value of
    # delta (we approximately get delta/2 for large 
    # numbers of samples M, so returning this maximum
    # distance for delta gives a conservative 
    # approximation). 
    dists = np.ones((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            pi, pj = w0s[i, :], w0s[j, :]
            dists[i, j] = np.linalg.norm(pi - pj)
    dists_min = np.min(dists, 1)
    dist_max = np.max(dists_min)
    delta = dist_max
    print("delta =", delta)
    # -----------------------------------------

    # -----------------------------------------
    M = 10000
    model = Model(M)
    us_samples = np.random.choice(
        [-u_max, u_max], (M, S, n_u))
    x_min = np.concatenate((x_ref[:4], -omega_max*np.ones(3)))
    x_max = np.concatenate((x_ref[:4], omega_max*np.ones(3)))
    x0s_samples = np.random.uniform(x_min, x_max, 
        (M, n_x))
    w0_samples = fibonacci_lattice(w_ball_radius, M)
    w0_samples = jnp.array(w0_samples)
    print("w0s =", w0_samples.shape)
    omegas_dw0s = model.extremal_omega_trajectories_dw0s(
        us_samples, x0s_samples, w0_samples)
    print("omegas_dw0s =", omegas_dw0s.shape) # (M, 11, 3, 3)
    controls_dw0s = model.extremal_closed_loop_control_trajectories_dw0s(
        us_samples, x0s_samples, w0_samples)
    print("controls_dw0s =", controls_dw0s.shape) # (M, 11, 3, 3)
    # Lipschitz constants
    L_omg_bar = jnp.linalg.norm(omegas_dw0s, axis=(2, 3))
    L_con_bar = jnp.linalg.norm(controls_dw0s, axis=(2, 3))
    L_omg_bar = jnp.max(L_omg_bar, axis=(0))
    L_con_bar = jnp.max(L_con_bar, axis=(0))
    # Hausdorff distance error bounds
    epsilons_omg = L_omg_bar * delta
    epsilons_con = L_con_bar * delta
    print("error bound: Hausdorff epsilons (omega) =",
        epsilons_omg)
    print("error bound: Hausdorff epsilons (control) =",
        epsilons_con)
    # Bound on the hessian of one-step discrete-time dynamics
    H_bar = vmap(model.next_nominal_state_closed_loop_hessian_ddxu)(
        x0s_samples, us_samples[:, 0, :])
    H_bar = H_bar[:, 4:8, 4:8, 4:8] # only in omega
    # Frobenius norm
    H_bar = jnp.max(jnp.sum(np.abs(H_bar), axis=(1, 2, 3)))
    with open('results/satellite_epsilons.npy', 
        'wb') as f:
        np.save(f, epsilons_omg)
        np.save(f, epsilons_con)
        np.save(f, H_bar)
    print("-----------------------------------------")