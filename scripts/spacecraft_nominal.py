"""Optimal control with nominal spacecraft dynamics (without disturbances)/"""
from time import time
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from chreach.controlled_dynamics import SpacecraftDynamics
from chreach.sets import Point, Ball
from chreach.optimal_control import OptimalControlAlgorithm
from chreach.utils.plotting.spacecraft_plotting import \
    plot_optimal_control_solution, plot_mpc_solutions, plot_computation_times
from chreach.utils.stats import uniformly_sample_points_in_unit_sphere, \
    uniformly_sample_points_in_rectangle

np.random.seed(0)


# hyperparameters for MPC
MPC_sim_horizon = 3# 60
num_mpc_repeats = 3# 20
num_computation_times_repeats = 3# 20
sample_sizes_for_computation_times = [20, 30, 50]

# initial state
rpy_0 = jnp.array([180, 45, 45])
quat_0 = Rotation.from_euler('xyz', rpy_0, degrees=True).as_quat()
omega_0 = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.
initial_state = jnp.concatenate((quat_0, omega_0))
reference_state = jnp.array([0, 0, 0, 1., 0, 0, 0])

dynamics = SpacecraftDynamics()
num_states = dynamics.num_states
num_controls = dynamics.num_controls
num_disturbances = dynamics.num_disturbances
initial_states_set = Point(initial_state)
disturbances_set = Ball(jnp.zeros(num_disturbances), 0.01)

cost_parameters = {
    "reference_state": reference_state,
    "state_quad_penalization_matrix": 10 * jnp.eye(num_states),
    "control_quad_penalization_matrix": jnp.eye(num_controls)
}
constraints_parameters = {
    "max_control": 0.1, # assumes infinity norm constraints
    "max_state": 0.1, # assumes infinity norm constraints
    "start_index_state_constraints": 4
}
solver_parameters = {
    "initial_guess_state": initial_state,
    "num_scp_iterations": 10,
    "verbose": False,
    "osqp_verbose": False,
    "osqp_tolerance": 1e-3,
    "osqp_polish": True,
    "warm_start": True
}
discretization_time = 1.
horizon = 10
oc_alg = OptimalControlAlgorithm(
    dynamics,
    initial_states_set,
    disturbances_set,
    cost_parameters,
    constraints_parameters,
    solver_parameters,
    discretization_time,
    horizon)



print("Solving the open-loop problem.")
start_time = time()
opt_variables, states, controls = oc_alg.define_and_solve(
    oc_alg.initial_guess(),
    initial_state)
print("Elapsed time = ", time() - start_time)

plot_optimal_control_solution(
    discretization_time,
    horizon,
    constraints_parameters,
    states,
    controls,
    reference_state)



print("Running in an MPC loop.")
states_closed_loop = jnp.zeros((
    num_mpc_repeats, MPC_sim_horizon + 1, num_states))
controls_closed_loop = jnp.zeros((
    num_mpc_repeats, MPC_sim_horizon, num_controls))
computation_time_per_iteration_define = np.zeros((
    num_mpc_repeats, MPC_sim_horizon))
computation_time_per_iteration_solve = np.zeros((
    num_mpc_repeats, MPC_sim_horizon))

initial_states = np.zeros((num_mpc_repeats, num_states))
initial_states[:, :4] = uniformly_sample_points_in_unit_sphere(
    4,
    num_mpc_repeats)
initial_states[:, 4:] = uniformly_sample_points_in_rectangle(
    np.zeros(3),
    constraints_parameters['max_state'] * np.ones(3),
    num_mpc_repeats)

for mpc_run in range(num_mpc_repeats):
    print("MPC repeat =", mpc_run)
    initial_state = initial_states[mpc_run]
    states_closed_loop = states_closed_loop.at[mpc_run, 0, :].set(
        initial_state.copy())
    # Initially, fully solve the problem
    oc_alg.solver_parameters["initial_guess_state"] = initial_state
    oc_alg.solver_parameters["verbose"] = False
    opt_variables, states, controls = oc_alg.define_and_solve(
        oc_alg.initial_guess(),
        initial_state)
    for t in range(MPC_sim_horizon):
        initial_state = states_closed_loop[mpc_run, t, :]

        # compute control input (run a single SCP iteration)
        define_time = time()
        oc_alg.update_problem(opt_variables, initial_state)
        define_time = time() - define_time

        solve_time = time()
        output = oc_alg.solve_one_scp_iteration()
        opt_variables, states_matrix, controls_matrix = output
        solve_time = time() - solve_time

        computation_time_per_iteration_define[mpc_run, t] = define_time
        computation_time_per_iteration_solve[mpc_run, t] = solve_time

        # simulate the system
        controls_closed_loop = controls_closed_loop.at[mpc_run, t, :].set(
            controls_matrix[0])
        disturbance = jnp.zeros(dynamics.num_disturbances)
        next_state = dynamics.next_state(
            discretization_time * t,
            states_closed_loop[mpc_run, t, :],
            controls_closed_loop[mpc_run, t, :],
            disturbance,
            discretization_time)
        states_closed_loop = states_closed_loop.at[mpc_run, t+1, :].set(
            next_state)
print("computation_time_per_iteration_define =",
    np.mean(computation_time_per_iteration_define))
print("computation_time_per_iteration_solve =",
    np.mean(computation_time_per_iteration_solve))
print("comp_time_total_per_iter =", 
    np.mean(computation_time_per_iteration_define +
            computation_time_per_iteration_solve))

plot_mpc_solutions(
    num_mpc_repeats,
    discretization_time,
    MPC_sim_horizon,
    constraints_parameters,
    states_closed_loop,
    controls_closed_loop,
    reference_state)








print("Running multiple times to evaluate computation time.")
computation_time_per_iteration_define = np.zeros((
    num_computation_times_repeats,
    len(sample_sizes_for_computation_times),
    solver_parameters['num_scp_iterations']))
computation_time_per_iteration_solve = np.zeros_like(
    computation_time_per_iteration_define)
computation_times_cumulative = np.zeros_like(
    computation_time_per_iteration_define)
accuracy_errors = np.zeros_like(
    computation_time_per_iteration_define)

initial_state = jnp.concatenate((quat_0, omega_0))

oc_alg.solver_parameters["initial_guess_state"] = initial_state
for i in range(len(sample_sizes_for_computation_times)):
    for repeat in range(num_computation_times_repeats):
        opt_variables = oc_alg.initial_guess()
        controls_matrix_prev = oc_alg.convert_controls_vector_to_controls_matrix(
            oc_alg.opt_variables_to_states_controls_vectors(
                opt_variables)[1])

        oc_alg.define_problem(opt_variables, initial_state)
        for scp_iter in range(solver_parameters['num_scp_iterations']):
            # compute control input (run a single SCP iteration)
            define_time = time()
            oc_alg.update_problem(opt_variables, initial_state)
            define_time = time() - define_time

            solve_time = time()
            output = oc_alg.solve_one_scp_iteration()
            opt_variables, states_matrix, controls_matrix = output
            solve_time = time() - solve_time

            computation_time_per_iteration_define[i, repeat, scp_iter] = define_time
            computation_time_per_iteration_solve[i, repeat, scp_iter] = solve_time

            if scp_iter == 0:
                computation_times_cumulative[repeat, i, scp_iter] = (
                    define_time + solve_time)
            else:
                computation_times_cumulative[repeat, i, scp_iter] = (
                    computation_times_cumulative[repeat, i, scp_iter-1] +
                    define_time + solve_time)
            error = oc_alg.distance_L2_normalized_controls(
                controls_matrix, controls_matrix_prev)
            accuracy_errors[repeat, i, scp_iter] = error

            controls_matrix_prev = controls_matrix

plot_computation_times(
    solver_parameters['num_scp_iterations'],
    sample_sizes_for_computation_times,
    solver_parameters,
    computation_time_per_iteration_define,
    computation_time_per_iteration_solve,
    computation_times_cumulative,
    accuracy_errors)
