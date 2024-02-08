"""Code to reproduce spacecraft robust NMPC results."""
from time import time
from scipy.spatial.transform import Rotation
import numpy as np
from matplotlib import rc, rcParams
import jax.numpy as jnp
from jax import vmap, jit
from chreach.controlled_dynamics import SpacecraftDynamics
from chreach.sets import Point, Ball
from chreach.reach import Algorithm1, LipschitzReachabilityAlgorithm, RandUP
from chreach.optimal_control import RobustSpacecraftOptimalControlAlgorithm
from chreach.utils.plotting.spacecraft_plotting import \
    plot_optimal_control_solution, \
    plot_mpc_solutions, \
    plot_computation_times, \
    plot_reachable_sets
from chreach.utils.stats import uniformly_sample_points_in_unit_sphere, \
    uniformly_sample_points_in_rectangle
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)


np.random.seed(0)

estimate_error_bounds = True
solve_open_loop = True
solve_mpc = True
solve_compute_times = True
plot_open_loop = True
plot_mpc = True
plot_compute_times = True
compare_reachable_sets = True


# hyperparameters for MPC
MPC_sim_horizon = 60
num_mpc_repeats = 100
num_computation_times_repeats = 50
sample_sizes_for_computation_times = [25, 50, 100]

# initial state
rpy_0 = jnp.array([180, 45, 45])
quat_0 = Rotation.from_euler('xyz', rpy_0, degrees=True).as_quat()
omega_0 = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.
initial_state = jnp.concatenate((quat_0, omega_0))
reference_state = jnp.array([0, 0, 0, 1., 0, 0, 0])

feedback_gain = jnp.concatenate([
    jnp.zeros((3, 4)),
    -jnp.diag(jnp.array([5., 2, 1]))], axis=1)

dynamics = SpacecraftDynamics(feedback_gain=feedback_gain)
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
    "warm_start": True,
    "sample_size": 50,
}

discretization_time = 1.
horizon = 10



if estimate_error_bounds:
    print("Computing the theoretical error bound.")
    oc_alg = RobustSpacecraftOptimalControlAlgorithm(
        dynamics,
        initial_states_set,
        disturbances_set,
        cost_parameters,
        constraints_parameters,
        solver_parameters,
        discretization_time,
        horizon,
        100 * jnp.ones(horizon+1),
        100 * jnp.ones(horizon))
    epsilons_velocities, epsilons_controls = \
        oc_alg.velocity_reachability_alg.estimation_errors(
            discretization_time,
            horizon,
            sample_size=solver_parameters["sample_size"],
            max_control=constraints_parameters["max_control"])
    with open('results/spacecraft_epsilons.npy',
        'wb') as f:
        np.save(f, epsilons_velocities)
        np.save(f, epsilons_controls)
        np.save(f, solver_parameters["sample_size"])


with open('results/spacecraft_epsilons.npy',
    'rb') as f:
    error_bounds_velocities = np.load(f)
    error_bounds_controls = np.load(f)
    sample_size = np.load(f)
    print("error_bounds_velocities =", error_bounds_velocities)
    print("error_bounds_controls =", error_bounds_controls)
    print("sample_size =", sample_size)





if solve_open_loop:
    print("Solving the open-loop problem.")

    oc_alg = RobustSpacecraftOptimalControlAlgorithm(
        dynamics,
        initial_states_set,
        disturbances_set,
        cost_parameters,
        constraints_parameters,
        solver_parameters,
        discretization_time,
        horizon,
        error_bounds_velocities,
        error_bounds_controls)

    start_time = time()
    opt_variables, states, controls = oc_alg.define_and_solve(
        oc_alg.initial_guess(),
        initial_state)
    print("Elapsed time = ", time() - start_time)

    with open('results/spacecraft_openloop_solution.npy',
        'wb') as f:
        np.save(f, discretization_time)
        np.save(f, horizon)
        np.save(f, reference_state)
        np.save(f, opt_variables)
        np.save(f, states)
        np.save(f, controls)


if plot_open_loop:
    with open('results/spacecraft_openloop_solution.npy',
        'rb') as f:
        discretization_time = float(np.load(f))
        horizon = int(np.load(f))
        reference_state = np.load(f)
        opt_variables = np.load(f)
        states = np.load(f)
        controls = np.load(f)


    controls_closed_loop = vmap(
        dynamics.closed_loop_control)(
            states[:-1], controls)
    plot_optimal_control_solution(
        discretization_time,
        horizon,
        constraints_parameters,
        states,
        controls_closed_loop,
        reference_state)







if solve_mpc:
    print("Running in an MPC loop.")

    solver_parameters = {
        "initial_guess_state": initial_state,
        "num_scp_iterations": 10,
        "verbose": False,
        "osqp_verbose": False,
        "osqp_tolerance": 1e-3,
        "osqp_polish": True,
        "warm_start": True,
        "sample_size": 50,
    }
    oc_alg = RobustSpacecraftOptimalControlAlgorithm(
        dynamics,
        initial_states_set,
        disturbances_set,
        cost_parameters,
        constraints_parameters,
        solver_parameters,
        discretization_time,
        horizon,
        error_bounds_velocities,
        error_bounds_controls)

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
            disturbance = disturbances_set.sample_random(1).flatten()
            state = states_closed_loop[mpc_run, t, :]
            next_state = dynamics.next_state(
                discretization_time * t,
                state,
                controls_matrix[0],
                disturbance,
                discretization_time)
            states_closed_loop = states_closed_loop.at[mpc_run, t+1, :].set(
                next_state)
            controls_closed_loop = controls_closed_loop.at[mpc_run, t, :].set(
                dynamics.closed_loop_control(
                    state,
                    controls_matrix[0]))
    print("computation_time_per_iteration_define =",
        np.mean(computation_time_per_iteration_define))
    print("computation_time_per_iteration_solve =",
        np.mean(computation_time_per_iteration_solve))
    print("comp_time_total_per_iter =",
        np.mean(computation_time_per_iteration_define +
                computation_time_per_iteration_solve))

    with open('results/spacecraft_mpc.npy',
        'wb') as f:
        np.save(f, num_mpc_repeats)
        np.save(f, discretization_time)
        np.save(f, MPC_sim_horizon)
        np.save(f, states_closed_loop)
        np.save(f, controls_closed_loop)
        np.save(f, reference_state)
        np.save(f, computation_time_per_iteration_define)
        np.save(f, computation_time_per_iteration_solve)


if plot_mpc:
    with open('results/spacecraft_mpc.npy',
        'rb') as f:
        num_mpc_repeats = np.load(f)
        discretization_time = float(np.load(f))
        MPC_sim_horizon = np.load(f)
        states_closed_loop = np.load(f)
        controls_closed_loop = np.load(f)
        reference_state = np.load(f)
        computation_time_per_iteration_define = np.load(f)
        computation_time_per_iteration_solve = np.load(f)

    plot_mpc_solutions(
        num_mpc_repeats,
        discretization_time,
        MPC_sim_horizon,
        constraints_parameters,
        states_closed_loop,
        controls_closed_loop,
        reference_state)




if solve_compute_times:
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

    for i in range(len(sample_sizes_for_computation_times)):
        solver_parameters = {
            "initial_guess_state": initial_state,
            "num_scp_iterations": 10,
            "verbose": False,
            "osqp_verbose": False,
            "osqp_tolerance": 1e-3,
            "osqp_polish": True,
            "warm_start": True,
            "sample_size": sample_sizes_for_computation_times[i],
        }
        oc_alg = RobustSpacecraftOptimalControlAlgorithm(
            dynamics,
            initial_states_set,
            disturbances_set,
            cost_parameters,
            constraints_parameters,
            solver_parameters,
            discretization_time,
            horizon,
            error_bounds_velocities,
            error_bounds_controls)
        for repeat in range(num_computation_times_repeats):
            opt_variables = oc_alg.initial_guess()
            controls_matrix_prev = oc_alg.convert_controls_vector_to_controls_matrix(
                oc_alg.opt_variables_to_states_controls_vectors(
                    opt_variables)[1])

            oc_alg.define_problem(opt_variables, initial_state)
            oc_alg.update_problem(opt_variables, initial_state)
            for scp_iter in range(solver_parameters['num_scp_iterations']):
                # compute control input (run a single SCP iteration)
                define_time = time()
                oc_alg.update_problem(opt_variables, initial_state)
                define_time = time() - define_time

                solve_time = time()
                output = oc_alg.solve_one_scp_iteration()
                opt_variables, states_matrix, controls_matrix = output
                solve_time = time() - solve_time

                computation_time_per_iteration_define[repeat, i, scp_iter] = define_time
                computation_time_per_iteration_solve[repeat, i, scp_iter] = solve_time

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

    with open('results/spacecraft_computation_times.npy',
        'wb') as f:
        np.save(f, solver_parameters['num_scp_iterations'])
        np.save(f, sample_sizes_for_computation_times)
        np.save(f, computation_time_per_iteration_define)
        np.save(f, computation_time_per_iteration_solve)
        np.save(f, computation_times_cumulative)
        np.save(f, accuracy_errors)


if plot_compute_times:
    with open('results/spacecraft_computation_times.npy',
        'rb') as f:
        num_scp_iterations = np.load(f)
        sample_sizes_for_computation_times = np.load(f)
        computation_time_per_iteration_define = np.load(f)
        computation_time_per_iteration_solve = np.load(f)
        computation_times_cumulative = np.load(f)
        accuracy_errors = np.load(f)

    plot_computation_times(
        num_scp_iterations,
        sample_sizes_for_computation_times,
        solver_parameters,
        computation_time_per_iteration_define,
        computation_time_per_iteration_solve,
        computation_times_cumulative,
        accuracy_errors)




if compare_reachable_sets:
    print("Comparing reachability analysis algorithms.")

    with open('results/spacecraft_epsilons.npy',
        'rb') as f:
        error_bounds_velocities = np.load(f)
        error_bounds_controls = np.load(f)
        sample_size = np.load(f)
        print("sample_size =", sample_size)

    solver_parameters = {
        "initial_guess_state": initial_state,
        "num_scp_iterations": 10,
        "verbose": False,
        "osqp_verbose": False,
        "osqp_tolerance": 1e-4,
        "osqp_polish": True,
        "warm_start": True,
        "sample_size": sample_size,
    }
    oc_alg = RobustSpacecraftOptimalControlAlgorithm(
        dynamics,
        initial_states_set,
        disturbances_set,
        cost_parameters,
        constraints_parameters,
        solver_parameters,
        discretization_time,
        horizon,
        error_bounds_velocities,
        error_bounds_controls)
    _, states_matrix, controls_matrix = oc_alg.define_and_solve(
        oc_alg.initial_guess(),
        initial_state)

    extremal_alg = oc_alg.velocity_reachability_alg
    reach_sets_extremal = extremal_alg.estimate_reachable_sets(
        discretization_time,
        horizon,
        sample_size=sample_size,
        controls_matrix=controls_matrix)

    lip_alg = LipschitzReachabilityAlgorithm(
        extremal_alg.dynamics,
        extremal_alg.initial_states_set,
        extremal_alg.disturbances_set)
    hessian_bound = lip_alg.evaluate_hessian_bound(
        discretization_time,
        max_state=constraints_parameters["max_state"],
        max_control=constraints_parameters["max_control"])
    ellipsoids_out = lip_alg.ellipsoidal_uncertainty_shape_matrices_trajectory(
        extremal_alg.initial_states_set.position,
        discretization_time,
        horizon,
        controls_matrix,
        hessian_bound)
    ellipsoids_centers, ellipsoids_shape_matrices = ellipsoids_out

    randup = RandUP(
        extremal_alg.dynamics,
        extremal_alg.initial_states_set,
        extremal_alg.disturbances_set)
    reach_sets_randup = randup.estimate_reachable_sets(
        discretization_time,
        horizon,
        sample_size=sample_size,
        controls_matrix=controls_matrix)


    plot_reachable_sets(
        discretization_time,
        horizon,
        constraints_parameters,
        reach_sets_extremal,
        error_bounds_velocities,
        ellipsoids_centers,
        ellipsoids_shape_matrices,
        reach_sets_randup)


    print("Computation times:")
    """
    We report the time to evaluate trajectories (as opposed to
    the time to compute the convex hulls), which makes sense in this 
    application, since we enforce one constraint per sample in the
    MPC formulation. In many applications, it is not necessary to
    actually compute the true convex hull to use Algorithm 1. 
    """
    num_repeats = 100
    @jit
    def get_min_max_states_extremal(
        controls_matrix, initial_directions):
        states, _, _ = extremal_alg.compute_extremal_trajectories_using_initial_directions(
            initial_directions,
            discretization_time, horizon, controls_matrix)
        states_min = jnp.min(states, axis=0)
        states_max = jnp.max(states, axis=0)
        return states_min, states_max
    @jit
    def get_min_max_states_lipschitz(
        controls_matrix, initial_state):
        out = lip_alg.ellipsoidal_uncertainty_shape_matrices_trajectory(
            initial_state,
            discretization_time,
            horizon,
            controls_matrix,
            hessian_bound)
        ellipsoids_centers, ellipsoids_shape_matrices = out
        ellipsoids_deltas = jnp.sqrt(jnp.diagonal(
            ellipsoids_shape_matrices, axis1=1, axis2=2))
        states_min = ellipsoids_centers - ellipsoids_deltas
        states_max = ellipsoids_centers + ellipsoids_deltas
        return states_min, states_max
    @jit
    def get_min_max_states_randup(
        controls_matrix, initial_states, disturbances_matrices):
        states = randup.solve_odes_using_initial_states_and_disturbances(
            initial_states, disturbances_matrices,
            discretization_time, horizon, controls_matrix)
        states_min = jnp.min(states, axis=0)
        states_max = jnp.max(states, axis=0)
        return states_min, states_max
    initial_directions = extremal_alg.sample_initial_directions(sample_size)
    initial_state = extremal_alg.initial_states_set.position
    initial_states = randup.initial_states_set.sample_random(sample_size)
    disturbances_matrices = jnp.reshape(
        randup.disturbances_set.sample_random(horizon * sample_size),
        (sample_size, horizon, -1))
    states_min, states_max = get_min_max_states_extremal(
        controls_matrix, initial_directions)
    states_min, states_max = get_min_max_states_lipschitz(
        controls_matrix, initial_state)
    states_min, states_max = get_min_max_states_randup(
        controls_matrix, initial_states, disturbances_matrices)
    start_time = time()
    for i in range(num_repeats):
        states_min, states_max = get_min_max_states_extremal(
            controls_matrix, initial_directions)
    print("Algorithm 1 - elapsed:",
        1e3 * (time()-start_time)/num_repeats, "ms")
    start_time = time()
    for i in range(num_repeats):
        states_min, states_max = get_min_max_states_lipschitz(
            controls_matrix, initial_state)
    print("Ellipsoidal tube - elapsed:",
        1e3 * (time()-start_time)/num_repeats, "ms")
    start_time = time()
    for i in range(num_repeats):
        states_min, states_max = get_min_max_states_randup(
            controls_matrix, initial_states, disturbances_matrices)
    print("RandUP - elapsed:",
        1e3 * (time()-start_time)/num_repeats, "ms")
