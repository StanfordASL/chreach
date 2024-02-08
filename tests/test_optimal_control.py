"""Tests optimal control code."""
import pytest

from scipy.spatial.transform import Rotation
import jax.numpy as jnp
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)

from chreach.sets import *
from chreach.dynamics import *
from chreach.controlled_dynamics import *
from chreach.optimal_control import *



@pytest.fixture
def spacecraft_cost_parameters():
    num_states = 7
    num_controls = 3
    cost_parameters = {
        "reference_state": jnp.zeros(num_states),
        "state_quad_penalization_matrix": 10 * jnp.eye(num_states),
        "control_quad_penalization_matrix": jnp.eye(num_controls)
    }
    return cost_parameters

@pytest.fixture
def spacecraft_constraints_parameters():
    constraints_parameters = {
        "max_control": 0.1, # assumes infinity norm constraints
        "max_state": 0.1, # assumes infinity norm constraints
        "start_index_state_constraints": 4
    }
    return constraints_parameters

@pytest.fixture
def spacecraft_solver_parameters():
    num_states = 7
    solver_parameters = {
        "initial_guess_state": jnp.zeros(num_states),
        "num_scp_iterations": 10,
        "verbose": False,
        "osqp_verbose": False,
        "osqp_tolerance": 1e-3,
        "osqp_polish": True,
        "warm_start": True
    }
    return solver_parameters


def test_oc_alg(
    spacecraft_cost_parameters,
    spacecraft_constraints_parameters,
    spacecraft_solver_parameters):
    """Tests for the optimal control algorithm class."""
    dynamics = SpacecraftDynamics()
    num_states = dynamics.num_states
    num_controls = dynamics.num_controls
    num_disturbances = dynamics.num_disturbances
    initial_states_set = Point(jnp.zeros(num_states))
    disturbances_set = Ball(jnp.zeros(num_disturbances), 1.)

    cost_parameters = spacecraft_cost_parameters
    constraints_parameters = spacecraft_constraints_parameters
    solver_parameters = spacecraft_solver_parameters
    discretization_time = 0.1
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

    assert jnp.all(num_states == oc_alg.num_states)
    assert jnp.all(num_controls == oc_alg.num_controls)

    key = random.PRNGKey(0)
    states = random.uniform(key, shape=(horizon + 1, num_states))
    controls = random.uniform(key, shape=(horizon, num_controls))
    states_vec = oc_alg.convert_states_matrix_to_states_vector(states)
    controls_vec = oc_alg.convert_controls_matrix_to_controls_vector(controls)
    assert len(states_vec) == oc_alg.num_states_opt_variables
    assert len(controls_vec) == oc_alg.num_controls_opt_variables
    _states = oc_alg.convert_states_vector_to_states_matrix(states_vec)
    _controls = oc_alg.convert_controls_vector_to_controls_matrix(controls_vec)
    assert jnp.all(states == _states)
    assert jnp.all(controls == _controls)
    opt_variables = oc_alg.states_controls_vectors_to_opt_variables(
        states_vec, controls_vec)
    assert len(opt_variables) == oc_alg.num_opt_variables
    _states_vec, _controls_vec = oc_alg.opt_variables_to_states_controls_vectors(
        opt_variables)
    assert jnp.all(states_vec == _states_vec)
    assert jnp.all(controls_vec == _controls_vec)

def test_oc_alg_solve(
    spacecraft_cost_parameters,
    spacecraft_constraints_parameters,
    spacecraft_solver_parameters):
    """Solving the optimal control problem should work."""
    dynamics = SpacecraftDynamics()
    num_states = dynamics.num_states
    num_controls = dynamics.num_controls
    num_disturbances = dynamics.num_disturbances
    initial_states_set = Point(jnp.zeros(num_states))
    disturbances_set = Ball(jnp.zeros(num_disturbances), 1.)

    cost_parameters = spacecraft_cost_parameters
    constraints_parameters = spacecraft_constraints_parameters
    solver_parameters = spacecraft_solver_parameters
    discretization_time = 0.1
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

    oc_alg.define_and_solve(
        oc_alg.initial_guess(),
        initial_state = jnp.zeros(num_states))

def test_mpc_spacecraft(
    spacecraft_cost_parameters,
    spacecraft_constraints_parameters,
    spacecraft_solver_parameters):
    """MPC should be recursively feasible."""
    # hyperparameters for MPC
    MPC_sim_horizon = 10
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

    cost_parameters = spacecraft_cost_parameters
    constraints_parameters = spacecraft_constraints_parameters
    solver_parameters = spacecraft_solver_parameters
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

    state = initial_state
    # Initially, fully solve the problem
    opt_variables, states, controls = oc_alg.define_and_solve(
        oc_alg.initial_guess(),
        initial_state)
    for t in range(MPC_sim_horizon):
        # compute control input (run a single SCP iteration)
        oc_alg.update_problem(opt_variables, state)

        output = oc_alg.solve_one_scp_iteration()
        opt_variables, _, controls_matrix = output

        disturbance = jnp.zeros(dynamics.num_disturbances)
        state = dynamics.next_state(
            discretization_time * t,
            state,
            controls_matrix[0],
            disturbance,
            discretization_time)
