"""Tests dynamics code."""
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from chreach.sets import *
from chreach.dynamics import *


def verify_dynamics(dynamics: Dynamics):
    """ General checks for dynamics."""
    assert isinstance(dynamics, Dynamics)
    num_states = dynamics.num_states
    num_disturbances = dynamics.num_disturbances

    assert isinstance(num_states, int)
    assert isinstance(num_disturbances, int)
    assert num_states >= 1
    assert num_disturbances >= 0

    time = 0
    state = jnp.ones(num_states)
    disturbance = jnp.ones(num_disturbances)

    f = dynamics.f(time, state)
    assert isinstance(f, jnp.ndarray)
    assert len(f.shape) == 1
    assert f.shape[0] == num_states

    g = dynamics.g(time, state)
    assert isinstance(g, jnp.ndarray)
    assert len(g.shape) == 2
    assert g.shape[0] == num_states
    assert g.shape[1] == num_disturbances

    fdx = dynamics.f_dx(time, state)
    assert isinstance(fdx, jnp.ndarray)
    assert len(fdx.shape) == 2
    assert fdx.shape[0] == num_states
    assert fdx.shape[1] == num_states

    gwdx = dynamics.gw_dx(time, state, disturbance)
    assert isinstance(gwdx, jnp.ndarray)
    assert len(gwdx.shape) == 2
    assert gwdx.shape[0] == num_states
    assert gwdx.shape[1] == num_states

    state_dot = dynamics.state_dot(time, state, disturbance)
    assert isinstance(state_dot, jnp.ndarray)
    assert len(state_dot.shape) == 1
    assert state_dot.shape[0] == num_states

    state_next = dynamics.next_state(time, state, disturbance, 0.1)
    assert isinstance(state_next, jnp.ndarray)
    assert len(state_next.shape) == 1
    assert state_next.shape[0] == num_states

def get_g_matrix_for_dubins(dim):
    assert dim == 1 or dim == 2 or dim == 3
    g_matrix = jnp.eye(3) # dim == 3
    if dim == 1:
        g_matrix = jnp.array([
            [0],
            [1.],
            [0]])
    elif dim == 2:
        g_matrix = jnp.array([
            [0, 0],
            [1., 0],
            [0, 1.]])
    return g_matrix

def test_dynamics():
    for dim in [1, 2, 3, 4]:
        dynamics = Dynamics(dim, dim)
        assert dynamics.num_states == dim
        assert dynamics.num_disturbances == dim

def test_attraction_repulsion():
    dynamics = AttractionRepulsionDynamics()
    verify_dynamics(dynamics)

    assert isinstance(dynamics.repulsion_position, jnp.ndarray)
    assert isinstance(dynamics.attraction_position, jnp.ndarray)
    assert len(dynamics.repulsion_position.shape) == 1
    assert len(dynamics.attraction_position.shape) == 1
    assert dynamics.repulsion_position.shape[0] == dynamics.num_states
    assert dynamics.attraction_position.shape[0] == dynamics.num_states

def test_dubins():
    dynamics = DubinsDynamics()
    verify_dynamics(dynamics)

    for dim in [1, 2, 3]:
        turning_speed = -0.5632
        forward_speed = 0.6123
        g_matrix = get_g_matrix_for_dubins(dim)
        dynamics = DubinsDynamics(
            turning_speed, forward_speed, g_matrix)
        verify_dynamics(dynamics)

        assert dynamics.num_disturbances == dim

        assert jnp.all(dynamics.g_matrix == g_matrix)
        assert dynamics.turning_speed == turning_speed
        assert dynamics.forward_speed == forward_speed

def test_g_added():
    smoothing_epsilon = 0.123
    for dim_to_add in [1, 2]:
        g_matrix = get_g_matrix_for_dubins(3 - dim_to_add)
        dynamics_base = DubinsDynamics(
            g_matrix=g_matrix)
        def g_to_add(time, state):
            if dim_to_add == 1:
                g = jnp.array([
                    [0],
                    [0],
                    [1.]])
            elif dim_to_add == 2:
                g = jnp.array([
                    [0, 0],
                    [1., 0],
                    [0, 1.]])
            return g
        dynamics = GaddedDynamics(
            dynamics_base,
            g_to_add,
            smoothing_epsilon)
        assert dynamics.num_disturbances_to_add == dim_to_add
        assert dynamics.epsilon == smoothing_epsilon
        verify_dynamics(dynamics)

def test_augmented():
    dim = 2
    for disturbances_set in [
        Ball(jnp.zeros(dim), 1.0),
        Ellipsoid(jnp.zeros(dim), jnp.eye(dim))]:
        for base_dynamics in [
            AttractionRepulsionDynamics(),
            DubinsDynamics(g_matrix=get_g_matrix_for_dubins(dim))]:
            dynamics = AugmentedDynamics(
                base_dynamics,
                disturbances_set)

            base_num_states = base_dynamics.num_states

            assert dynamics.num_base_dynamics_states == base_num_states
            assert dynamics.num_states == 2 * base_num_states
            assert dynamics.num_disturbances == 0

            time = 0
            state = jnp.ones(base_num_states)
            adjoint = jnp.ones(base_num_states)

            augmented_state = dynamics.augmented_state_from_state_adjoint(
                state, adjoint)
            _state, _adjoint = dynamics.state_adjoint_from_augmented_state(
                augmented_state)
            assert jnp.all(state == _state)
            assert jnp.all(adjoint == _adjoint)

            verify_dynamics(dynamics)
