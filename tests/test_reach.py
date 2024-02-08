"""Tests reachability analysis code."""
import jax.numpy as jnp
from jax import vmap, jacfwd, random
from jax.config import config
config.update("jax_enable_x64", True)

from chreach.dynamics import *
from chreach.sets import *
from chreach.reach import *


def test_algorithm1_attraction_repulsion():
    final_time = 7.5
    prediction_horizon = 10
    discretization_time = final_time / prediction_horizon
    sample_size = 30

    dynamics = AttractionRepulsionDynamics()
    initial_states_set = Point(
        position=jnp.array([-1., 0.]))
    disturbances_set = Ball(
        center=jnp.zeros(2), radius=0.1)
    reach_alg = Algorithm1(
        dynamics, initial_states_set, disturbances_set)

    # sample initial directions
    theta_vals = jnp.linspace(0, jnp.pi-1e-2, sample_size)
    d0s_x, d0s_y = jnp.cos(theta_vals), jnp.sin(theta_vals)
    initial_directions = jnp.stack((d0s_x, d0s_y)).T

    reach_alg.estimate_reachable_sets_using_initial_directions(
        initial_directions,
        discretization_time,
        prediction_horizon)
    reach_alg.estimate_reachable_sets(
        discretization_time,
        prediction_horizon,
        sample_size)

def test_algorithm1_dubins_full_rank():
    discretization_time = 0.5
    prediction_horizon = 14
    sample_size = 20

    epsilon = 0.2

    # initialize problem
    g_matrix = jnp.array([
        [1., 0.],
        [0., 0.],
        [0., 1.]])
    dynamics = DubinsDynamics(
        turning_speed=0.5,
        forward_speed=0.5,
        g_matrix=g_matrix)
    dim = dynamics.num_states
    initial_states_set = Ellipsoid(
        center=jnp.zeros(dim),
        shape_matrix=jnp.diag(
            0.001 * jnp.array([1., 1., 0.1])))
    center = jnp.zeros(dim)
    radius = 0.01
    disturbances_set = Ball(center=center, radius=radius)

    smooth_disturbances_set_over = FullDimSmoothConvexSet(
        disturbances_set,
        num_variables_to_add=1)
    def g_to_add(time, state):
        g3 = jnp.array([
            [0],
            [1.],
            [0]])
        return g3
    dynamics_invertible = GaddedDynamics(
        dynamics,
        g_to_add,
        epsilon)
    reachability_alg = Algorithm1(
        dynamics_invertible,
        initial_states_set,
        smooth_disturbances_set_over)
    _ = reachability_alg.estimate_reachable_sets(
        discretization_time,
        prediction_horizon,
        sample_size)

def test_algorithm1_dubins_rectangle():
    discretization_time = 0.5
    prediction_horizon = 14
    sample_size = 10
    smoothing_parameter = 5
    dynamics = DubinsDynamics(
        turning_speed=0.5,
        forward_speed=0.5,
        g_matrix=jnp.eye(3))
    dim = dynamics.num_states
    initial_states_set = Ellipsoid(
        center=jnp.zeros(dim),
        shape_matrix=jnp.diag(
            0.001 * jnp.array([1., 1., 0.1])))
    center = jnp.zeros(dim)
    deltas = 0.01 * jnp.ones(3)
    disturbances_set = Rectangle(
        center=center,
        deltas=deltas)
    smooth_disturbances_set_over = SmoothRectangle(
        disturbances_set,
        smoothing_parameter=smoothing_parameter)
    reachability_alg = Algorithm1(
        dynamics,
        initial_states_set,
        smooth_disturbances_set_over)
    sets = reachability_alg.estimate_reachable_sets(
        discretization_time,
        prediction_horizon,
        sample_size)
    sphere = UnitSphere(dim)
    initial_directions = sphere.sample(sample_size)
    vmap(reachability_alg.solve_augmented_ode,
        in_axes=(0, None, None))(
        initial_directions,
        discretization_time,
        prediction_horizon)

def test_diffeomorphism():
    final_time = 2
    prediction_horizon = 10
    discretization_time = final_time / prediction_horizon
    sample_size = 3000

    dim = 2
    dynamics = AttractionRepulsionDynamics()
    initial_states_set = Point(
        position=jnp.ones(dim))
    disturbances_set = Ball(
        center=jnp.zeros(dim), radius=0.1)
    initial_directions = UnitSphere(dim).sample(sample_size)
    reach_alg = Algorithm1(
        dynamics, initial_states_set, disturbances_set)

    directions = vmap(reach_alg.diffeomorphism_sphere)(
        initial_directions)
    assert jnp.linalg.norm(directions - initial_directions) < 1e-9


    # check that reachable sets are close to each other for large sample sizes
    # even if the diffeomorphism map is not the identity
    sets_1 = reach_alg.estimate_reachable_sets_using_initial_directions(
        initial_directions,
        discretization_time,
        prediction_horizon)
    reach_alg = Algorithm1(
        dynamics, initial_states_set, disturbances_set,
        diffeomorphism_vector=jnp.array([2., 4., 6.]))
    sets_2 = reach_alg.estimate_reachable_sets_using_initial_directions(
        directions,
        discretization_time,
        prediction_horizon)

    states_1 = sets_1.state_trajectories
    states_2 = sets_2.state_trajectories
    # trajectories are different
    assert jnp.linalg.norm(states_1 - states_2) > 1e-3
    # convex hulls (rectangular over-approximations) are the same
    for t in [1, 4, 6, prediction_horizon]:
        states_1_t = states_1[:, t, :]
        states_2_t = states_2[:, t, :]
        assert jnp.all(jnp.abs(
            jnp.min(states_1_t, 0) - jnp.min(states_2_t, 0)) < 1e-6)
        assert jnp.all(jnp.abs(
            jnp.max(states_1_t, 0) - jnp.max(states_2_t, 0)) < 1e-6)

def test_randup_attraction_repulsion():
    final_time = 7.5
    prediction_horizon = 10
    discretization_time = final_time / prediction_horizon
    sample_size = 30

    dynamics = AttractionRepulsionDynamics()
    initial_states_set = Point(
        position=jnp.array([-1., 0.]))
    disturbances_set = Ball(
        center=jnp.zeros(2), radius=0.1)
    reach_alg = RandUP(
        dynamics, initial_states_set, disturbances_set)
    reach_alg.estimate_reachable_sets(
        discretization_time,
        prediction_horizon,
        sample_size)

def test_randup_dubins_full_rank():
    discretization_time = 0.5
    prediction_horizon = 14
    sample_size = 20
    epsilon = 0.2
    # initialize problem
    g_matrix = jnp.array([
        [1., 0.],
        [0., 0.],
        [0., 1.]])
    dynamics = DubinsDynamics(
        turning_speed=0.5,
        forward_speed=0.5,
        g_matrix=g_matrix)
    dim = dynamics.num_states
    initial_states_set = Ball(
        center=jnp.zeros(dim),
        radius=0.01)
    disturbances_set = Ball(
        center=jnp.zeros(dim+1), radius=0.01)
    def g_to_add(time, state):
        g3 = jnp.array([
            [0],
            [1.],
            [0]])
        return g3
    dynamics_invertible = GaddedDynamics(
        dynamics,
        g_to_add,
        epsilon)
    reach_alg = RandUP(
        dynamics, initial_states_set, disturbances_set)
    reach_alg.estimate_reachable_sets(
        discretization_time,
        prediction_horizon,
        sample_size)

def test_randup_dubins_rectangle():
    discretization_time = 0.5
    prediction_horizon = 14
    sample_size = 10
    dynamics = DubinsDynamics(
        turning_speed=0.5,
        forward_speed=0.5,
        g_matrix=jnp.eye(3))
    dim = dynamics.num_states
    initial_states_set = Ball(
        center=jnp.zeros(dim),
        radius=0.01)
    center = jnp.zeros(dim)
    deltas = 0.01 * jnp.ones(dim)
    disturbances_set = Rectangle(
        center=center,
        deltas=deltas)
    reach_alg = RandUP(
        dynamics, initial_states_set, disturbances_set)
    reach_alg.estimate_reachable_sets(
        discretization_time,
        prediction_horizon,
        sample_size)
