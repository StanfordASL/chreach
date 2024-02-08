"""Optimal control classes."""
from functools import partial
from typing import Tuple, Dict, Any

import osqp
from scipy.sparse import csc_matrix
import numpy as np
import jax.numpy as jnp
from jax import vmap, jacfwd, jit

from chreach.controlled_dynamics import ControlledDynamics, SpacecraftDynamics
from chreach.sets import Point, Ball
from chreach.reach import Algorithm1


num_states = 1
num_controls = 1
cost_parameters = {
    "reference_state": jnp.zeros(num_states),
    "state_quad_penalization_matrix": 10 * jnp.eye(num_states),
    "control_quad_penalization_matrix": jnp.eye(num_controls)
}
constraints_parameters = {
    "max_control": 0.1, # assumes infinity norm constraints
    "max_state": 0.1, # assumes infinity norm constraints
    "start_index_state_constraints": 4
}
solver_parameters = {
    "initial_guess_state": jnp.zeros(num_states),
    "num_scp_iterations": 10,
    "verbose": False,
    "osqp_verbose": False,
    "osqp_tolerance": 1e-3,
    "osqp_polish": True,
    "warm_start": True,
    "sample_size": 10,
}


class OptimalControlAlgorithm:
    """Optimal control algorithm."""
    def __init__(
        self,
        dynamics: ControlledDynamics,
        initial_states_set: Point,
        disturbances_set: Ball,
        cost_parameters: Dict[str, jnp.ndarray],
        constraints_parameters: Dict[str, jnp.ndarray],
        solver_parameters: Dict[str, Any],
        discretization_time: float,
        horizon: int):
        """
        Initializes the class.

        Args:
            dynamics: dynamics class to extend.
                (ControlledDynamics)
            initial_states_set: set of initial states
                (Set)
            disturbances_set: set of disturbances
                (Set)
            cost_parameters: list of parameters for the cost function
                Dict[str, jnp.ndarray]
            constraints_parameters: list of parameters for the constraints
                Dict[str, jnp.ndarray]
            solver_parameters: list of parameters for the solver
                Dict[str, Any]
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
        """
        assert isinstance(dynamics, ControlledDynamics)
        assert isinstance(initial_states_set, Point)
        assert isinstance(disturbances_set, Ball)
        assert isinstance(discretization_time, float)
        assert isinstance(horizon, int)
        self._dynamics = dynamics
        self._initial_states_set = initial_states_set
        self._disturbances_set = disturbances_set
        self._cost_parameters = cost_parameters
        self._constraints_parameters = constraints_parameters
        self._solver_parameters = solver_parameters
        self._discretization_time = discretization_time
        self._horizon = horizon

        # jit-precompile
        opt_variables = self.initial_guess()
        state = jnp.zeros(self.num_states)
        control = jnp.zeros(self.num_controls)
        states_vec = jnp.zeros(self.num_states_opt_variables)
        controls_vec = jnp.zeros(self.num_controls_opt_variables)
        states_mat = jnp.zeros((self.horizon + 1, self.num_states))
        controls_mat = jnp.zeros((self.horizon, self.num_controls))
        self.get_objective_quadratized_coefficients()
        self.get_constraints_linearized_coefficients(
            opt_variables, state)
        self.distance_L2_normalized_controls(control, control)
        self.opt_variables_to_states_controls_vectors(opt_variables)
        self.convert_states_vector_to_states_matrix(states_vec)
        self.convert_controls_vector_to_controls_matrix(controls_vec)
        self.states_controls_vectors_to_opt_variables(states_vec, controls_vec)
        self.convert_states_matrix_to_states_vector(states_mat)
        self.convert_controls_matrix_to_controls_vector(controls_mat)

    @property
    def dynamics(self) -> ControlledDynamics:
        """Returns the dynamics."""
        return self._dynamics

    @property
    def initial_states_set(self) -> Point:
        """Returns the set of initial states."""
        return self._initial_states_set

    @property
    def disturbances_set(self) -> Ball:
        """Returns the set of disturbances."""
        return self._disturbances_set

    @property
    def cost_parameters(self) -> Dict[str, jnp.ndarray]:
        """Returns the list of parameters for the cost function."""
        return self._cost_parameters

    @property
    def constraints_parameters(self) -> Dict[str, jnp.ndarray]:
        """Returns the list of parameters for the constraints."""
        return self._constraints_parameters

    @property
    def solver_parameters(self) -> Dict[str, Any]:
        """Returns the list of parameters for the solver."""
        return self._solver_parameters

    @property
    def discretization_time(self) -> float:
        """Returns the discretization time."""
        return self._discretization_time

    @property
    def horizon(self) -> int:
        """Returns the prediction horizon."""
        return self._horizon

    @property
    def num_states(self) -> int:
        """Returns the number of states."""
        return self.dynamics.num_states

    @property
    def num_controls(self) -> int:
        """Returns the number of controls."""
        return self.dynamics.num_controls

    @property
    def num_states_opt_variables(self) -> int:
        """Returns the number of state optimization variables."""
        return (self.horizon + 1) * self.num_states

    @property
    def num_controls_opt_variables(self) -> int:
        """Returns the number of control optimization variables."""
        return self.horizon * self.num_controls

    @property
    def num_opt_variables(self) -> int:
        """Returns the number of optimization variables."""
        num = self.num_states_opt_variables
        num += self.num_controls_opt_variables
        return num

    @partial(jit, static_argnums=(0,))
    def opt_variables_to_states_controls_vectors(
        self, opt_variables: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Converts the optimization variables into the state-control trajectories.

        opt_variables = (states, controls)
        where
            states is nominal state trajectory of shape (horizon + 1) * num_states
            controls is nominal control trajectory of shape horizon*n_u

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array

        Returns:
            states: state vector
                ((horizon + 1) * num_states) array
            controls: control vector
                (horizon * num_controls) array
        """
        states_vec = opt_variables[:self.num_states_opt_variables]
        controls_vec = opt_variables[self.num_states_opt_variables:]
        return states_vec, controls_vec

    @partial(jit, static_argnums=(0,))
    def states_controls_vectors_to_opt_variables(
        self,
        states_vec: jnp.ndarray,
        controls_vec: jnp.ndarray) -> jnp.ndarray:
        """
        Converts the state-control trajectories into the optimization variables.

        Args:
            states_vec: state vector
                ((horizon + 1) * num_states) array
            controls_vec: control vector
                (horizon * num_states) array

        Returns:
            opt_variables: optimization variables.
                (num_opt_variabless) array
        """
        opt_variables = jnp.concatenate([states_vec, controls_vec])
        return opt_variables

    @partial(jit, static_argnums=(0,))
    def convert_states_vector_to_states_matrix(
        self, states_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Converts the states vector a states matrix.

        Args:
            states_vector: states vector.
                ((horizon + 1) * num_states) array

        Returns:
            states_matrix: states matrix
                (horizon + 1, num_states) array
        """
        states_matrix = jnp.reshape(
            states_vector,
            (self.num_states, self.horizon + 1), 'F')
        states_matrix = states_matrix.T
        return states_matrix

    @partial(jit, static_argnums=(0,))
    def convert_controls_vector_to_controls_matrix(
        self, controls_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Converts the controls vector a controls matrix.

        Args:
            controls_vector: controls vector.
                (horizon * num_states) array

        Returns:
            controls_matrix: controls matrix
                (horizon, num_controls) array
        """
        controls_matrix = jnp.reshape(
            controls_vector,
            (self.num_controls, self.horizon), 'F')
        controls_matrix = controls_matrix.T
        return controls_matrix

    @partial(jit, static_argnums=(0,))
    def convert_states_matrix_to_states_vector(
        self, states_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Converts the controls matrix a controls matrix.

        Args:
            states_matrix: states matrix
                (horizon + 1, num_states) array

        Returns:
            states_vec: states vector.
                ((horizon + 1) * num_states) array
        """
        states_vec = jnp.reshape(
            states_matrix,
            (self.num_states * (self.horizon + 1)), 'C')
        return states_vec

    @partial(jit, static_argnums=(0,))
    def convert_controls_matrix_to_controls_vector(
        self, controls_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Converts the controls matrix a controls matrix.

        Args:
            controls_matrix: controls matrix
                (horizon, num_controls) array

        Returns:
            controls_vector: controls vector.
                (horizon * num_states) array
        """
        controls_vector = jnp.reshape(
            controls_matrix,
            (self.num_controls * self.horizon), 'C')
        return controls_vector

    def initial_guess(self) -> jnp.ndarray:
        """
        Returns an initial guess for the optimization variable.

        Returns:
            opt_variables: initial guess for the optimization variables.
                (num_opt_variabless)
        """
        initial_guess_states = jnp.tile(
            self.solver_parameters["initial_guess_state"],
            self.horizon + 1) + 1e-6
        initial_guess_controls = jnp.zeros(
            self.num_controls_opt_variables) + 1e-6
        opt_variables = self.states_controls_vectors_to_opt_variables(
                initial_guess_states, initial_guess_controls)
        return opt_variables

    @partial(jit, static_argnums=(0,))
    def nominal_initial_constraints(
        self,
        opt_variables: jnp.ndarray,
        initial_state: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ 
        Returns (A, l, u) so that the constraints
            l <= A @ opt_variables <= u
        correspond to the initial constraints.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array
            initial_state: initial state
                (num_states) array

        Returns:
            A: matrix for linear constraint l <= A @ opt_variables <= u
                (num_constraints, num_opt_variables) array
            l: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
            u: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
        """
        A = jnp.zeros((self.num_states, self.num_opt_variables))
        A = A.at[:, :self.num_states].set(jnp.eye(self.num_states))
        l = initial_state
        u = l
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def nominal_dynamics_constraints(
        self,
        opt_variables: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ 
        Returns (A, l, u) so that the constraints
            l <= A @ opt_variables <= u
        correspond to the (linearized) dynamics equality constraints.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array

        Returns:
            A: matrix for linear constraint l <= A @ opt_variables <= u
                (num_constraints, num_opt_variables) array
            l: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
            u: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
        """
        def dynamics_constraint_state_control(time, state, control, next_state):
            predicted_next_state = self.dynamics.next_state(
                time,
                state,
                control,
                jnp.zeros(self.dynamics.num_disturbances),
                self.discretization_time)
            constraint = predicted_next_state - next_state
            return constraint
        def dynamics_constraints(opt_variables):
            times = self.discretization_time * jnp.arange(self.horizon)
            xs, us = self.opt_variables_to_states_controls_vectors(
                opt_variables)
            states_matrix = self.convert_states_vector_to_states_matrix(
                xs)
            controls_matrix = self.convert_controls_vector_to_controls_matrix(
                us)
            constraints = vmap(dynamics_constraint_state_control)(
                times, states_matrix[:-1], controls_matrix, states_matrix[1:])
            constraints = constraints.flatten()
            return constraints
        def dynamics_constraints_dopt(opt_variables):
            constraints_dopt = jacfwd(dynamics_constraints)(
                opt_variables)
            return constraints_dopt
        con = dynamics_constraints(opt_variables)
        con_dopt = dynamics_constraints_dopt(opt_variables)
        """
        con(z) = 0
        => linearize:
        con(zp) + con_dz(zp) @ (z - zp) = 0
        con_dz(zp) @ z = - (con(z) - con_dz(zp) @ zp)
        """
        A = con_dopt
        l = -(con - con_dopt @ opt_variables)
        u = l
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def nominal_state_constraints(
        self,
        opt_variables: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ 
        Returns (A, l, u) so that the constraints
            l <= A @ opt_variables <= u
        correspond to the state inequality constraints.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array

        Returns:
            A: matrix for linear constraint l <= A @ opt_variables <= u
                (num_constraints, num_opt_variables) array
            l: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
            u: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
        """
        def state_constraint_value(state):
            index = self.constraints_parameters["start_index_state_constraints"]
            states_to_constrain = state[index:]
            return states_to_constrain
        def state_constraints_values(opt_variables):
            xs, _ = self.opt_variables_to_states_controls_vectors(opt_variables)
            states_matrix = self.convert_states_vector_to_states_matrix(xs)
            constraints = vmap(state_constraint_value)(states_matrix[1:])
            constraints = constraints.flatten()
            return constraints
        def state_constraints_values_dopt(opt_variables):
            constraints_dopt = jacfwd(state_constraints_values)(
                opt_variables)
            return constraints_dopt
        con = state_constraints_values(opt_variables)
        con_dopt = state_constraints_values_dopt(opt_variables)
        """
        con(z) = 0
        => linearize:
        con(zp) + con_dz(zp) @ (z - zp) = 0
        con_dz(zp) @ z = - (con(z) - con_dz(zp) @ zp)
        """
        max_state_value = self.constraints_parameters["max_state"]
        A = con_dopt
        lu_term = -(con - con_dopt @ opt_variables)
        l = lu_term - max_state_value
        u = lu_term + max_state_value
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def nominal_control_constraints(
        self,
        opt_variables: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ 
        Returns (A, l, u) so that the constraints
            l <= A @ opt_variables <= u
        correspond to the control inequality constraints.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array

        Returns:
            A: matrix for linear constraint l <= A @ opt_variables <= u
                (num_constraints, num_opt_variables) array
            l: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
            u: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
        """
        def control_constraint_value(control):
            return control
        def control_constraints_values(opt_variables):
            _, us = self.opt_variables_to_states_controls_vectors(opt_variables)
            controls_matrix = self.convert_controls_vector_to_controls_matrix(us)
            constraints = vmap(control_constraint_value)(controls_matrix)
            constraints = constraints.flatten()
            return constraints
        def control_constraints_values_dopt(opt_variables):
            constraints_dopt = jacfwd(control_constraints_values)(
                opt_variables)
            return constraints_dopt
        con = control_constraints_values(opt_variables)
        con_dopt = control_constraints_values_dopt(opt_variables)
        """
        con(z) = 0
        => linearize:
        con(zp) + con_dz(zp) @ (z - zp) = 0
        con_dz(zp) @ z = - (con(z) - con_dz(zp) @ zp)
        """
        max_control_value = self.constraints_parameters["max_control"]
        A = con_dopt
        lu_term = -(con - con_dopt @ opt_variables)
        l = lu_term - max_control_value
        u = lu_term + max_control_value
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def get_objective_quadratized_coefficients(self) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns (P, q) corresponding to the objective
               min_z (1/2 z^T P z + q^T z)
        where z = opt_variables is the vector of optimization variables.

        (x-x_ref).T @ Q (x-x_ref)
        => x.T @ Q @ x + (-2 * x_ref.T @ Q) @ x + constant
                (P)      (2  *    q       ) 
        => pack onto z = (x_vec, u_vec)

        Returns:
            cost_matrix: cost matrix (P variable).
                (num_opt_variables, num_opt_variables) array
            cost_vector: cost vector (q variable)
                (num_opt_variables) array
        """
        # Quadratic Objective
        cost_mat = jnp.block([
            [
            jnp.kron(
                jnp.eye(self.horizon + 1),
                self.cost_parameters['state_quad_penalization_matrix']),
            jnp.zeros((
                self.num_states_opt_variables,
                self.num_controls_opt_variables))
            ],
            [
            jnp.zeros((
                self.num_controls_opt_variables,
                self.num_states_opt_variables)),
            jnp.kron(
                jnp.eye(self.horizon),
                self.cost_parameters['control_quad_penalization_matrix'])
            ]
            ])
        cost_mat = 2 * cost_mat

        cost_vec = -2. * (
            self.cost_parameters['reference_state'].T @
            self.cost_parameters['state_quad_penalization_matrix'])
        cost_vec = jnp.tile(
            cost_vec, self.horizon + 1)
        cost_vec = jnp.concatenate([
            cost_vec,
            jnp.zeros(self.num_controls_opt_variables)])
        return cost_mat, cost_vec

    @partial(jit, static_argnums=(0,))
    def get_constraints_linearized_coefficients(
        self,
        opt_variables: jnp.array,
        initial_state: jnp.array
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns (A, l, u) so that the constraint
            l <= A @ opt_variables <= u
        corresponds to all the constraints.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array
            initial_state: initial state
                (num_states) array

        Returns:
            A: matrix for linear constraint l <= A @ opt_variables <= u
                (num_constraints, num_opt_variables) array
            l: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
            u: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
        """
        A1, l1, u1 = self.nominal_initial_constraints(
            opt_variables, initial_state)
        A2, l2, u2 = self.nominal_dynamics_constraints(opt_variables)
        A3, l3, u3 = self.nominal_state_constraints(opt_variables)
        A4, l4, u4 = self.nominal_control_constraints(opt_variables)
        A = jnp.concatenate([A1, A2, A3, A4], axis=0)
        l = jnp.concatenate([l1, l2, l3, l4])
        u = jnp.concatenate([u1, u2, u3, u4])
        return A, l, u

    def define_problem(
        self,
        opt_variables: jnp.array,
        initial_state: jnp.array):
        """
        Defines the optimization problem.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array
            initial_state: initial state
                (num_states) array
        """
        P, q = self.get_objective_quadratized_coefficients()
        A, l, u = self.get_constraints_linearized_coefficients(
            opt_variables, initial_state)
        P, A = csc_matrix(P), csc_matrix(A)
        q, l, u = np.asarray(q), np.asarray(l), np.asarray(u)

        # Setup OSQP problem
        self.osqp_prob = osqp.OSQP()
        self.osqp_prob.setup(
            P, q, A, l, u,
            eps_abs=self.solver_parameters["osqp_tolerance"],
            eps_rel=self.solver_parameters["osqp_tolerance"],
            warm_start=self.solver_parameters["warm_start"],
            verbose=self.solver_parameters["osqp_verbose"],
            polish=self.solver_parameters["osqp_polish"])

    def update_problem(
        self,
        opt_variables: jnp.array,
        initial_state: jnp.array):
        """
        Updates the optimization problem.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array
            initial_state: initial state
                (num_states) array
        """
        A, l, u = self.get_constraints_linearized_coefficients(
            opt_variables, initial_state)
        A = csc_matrix(A)
        l, u = np.asarray(l), np.asarray(u)
        self.osqp_prob.update(l=l, u=u)
        self.osqp_prob.update(Ax=A.data)

    @partial(jit, static_argnums=(0,))
    def distance_L2_normalized_controls(
        self,
        controls_matrix: jnp.array,
        controls_matrix_prev: jnp.array) -> float:
        """
        Computes the normalized squared norm error between the controls.

        Args:
            controls_matrix: controls matrix
                (horizon, num_controls) array
            controls_matrix_prev: previous controls matrix
                (horizon, num_controls) array

        Returns:
            error: squared norm (L2) error
                (float)
        """
        error = jnp.mean(
            jnp.linalg.norm(
                controls_matrix - controls_matrix_prev, axis=-1))
        error = error / jnp.mean(
            jnp.linalg.norm(
                controls_matrix, axis=-1))
        return error

    def solve_one_scp_iteration(
        self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Solves one convexification of the optimal control problem.

        Returns:
            opt_variables: optimization variable
                (num_opt_variables) array
            states_matrix: states matrix
                (horizon + 1, num_states) array
            controls_matrix: controls matrix
                (horizon, num_controls) array
        """
        self.res = self.osqp_prob.solve()
        if self.res.info.status != 'solved':
            if self.solver_parameters["verbose"]:
                print("[solve]: Problem infeasible.")
        opt_variables = self.res.x
        xs, us = self.opt_variables_to_states_controls_vectors(opt_variables)
        states_matrix = self.convert_states_vector_to_states_matrix(xs)
        controls_matrix = self.convert_controls_vector_to_controls_matrix(us)
        return opt_variables, states_matrix, controls_matrix

    def define_and_solve(
        self,
        initial_guess_opt_variables: jnp.ndarray,
        initial_state: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Define and solve the optimal control problem.

        Args:
            initial_guess_opt_variables: initial guess
                (num_opt_variables) array
            initial_state: initial state
                (num_states) array
        Returns:
            opt_variables: optimization variable
                (num_opt_variables) array
            states_matrix: states matrix
                (horizon + 1, num_states) array
            controls_matrix: controls matrix
                (horizon, num_controls) array
        """
        num_scp_iterations = self.solver_parameters["num_scp_iterations"]

        opt_variables = initial_guess_opt_variables
        self.define_problem(opt_variables, initial_state)

        _, us = self.opt_variables_to_states_controls_vectors(opt_variables)
        controls_prev = self.convert_controls_vector_to_controls_matrix(us)
        for scp_iter in range(num_scp_iterations):
            if self.solver_parameters["verbose"]:
                print("scp_iter =", scp_iter)
            opt_variables, states, controls = self.solve_one_scp_iteration()
            if self.solver_parameters["verbose"]:
                print("L2 error =",
                    self.distance_L2_normalized_controls(
                        controls, controls_prev))
            controls_prev = controls
            if scp_iter < num_scp_iterations - 1:
                self.update_problem(opt_variables, initial_state)

        return opt_variables, states, controls


class RobustSpacecraftOptimalControlAlgorithm(OptimalControlAlgorithm):
    """
    Robust optimal control problem solver with reachable sets from Algorithm 1
    for the spacecraft system.

    Generalizing to arbitrary dynamcis should be possible. Here, we use the fact
    that only uncertainty in angular velocity needs to be propagated. Thus, we
    wrap the velocity dynamics of the spacecraft with Algorithm 1.
    """
    def __init__(
        self,
        dynamics: SpacecraftDynamics,
        initial_states_set: Point,
        disturbances_set: Ball,
        cost_parameters: Dict[str, jnp.ndarray],
        constraints_parameters: Dict[str, jnp.ndarray],
        solver_parameters: Dict[str, Any],
        discretization_time: float,
        horizon: int,
        error_bounds_velocities: jnp.ndarray,
        error_bounds_controls: jnp.ndarray):
        """Initializes the class."""
        assert isinstance(dynamics, SpacecraftDynamics)
        assert len(error_bounds_velocities) == horizon + 1
        assert len(error_bounds_controls) == horizon
        self._error_bounds_velocities = error_bounds_velocities
        self._error_bounds_controls = error_bounds_controls

        start_index_velocity_states = 4
        velocity_initial_states_set = Point(
            initial_states_set.position[start_index_velocity_states:])
        velocity_disturbances_set = Ball(
            disturbances_set.center,
            disturbances_set.radius)
        self._velocity_reachability_alg = Algorithm1(
            dynamics.velocity_dynamics,
            velocity_initial_states_set,
            velocity_disturbances_set,
            diffeomorphism_vector=jnp.diag(dynamics.inertia))
        self._initial_directions = \
            self.velocity_reachability_alg.sample_initial_directions(
                solver_parameters['sample_size'])

        super().__init__(
            dynamics,
            initial_states_set,
            disturbances_set,
            cost_parameters,
            constraints_parameters,
            solver_parameters,
            discretization_time,
            horizon)

    @property
    def velocity_reachability_alg(self) -> Algorithm1:
        """Returns the algorithm to analyze velocity reachable sets."""
        return self._velocity_reachability_alg

    @property
    def sample_size(self):
        """Returns the sample size used to robustly enforce constraints."""
        return self.solver_parameters['sample_size']

    @property
    def initial_directions(self):
        """Returns the initial directions used to robustly enforce constraints."""
        return self._initial_directions

    @property
    def error_bounds_velocities(self) -> jnp.ndarray:
        """Returns the velocity reachable set error bounds."""
        return self._error_bounds_velocities

    @property
    def error_bounds_controls(self) -> jnp.ndarray:
        """Returns the control reachable set error bounds."""
        return self._error_bounds_controls

    @partial(jit, static_argnums=(0,))
    def robust_constraints(
        self,
        opt_variables: jnp.ndarray,
        initial_state: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ 
        Returns (A, l, u) so that the constraints
            l <= A @ opt_variables <= u
        correspond to the robust inequality constraints.

        Args:
            opt_variables: optimization variables.
                (num_opt_variables) array
            initial_state: initial state
                (num_states) array

        Returns:
            A: matrix for linear constraint l <= A @ opt_variables <= u
                (num_constraints, num_opt_variables) array
            l: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
            u: vector for linear constraint l <= A @ opt_variables <= u
                (num_constraints) array
        """
        def robust_state_constraint_value(velocity_state):
            return velocity_state
        def robust_control_constraint_value(velocity_state, control):
            velocity_dynamics = self.dynamics.velocity_dynamics
            control_closed_loop = velocity_dynamics.closed_loop_control(
                velocity_state, control)
            return control_closed_loop
        def robust_constraints_values(opt_variables):
            _, us = self.opt_variables_to_states_controls_vectors(
                opt_variables)
            controls_matrix = self.convert_controls_vector_to_controls_matrix(
                us)
            initial_omega_state = initial_state[4:]
            # reachability here
            reach = self.velocity_reachability_alg
            omegas_matrices, _, _ = vmap(
                reach.solve_augmented_ode_fixed_initial_state,
                in_axes=(None, 0, None, None, None))(
                    initial_omega_state,
                    self.initial_directions,
                    self.discretization_time,
                    self.horizon,
                    controls_matrix)
            velocity_constraints = vmap(vmap(robust_state_constraint_value))(
                omegas_matrices[:, 1:])
            control_constraints = vmap(vmap(robust_control_constraint_value),
                in_axes=(0, None))(
                omegas_matrices[:, :-1],
                controls_matrix)
            velocity_constraints = velocity_constraints.flatten()
            control_constraints = control_constraints.flatten()
            return velocity_constraints, control_constraints
        def robust_constraints_values_dopt(opt_variables):
            constraints_dopt = jacfwd(robust_constraints_values)(
                opt_variables)
            return constraints_dopt
        c_vel, c_con = robust_constraints_values(
            opt_variables)
        c_dopt_vel, c_dopt_con = robust_constraints_values_dopt(
            opt_variables)
        """
        c(z) = 0
        => linearize:
        c(zp) + c_dz(zp) @ (z - zp) = 0
        c_dz(zp) @ z = - (c(z) - c_dz(zp) @ zp)
        """
        max_state_value = self.constraints_parameters["max_state"]
        max_control_value = self.constraints_parameters["max_control"]
        A_velocity = c_dopt_vel
        lu_velocity = -(c_vel - c_dopt_vel @ opt_variables)
        l_velocity = lu_velocity - max_state_value
        u_velocity = lu_velocity + max_state_value
        A_control = c_dopt_con
        lu_control = -(c_con - c_dopt_con @ opt_variables)
        l_control = lu_control - max_control_value
        u_control = lu_control + max_control_value

        # add approximation errors due to sampling
        epsilons_vel = self.error_bounds_velocities[1:] # (horizon,)
        epsilons_con = self.error_bounds_controls # (horizon,)
        epsilons_vel = jnp.repeat( # (horizon, 3)
            epsilons_vel[:, jnp.newaxis], 3, axis=1)
        epsilons_con = jnp.repeat( # (horizon, 3)
            epsilons_con[:, jnp.newaxis], 3, axis=1)
        epsilons_vel = jnp.repeat( # (sample_size, horizon, 3)
            epsilons_vel[jnp.newaxis], self.sample_size, axis=0)
        epsilons_con = jnp.repeat( # (sample_size, horizon, 3)
            epsilons_con[jnp.newaxis], self.sample_size, axis=0)
        epsilons_vel = epsilons_vel.flatten()
        epsilons_con = epsilons_con.flatten()
        l_velocity += epsilons_vel
        u_velocity -= epsilons_vel
        l_control += epsilons_con
        u_control -= epsilons_con

        A = jnp.vstack([A_velocity, A_control])
        l = jnp.hstack([l_velocity, l_control])
        u = jnp.hstack([u_velocity, u_control])
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def get_constraints_linearized_coefficients(
        self,
        opt_variables: jnp.array,
        initial_state: jnp.array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        A1, l1, u1 = self.nominal_initial_constraints(
            opt_variables, initial_state)
        A2, l2, u2 = self.nominal_dynamics_constraints(
            opt_variables)
        A3, l3, u3 = self.robust_constraints(
            opt_variables, initial_state)
        A = jnp.vstack([A1, A2, A3])
        l = jnp.hstack([l1, l2, l3])
        u = jnp.hstack([u1, u2, u3])
        return A, l, u
