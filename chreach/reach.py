"""Reachability analysis algorithms."""
import enum
from typing import Tuple, List

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import numpy as np
import jax.numpy as jnp
from jax import vmap, jacfwd
from jax.lax import scan

from chreach.dynamics import Dynamics, AugmentedDynamics
from chreach.sets import Set, UnitSphere, SmoothConvexSet, Point, Ball
from chreach.controlled_dynamics import \
    ControlledDynamics, ControlledFeedbackDynamics, AugmentedControlledDynamics
from chreach.utils.stats import uniformly_sample_points_in_rectangle


class PlotType(enum.IntEnum):
    """Type of plot"""
    SCATTERPLOT = 0
    PLOT = 1
    CONVEXHULLPLOT = 2


class SampledReachableSetTrajectory:
    """Trajectory of sample-based approximations of reachable sets.""" 

    def __init__(
        self,
        state_trajectories: jnp.ndarray,
        disturbance_trajectories: jnp.ndarray):
        """
        Initializes the class.

        Args:
            state_trajectories: state trajectory
                (sample_size, horizon+1, num_states) array
            disturbance_trajectories: disturbance trajectory
                (sample_size, horizon+1, num_states) array
        """
        self._sample_size = state_trajectories.shape[0]
        self._horizon = state_trajectories.shape[1] - 1
        self._num_states = state_trajectories.shape[2]
        self._num_disturbances = disturbance_trajectories.shape[1]
        self._state_trajectories = state_trajectories
        self._disturbance_trajectories = disturbance_trajectories

    @property
    def sample_size(self) -> int:
        """Returns the sample size."""
        return self._sample_size

    @property
    def horizon(self) -> int:
        """Returns the horizon."""
        return self._horizon

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._num_states

    @property
    def num_disturbances(self) -> int:
        """Returns the number of disturbance variables."""
        return self._num_disturbances

    @property
    def state_trajectories(self) -> jnp.ndarray:
        """Returns the state trajectories."""
        return self._state_trajectories

    @property
    def disturbance_trajectories(self) -> jnp.ndarray:
        """Returns the disturbance_trajectories."""
        return self._disturbance_trajectories

    def plot(
        self,
        ax,
        times_to_plot: List[int] = None,
        axes_to_plot: List[int] = [0, 1],
        plot_type: PlotType = PlotType.SCATTERPLOT,
        color='k', alpha=0.25, linestyle='-', linewidth=2, markersize=1):
        num_axes_to_plot = len(axes_to_plot)
        if num_axes_to_plot != 2 and num_axes_to_plot != 3:
            raise NotImplementedError
        if num_axes_to_plot == 3 and plot_type != PlotType.SCATTERPLOT:
            raise NotImplementedError
        if not(
            plot_type is PlotType.PLOT or
            plot_type is PlotType.SCATTERPLOT or
            plot_type is PlotType.CONVEXHULLPLOT):
            raise NotImplementedError

        if times_to_plot is None:
            times_to_plot = range(self.horizon+1)
            # if plot_type is PlotType.CONVEXHULLPLOT:
            #     # Skip first time that may be a single point. This would
            #     # cause the ConvexHull() function to throw an error.
            #     times_to_plot = range(1, self.horizon+1)
        for t in times_to_plot:
            if num_axes_to_plot == 2:
                if plot_type is PlotType.PLOT:
                    ax.plot(
                        self.state_trajectories[:, t, axes_to_plot[0]].T,
                        self.state_trajectories[:, t, axes_to_plot[1]].T,
                        color=color, alpha=alpha,
                        linestyle=linestyle, linewidth=linewidth)
                elif plot_type is PlotType.SCATTERPLOT:
                    ax.scatter(
                        self.state_trajectories[:, t, axes_to_plot[0]],
                        self.state_trajectories[:, t, axes_to_plot[1]],
                        color=color, alpha=alpha, s=markersize)
                elif plot_type is PlotType.CONVEXHULLPLOT:
                    state_traj = self.state_trajectories
                    pts = jnp.concatenate([
                        state_traj[:, t, axes_to_plot[0]][:, jnp.newaxis],
                        state_traj[:, t, axes_to_plot[1]][:, jnp.newaxis]],
                        axis=1)
                    if jnp.min(jnp.linalg.eigvals(jnp.cov(pts.T))) > 1e-6:
                        hull = ConvexHull(pts)
                        for s in hull.simplices:
                            plt.plot(
                                pts[s, 0], pts[s, 1],
                                color=color, alpha=alpha,
                                linestyle=linestyle, linewidth=linewidth)
                    else:
                        # Not enough difference between states:
                        # ConvexHull() would throw an error
                        # Plot a point instead.
                        ax.scatter(
                            self.state_trajectories[0, t, axes_to_plot[0]],
                            self.state_trajectories[0, t, axes_to_plot[1]],
                            color=color, alpha=alpha)
                else:
                    raise NotImplementedError
            elif num_axes_to_plot == 3:
                if plot_type is PlotType.SCATTERPLOT:
                    ax.scatter(
                        self.state_trajectories[:, t, axes_to_plot[0]],
                        self.state_trajectories[:, t, axes_to_plot[1]],
                        self.state_trajectories[:, t, axes_to_plot[2]],
                        color=color, alpha=alpha, s=markersize)
                else:
                    raise NotImplementedError


class ReachabilityAlgorithm:
    """Reachability analysis algorithm."""
    def __init__(
        self,
        dynamics: Dynamics,
        initial_states_set: Set,
        disturbances_set: Set):
        """
        Initializes the class.

        Args:
            dynamics: dynamics class to extend.
                (Dynamics)
            initial_states_set: set of initial states
                (Set)
            disturbances_set: set of disturbances
                (Set)
        """
        print("Initializing reachability algorithm with")
        print("> dynamics           =", dynamics)
        print("> initial_states_set =", initial_states_set)
        print("> disturbances_set   =", disturbances_set)
        self._dynamics_are_controlled = False
        if isinstance(dynamics, ControlledDynamics):
            self._dynamics_are_controlled = True

        self._dynamics = dynamics
        self._initial_states_set = initial_states_set
        self._disturbances_set = disturbances_set

    @property
    def dynamics(self) -> Dynamics:
        """Returns the dynamics of the class."""
        return self._dynamics

    @property
    def dynamics_are_controlled(self) -> bool:
        """Used to check if control inputs should be used."""
        return self._dynamics_are_controlled

    @property
    def initial_states_set(self) -> Set:
        """Returns the set of initial states of the class."""
        return self._initial_states_set

    @property
    def disturbances_set(self) -> Set:
        """Returns the set of disturbances of the class."""
        return self._disturbances_set

    def estimate_reachable_sets(
        self,
        discretization_time: float,
        horizon: int,
        sample_size: int,
        controls_matrix: jnp.ndarray) -> SampledReachableSetTrajectory:
        """
        Estimates the reachable sets.

        Args:
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            sample_size: number of samples used for the estimate
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            reachable_sets: trajectory of sample-based approx. of reachable sets
                (SampledReachableSetTrajectory) class
        """
        raise NotImplementedError


class Algorithm1(ReachabilityAlgorithm):
    """Algorithm returning the convex hull of a sample of extremal trajectories."""
    def __init__(
        self,
        dynamics: Dynamics,
        initial_states_set: SmoothConvexSet,
        disturbances_set: SmoothConvexSet,
        diffeomorphism_vector: jnp.array = jnp.ones(100)):
        """Initializes the class."""
        self._diffeomorphism_diagonal_matrix = jnp.diag(
            diffeomorphism_vector[:dynamics.num_states])

        if not isinstance(dynamics, ControlledDynamics):
            self._augmented_dynamics = AugmentedDynamics(
                dynamics,
                disturbances_set)
        else:
            self._augmented_dynamics = AugmentedControlledDynamics(
                dynamics,
                disturbances_set)

        super().__init__(dynamics, initial_states_set, disturbances_set)

    @property
    def augmented_dynamics(self) -> AugmentedDynamics:
        """Returns the augmented dynamics of the class."""
        return self._augmented_dynamics

    @property
    def diffeomorphism_diagonal_matrix(self) -> jnp.ndarray:
        """Returns a diagonal matrix. See @diffeomorphism_sphere(). """
        return self._diffeomorphism_diagonal_matrix

    def sample_initial_directions(self, sample_size: int) -> jnp.ndarray:
        """
        Returns a sample of initial directions to use for reachability analysis.

        Args:
            sample_size: number of initial directions to sample
                (int)

        Returns:
            initial_directions: initial directions (points on the sphere)
                (sample_size, _num_adjoint_variables) array
        """
        sphere = UnitSphere(
            self.augmented_dynamics.num_base_dynamics_states)
        initial_directions = sphere.sample(sample_size)
        return initial_directions

    def diffeomorphism_sphere(self, direction: jnp.ndarray) -> jnp.ndarray:
        # Diffeomorphism from the sphere to the sphere
        # d0\in\cS^{n-1} |-> diffeo(d0)\in\cS^{n-1}
        # The theory is unchanged: the map
        #   F(diffeo(.), t))
        # can still be used to reconstruct the reachable convex hull.
        # However, this diffeomorphism can help with the error bound:
        # by reducing the Lipschitz constants. This makes sense, since
        # our bounds are naive by considering the worst-case error in
        # every direction. Using this diffeomorphism, we can make the
        # function $F$ have approximately the same smoothness properties
        # in every direction.
        new_direction = self.diffeomorphism_diagonal_matrix @ direction
        new_direction /= jnp.linalg.norm(new_direction)
        return new_direction

    def solve_augmented_ode_fixed_initial_state(
        self,
        initial_state: jnp.ndarray,
        initial_direction: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Solves the augmented ODE_{initial_direction}.

        Args:
            initial_state: initial state
                (num_states) array
            initial_direction: initial direction parameterizing the augmented ODE
                (num_states) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            state_trajectory: state trajectory
                (horizon+1, num_states) array
            adjoint_trajectory: disturbance trajectory
                (horizon+1, num_states) array
            disturbance_trajectory: disturbance trajectory
                (horizon+1, num_disturbances) array
        """
        time_indices = jnp.arange(horizon + 1)
        times = time_indices * discretization_time
        # initial conditions
        initial_adjoint = self.diffeomorphism_sphere(initial_direction)
        initial_augmented_state = self.augmented_dynamics.augmented_state_from_state_adjoint(
            initial_state, initial_adjoint)
        # solve the ode
        def next_augmented_state_scan(augmented_state, time_index_and_control):
            time_index = time_index_and_control[0]
            disturbance = jnp.zeros(
                self.augmented_dynamics.num_disturbances)
            if not self.dynamics_are_controlled:
                next_augmented_state = self.augmented_dynamics.next_state(
                    discretization_time * time_index,
                    augmented_state,
                    disturbance,
                    discretization_time)
            else:
                control = time_index_and_control[1:]
                next_augmented_state = self.augmented_dynamics.next_state(
                    discretization_time * time_index,
                    augmented_state,
                    control,
                    disturbance,
                    discretization_time)
            return next_augmented_state, next_augmented_state
        time_indices = time_indices[:-1, jnp.newaxis]
        if not self.dynamics_are_controlled:
            # no control inputs
            time_indices_and_controls = time_indices
        else:
            time_indices_and_controls = jnp.concatenate([
                time_indices, controls_matrix], axis=1)
        _, augmented_states = scan(
            next_augmented_state_scan,
            initial_augmented_state,
            time_indices_and_controls)
        augmented_states = jnp.concatenate([
            initial_augmented_state[jnp.newaxis, :],
            augmented_states],
            axis=0)
        states, adjoints = vmap(
            self.augmented_dynamics.state_adjoint_from_augmented_state)(
            augmented_states)
        # retrieve disturbances
        disturbances_from_augmented_states = vmap(
            self.augmented_dynamics.disturbance_from_augmented_state)
        if not self.dynamics_are_controlled:
            disturbances = disturbances_from_augmented_states(
                times, augmented_states)
        else:
            controls_matrix = jnp.concatenate([
                controls_matrix, controls_matrix[-1:]],
                axis=0)
            disturbances = disturbances_from_augmented_states(
                times, augmented_states, controls_matrix)
        return states, adjoints, disturbances

    def solve_augmented_ode(
        self,
        initial_direction: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Solves the augmented ODE_{initial_direction}.

        Args:
            initial_direction: initial direction parameterizing the augmented ODE
                (num_states) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            state_trajectory: state trajectory
                (horizon+1, num_states) array
            adjoint_trajectory: disturbance trajectory
                (horizon+1, num_states) array
            disturbance_trajectory: disturbance trajectory
                (horizon+1, num_disturbances) array
        """
        initial_state = self.initial_states_set.gauss_map_inverse(
            -self.diffeomorphism_sphere(initial_direction))
        output = self.solve_augmented_ode_fixed_initial_state(
                initial_state,
                initial_direction,
                discretization_time,
                horizon,
                controls_matrix)
        return output

    def compute_extremal_trajectories_using_initial_directions(
        self,
        initial_directions: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Solves the augmented ODE from different initial directions.

        Args:
            initial_directions: initial directions of the augmented ODE
                (sample_size, num_states) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            state_trajectories: state trajectory
                (sample_size, horizon+1, num_states) array
            adjoint_trajectories: disturbance trajectory
                (sample_size, horizon+1, num_states) array
            disturbance_trajectories: disturbance trajectory
                (sample_size, horizon+1, num_disturbances) array
        """
        if not self.dynamics_are_controlled:
            controls_matrix = jnp.array([None] * horizon)

        # evaluate solutions of the augmented ODE
        states, adjoints, disturbances = vmap(
            self.solve_augmented_ode,
            in_axes=(0, None, None, None))(
            initial_directions, discretization_time, horizon, controls_matrix)
        return states, adjoints, disturbances

    def estimate_reachable_sets_using_initial_directions(
        self,
        initial_directions: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) \
        -> SampledReachableSetTrajectory:
        """
        Estimates the reachable sets.

        Args:
            initial_directions: initial directions of the augmented ODE
                (sample_size, num_states) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            reachable_sets: trajectory of sample-based approx. of reachable sets
                (SampledReachableSetTrajectory) class
        """
        trajs = self.compute_extremal_trajectories_using_initial_directions(
            initial_directions, discretization_time, horizon, controls_matrix)
        states, adjoints, disturbances = trajs
        reachable_sets = SampledReachableSetTrajectory(
            states, disturbances)
        return reachable_sets

    def estimate_reachable_sets(
        self,
        discretization_time: float,
        horizon: int,
        sample_size: int,
        controls_matrix: jnp.ndarray = None) \
        -> SampledReachableSetTrajectory:
        """
        Estimates the reachable sets.

        Args:
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            sample_size: number of samples used for the estimate
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            reachable_sets: trajectory of sample-based approx. of reachable sets
                (SampledReachableSetTrajectory) class
        """
        initial_directions = self.sample_initial_directions(sample_size)
        reachable_sets = self.estimate_reachable_sets_using_initial_directions(
            initial_directions,
            discretization_time,
            horizon,
            controls_matrix)
        return reachable_sets

    def estimation_errors(
        self,
        discretization_time: float,
        horizon: int,
        sample_size: int,
        sample_size_for_lipschitz_estimation: int = 10000,
        max_control: float = 0.0) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Solves the augmented ODE_{initial_direction}.

        Args:
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            sample_size: number of samples used for the reachable set estimates
                (int)
            sample_size_for_lipschitz_estimation: for error estimation
                (int)
            max_control: maximum magnitude of the control (infinity norm)
                (float)

        Returns:
            error_bounds_states: error bounds of the reachable set approximations
                (horizon + 1,) array
            error_bounds_controls: error bounds for the reachable controls
                (horizon,) array
        """
        base_dynamics = self.augmented_dynamics.base_dynamics
        num_directions = self.augmented_dynamics.num_base_dynamics_states
        assert num_directions == 3
        sphere = UnitSphere(num_directions)
        delta = sphere.get_internal_covering_delta(sample_size)

        # evaluate Lipschitz constants using Monte Carlo
        sample_size = sample_size_for_lipschitz_estimation
        initial_directions = sphere.sample(sample_size)
        # only used if self.dynamics_are_controlled
        controls_matrices = np.random.choice(
            [-max_control, max_control],
            (sample_size, horizon, base_dynamics.num_controls))

        def extremal_states(
            initial_direction, controls_matrix):
            states, adjoints, disturbances = self.solve_augmented_ode(
                initial_direction,
                discretization_time,
                horizon,
                controls_matrix)
            return states
        def extremal_states_d_d0(
            initial_direction, controls_matrix):
            jacobians = jacfwd(extremal_states, argnums=(0))(
                initial_direction, controls_matrix)
            return jacobians
        def extremal_states_hessian_dd_d0(
            initial_direction, controls_matrix):
            hessians = jacfwd(extremal_states_d_d0, argnums=(0))(
                initial_direction, controls_matrix)
            return hessians
        states_d_d0s = vmap(extremal_states_d_d0)(
            initial_directions, controls_matrices)
        states_dd_d0s = vmap(extremal_states_hessian_dd_d0)(
            initial_directions, controls_matrices)
        lipschitz_states = jnp.max(
            jnp.linalg.norm(states_d_d0s, axis=(2, 3)),
            axis=0)
        lipschitz_jacobian_states = jnp.max(
            jnp.sqrt(jnp.sum(jnp.square(states_dd_d0s), axis=(2, 3, 4))),
            axis=0)
        epsilons_states = lipschitz_states * delta
        epsilons_states_2nd_order = 0.5 * (
            lipschitz_states + lipschitz_jacobian_states) * delta**2
        print("error bound (1st order): Hausdorff epsilons (states) =",
            epsilons_states)
        print("error bound (2nd order): Hausdorff epsilons (states) =",
            epsilons_states_2nd_order)
        error_bounds_states = epsilons_states_2nd_order

        # same for closed-loop control trajectories
        epsilons_controls_2nd_order = jnp.zeros(horizon)
        if isinstance(base_dynamics, ControlledFeedbackDynamics):
            def extremal_controls(
                initial_direction, controls_matrix):
                states, adjoints, disturbances = self.solve_augmented_ode(
                    initial_direction,
                    discretization_time,
                    horizon,
                    controls_matrix)
                controls_closed_loop = base_dynamics.closed_loop_controls(
                    states[:-1], controls_matrix)
                return controls_closed_loop
            def extremal_controls_d_d0(
                initial_direction, controls_matrix):
                jacobians = jacfwd(extremal_controls, argnums=(0))(
                    initial_direction, controls_matrix)
                return jacobians
            def extremal_controls_hessian_dd_d0(
                initial_direction, controls_matrix):
                hessians = jacfwd(extremal_controls_d_d0, argnums=(0))(
                    initial_direction, controls_matrix)
                return hessians
            controls_d_d0s = vmap(extremal_controls_d_d0)(
                initial_directions, controls_matrices)
            controls_dd_d0s = vmap(extremal_controls_hessian_dd_d0)(
                initial_directions, controls_matrices)
            lipschitz_controls = jnp.max(
                jnp.linalg.norm(controls_d_d0s, axis=(2, 3)),
                axis=0)
            lipschitz_jacobian_controls = jnp.max(
                jnp.sqrt(jnp.sum(jnp.square(controls_dd_d0s), axis=(2, 3, 4))),
                axis=0)
            epsilons_controls = lipschitz_controls * delta
            epsilons_controls_2nd_order = 0.5 * (
                lipschitz_controls + lipschitz_jacobian_controls) * delta**2
            print("error bound (1st order): Hausdorff epsilons (controls) =",
                epsilons_controls)
            print("error bound (2nd order): Hausdorff epsilons (controls) =",
                epsilons_controls_2nd_order)
            error_bounds_controls = epsilons_controls_2nd_order

        return error_bounds_states, error_bounds_controls



class RandUP(ReachabilityAlgorithm):
    """Algorithm returning the convex hull of a sample of state trajectories."""
    def __init__(
        self,
        dynamics: Dynamics,
        initial_states_set: SmoothConvexSet,
        disturbances_set: SmoothConvexSet):
        """Initializes the class."""
        # check that the sample function is defined
        initial_states_set.sample_random(sample_size=10)
        disturbances_set.sample_random(sample_size=10)

        super().__init__(dynamics, initial_states_set, disturbances_set)

    def solve_ode(
        self,
        initial_state: jnp.ndarray,
        disturbances_matrix: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) -> jnp.ndarray:
        """
        Solves the dynamics ODE.

        Args:
            initial_state: initial state
                (num_states) array
            disturbances_matrix: disturbance trajectory
                (horizon, num_disturbances) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            states_matrix: state trajectory
                (horizon+1, num_states) array
        """
        num_disturbances = self.dynamics.num_disturbances
        times = discretization_time * jnp.arange(horizon + 1)
        def next_state_scan(state, time_disturbance_control):
            time = time_disturbance_control[0]
            disturbance = time_disturbance_control[1:1+num_disturbances]
            if not self.dynamics_are_controlled:
                next_state = self.dynamics.next_state(
                    time,
                    state,
                    disturbance,
                    discretization_time)
            else:
                control = time_disturbance_control[1+num_disturbances:]
                next_state = self.dynamics.next_state(
                    time,
                    state,
                    control,
                    disturbance,
                    discretization_time)
            return next_state, next_state
        times_controls_disturbances = jnp.concatenate([
            times[:-1, jnp.newaxis],
            disturbances_matrix], axis=1)
        if self.dynamics_are_controlled:
            times_controls_disturbances = jnp.concatenate([
                times_controls_disturbances,
                controls_matrix], axis=1)
        _, states_matrix = scan(
            next_state_scan,
            initial_state,
            times_controls_disturbances)
        states_matrix = jnp.concatenate([
            initial_state[jnp.newaxis, :],
            states_matrix], axis=0)
        return states_matrix

    def solve_odes_using_initial_states_and_disturbances(
        self,
        initial_states: jnp.ndarray,
        disturbances_matrices: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) -> jnp.ndarray:
        """
        Solves the augmented ODE from different initial directions.

        Args:
            initial_states: initial states of the ODE
                (sample_size, num_states) array
            disturbances_matrices: disturbance trajectories
                (sample_size, horizon, num_disturbances) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            states_matrices: state trajectories
                (sample_size, horizon+1, num_states) array
        """
        if not self.dynamics_are_controlled:
            controls_matrix = jnp.array([None] * horizon)
        states_matrices = vmap(
            self.solve_ode,
            in_axes=(0, 0, None, None, None))(
            initial_states, disturbances_matrices,
            discretization_time, horizon, controls_matrix)
        return states_matrices

    def estimate_reachable_sets_using_initial_states_and_disturbances(
        self,
        initial_states: jnp.ndarray,
        disturbances_matrices: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray = None) \
        -> SampledReachableSetTrajectory:
        """
        Estimates the reachable sets.

        Args:
            initial_states: initial states of the ODE
                (sample_size, num_states) array
            disturbances_matrices: disturbance trajectories
                (sample_size, horizon, num_disturbances) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            reachable_sets: trajectory of sample-based approx. of reachable sets
                (SampledReachableSetTrajectory) class
        """
        # evaluate solutions of the augmented ODE
        states = self.solve_odes_using_initial_states_and_disturbances(
            initial_states, disturbances_matrices,
            discretization_time, horizon, controls_matrix)
        # append timestep so of size (sample_size, horizon+1, num_disturbances)
        disturbances = jnp.concatenate([
            disturbances_matrices,
            disturbances_matrices[:, -1:, :]], axis=1)
        # return as a reachable set
        reachable_sets = SampledReachableSetTrajectory(states, disturbances)
        return reachable_sets

    def estimate_reachable_sets(
        self,
        discretization_time: float,
        horizon: int,
        sample_size: int,
        controls_matrix: jnp.ndarray = None) \
        -> SampledReachableSetTrajectory:
        """
        Estimates the reachable sets.

        Args:
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            sample_size: number of samples used for the estimate
                (int)
            controls_matrix: controls matrix. 
                Only used if dynamics are ControlledDynamics
                (horizon, num_controls) array

        Returns:
            reachable_sets: trajectory of sample-based approx. of reachable sets
                (SampledReachableSetTrajectory) class
        """
        initial_states = self.initial_states_set.sample_random(sample_size)
        disturbances_matrices = self.disturbances_set.sample_random(
            horizon * sample_size)
        disturbances_matrices = jnp.reshape(
            disturbances_matrices, (sample_size, horizon, -1))
        rsets = self.estimate_reachable_sets_using_initial_states_and_disturbances(
            initial_states, disturbances_matrices,
            discretization_time, horizon, controls_matrix)
        return rsets


class LipschitzReachabilityAlgorithm(ReachabilityAlgorithm):
    """
    Lipschitz-based reachability analysis algorithm."""
    def __init__(
        self,
        dynamics: ControlledDynamics,
        initial_states_set: Point,
        disturbances_set: Ball):
        """
        Initializes the class.
        """
        if not isinstance(dynamics, ControlledDynamics):
            raise NotImplementedError
        if not isinstance(initial_states_set, Point):
            raise NotImplementedError
        if not isinstance(disturbances_set, Ball):
            raise NotImplementedError

        super().__init__(dynamics, initial_states_set, disturbances_set)

    def evaluate_hessian_bound(
        self,
        discretization_time: float,
        sample_size_for_lipschitz_estimation: int = 10000,
        max_state: float = 0.0,
        max_control: float = 0.0) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes a bound on the lipschitz constant of the gradient of the one 
        step state prediction.

        Args:
            discretization_time: discretization time (dt)
                (float)
            sample_size_for_lipschitz_estimation: for error estimation
                (int)
            max_state: maximum magnitude of the state (infinity norm)
                (float)
            max_control: maximum magnitude of the control (infinity norm)
                (float)

        Returns:
            hessian_bound: bound on the lipschitz constant of the gradient of
                the one step state prediction.
                (float)
        """
        num_states = self.dynamics.num_states
        num_controls = self.dynamics.num_controls
        states = uniformly_sample_points_in_rectangle(
            np.zeros(self.dynamics.num_states),
            max_state * np.ones(num_states),
            sample_size_for_lipschitz_estimation)
        controls = np.random.choice(
            [-max_control, max_control],
            (sample_size_for_lipschitz_estimation, num_controls))

        def next_state_hessian(state, control):
            def next_state(state, control):
                time = 0 # assumes time invariant dynamics
                next_state = self.dynamics.next_state(
                    time,
                    state,
                    control,
                    jnp.zeros(self.dynamics.num_disturbances),
                    discretization_time)
                return next_state
            hessian = jacfwd(jacfwd(next_state, argnums=(0)), argnums=(0))(
                state, control)
            return hessian

        hessians = vmap(next_state_hessian)(states, controls)
        hessian_bound = jnp.max( # Frobenius norm
            jnp.sum(jnp.abs(hessians), axis=(1, 2, 3)))
        return hessian_bound

    def next_ellipsoidal_uncertainty_shape_matrix(
        self,
        time: float,
        state: jnp.ndarray,
        state_shape_matrix: jnp.ndarray,
        control: jnp.ndarray,
        discretization_time: float,
        hessian_bound: float):
        """
        Propagates the ellipsoidal uncertainty set forward in time.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            shape_matrix: shape matrix represented the ellipsoidal set (Q var)
                (num_states, num_states) array
            control: control applied to the system (u variable)
                (num_controls) array
            discretization_time: discretization time (dt)
                (float)
            hessian_bound: bound on the lipschitz constant of the gradient of
                the one step state prediction.
                (float)
        Returns:
            shape_matrix_next: shape matrix at the next timestep
                (num_states, num_states) array
        """
        # linear component
        jacobian = jacfwd(self.dynamics.next_state, argnums=1)(
            time,
            state,
            control,
            jnp.zeros(self.dynamics.num_disturbances),
            discretization_time)
        shape_matrix_nominal_next = jacobian @ state_shape_matrix @ jacobian.T
        # additive disturbance
        deltas_disturbances = discretization_time * self.disturbances_set.radius
        # linearization error bound via Taylor remainder
        shape_max_eigenvalue = jnp.max(jnp.linalg.eigh(state_shape_matrix)[0])
        deltas_state_taylor = 0.5 * hessian_bound * shape_max_eigenvalue
        # disturbance + linearization parts
        num_states = self.dynamics.num_states
        shape_matrix_disturbance_taylor = num_states * jnp.eye(num_states) * (
            deltas_state_taylor + deltas_disturbances)**2
        # bound sum of ellipsoids
        # 1) nominal + (disturbance + linearization error)
        c  = jnp.sqrt(jnp.trace(shape_matrix_nominal_next) /
                      jnp.trace(shape_matrix_disturbance_taylor))
        shape_matrix_next = ((c + 1) / c) * shape_matrix_nominal_next + \
                            (1 + c) * shape_matrix_disturbance_taylor
        return shape_matrix_next

    def ellipsoidal_uncertainty_shape_matrices_trajectory(
        self,
        initial_state: jnp.ndarray,
        discretization_time: float,
        horizon: int,
        controls_matrix: jnp.ndarray,
        hessian_bound: float = 0.0) -> jnp.ndarray:
        """
        Solves the augmented ODE_{initial_direction}.

        Args:
            initial_state: initial state
                (num_states) array
            discretization_time: discretization time (dt)
                (float)
            horizon: prediction horizon (N)
                (int)
            hessian_bound: bound on the lipschitz constant of the gradient of
                the one step state prediction.
                (float)
            controls_matrix: controls matrix. 
                (horizon, num_controls) array

        Returns:
            states_matrix: trajectory of states (centers of the ellipsoids)
                (horizon+1, num_states) array
            shape_matrices: trajectory of shape matrices
                (horizon+1, num_states, num_states) array
        """
        num_states = self.dynamics.num_states
        times = discretization_time * jnp.arange(horizon + 1)
        # get nominal state trajectory
        def next_state_scan(state, time_control):
            time = time_control[0]
            control = time_control[1:]
            next_state = self.dynamics.next_state(
                time,
                state,
                control,
                jnp.zeros(self.dynamics.num_disturbances),
                discretization_time)
            return next_state, next_state
        times_controls = jnp.concatenate([
            times[:-1, jnp.newaxis],
            controls_matrix], axis=1)
        _, states_matrix = scan(
            next_state_scan,
            initial_state,
            times_controls)
        states_matrix = jnp.concatenate([
            initial_state[jnp.newaxis, :],
            states_matrix], axis=0)
        # get shape matrix trajectory
        initial_shape_matrix = jnp.zeros((num_states, num_states))
        first_shape_matrix = (
            discretization_time *
            self.disturbances_set.radius**2 *
            np.eye(num_states))
        def next_shape_matrix_scan(shape_matrix, time_state_control):
            time = time_state_control[0]
            state = time_state_control[1:1+num_states]
            control = time_state_control[1+num_states:]
            next_shape_matrix = self.next_ellipsoidal_uncertainty_shape_matrix(
                time,
                state,
                shape_matrix,
                control,
                discretization_time,
                hessian_bound)
            return next_shape_matrix, next_shape_matrix
        times_states_controls = jnp.concatenate([
            times[1:-1, jnp.newaxis],
            states_matrix[1:-1],
            controls_matrix[1:]], axis=1)
        _, shape_matrices = scan(
            next_shape_matrix_scan,
            first_shape_matrix,
            times_states_controls)
        shape_matrices = jnp.concatenate([
            initial_shape_matrix[jnp.newaxis, :],
            first_shape_matrix[jnp.newaxis, :],
            shape_matrices],
            axis=0)
        return states_matrix, shape_matrices
