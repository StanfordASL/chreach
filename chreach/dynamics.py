from typing import Tuple, Callable

import jax.numpy as jnp
from jax import vmap, jacfwd
from jax.lax import fori_loop
from jax.config import config

from chreach.sets import SmoothConvexSet


class Dynamics:
    """
    Dynamics class for dynamical system represented by the ODE
    d/dt x(t) = f(t, x(t)) + g(t, x(t))w(t).
    """
    def __init__(
        self,
        num_states: int,
        num_disturbances: int):
        """
        Initializes the class.

        Args:
            num_states: number of states
                (int)
            num_disturbances: number of disturbances
                (int)
        """
        print("Initializing dynamics with")
        print("> num_states       =", num_states)
        print("> num_disturbances =", num_disturbances)
        self._num_states = num_states
        self._num_disturbances = num_disturbances

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._num_states

    @property
    def num_disturbances(self) -> int:
        """Returns the number of disturbance variables."""
        return self._num_disturbances

    def f(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector field value f(time, state) of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array

        Returns:
            f_value: value f(t, x)
                (num_states) array
        """
        raise NotImplementedError

    def g(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector field value g(time, state) of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array

        Returns:
            g_value: value g(t, x)
                (num_states, num_disturbances) array
        """
        raise NotImplementedError

    def f_dx(
        self,
        time: float,
        state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian of f(time, state) of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array

        Returns:
            f_dx_value: value of the Jacobian of f(t, x)
                (num_states, num_states) array
        """
        f_dx_value = jacfwd(self.f, argnums=(1))(time, state)
        return f_dx_value

    def gw_dx(
        self,
        time: float,
        state: jnp.ndarray,
        disturbance: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian of g(time, state)@disturbance of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            disturbance: disturbance applied to the system (w variable)
                (num_disturbances) array

        Returns:
            gw_dx_value: value of the Jacobian of g(t, x)@w
                (num_states, num_states) array
        """
        def gw(time: float, state: jnp.ndarray, disturbance: jnp.ndarray):
            gw_value = self.g(time, state) @ disturbance
            return gw_value
        gw_dx_value = jacfwd(gw, argnums=(1))(time, state, disturbance)
        return gw_dx_value

    def state_dot(
        self,
        time: float,
        state: jnp.ndarray,
        disturbance: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the time derivative of the state
        d/dt x(t) = f(t, x(t)) + g(t, x(t))w(t).

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            disturbance: disturbance applied to the system (w variable)
                (num_disturbances) array

        Returns:
            state_dot_value: value of the state time derivative (d/dt x(t))
                (num_states) array
        """
        state_dot = self.f(time, state) + self.g(time, state) @ disturbance
        return state_dot

    def next_state(
        self,
        time: float,
        state: jnp.ndarray,
        disturbance: jnp.ndarray,
        discretization_time: float) -> jnp.ndarray:
        """
        Integrates the dynamics forward to get x(t+dt) from (t, x(t), w(t))

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            disturbance: disturbance applied to the system (w variable)
                (num_disturbances) array
            discretization_time: discretization time (dt)
                (float)

        Returns:
            next_state_value: value of the state at time t+dt
                (num_states) array
        """
        dt = discretization_time
        k1 = self.state_dot(
            time, state, disturbance)
        k2 = self.state_dot(
            time + 0.5 * dt, state + 0.5 * dt * k1, disturbance)
        k3 = self.state_dot(
            time + 0.5 * dt, state + 0.5 * dt * k2, disturbance)
        k4 = self.state_dot(
            time + dt, state + dt * k3, disturbance)
        state_next = state + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4) * dt
        return state_next


class AttractionRepulsionDynamics(Dynamics):
    """Simple 2-dimensional system with attraction - repulsion terms."""
    def __init__(self):
        """Initializes the class."""
        num_states = 2
        num_disturbances = 2

        self._repulsion_position = jnp.zeros(2)
        self._attraction_position = jnp.array([1., 0.])

        super().__init__(num_states, num_disturbances)

    @property
    def repulsion_position(self) -> jnp.ndarray:
        """Returns the repulsion position."""
        return self._repulsion_position

    @property
    def attraction_position(self) -> jnp.ndarray:
        """Returns the attraction position."""
        return self._attraction_position

    def f(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        r_attraction = self.attraction_position - state
        r_repulsion = self.repulsion_position - state
        f_attraction = r_attraction / jnp.linalg.norm(r_attraction)
        f_repulsion = -r_repulsion / jnp.linalg.norm(r_repulsion)
        f_value = f_attraction + f_repulsion
        return f_value

    def g(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        g_value = jnp.eye(self.num_states)
        return g_value


class DubinsDynamics(Dynamics):
    """
    Dubins car dynamics, with state = (position_x, position_y, theta_angle).
    """
    def __init__(
        self,
        turning_speed: float = 0.1,
        forward_speed: float = 0.3,
        g_matrix: jnp.ndarray = jnp.eye(3)):
        """
        Initializes the class.

        Args:
            turning_speed: turning speed
                (float)
            forward_speed: forward speed
                (float)
            g_matrix: matrix multiplying the disturbances
                (num_states, num_disturbances)
        """
        num_states = 3
        if len(g_matrix.shape) != 2:
            raise ValueError("g_matrix should be a matrix.")
        if g_matrix.shape[0] != num_states:
            raise ValueError("g_matrix.shape[0] should be num_states.")
        num_disturbances = g_matrix.shape[1]
        if num_disturbances < 1 or num_disturbances > num_states:
            raise ValueError("g_matrix.shape[1] should be in {1, 2, 3}.")

        self._turning_speed = turning_speed
        self._forward_speed = forward_speed
        self._g_matrix = g_matrix

        super().__init__(num_states, num_disturbances)

    @property
    def turning_speed(self) -> float:
        """Returns the turning speed."""
        return self._turning_speed

    @property
    def forward_speed(self) -> float:
        """Returns the forward speed."""
        return self._forward_speed

    @property
    def g_matrix(self) -> jnp.ndarray:
        """Returns the g_matrix defining the dynamics."""
        return self._g_matrix

    def f(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        pos_x, pos_y, theta = state
        # dynamics
        pos_x_dot = self.forward_speed * jnp.cos(theta)
        pos_y_dot = self.forward_speed * jnp.sin(theta)
        theta_dot = self.turning_speed
        f_value = jnp.array([pos_x_dot, pos_y_dot, theta_dot])
        return f_value

    def g(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        g_value = self.g_matrix
        return g_value


class GaddedDynamics(Dynamics):
    """
    Dynamics class for a system with additional disturbances.

    Given an original system with dynamics
        d/dt x(t) = f(t, x(t)) + g(t, x(t)) w(t),
    defines a new system with dynamics
    >>>>   d/dt x(t) = f(t, x(t)) + g(t, x(t)) w(t) +         <<<<
    >>>>               epsilon * g_to_add(t, x(t)) w_to_add   <<<<
    where g_to_add multiplies the additional disturbances w_to_add.
    """
    def __init__(
        self,
        dynamics: Dynamics,
        g_to_add: Callable[[float, jnp.ndarray], jnp.ndarray],
        epsilon: float):
        """
        Initializes the class.

        Args:
            dynamics: dynamics class to extend.
                (Dynamics)
            g_to_add: 
                with the following definition:
                    def g_to_add(time: float, state: jnp.ndarray) -> jnp.ndarray
                    Args:
                        time: current time (t variable)
                            (float)
                        state: state of the system of dynamics (x variable)
                            (num_states) array
                    Returns:
                        g_to_add_value: value g_to_add(t, x)
                            (num_states, num_disturbances_to_add) array
            epsilon: smoothing parameter multiplying g_to_add
                (float)
        """
        num_states = dynamics.num_states
        self._base_dynamics = dynamics
        self._g_to_add = g_to_add
        self._epsilon = epsilon
        self._num_disturbances_to_add = self.g_to_add(
            time=0, state=jnp.zeros(num_states)).shape[1]

        super().__init__(
            num_states = num_states,
            num_disturbances = (
                self.num_disturbances_base +
                self.num_disturbances_to_add)
            )

    @property
    def base_dynamics(self) -> Dynamics:
        """Returns the base dynamics."""
        return self._base_dynamics

    @property
    def num_disturbances_base(self) -> int:
        """Returns the number of state variables of the dynamics."""
        return self.base_dynamics.num_disturbances

    @property
    def num_disturbances_to_add(self) -> int:
        """Returns the number of added disturbances."""
        return self._num_disturbances_to_add

    @property
    def epsilon(self) -> float:
        """Returns the epsilon smoothing parameter."""
        return self._epsilon

    def g_to_add(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the additional vector field value."""
        return self._g_to_add(time, state)

    def g(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        g_base_value = self.base_dynamics.g(time, state)
        g_to_add_value = self.epsilon * self.g_to_add(time, state)
        g_value = jnp.concatenate([
            g_base_value, g_to_add_value], axis=1)
        return g_value

    def f(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        return self.base_dynamics.f(time, state)


class AugmentedDynamics(Dynamics):
    """
    Dynamics class for the system represented by the augmented ODE

    d/dt x(t) = f(t, x(t)) + g(t, x(t))w(t).
    d/dt p(t) = -( ∇f(t, x(t)) + ∇g(t, x(t))w(t) )^T p(t),
         w(t) = (n^∂W)^{-1}( g(t, x(t))^T p(t) / ||g(t, x(t))^T p(t)||)
    """
    def __init__(
        self,
        dynamics: Dynamics,
        disturbances_set: SmoothConvexSet):
        """
        Initializes the class.

        Args:
            dynamics: dynamics class to extend.
                (Dynamics)
            disturbances_set: set of disturbances
                (SmoothConvexSet)
        """
        print("Initializing augmented dynamics with")
        print("> dynamics         =", dynamics)
        print("> disturbances_set =", disturbances_set)
        self._base_dynamics = dynamics
        self._disturbances_set = disturbances_set

        num_augmented_states = 2 * self.num_base_dynamics_states
        num_disturbances = 0
        self._num_dynamics_states = self.num_base_dynamics_states
        super().__init__(
            num_states = num_augmented_states,
            num_disturbances = num_disturbances)

    @property
    def base_dynamics(self) -> Dynamics:
        """Returns the base dynamics."""
        return self._base_dynamics

    @property
    def disturbances_set(self) -> SmoothConvexSet:
        """Returns the set of disturbances."""
        return self._disturbances_set

    @property
    def num_base_dynamics_states(self) -> int:
        """Returns the number of state variables of the base dynamics."""
        return self.base_dynamics.num_states

    def augmented_state_from_state_adjoint(
        self,
        state: jnp.ndarray,
        adjoint: jnp.ndarray) -> jnp.ndarray:
        augmented_state = jnp.concatenate([state, adjoint])
        return augmented_state

    def state_adjoint_from_augmented_state(
        self,
        augmented_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        state = augmented_state[:self.num_base_dynamics_states]
        adjoint = augmented_state[self.num_base_dynamics_states:]
        return state, adjoint

    def disturbance_from_augmented_state(
        self, time: float, augmented_state: jnp.ndarray) -> jnp.ndarray:
        state, adjoint = self.state_adjoint_from_augmented_state(
            augmented_state)
        g_value = self.base_dynamics.g(time, state)
        gp_value = g_value.T @ adjoint
        disturbance = self.disturbances_set.gauss_map_inverse(
            -gp_value / jnp.linalg.norm(gp_value))
        return disturbance

    def f_augmented(
        self,
        time: float,
        augmented_state: jnp.ndarray) -> jnp.ndarray:
        state, adjoint = self.state_adjoint_from_augmented_state(
            augmented_state)
        disturbance = self.disturbance_from_augmented_state(
            time, augmented_state)

        state_dot = self.base_dynamics.state_dot(
            time, state, disturbance)
        adjoint_dot = -(
            self.base_dynamics.f_dx(time, state) +
            self.base_dynamics.gw_dx(time, state, disturbance)).T @ adjoint

        aug_state_dot = self.augmented_state_from_state_adjoint(
            state_dot, adjoint_dot)
        return aug_state_dot

    def f(self, time: float, augmented_state: jnp.ndarray) -> jnp.ndarray:
        return self.f_augmented(time, augmented_state)

    def g(self, time: float, state: jnp.ndarray) -> jnp.ndarray:
        g_value = jnp.zeros((self.num_states, self.num_disturbances))
        return g_value
