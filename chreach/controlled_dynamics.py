"""Controlled dynamics classes."""
from typing import Tuple

import jax.numpy as jnp
from jax import vmap, jacfwd

from chreach.sets import SmoothConvexSet


class ControlledDynamics:
    """
    Dynamics class for dynamical system represented by the ODE
    d/dt x(t) = f(t, x(t), u(t)) + g(t, x(t), u(t))w(t).
    """
    def __init__(
        self,
        num_states: int,
        num_controls: int,
        num_disturbances: int):
        """
        Initializes the class.

        Args:
            num_states: number of states
                (int)
            num_controls: number of controls
                (int)
            num_disturbances: number of disturbances
                (int)
        """
        print("Initializing controlled dynamics with")
        print("> num_states       =", num_states)
        print("> num_controls     =", num_controls)
        print("> num_disturbances =", num_disturbances)
        self._num_states = num_states
        self._num_controls = num_controls
        self._num_disturbances = num_disturbances

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._num_states

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self._num_controls

    @property
    def num_disturbances(self) -> int:
        """Returns the number of disturbance variables."""
        return self._num_disturbances

    def f(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector field value f(time, state, control) of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: control applied to the system (u variable)
                (num_controls) array

        Returns:
            f_value: value f(t, x, u)
                (num_states) array
        """
        raise NotImplementedError

    def g(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector field value g(time, state, control) of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: control applied to the system (u variable)
                (num_controls) array

        Returns:
            g_value: value g(t, x, u)
                (num_states, num_disturbances) array
        """
        raise NotImplementedError

    def f_dx(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian of f(time, state, control) of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: control applied to the system (u variable)
                (num_controls) array

        Returns:
            f_dx_value: value of the Jacobian of f(t, x, u)
                (num_states, num_states) array
        """
        f_dx_value = jacfwd(self.f, argnums=1)(time, state, control)
        return f_dx_value

    def gw_dx(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray,
        disturbance: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian of g(time, state, control) @ disturbance.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: control applied to the system (u variable)
                (num_controls) array
            disturbance: disturbance applied to the system (w variable)
                (num_disturbances) array

        Returns:
            gw_dx_value: value of the Jacobian of g(t, x, u)@w
                (num_states, num_states) array
        """
        def gw(time, state, control, disturbance):
            gw_value = self.g(time, state, control) @ disturbance
            return gw_value
        gw_dx_value = jacfwd(gw, argnums=1)(time, state, control, disturbance)
        return gw_dx_value

    def state_dot(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray,
        disturbance: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the time derivative of the state
        d/dt x(t) = f(t, x(t), u(t)) + g(t, x(t), u(t))w(t).

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: control applied to the system (u variable)
                (num_controls) array
            disturbance: disturbance applied to the system (w variable)
                (num_disturbances) array

        Returns:
            state_dot_value: value of the state time derivative (d/dt x(t))
                (num_states) array
        """
        state_dot = self.f(time, state, control)
        state_dot = state_dot + self.g(time, state, control) @ disturbance
        return state_dot

    def next_state(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray,
        disturbance: jnp.ndarray,
        discretization_time: float) -> jnp.ndarray:
        """
        Integrates the dynamics forward to get x(t+dt) from (t, x(t), w(t))

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: control applied to the system (u variable)
                (num_controls) array
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
            time,            state,                 control, disturbance)
        k2 = self.state_dot(
            time + 0.5 * dt, state + 0.5 * dt * k1, control, disturbance)
        k3 = self.state_dot(
            time + 0.5 * dt, state + 0.5 * dt * k2, control, disturbance)
        k4 = self.state_dot(
            time + dt,        state + dt * k3,      control, disturbance)
        state_next = state + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4) * dt
        return state_next


class ControlledFeedbackDynamics(ControlledDynamics):
    """
    Controlled dynamics with linear feedback.

    The dynamics are given by
        state_dot = f(time, state, control + feedback_gain @ state)
                + g(time, state, control + feedback_gain @ state) @ disturbance
    """
    def __init__(
        self,
        num_states: int,
        num_controls: int,
        num_disturbances: int,
        feedback_gain: jnp.ndarray = jnp.zeros((3, 3))):
        """
        Initializes the class.

        """
        self._feedback_gain = feedback_gain
        assert feedback_gain.shape[0] == num_controls
        assert feedback_gain.shape[1] == num_states

        super().__init__(num_states, num_controls, num_disturbances)

    @property
    def feedback_gain(self) -> jnp.ndarray:
        """Returns the feedback gain."""
        return self._feedback_gain

    def closed_loop_control(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """
        Closed-loop feedback control applied to the system.

        Args:
            state: state of the system (x variable)
                (num_states) array
            control: nominal control applied to the system (u variable)
                (num_controls) array

        Returns:
            control: feedback control applied to the system (u variable)
                (num_controls) array
        """
        control_fb = control + self.feedback_gain @ state
        return control_fb

    def closed_loop_controls(
        self,
        states_matrix: jnp.ndarray,
        controls_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Adds state feedback to the control inputs.

        Args:
            states_matrix: states matrix
                (horizon, num_states) array
            controls_matrix: controls matrix
                (horizon, num_controls) array

        Returns:
            controls_matrix: controls matrix with feedback
                (horizon, num_controls) array
        """
        controls_matrix = vmap(self.closed_loop_control)(
            states_matrix, controls_matrix)
        return controls_matrix

    def f(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector field value
            f(time, state, control + feedback_gain @ state)
        of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: open-loop control applied to the system (u variable)
                (num_controls) array

        Returns:
            f_value: value f(t, x, u + feedback_gain @ state)
                (num_states) array
        """
        raise NotImplementedError

    def g(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector field value 
            g(time, state, control + feedback_gain @ state)
        of the dynamics.

        Args:
            time: current time (t variable)
                (float)
            state: state of the system (x variable)
                (num_states) array
            control: open-loop control applied to the system (u variable)
                (num_controls) array

        Returns:
            g_value: value g(t, x, u + feedback_gain @ state)
                (num_states, num_disturbances) array
        """
        raise NotImplementedError


class SpacecraftVelocityDynamics(ControlledFeedbackDynamics):
    """Controlled velocity dynamics."""
    def __init__(
        self,
        inertia_matrix: jnp.ndarray = jnp.diag(jnp.array([5., 2., 1.])),
        feedback_gain: jnp.ndarray = jnp.zeros((3, 3))):
        """
        Initializes the class.

        The dynamics are given by
        state_dot = f(time, state, control + feedback_gain @ state)
                + g(time, state, control + feedback_gain @ state) @ disturbance
        """
        num_states = 3
        num_controls = 3
        num_disturbances = 3

        self._inertia = inertia_matrix
        self._inertia_inverse = jnp.linalg.inv(inertia_matrix)
        self._feedback_gain = feedback_gain
        assert feedback_gain.shape[0] == num_controls
        assert feedback_gain.shape[1] == num_states

        super().__init__(
            num_states,
            num_controls,
            num_disturbances,
            feedback_gain)

    @property
    def inertia(self) -> jnp.ndarray:
        """Returns the inertia matrix of the spacecraft."""
        return self._inertia

    @property
    def inertia_inverse(self) -> jnp.ndarray:
        """Returns the inverse of the inertia matrix of the spacecraft."""
        return self._inertia_inverse

    def f(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        omega = state
        ox, oy, oz = omega
        omega_cross = jnp.array([
            [0, -oz, oy],
            [oz, 0, -ox],
            [-oy, ox, 0]])

        control_feedback = self.closed_loop_control(state, control)

        omega_dot = self.inertia_inverse @ (
            control_feedback - omega_cross @ self.inertia @ omega)
        return omega_dot

    def g(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        g_value = self.inertia_inverse
        return g_value


class SpacecraftDynamics(ControlledFeedbackDynamics):
    """Spacecraft dynamics."""
    def __init__(
        self,
        inertia_matrix: jnp.ndarray = jnp.diag(jnp.array([5., 2., 1.])),
        feedback_gain: jnp.ndarray = jnp.zeros((3, 7))):
        """
        Initializes the class.

        The dynamics are given by
        state_dot = f(time, state, control + feedback_gain @ state)
                + g(time, state, control + feedback_gain @ state) @ disturbance
        """
        num_states = 7
        num_controls = 3
        num_disturbances = 3

        self._inertia = inertia_matrix
        self._inertia_inverse = jnp.linalg.inv(inertia_matrix)
        self._velocity_dynamics = SpacecraftVelocityDynamics(
            inertia_matrix, feedback_gain[:, 4:])
        self._feedback_gain = feedback_gain
        assert feedback_gain.shape[0] == num_controls
        assert feedback_gain.shape[1] == num_states

        super().__init__(
            num_states,
            num_controls,
            num_disturbances,
            feedback_gain)

    @property
    def velocity_dynamics(self) -> ControlledDynamics:
        """Returns the rotational dynamics of the spacecraft."""
        return self._velocity_dynamics

    @property
    def inertia(self) -> jnp.ndarray:
        """Returns the inertia matrix of the spacecraft."""
        return self._inertia

    @property
    def inertia_inverse(self) -> jnp.ndarray:
        """Returns the inverse of the inertia matrix of the spacecraft."""
        return self._inertia_inverse

    @property
    def feedback_gain(self) -> jnp.ndarray:
        """Returns the feedback gain."""
        return self._feedback_gain

    def state_to_quaternion_and_angular_velocity(
        self,
        state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the quaternion and angular velocity components of the state.

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            quaternion: quaternion
                (4) array
            omega: angular velocity
                (3) array
        """
        quaternion, omega = state[:4], state[4:]
        return quaternion, omega

    def f(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        quaternion, omega = self.state_to_quaternion_and_angular_velocity(state)
        ox, oy, oz = omega

        # control_feedback = self.closed_loop_control(state, control)

        omega_matrix = 0.5 * jnp.array([
            [0, -ox, -oy, -oz],
            [ox, 0, oz, -oy],
            [oy, -oz, 0, ox],
            [oz, oy, -ox, 0]])
        quaternion_dot = omega_matrix @ quaternion

        omega_dot = self.velocity_dynamics.f(time, omega, control)

        state_dot = jnp.concatenate([quaternion_dot, omega_dot])
        return state_dot

    def g(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        g_value = jnp.concatenate([
            jnp.zeros((4, 3)),
            self.velocity_dynamics.g(time, state, control)],
            axis=0)
        return g_value


class AugmentedControlledDynamics(ControlledDynamics):
    """
    Dynamics class for the system represented by the controlled augmented ODE

    d/dt x(t) = f(t, x(t), u(t)) + g(t, x(t), u(t))w(t).
    d/dt p(t) = -( ∇f(t, x(t), u(t)) + ∇g(t, x(t), u(t))w(t) )^T p(t),
         w(t) = (n^∂W)^{-1}( g(t, x(t), u(t))^T p(t) / ||g(t, x(t), u(t))^T p(t)||)
    """
    def __init__(
        self,
        dynamics: ControlledDynamics,
        disturbances_set: SmoothConvexSet):
        """
        Initializes the class.

        Args:
            dynamics: dynamics class to extend.
                (ControlledDynamics)
            disturbances_set: set of disturbances
                (SmoothConvexSet)
        """
        assert isinstance(dynamics, ControlledDynamics)
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
            num_controls = self.base_dynamics.num_controls,
            num_disturbances = num_disturbances)

    @property
    def base_dynamics(self) -> ControlledDynamics:
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
        """Gets the state variables from the augmented state-adjoint vector."""
        augmented_state = jnp.concatenate([state, adjoint])
        return augmented_state

    def state_adjoint_from_augmented_state(
        self,
        augmented_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Gets the adjoint variables from the augmented state-adjoint."""
        state = augmented_state[:self.num_base_dynamics_states]
        adjoint = augmented_state[self.num_base_dynamics_states:]
        return state, adjoint

    def disturbance_from_augmented_state(
        self,
        time: float,
        augmented_state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """Gets the disturbance variables from the augmented state-adjoint."""
        state, adjoint = self.state_adjoint_from_augmented_state(
            augmented_state)
        g_value = self.base_dynamics.g(time, state, control)
        gp_value = g_value.T @ adjoint
        disturbance = self.disturbances_set.gauss_map_inverse(
            -gp_value / jnp.linalg.norm(gp_value))
        return disturbance

    def f_augmented(
        self,
        time: float,
        augmented_state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        """Augmented vector field. Appends the adjoint dynamics."""
        state, adjoint = self.state_adjoint_from_augmented_state(
            augmented_state)
        disturbance = self.disturbance_from_augmented_state(
            time, augmented_state, control)

        state_dot = self.base_dynamics.state_dot(
            time, state, control, disturbance)
        adjoint_dot = -(
            self.base_dynamics.f_dx(time, state, control) +
            self.base_dynamics.gw_dx(
                time, state, control, disturbance)).T @ adjoint

        aug_state_dot = self.augmented_state_from_state_adjoint(
            state_dot, adjoint_dot)
        return aug_state_dot

    def f(
        self,
        time: float,
        augmented_state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        return self.f_augmented(time, augmented_state, control)

    def g(
        self,
        time: float,
        state: jnp.ndarray,
        control: jnp.ndarray) -> jnp.ndarray:
        g_value = jnp.zeros((self.num_states, self.num_disturbances))
        return g_value
