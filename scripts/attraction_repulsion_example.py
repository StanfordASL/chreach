"""Example with attraction-repulsion dynamics."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chreach.dynamics import AttractionRepulsionDynamics
from chreach.sets import Point, Ball
from chreach.reach import Algorithm1, PlotType

# hyperparameters
final_time = 7.5
prediction_horizon = 200
discretization_time = final_time / prediction_horizon
sample_size = 3000

# instantiate the problem
dynamics = AttractionRepulsionDynamics()
initial_states_set = Point(
    position=jnp.array([-1., 0.]))
disturbances_set = Ball(
    center=jnp.zeros(2), radius=0.1)
reach_alg = Algorithm1(
    dynamics, initial_states_set, disturbances_set)

# sample initial values of d0
assert dynamics.num_states == 2
theta_vals1 = jnp.linspace(0, jnp.pi-1e-2, sample_size)
theta_vals2 = jnp.linspace(jnp.pi-1e-2, jnp.pi+1e-2, sample_size+1)
theta_vals3 = jnp.linspace(jnp.pi+1e-2, 2*jnp.pi, sample_size)
theta_vals = jnp.concatenate((
    theta_vals1, theta_vals2, theta_vals3))
d0s_x, d0s_y = jnp.cos(theta_vals), jnp.sin(theta_vals)
initial_directions = jnp.stack((d0s_x, d0s_y)).T

# estimate the reachable sets
sets = reach_alg.estimate_reachable_sets_using_initial_directions(
    initial_directions,
    discretization_time,
    prediction_horizon)
print("Finished computing reachable sets.")

# plot
fig = plt.figure(figsize=[5, 5])
ax = plt.gca()
sets.plot(
    ax,
    times_to_plot=[-1],
    plot_type=PlotType.PLOT,
    color='b', alpha=1.0)
sets.plot(
    ax,
    times_to_plot=range(0, prediction_horizon, 20),
    plot_type=PlotType.PLOT)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])
plt.yticks([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2])
plt.xlim([-2.1, 2.1])
plt.ylim([-2.1, 2.1])
plt.tight_layout()
plt.show()