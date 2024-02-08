"""Dubins car reachability with rectangular uncertainty sets."""
from time import time
import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc, rcParams

from chreach.dynamics import DubinsDynamics
from chreach.sets import Ellipsoid, UnitSphere, Rectangle
from chreach.sets import SmoothRectangle, SmoothRectangleUnder
from chreach.reach import Algorithm1, RandUP, PlotType

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)


discretization_time = 0.5
prediction_horizon = 14
sample_size = 1000

smoothing_parameters = [2, 2.5, 3, 3.5, 5, 10., 20., 200.]
num_smoothing_parameters = len(smoothing_parameters)

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

list_reachable_sets_under = []
list_reachable_sets_over = []
for i, smoothing_parameter in enumerate(smoothing_parameters):
    print("smoothing_parameter =", smoothing_parameter)
    # over-approximations
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
    list_reachable_sets_over.append(sets)
    # under-approximations
    smooth_disturbances_set_under = SmoothRectangleUnder(
        disturbances_set,
        smoothing_parameter=smoothing_parameter)
    reachability_alg = Algorithm1(
        dynamics,
        initial_states_set,
        smooth_disturbances_set_under)
    sets = reachability_alg.estimate_reachable_sets(
            discretization_time,
            prediction_horizon,
            sample_size)
    list_reachable_sets_under.append(sets)
    # estimate computation time
    if i == 0:
        estimate_trajectories = jit(
            lambda initial_directions:
            vmap(reachability_alg.solve_augmented_ode,
                in_axes=(0, None, None))(
                initial_directions,
                discretization_time,
                prediction_horizon))
        sphere = UnitSphere(dim)
        initial_directions = sphere.sample(sample_size)
        estimate_trajectories(initial_directions)
        start_time = time()
        for _ in range(10):
            estimate_trajectories(initial_directions)
        print("elapsed =", (time() - start_time) / 10)


# naive Monte Carlo
randup = RandUP(
    dynamics,
    initial_states_set,
    disturbances_set)
reach_sets_randup = randup.estimate_reachable_sets(
    discretization_time,
    prediction_horizon,
    sample_size=10000)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cmap = plt.get_cmap('jet')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(jnp.linspace(minval, maxval, n)))
    return new_cmap
cmap = truncate_colormap(cmap, 0.015, 0.9)
colormap_values = jnp.sqrt(jnp.linspace(
    0, 1, num_smoothing_parameters))
colors_under = cmap(colormap_values)
colors_over = colors_under
for i, smoothing_parameter in enumerate(smoothing_parameters):
    sets_over = list_reachable_sets_over[i]
    sets_under = list_reachable_sets_under[i]
    sets_over.plot(
        ax,
        # times_to_plot=[-1],
        plot_type=PlotType.CONVEXHULLPLOT,
        color=colors_over[i],
        alpha=1.0, linewidth=4)
    sets_under.plot(
        ax,
        # times_to_plot=[-1],
        plot_type=PlotType.CONVEXHULLPLOT,
        color=colors_under[i],
        alpha=1.0, linestyle='dotted', linewidth=4)
reach_sets_randup.plot(
    ax,
    # times_to_plot=[-1],
    plot_type=PlotType.SCATTERPLOT,
    color='k', alpha=1)
reach_sets_randup.plot(
    ax,
    # times_to_plot=[-1],
    plot_type=PlotType.CONVEXHULLPLOT,
    color='k', alpha=1, linewidth=1)
# colormap
cbar = plt.colorbar(
    plt.contourf(
        [[1e3, 1e3], [1e3, 1e3]], jnp.linspace(0, 1, 100),
        cmap=cmap),
    ticks=[colormap_values[i] for i in range(num_smoothing_parameters)]
    )
lams = smoothing_parameters
cbar.ax.set_yticklabels([
    round(lams[i], 1) if \
    round(lams[i], 1) != int(lams[i]) \
    else int(lams[i]) \
    for i in range(num_smoothing_parameters)])
cbar.set_label(
    r'$\lambda$',
    fontsize=30, rotation='horizontal', labelpad=16)
cbar.ax.tick_params(labelsize=26)
ax.tick_params(labelsize=26)
ax.set_xlabel(r'$p_1$', fontsize=32)
ax.set_ylabel(r'$p_2$', fontsize=32, rotation=0, labelpad=30)
plt.tight_layout()
plt.show()
