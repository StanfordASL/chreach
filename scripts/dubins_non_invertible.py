"""Dubins car reachability with non-invertible g(t, x) dynamics function."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc, rcParams

from chreach.dynamics import DubinsDynamics
from chreach.dynamics import GaddedDynamics
from chreach.sets import Ball, Ellipsoid
from chreach.sets import FullDimSmoothConvexSet
from chreach.reach import Algorithm1, PlotType, RandUP

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)


discretization_time = 0.5
prediction_horizon = 14
sample_size = 1000

epsilon_parameters = [1, 1/2, 1/3, 1/5, 1/10, 1/100, 1/1000]
num_epsilon_parameters = len(epsilon_parameters)

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

list_reachable_sets_under = []
list_reachable_sets_over = []

for i, smoothing_epsilon in enumerate(epsilon_parameters):
    print("smoothing_epsilon =", smoothing_epsilon)
    # over-approximations
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
        smoothing_epsilon)
    reachability_alg = Algorithm1(
        dynamics_invertible,
        initial_states_set,
        smooth_disturbances_set_over)
    sets = reachability_alg.estimate_reachable_sets(
            discretization_time,
            prediction_horizon,
            sample_size)
    list_reachable_sets_over.append(sets)

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
    0, 1, num_epsilon_parameters))
colors = cmap(colormap_values)
for i in range(num_epsilon_parameters):
    sets_over = list_reachable_sets_over[i]
    sets_over.plot(
        ax,
        # times_to_plot=[-1],
        plot_type=PlotType.CONVEXHULLPLOT,
        color=colors[i],
        alpha=1.0, linewidth=4)
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
    ticks=[colormap_values[i] for i in range(num_epsilon_parameters)]
    )
epss = epsilon_parameters
cbar.ax.set_yticklabels([
    round(epss[i], 3) for i in range(num_epsilon_parameters)])
cbar.set_label(
    r'$\epsilon$',
    fontsize=30, rotation='horizontal', labelpad=16)
cbar.ax.tick_params(labelsize=26)
ax.tick_params(labelsize=26)
ax.set_xlabel(r'$p_1$', fontsize=32)
ax.set_ylabel(r'$p_2$', fontsize=32, rotation=0, labelpad=30)
plt.tight_layout()
plt.show()
