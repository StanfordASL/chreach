"""Plots the rectangular set relaxation."""
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from chreach.sets import Rectangle, UnitSphere
from chreach.sets import SmoothRectangle, SmoothRectangleUnder

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)

dim = 2
center = jnp.array([2, 1])
deltas = jnp.array([2.5, 1])
rectangle = Rectangle(
    center=center,
    deltas=deltas)

# generate points on boundaries of smooth rectangles
smoothing_parameters = [2, 3, 5, 10., 20.]
num_smoothing_parameters = len(smoothing_parameters)
sample_size = 1000
sphere = UnitSphere(dim)
ds = sphere.sample(sample_size)

xs_under = jnp.zeros(
    (num_smoothing_parameters, sample_size, dim))
xs_over = jnp.zeros_like(xs_under)
for i, smoothing_parameter in enumerate(smoothing_parameters):
    print("smoothing_parameter =", smoothing_parameter)
    smooth_rectangle_over = SmoothRectangle(
        rectangle,
        smoothing_parameter=smoothing_parameter)
    xs_over = xs_over.at[i, :].set(vmap(
        smooth_rectangle_over.gauss_map_inverse)(
        ds))
    smooth_rectangle_under = SmoothRectangleUnder(
        rectangle,
        smoothing_parameter=smoothing_parameter)
    xs_under = xs_under.at[i, :].set(vmap(
        smooth_rectangle_under.gauss_map_inverse)(
        ds))

# plot
fig, ax = plt.subplots(figsize=[10, 6])
rectangle = plt.Rectangle(
    center-deltas, 
    2 * deltas[0], 
    2 * deltas[1], 
    alpha=0.05, color='k', label=r'$C$')
ax.add_patch(rectangle)
rectangle = plt.Rectangle(
    center - deltas, 
    2 * deltas[0], 
    2 * deltas[1], 
    alpha=1, facecolor='none',
    edgecolor='black', linewidth=3)
ax.add_patch(rectangle)
for i in range(num_smoothing_parameters):
    if i==0:
        plt.plot(
            xs_over[i, :, 0], xs_over[i, :, 1], 
            'b', label=r'$\partial\overline{C_\lambda}$', alpha=1)
        plt.plot(
            xs_under[i, :, 0], xs_under[i, :, 1], 
            'r', label=r'$\partial\underline{C_\lambda}$', alpha=1)
    else:
        plt.plot(
            xs_under[i, :, 0], xs_under[i, :, 1], 
            'r', alpha=1)
        plt.plot(
            xs_over[i, :, 0], xs_over[i, :, 1], 
            'b', alpha=1)
plt.xlim([
    (center-1.5*deltas)[0], 
    (center+2.00*deltas)[0]])
plt.ylim([
    (center-1.75*deltas)[1], 
    (center+1.75*deltas)[1]])
plt.tick_params(
    axis='both', which='major', labelsize=24)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 0, 1]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    fontsize=24, labelspacing=0.2, handlelength=1,
    loc='lower right')
plt.savefig('Clambda_approx.png', dpi=400)
plt.show()
