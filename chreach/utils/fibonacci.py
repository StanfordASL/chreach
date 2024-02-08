"""
Construct a Fibonacci lattice, see
https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
"""
from typing import Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt


def fibonacci_lattice_3d(
    sphere_radius: float,
    sample_size: int) -> jnp.array:
    """
    Returns points on a Fibonacci lattice on a sphere in R^3.

    Args:
        sphere_radius: radius of the sphere
            (float)
        sample_size: number of points on the lattice
            (int)

    Returns:
        points: points on the lattice
            (sample_size, 3) array
    """
    indices = jnp.arange(
        0, sample_size, dtype=float) + 0.5
    phi = jnp.arccos(1 - 2*indices / sample_size)
    theta = jnp.pi * (1 + 5**0.5) * indices
    x = sphere_radius * jnp.cos(theta) * jnp.sin(phi)
    y = sphere_radius * jnp.sin(theta) * jnp.sin(phi)
    z = sphere_radius * jnp.cos(phi)
    pts = jnp.stack((x, y, z)).T
    return pts

def fibonacci_lattice_3d_with_delta_covering_distance(
    sphere_radius: float,
    sample_size: int) -> Tuple[jnp.array, float]:
    """
    Returns points on a Fibonacci lattice on a sphere in R^3 and covering radius

    Args:
        sphere_radius: radius of the sphere
            (float)
        sample_size: number (M) of points on the lattice 
            (int)

    Returns:
        points: points on the lattice
            (sample_size, 3) array
        delta: value of delta such that points form an internal delta-covering
            of the sphere.
            (float)
    """
    pts = fibonacci_lattice_3d(sphere_radius, sample_size)
    sample_size = pts.shape[0]

    # Compute maximal minimal distance between points
    # as a conservative approximation of delta such
    # that the M points of the lattice form a
    # delta-covering.
    # Indeed, points are approximately evenly spread
    # on the surface of the sphere. Thus, the distance
    # between neighbors (the min distance for each
    # point) is approximately always the same. By
    # taking the maximum such distance for each point,
    # we get an over-approximation of the value of
    # delta (we approximately get delta/2 for large
    # numbers of samples M, so returning this maximum
    # distance for delta gives a conservative
    # approximation).
    dists = jnp.ones((sample_size, sample_size))
    for i in range(sample_size):
        for j in range(sample_size):
            if i == j:
                continue
            pi, pj = pts[i, :], pts[j, :]
            dists = dists.at[i, j].set(jnp.linalg.norm(pi - pj))
    dists_min = jnp.min(dists, 1)
    dist_max = jnp.max(dists_min)
    delta = dist_max
    return pts, delta


if __name__=="__main__":
    sample_size = 50
    points, delta = fibonacci_lattice_3d_with_delta_covering_distance(
        1, sample_size)
    print("delta =", delta)
    plt.figure().add_subplot(111, projection='3d').scatter(
        points[:, 0], points[:, 1], points[:, 2])
    plt.show()
