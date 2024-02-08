"""Tests for stats.py in utils"""

import jax.numpy as jnp
from jax import random, vmap
from jax.config import config
config.update("jax_enable_x64", True)

from chreach.sets import *
from chreach.utils.stats import *


def test_uniformly_sample_points_in_unit_sphere():
    for sample_size in [1, 2, 10]:
        for dim in [1, 2, 3]:
            sphere = UnitSphere(dim)

            points = uniformly_sample_points_in_unit_sphere(
                dim, sample_size)
            assert len(points.shape) == 2
            assert points.shape[0] == sample_size
            assert points.shape[1] == dim

            are_in = vmap(sphere.is_in_the_set)(points)
            assert jnp.sum(are_in) == sample_size

def test_uniformly_sample_points_in_unit_ball():
    for sample_size in [1, 2, 10]:
        for dim in [1, 2, 3]:
            ball = Ball(jnp.zeros(dim), 1.0)

            points = uniformly_sample_points_in_unit_ball(
                dim, sample_size)
            assert len(points.shape) == 2
            assert points.shape[0] == sample_size
            assert points.shape[1] == dim

            are_in = vmap(ball.is_in_the_set)(points)
            assert jnp.sum(are_in) == sample_size

def test_uniformly_sample_points_in_ellipsoid():
    for sample_size in [1, 2, 10]:
        for dim in [1, 2, 3]:
            key = random.PRNGKey(0 + dim)
            center = 3 * random.uniform(key, shape=(dim,))
            key = random.PRNGKey(123 + dim)
            shape_matrix = jnp.diag(
                2 * random.uniform(key, shape=(dim,)))
            sset = Ellipsoid(center, shape_matrix)

            points = uniformly_sample_points_in_ellipsoid(
                center, shape_matrix, sample_size)
            assert len(points.shape) == 2
            assert points.shape[0] == sample_size
            assert points.shape[1] == dim

            are_in = vmap(sset.is_in_the_set)(points)
            assert jnp.sum(are_in) == sample_size

def test_uniformly_sample_points_in_rectangle():
    for sample_size in [1, 2, 10]:
        for dim in [1, 2, 3]:
            center = random.uniform(
                random.PRNGKey(0 + dim), shape=(dim,))
            deltas = random.uniform(
                random.PRNGKey(123 + dim), shape=(dim,))

            rectangle = Rectangle(center, deltas)

            points = uniformly_sample_points_in_rectangle(
                center, deltas, sample_size)
            assert len(points.shape) == 2
            assert points.shape[0] == sample_size
            assert points.shape[1] == dim

            are_in = vmap(rectangle.is_in_the_set)(points)
            assert jnp.sum(are_in) == sample_size
