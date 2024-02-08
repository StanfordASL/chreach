"""Tests sets code."""
import jax.numpy as jnp
from jax import vmap, jacfwd, random
from jax.config import config
config.update("jax_enable_x64", True)

from chreach.sets import *


def test_set():
    for dim in range(3):
        sset = Set(dim)
        assert sset.num_variables == dim

def test_unit_sphere():
    sample_size = 20

    for dim in [1, 2, 3]:
        sphere = UnitSphere(dim)
        assert sphere.ambient_dimension == dim

        points = sphere.sample(sample_size)

        # check dimensions
        assert points.shape[0] == sample_size
        assert points.shape[1] == dim

        # check that sampled points are on the set
        are_in = vmap(sphere.is_in_the_set)(points)
        assert jnp.sum(are_in) == sample_size

    dim = 3
    sphere = UnitSphere(dim)
    delta = sphere.get_internal_covering_delta(sample_size)
    print(delta)
    assert delta > 0
    assert delta < 1.0

def test_smooth_convex_set():
    for dim in [1, 2, 3]:
        sset = SmoothConvexSet(dim)
        assert sset.num_variables == dim

def test_smooth_convex_sub_level_set():
    for dim in [1, 2, 3]:
        sset = SmoothConvexSubLevelSet(dim)
        assert sset.num_variables == dim

def test_point():
    for dim in [1, 2, 3]:
        key = random.PRNGKey(0 + dim)
        position = random.uniform(key, shape=(dim,))
        sset = Point(position)
        assert sset.num_variables == dim
        assert jnp.all(sset.position == position)

        # test inverse gauss map
        sample_size = 10
        sphere = UnitSphere(dim)
        ds = sphere.sample(sample_size)
        ps = vmap(sset.gauss_map_inverse)(ds)
        assert jnp.linalg.norm(ps - position) <= 1e-6

        points = sset.sample_random(sample_size)
        assert jnp.all(vmap(sset.is_in_the_set)(points))

def verify_level_set_condition(sset: SmoothConvexSubLevelSet):
    """
    Check a condition for the sublevel sets, that enables 
    easily computing the inverse gauss map of these sets.
    """
    assert isinstance(sset, SmoothConvexSubLevelSet)

    sample_size = 10
    sphere = UnitSphere(sset.num_variables)
    ds = sphere.sample(sample_size)

    # get points on the boundary
    ps = vmap(sset.gauss_map_inverse)(ds)

    left = vmap(sset.level_set_function)(
        vmap(sset.level_set_function_gradient_inverse)(ds))
    right = 1. / jnp.sum(
        vmap(sset.level_set_function_gradient)(ps)**2,
        axis=-1)

    assert jnp.linalg.norm(left - right) <= 1e-6

def verify_level_set_functions(sset: SmoothConvexSubLevelSet):
    """
    Tests for the level set functions of sublevel sets.
    """
    assert isinstance(sset, SmoothConvexSubLevelSet)

    dim = sset.num_variables
    sample_size = 10
    key = random.PRNGKey(0 + dim)
    points = -3 + 10*random.uniform(key, shape=(sample_size, dim))

    # check if level_set_function_gradient works
    grads = vmap(sset.level_set_function_gradient)(points)
    grads_jax = vmap(jacfwd(sset.level_set_function))(points)
    assert jnp.linalg.norm(grads - grads_jax) <= 1e-6

    # check if level_set_function_gradient_inverse is a correct inverse
    grads = vmap(sset.level_set_function_gradient)(
        points)
    grads_inv = vmap(sset.level_set_function_gradient_inverse)(
        grads)
    assert jnp.linalg.norm(points - grads_inv) <= 1e-6

    grads_inv = vmap(sset.level_set_function_gradient_inverse)(
        points)
    grads_inv_inv = vmap(sset.level_set_function_gradient)(
        grads_inv)
    assert jnp.linalg.norm(points - grads_inv_inv) <= 1e-6

def verify_gauss_map_inverse(sset: SmoothConvexSubLevelSet):
    """
    Tests for the inverse Gauss map of sublevel sets.
    """
    assert isinstance(sset, SmoothConvexSubLevelSet)

    sample_size = 10
    sphere = UnitSphere(sset.num_variables)
    ds = sphere.sample(sample_size)

    # check if if the inverse gauss map is a good right inverse
    # i.e., checks if n(n^{-1}(d)) = d
    ps = vmap(sset.gauss_map_inverse)(ds)
    ds_gm = vmap(sset.gauss_map)(ps)
    delta_ds = jnp.linalg.norm(ds - ds_gm)
    assert delta_ds <= 1e-6

    # check again if n^{-1}(n(n^{-1}(d))) = n^{-1}(d)
    ps_inv = vmap(sset.gauss_map_inverse)(ds_gm)
    delta_ps = jnp.linalg.norm(ps_inv - ps)
    assert delta_ps <= 1e-6

    # check if inverse gauss map returns points on ∂C
    level_set_values = vmap(sset.level_set_function)(ps)
    print("level_set_values =", level_set_values)
    are_in_set = vmap(sset.is_in_the_set)(ps)
    num_in_set = jnp.sum(are_in_set)
    assert num_in_set == sample_size
    are_in_boundary = vmap(sset.is_in_the_boundary)(ps)
    num_in_boundary = jnp.sum(are_in_boundary)
    assert num_in_boundary == sample_size

    # check if gauss map is grad(h) / ||grad(h)|| 
    grads = vmap(sset.level_set_function_gradient)(ps)
    grads_norm = jnp.linalg.norm(grads, axis=-1)
    ds_grad = (grads.T / grads_norm).T
    ds_gm = vmap(sset.gauss_map)(ps)
    delta_ds = jnp.linalg.norm(ds_grad - ds_gm)
    assert delta_ds <= 1e-6

def test_ball():
    for dim in [1, 2, 3]:
        key = random.PRNGKey(0 + dim)
        center = 3 * random.uniform(key, shape=(dim,))
        key = random.PRNGKey(123 + dim)
        radius = 2 * random.uniform(key)
        sset = Ball(center, radius)
        assert sset.num_variables == dim
        assert jnp.all(sset.center == center)
        assert jnp.all(sset.radius == radius)

        # check level set functions
        verify_level_set_functions(sset)

        # check condition that allows easily computing 
        # inverse gauss map
        verify_level_set_condition(sset)

        # checks for inverse gauss map
        verify_gauss_map_inverse(sset)

        # check if n^{-1}(n(p)) = p for all p on ∂C
        sample_size = 10
        sphere = UnitSphere(dim)
        ds = sphere.sample(sample_size)
        ps = center + ds * radius
        ds = vmap(sset.gauss_map)(ps)
        ps_gm_inv = vmap(sset.gauss_map_inverse)(ds)
        assert jnp.linalg.norm(ps - ps_gm_inv) <= 1e-6

        points = sset.sample_random(sample_size)
        points_new = sset.sample_random(sample_size)
        # should be inside
        assert jnp.all(vmap(sset.is_in_the_set)(points))
        # should be different
        assert jnp.all(jnp.abs(points - points_new) > 1e-6)

def test_ellipsoid():
    for dim in [1, 2, 3]:
        key = random.PRNGKey(0 + dim)
        center = 3 * random.uniform(key, shape=(dim,))
        key = random.PRNGKey(123 + dim)
        shape_matrix = jnp.diag(
            2 * random.uniform(key, shape=(dim,)))
        sset = Ellipsoid(center, shape_matrix)
        assert sset.num_variables == dim
        assert jnp.all(sset.center == center)
        assert jnp.all(sset.shape_matrix == shape_matrix)

        # check level set functions
        verify_level_set_functions(sset)

        # check condition that allows easily computing 
        # inverse gauss map
        verify_level_set_condition(sset)

        # checks for inverse gauss map
        verify_gauss_map_inverse(sset)

        sample_size = 5
        points = sset.sample_random(sample_size)
        points_new = sset.sample_random(sample_size)
        # should be inside
        assert jnp.all(vmap(sset.is_in_the_set)(points))
        # should be different
        assert jnp.all(jnp.abs(points - points_new) > 1e-6)


def test_remark_condition_in_inverse_gauss_maps_sublevel_sets():
    """
    Checks the counter example that shows that the condition
    is necessary.
    """
    dim = 3
    ball = Ball(jnp.zeros(dim), 1)

    sample_size = 10
    sphere = UnitSphere(dim)
    ds = sphere.sample(sample_size)

    # get points on the boundary
    def level_set_squared(x):
        # ||x||^4
        h_value = jnp.sum(x**2)**2
        return h_value
    def level_set_squared_gradient(x):
        h_gradient = 4 * jnp.sum(x**2) * x
        return h_gradient
    def level_set_squared_gradient_jax(x):
        h_gradient = jacfwd(level_set_squared)(x)
        return h_gradient
    def level_set_squared_gradient_inverse(y):
        h_gradient_inv = (1. / (4.**(1/3))) * y / jnp.sum(y**2)**(1./3)
        return h_gradient_inv
    def incorrect_gauss_map_inverse(d):
        x = level_set_squared_gradient_inverse(
            d / jnp.sqrt(
                level_set_squared(
                    level_set_squared_gradient_inverse(d))))
        return x
    def incorrect_gauss_map_inverse_hand_derivations(d):
        x = (1. / 2**(1. / 9)) * d
        return x
    def condition_left(x):
        left = level_set_squared(
            level_set_squared_gradient_inverse(
                ball.gauss_map(x)))
        a = ball.gauss_map(x)
        b = level_set_squared_gradient_inverse(
                ball.gauss_map(x))
        return left
    def condition_right(x):
        right = 1. / jnp.sum(
            level_set_squared_gradient_jax(x)**2)
        return right

    xs = vmap(ball.gauss_map_inverse)(ds)

    # check the condition is violated
    error = jnp.linalg.norm(
        vmap(condition_left)(xs) - 
        vmap(condition_right)(xs))
    print("error =", error)
    assert error > 1e-6

    # check the inverse gauss map from theory is exactly d / 2**1/9
    error = jnp.linalg.norm(
        ds - 
        vmap(incorrect_gauss_map_inverse_hand_derivations)(ds))
    assert error > 1e-6
    error = jnp.linalg.norm(
        (1. / 2**(1. / 9)) * ds - 
        vmap(incorrect_gauss_map_inverse_hand_derivations)(ds))
    assert error < 1e-6

    # check the inverse gauss map from theory is incorrect
    # (which is ok, as the condition does not hold)
    error = jnp.linalg.norm(
        vmap(ball.gauss_map_inverse)(ds) -
        vmap(incorrect_gauss_map_inverse)(ds))
    assert error > 1e-6

def test_rectangle():
    for dim in [1, 2, 3]:
        key = random.PRNGKey(0 + dim)
        center = random.uniform(key, shape=(dim,))
        key = random.PRNGKey(123 + dim)
        deltas = random.uniform(key, shape=(dim,))
        sset = Rectangle(center, deltas)
        assert sset.num_variables == dim
        assert jnp.all(sset.center == center)
        assert jnp.all(sset.deltas == deltas)
        assert sset.is_in_the_set(sset.center)
        assert sset.is_in_the_set(sset.center+sset.deltas)
        assert sset.is_in_the_set(sset.center-sset.deltas)
        assert sset.is_in_the_set(sset.center-0.5*sset.deltas)

        points = sset.sample_random(20)
        points_new = sset.sample_random(20)
        # should be inside
        assert jnp.all(vmap(sset.is_in_the_set)(points))
        # should be different
        assert jnp.all(jnp.abs(points - points_new) > 1e-6)



def test_smooth_rectangle():
    for dim in [1, 2, 3]:
        center = random.uniform(
            random.PRNGKey(0 + dim),
            shape=(dim,))
        deltas = random.uniform(
            random.PRNGKey(123 + dim),
            shape=(dim,))
        rectangle = Rectangle(center, deltas)
        smoothing_parameter = (
            1 + 30 * random.uniform(
            random.PRNGKey(123 + dim)))
        sset = SmoothRectangle(
            rectangle, smoothing_parameter)
        assert sset.num_variables == dim
        assert jnp.all(sset.center == center)
        assert jnp.all(sset.deltas == deltas)
        assert jnp.all(sset.smoothing_parameter == smoothing_parameter)

        # check level set functions
        verify_level_set_functions(sset)

        # checks for inverse gauss map
        verify_gauss_map_inverse(sset)

        # check condition that allows easily computing 
        # inverse gauss map
        verify_level_set_condition(sset)

        # check that points on edges of rectangle are inside 
        # smooth approximation
        p = center + deltas
        assert sset.is_in_the_set(p)
        p = center - deltas
        assert sset.is_in_the_set(p)

def test_smooth_rectangle_under():
    for dim in [1, 2, 3]:
        center = random.uniform(
            random.PRNGKey(0 + dim),
            shape=(dim,))
        deltas = random.uniform(
            random.PRNGKey(123 + dim),
            shape=(dim,))
        rectangle = Rectangle(center, deltas)
        smoothing_parameter = (
            1 + 30 * random.uniform(
            random.PRNGKey(123 + dim)))
        sset = SmoothRectangleUnder(
            rectangle, smoothing_parameter)
        assert sset.num_variables == dim
        assert jnp.all(sset.center == center)
        assert jnp.all(sset.deltas == deltas)
        assert jnp.all(sset.smoothing_parameter == smoothing_parameter)

        # check level set functions
        verify_level_set_functions(sset)

        # checks for inverse gauss map
        verify_gauss_map_inverse(sset)

        # check condition that allows easily computing 
        # inverse gauss map
        verify_level_set_condition(sset)

        # check that points on edges of rectangle are outside 
        # smooth approximation
        if dim > 1:
            # rectangle corners are not inside
            p = center + deltas
            assert not sset.is_in_the_set(p)
            p = center - deltas
            assert not sset.is_in_the_set(p)
            # rectangle edges are inside (on the boundary)
            p = center + jnp.concatenate([
                deltas[0:1], jnp.zeros_like(deltas[1:])])
            assert sset.is_in_the_set(p)
            assert sset.is_in_the_boundary(p)
            p = center - jnp.concatenate([
                deltas[0:1], jnp.zeros_like(deltas[1:])])
            assert sset.is_in_the_set(p)
            assert sset.is_in_the_boundary(p)

def test_full_dim_smooth_convex_set_init():
    for base_dim in [1, 2]:
        key1 = random.PRNGKey(0 + base_dim)
        key2 = random.PRNGKey(123 + base_dim)
        key3 = random.PRNGKey(456 + base_dim)
        for base_sset in [
            Ball(
                center=random.uniform(key1, shape=(base_dim,)),
                radius=random.uniform(key2)),
            SmoothRectangle(
                rectangle=Rectangle(
                    center=random.uniform(key1, shape=(base_dim,)),
                    deltas=random.uniform(key2, shape=(base_dim,))),
                smoothing_parameter=(1 + 30 * random.uniform(key3))
                )
            ]:
            for num_dims_to_add in [1]:
                sset = FullDimSmoothConvexSet(
                    base_sset, num_dims_to_add)
                dim = base_dim + num_dims_to_add
                assert sset.num_variables == dim

def test_full_dim_smooth_convex_set_level_set():
    for base_dim in [1, 2]:
        key1 = random.PRNGKey(0 + base_dim)
        key2 = random.PRNGKey(123 + base_dim)
        key3 = random.PRNGKey(456 + base_dim)
        for base_sset in [
            Ball(
                center=random.uniform(key1, shape=(base_dim,)),
                radius=random.uniform(key2)),
            SmoothRectangle(
                rectangle=Rectangle(
                    center=random.uniform(key1+1, shape=(base_dim,)),
                    deltas=random.uniform(key2+1, shape=(base_dim,))),
                smoothing_parameter=(1 + 30 * random.uniform(key3))
                )
            ]:
            for num_dims_to_add in [1]:
                sset = FullDimSmoothConvexSet(
                    base_sset, num_dims_to_add)

                # check level set functions
                verify_level_set_functions(sset)

                # check condition that allows easily computing 
                # inverse gauss map
                verify_level_set_condition(sset)

def test_full_dim_smooth_convex_set_gauss_map():
    for base_dim in [2]:
        key1 = random.PRNGKey(0 + base_dim)
        key2 = random.PRNGKey(123 + base_dim)
        key3 = random.PRNGKey(456 + base_dim)
        for base_sset in [
            Ball(
                center=random.uniform(key1, shape=(base_dim,)),
                radius=random.uniform(key2)),
            SmoothRectangle(
                rectangle=Rectangle(
                    center=-1+2*random.uniform(key1+13, shape=(base_dim,)),
                    deltas=4*random.uniform(key2+16, shape=(base_dim,))),
                smoothing_parameter=(1 + 30 * random.uniform(key3))
                )
            ]:
            for num_dims_to_add in [1]:
                sset = FullDimSmoothConvexSet(
                    base_sset, num_dims_to_add)
                print(20*"-")
                print("base_dim =", base_dim)
                print("dim      =", sset.num_variables)
                print("base_sset =", base_sset)

                # checks for inverse gauss map
                verify_gauss_map_inverse(sset)
