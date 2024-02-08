"""Set classes."""
import jax.numpy as jnp
from jax import jacfwd

from chreach.utils.fibonacci import \
    fibonacci_lattice_3d, \
    fibonacci_lattice_3d_with_delta_covering_distance
from chreach.utils.stats import \
    uniformly_sample_points_in_unit_ball, \
    uniformly_sample_points_in_ellipsoid, \
    uniformly_sample_points_in_rectangle


class Set:
    """Base class set."""
    def __init__(
        self,
        num_variables: int):
        self._num_variables = num_variables

    @property
    def num_variables(self) -> int:
        """Returns the number of variables n."""
        return self._num_variables

    def is_in_the_set(self, p: jnp.ndarray) -> bool:
        """
        Returns true if p is in the set.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            is_in: true if p is in the set
                (bool)
        """
        raise NotImplementedError

    def plot(self, ax):
        """Plots the set."""
        raise NotImplementedError


class UnitSphere:
    """
    Sphere S^{n-1} in R^n defined as 
    S^{n-1} = {d in R^n : ||d||=1}.
    """
    def __init__(self, ambient_dimension):
        self._ambient_dimension = ambient_dimension

    @property
    def ambient_dimension(self) -> int:
        """Returns the dimension n."""
        return self._ambient_dimension

    def sample(self, sample_size: int) -> jnp.ndarray:
        """
        Samples points on the sphere.

        Args:
            sample_size: number of points to sample from S^{n-1}
                (int)

        Returns:
            points: points on the sphere
                (sample_size, ambient_dimension) array
        """
        if self.ambient_dimension == 1:
            ones = jnp.ones(sample_size)
            middle = int(sample_size / 2)
            ds = jnp.concatenate([ones[:middle], -ones[middle:]])
            ds = ds[:, jnp.newaxis]
        elif self.ambient_dimension == 2:
            theta_vals = jnp.linspace(
                0, 2*jnp.pi+1e-9, sample_size)
            ds_x = jnp.cos(theta_vals)
            ds_y = jnp.sin(theta_vals)
            ds = jnp.stack((ds_x, ds_y)).T
        elif self.ambient_dimension == 3:
            ds = fibonacci_lattice_3d(1., sample_size)
        else:
            raise NotImplementedError
        return ds

    def get_internal_covering_delta(self, sample_size: int) -> float:
        """
        Assuming sample_size points are sampled on the sphere, returns the value
        of delta such that the sample is an inner delta-covering of the sphere.

        Args:
            sample_size: number of points to sample from S^{n-1}
                (int)

        Returns:
            delta: radius of the covering using sample_size points
                (float)
        """
        if self.ambient_dimension == 3:
            _, delta = fibonacci_lattice_3d_with_delta_covering_distance(
                1., sample_size)
        else:
            raise NotImplementedError
        return delta

    def level_set_function(self, p: jnp.ndarray) -> float:
        """
        Evaluate the level set function h(.) at p, where
        the sphere is represented as
        S^{n-1} = {p in R^n : h(p) = 1}
        with h(p) = ||p||

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            h(p): value of the level set function at p
                (float)
        """
        h_value = jnp.linalg.norm(p)
        return h_value

    def is_in_the_set(self, p: jnp.ndarray) -> bool:
        """
        Returns true if p is in the sphere S^{n-1}.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            is_in: true if p is in the sphere S^{n-1}
                (bool)
        """
        is_in = jnp.abs(self.level_set_function(p) - 1) <= 1e-6
        return is_in


class SmoothConvexSet(Set):
    """
    Compact convex subset C of R^n with smooth boundary ∂C.
    """
    def __init__(
        self,
        num_variables: int):
        """Class initializer"""
        print("Initializing smooth convex set with")
        print("> num_variables =", num_variables)
        super().__init__(num_variables)

    def gauss_map(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Gauss map of the set C, defined as the map 
            n: ∂C -> S^{n-1}
        that gives the unit-norm outward-pointing vector n(p) 
        at any point p on the boundary ∂C.

        Args:
            p: point on the boundary ∂C 
                (num_variables) array

        Returns:
            d: value n(x) of the gauss map at p 
                (num_variables) array
        """
        raise NotImplementedError

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse of the Gauss map (n^{-1}: S^{n-1} -> ∂C).

        Args:
            d: direction (unit-norm vector)
                (num_variables) array

        Returns:
            p: point on the boundary ∂C such that p = n^{-1}(d)
                (num_variables) array
        """
        raise NotImplementedError


class SmoothConvexSubLevelSet(SmoothConvexSet):
    """
    Compact convex subset C of R^n with smooth boundary ∂C
    represented as the sublevel set
    {p in R^n : h(p) <= 1}
    for some smooth convex function h: R^n -> R.
    """
    def level_set_function(self, p: jnp.ndarray) -> float:
        """
        Evaluate the level set function h(.) at p.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            h(p): value of the level set function at p
                (float)
        """
        raise NotImplementedError

    def level_set_function_gradient(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the gradient of the level set function h(.) at p.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            h_dp(p): gradient of the level set function at p
                (num_variables) array
        """
        h_dp_value = jacfwd(self.level_set_function)(p)
        return h_dp_value

    def level_set_function_gradient_inverse(
        self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the inverse of the gradient of the level set function h() at p.

        Args:
            y: point in R^n
                (num_variables) array

        Returns:
            inv(h_dp)(y): inverse of the gradient of the level set function at p
                (num_variables) array
        """
        raise NotImplementedError

    def is_in_the_set(self, p: jnp.ndarray) -> bool:
        """
        Returns true if p is in the set C.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            is_in: true if p is in the set C
                (bool)
        """
        is_in = self.level_set_function(p) - 1 <= 1e-6
        return is_in

    def is_in_the_boundary(self, p: jnp.ndarray) -> bool:
        """
        Returns true if p is in the boundary ∂C of the set C.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            is_in: true if p is in the boundary ∂C
                (bool)
        """
        is_in = jnp.abs(self.level_set_function(p) - 1) <= 1e-6
        return is_in

    def gauss_map(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Gauss map of the set C, defined as the map 
            n: ∂C -> S^{n-1}
        that gives the unit-norm outward-pointing vector n(p) 
        at any point p on the boundary ∂C.

        Args:
            p: point on the boundary ∂C 
                (num_variables) array

        Returns:
            d: value n(x) of the gauss map at p 
                (num_variables) array
        """
        gradient = self.level_set_function_gradient(p)
        gradient_norm = jnp.linalg.norm(gradient)
        direction = gradient / gradient_norm
        return direction


class Point(SmoothConvexSet):
    """Point set."""
    def __init__(self, position: jnp.ndarray):
        self._position = position
        num_variables = len(position)
        super().__init__(num_variables)

    @property
    def position(self) -> jnp.ndarray:
        """Returns the position of the point."""
        return self._position

    def is_in_the_set(self, p: jnp.ndarray) -> bool:
        is_in = jnp.linalg.norm(p - self.position) <= 1e-6
        return is_in

    def gauss_map(self, p: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        return self.position

    def sample_random(self, sample_size: int) -> jnp.ndarray:
        """
        Samples points in the set.

        Args:
            sample_size: number of points to sample in the set
                (int)

        Returns:
            points: points in the set
                (sample_size, _num_variables) array
        """
        points = jnp.repeat(self.position[None, :], sample_size, axis=0)
        return points


class Ball(SmoothConvexSubLevelSet):
    """
    Ball represented as the set of points
    {p in R^n : ||p - center||^2 <= radius^2}.
    """
    def __init__(self, center: jnp.ndarray, radius: float):
        self._center = center
        self._radius = radius
        num_variables = len(center)
        super().__init__(num_variables)

    @property
    def center(self) -> jnp.ndarray:
        """Returns the center of the ball."""
        return self._center

    @property
    def radius(self) -> float:
        """Returns the radius of the ball."""
        return self._radius

    def gauss_map(self, p: jnp.ndarray) -> jnp.ndarray:
        delta = p - self.center
        distance = jnp.linalg.norm(delta)
        d = delta / distance
        return d

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        p = self.center + self.radius * d
        return p

    def level_set_function(self, p: jnp.ndarray) -> float:
        delta = p - self.center
        distance_squared = jnp.sum(delta**2)
        h_value = distance_squared / self.radius**2
        return h_value

    def level_set_function_gradient(
        self, p: jnp.ndarray) -> jnp.ndarray:
        delta = p - self.center
        h_gradient = 2 * delta / self.radius**2
        return h_gradient

    def level_set_function_gradient_inverse(
        self, y: jnp.ndarray) -> jnp.ndarray:
        h_gradient_inv = self.center + (self.radius**2 / 2.0) * y
        return h_gradient_inv

    def sample_random(self, sample_size: int) -> jnp.ndarray:
        """
        Samples points in the ball.

        Args:
            sample_size: number of points to sample from the set
                (int)

        Returns:
            points: points in the ball
                (sample_size, _num_variables) array
        """
        points = uniformly_sample_points_in_unit_ball(
            self.num_variables, sample_size)
        points = self.center + self.radius * points
        return points


class Ellipsoid(SmoothConvexSubLevelSet):
    """
    Ellpsoid represented as the set of points
    {p in R^n : (p - center).T @ shape_matrix^{-1} (p- center) <= 1}.
    """
    def __init__(self, center: jnp.ndarray, shape_matrix: jnp.ndarray):
        if len(center.shape) != 1:
            raise ValueError("center should be a vector.")
        if len(shape_matrix.shape) != 2:
            raise ValueError("shape_matrix should be a matrix.")
        if not jnp.all(shape_matrix - shape_matrix.T == 0):
            raise ValueError("shape_matrix should be symmetric.")
        if not jnp.all(jnp.linalg.eigvals(shape_matrix) > 0):
            raise ValueError("shape_matrix should be positive definite.")
        self._center = center
        self._shape_matrix = shape_matrix
        self._shape_matrix_inverse = jnp.linalg.inv(shape_matrix)
        num_variables = len(center)
        super().__init__(num_variables)

    @property
    def center(self) -> jnp.ndarray:
        """Returns the center of the ellipsoid."""
        return self._center

    @property
    def shape_matrix(self) -> float:
        """Returns the shape_matrix of the ellipsoid."""
        return self._shape_matrix

    @property
    def shape_matrix_inverse(self) -> float:
        """Returns the inverse of the shape_matrix of the ellipsoid."""
        return self._shape_matrix_inverse

    def gauss_map(self, p: jnp.ndarray) -> jnp.ndarray:
        delta = p - self.center
        Qinv_delta = self.shape_matrix_inverse @ delta
        Qinv_delta_norm = jnp.linalg.norm(Qinv_delta)
        d = Qinv_delta / Qinv_delta_norm
        return d

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        Q_d = self.shape_matrix @ d
        inv_sqrt_d_Q_d = 1. / jnp.sqrt(d.T @ Q_d)
        p = self.center + inv_sqrt_d_Q_d * Q_d
        return p

    def level_set_function(self, p: jnp.ndarray) -> float:
        delta = p - self.center
        Qinv_delta = self.shape_matrix_inverse @ delta
        h_value = delta.T @ Qinv_delta
        return h_value

    def level_set_function_gradient(self, p: jnp.ndarray) -> jnp.ndarray:
        delta = p - self.center
        h_gradient = 2. * self.shape_matrix_inverse @ delta
        return h_gradient

    def level_set_function_gradient_inverse(
        self, y: jnp.ndarray) -> jnp.ndarray:
        h_gradient_inv = self.center + 0.5 * self.shape_matrix @ y
        return h_gradient_inv

    def sample_random(self, sample_size: int) -> jnp.ndarray:
        """
        Samples points in the ellipsoid.

        Args:
            sample_size: number of points to sample from the set
                (int)

        Returns:
            points: points in the ellipsoid
                (sample_size, _num_variables) array
        """
        points = uniformly_sample_points_in_ellipsoid(
            self.center, self.shape_matrix, sample_size)
        return points


class Rectangle(Set):
    """
    Rectangular set represented as
    R = {x in R^n : |x[i] - center[i]| <= deltas[i] for all i}
    """
    def __init__(
        self,
        center: jnp.ndarray,
        deltas: jnp.ndarray):
        self._center = center
        self._deltas = deltas
        num_variables = len(center)
        super().__init__(num_variables)

    @property
    def center(self) -> jnp.ndarray:
        """Returns the center of the rectangle."""
        return self._center

    @property
    def deltas(self) -> jnp.ndarray:
        """Returns the deltas of the rectangle."""
        return self._deltas

    def level_set_function(self, p: jnp.ndarray) -> float:
        delta = p - self.center
        distances_abs = jnp.abs(delta) - self.deltas
        h_value = jnp.max(distances_abs)
        return h_value

    def is_in_the_set(self, p: jnp.ndarray) -> bool:
        """
        Returns true if p is in the rectangle.

        Args:
            p: point in R^n
                (num_variables) array

        Returns:
            is_in: true if p is in the rectangleC
                (bool)
        """
        is_in = self.level_set_function(p) - 1 <= 1e-6
        return is_in

    def sample_random(self, sample_size: int) -> jnp.ndarray:
        """
        Samples points in the rectangle.

        Args:
            sample_size: number of points to sample from the set
                (int)

        Returns:
            points: points in the rectangle
                (sample_size, _num_variables) array
        """
        points = uniformly_sample_points_in_rectangle(
            self.center, self.deltas, sample_size)
        return points


class SmoothRectangle(SmoothConvexSubLevelSet):
    """
    Smooth approximation of rectangular set represented as
    C = {x in R^n : 
         (sum_i |x[i] - center[i]| / deltas[i])^lam)^{1/lam} <= n^(1/lam)}
    where lam is a smoothing parameter.

    The larger the smoothing parameter, the closer the approximation is to 
    the true rectangle.
    """
    def __init__(
        self,
        rectangle: Rectangle,
        smoothing_parameter: float):
        self._rectangle = rectangle
        self._smoothing_parameter = float(smoothing_parameter)
        num_variables = len(self.rectangle.center)
        super().__init__(num_variables)

    @property
    def rectangle(self) -> Rectangle:
        """Returns the rectangle without smoothing."""
        return self._rectangle

    @property
    def smoothing_parameter(self) -> float:
        """Returns the smoothing_parameter."""
        return self._smoothing_parameter

    @property
    def center(self) -> jnp.ndarray:
        """Returns the center of the rectangle."""
        return self.rectangle.center

    @property
    def deltas(self) -> jnp.ndarray:
        """Returns the deltas of the rectangle."""
        return self.rectangle.deltas

    def level_set_function_under(self, p: jnp.ndarray) -> float:
        lam = self.smoothing_parameter

        p_diff = p - self.center
        dist = jnp.linalg.norm(p_diff / self.deltas, ord=lam)
        h_value = dist**2
        return h_value

    def level_set_function_over(self, p: jnp.ndarray) -> float:
        lam = self.smoothing_parameter

        dist = self.level_set_function_under(p)
        h_value = dist / (self.num_variables**(2 / lam))
        return h_value

    def level_set_function(self, p: jnp.ndarray) -> float:
        h_value = self.level_set_function_over(p)
        return h_value

    def level_set_function_gradient(self, p: jnp.ndarray) -> jnp.ndarray:
        lam = self.smoothing_parameter
        dp = p - self.center
        up = dp * jnp.abs(dp)**(lam - 2) / (self.deltas**lam)
        down = jnp.linalg.norm(dp / self.deltas, ord=lam)**(lam - 2)
        h_gradient = 2. * up / down
        h_gradient = h_gradient / ((self.num_variables**(1 / lam))**2)
        return h_gradient

    def level_set_function_gradient_inverse(
        self, y: jnp.ndarray) -> jnp.ndarray:
        # This formula is not documented in the paper,
        # but is tested in test_sets.py
        lam = self.smoothing_parameter
        beta = 0.5 * jnp.linalg.norm(
            jnp.abs(y * self.deltas)**(1 / (lam - 1)), ord=lam)**(lam - 2)
        value = beta * jnp.sign(y) * jnp.abs(y)**(1 / (lam - 1)) * (
            self.deltas**(lam / (lam - 1)))
        value = value * ((self.num_variables**(1 / lam))**2)
        h_gradient_inv = self.center + value
        return h_gradient_inv

    def gauss_map_inverse_under(self, d: jnp.ndarray) -> jnp.ndarray:
        lam = self.smoothing_parameter

        d_deltas_lambda = jnp.abs(d * self.deltas)**(
            1. / (lam - 1))
        n_inv = d_deltas_lambda / jnp.linalg.norm(
            d_deltas_lambda, ord=lam)
        n_inv = self.deltas * n_inv
        n_inv = jnp.sign(d) * n_inv
        n_inv = self.center + n_inv
        return n_inv

    def gauss_map_inverse_over(self, d: jnp.ndarray) -> jnp.ndarray:
        lam = self.smoothing_parameter

        d_deltas_lambda = jnp.abs(d * self.deltas)**(
            1. / (lam - 1))
        n_inv = d_deltas_lambda / jnp.linalg.norm(
            d_deltas_lambda, ord=lam)
        n_inv = self.deltas * n_inv
        n_inv = jnp.sign(d) * n_inv
        n_inv = self.num_variables**(1 / lam) * n_inv
        n_inv = self.center + n_inv
        return n_inv

    def gauss_map(self, p: jnp.ndarray) -> jnp.ndarray:
        lam = self.smoothing_parameter

        p_diff = p - self.center
        deltas_lam_inv = 1. / (self.deltas**lam)

        delta_abs_lam_1 = jnp.abs(p_diff)**(lam-1)
        delta_abs_lam_1 = delta_abs_lam_1 * deltas_lam_inv
        delta_abs_lam_1_norm = jnp.linalg.norm(
            delta_abs_lam_1)
        direction = delta_abs_lam_1 / delta_abs_lam_1_norm
        direction = jnp.sign(p_diff) * direction
        return direction

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        return self.gauss_map_inverse_over(d)


class SmoothRectangleUnder(SmoothRectangle):
    """
    Smooth under-approximation of rectangular set represented as
    C = {x in R^n : 
         (sum_i |x[i] - center[i]| / deltas[i])^lam)^{1/lam} <= 1}
    where lam is a smoothing parameter.

    The larger the smoothing parameter, the closer the approximation is to 
    the true rectangle.
    """

    def level_set_function(self, p: jnp.ndarray) -> float:
        h_value = self.level_set_function_under(p)
        return h_value

    def level_set_function_gradient(self, p: jnp.ndarray) -> jnp.ndarray:
        lam = self.smoothing_parameter
        dp = p - self.center
        up = dp * jnp.abs(dp)**(lam - 2) / (self.deltas**lam)
        down = jnp.linalg.norm(dp / self.deltas, ord=lam)**(lam - 2)
        h_gradient = 2. * up / down
        return h_gradient

    def level_set_function_gradient_inverse(
        self, y: jnp.ndarray) -> jnp.ndarray:
        # This formula is not documented in the paper,
        # but is tested in test_sets.py
        lam = self.smoothing_parameter
        beta = 0.5 * jnp.linalg.norm(
            jnp.abs(y * self.deltas)**(1 / (lam - 1)), ord=lam)**(lam - 2)
        value = beta * jnp.sign(y) * jnp.abs(y)**(1 / (lam - 1)) * (
            self.deltas**(lam / (lam - 1)))
        h_gradient_inv = self.center + value
        return h_gradient_inv

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        return self.gauss_map_inverse_under(d)


class FullDimSmoothConvexSet(SmoothConvexSubLevelSet):
    """
    Smooth convex set represented as the sublevel set
    {p = (p_1, ..., p_m, p_(m+1), ..., p_n) in R^n : 
     h(p_1, ..., p_m) + 0.5 ||(p_(m+1), ..., p_n)||^2 <= 1}
    given a sublevel in R^m represented by the 
    smooth convex function h: R^m -> R.
    """
    def __init__(
        self,
        sub_level_set: SmoothConvexSubLevelSet,
        num_variables_to_add: int = 1):
        self._base_sub_level_set = sub_level_set
        num_variables = self.base_sub_level_set.num_variables
        num_variables += num_variables_to_add
        super().__init__(num_variables)

    @property
    def base_sub_level_set(self) -> SmoothConvexSubLevelSet:
        """Returns the base level set without added dimension."""
        return self._base_sub_level_set

    def level_set_function(self, p: jnp.ndarray) -> float:
        p_base = p[:self.base_sub_level_set.num_variables]
        p_added = p[self.base_sub_level_set.num_variables:]
        base_level_set_fn_value = self.base_sub_level_set.level_set_function(
            p_base)
        added_level_set_fn_value = 0.5 * jnp.sum(p_added**2)
        h_value = base_level_set_fn_value + added_level_set_fn_value
        return h_value

    def level_set_function_gradient(self, p: jnp.ndarray) -> jnp.ndarray:
        p_base = p[:self.base_sub_level_set.num_variables]
        p_added = p[self.base_sub_level_set.num_variables:]
        h_base_gradient = self.base_sub_level_set.level_set_function_gradient(
            p_base)
        h_gradient = jnp.concatenate([
            h_base_gradient, p_added])
        return h_gradient

    def level_set_function_gradient_inverse(
        self, y: jnp.ndarray) -> jnp.ndarray:
        y_base = y[:self.base_sub_level_set.num_variables]
        y_added = y[self.base_sub_level_set.num_variables:]
        h_base_gradient_inv = (
            self.base_sub_level_set.level_set_function_gradient_inverse(
                y_base))
        h_gradient_inv = jnp.concatenate([
            h_base_gradient_inv, y_added])
        return h_gradient_inv

    def gauss_map_inverse(self, d: jnp.ndarray) -> jnp.ndarray:
        n_inv = self.level_set_function_gradient_inverse(d / (
            jnp.sqrt(self.level_set_function(
                self.level_set_function_gradient_inverse(d)))))
        return n_inv
