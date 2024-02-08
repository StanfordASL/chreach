"""Statistics helper functions."""
import numpy as np


def uniformly_sample_points_in_unit_sphere(
    num_variables: int, sample_size: int) -> np.ndarray:
    """
    Uniformly samples points on a sphere (boundary of a ball) represented as
        {x in R^{num_variables} : ||x||_2 = 1}.

    Args:
        num_variables: dimension of the ambient space of the sphere
            (int) 
    `   sample_size - number of points to sample
            (int)

    Returns:
        points: point samples uniformly distributed in the sphere
            (sample_size, num_variables)
    """
    u = np.random.normal(0, 1, (num_variables, sample_size))
    d = np.sum(u**2, axis=0) **(0.5)
    pts = u / d
    pts = pts.T
    return pts

def uniformly_sample_points_in_unit_ball(
    num_variables: int, sample_size: int) -> np.ndarray:
    """
    Uniformly samples points in a ball represented as
        {x in R^{num_variables} : ||x||_2 <= 1}.

    Args:
        num_variables: dimension of the ambient space of the sphere
            (int) 
    `   sample_size - number of points to sample
            (int)

    Returns:
        points: point samples uniformly distributed in the ball
            (sample_size, num_variables)
    """
    us    = np.random.normal(0, 1, (num_variables, sample_size))
    norms = np.linalg.norm(us, 2, axis=0)
    rs    = np.random.random(sample_size)**(1.0 / num_variables)
    pts   = rs*us / norms
    pts = pts.T
    return pts

def uniformly_sample_points_in_ellipsoid(
    center: np.ndarray,
    shape_matrix: np.ndarray,
    sample_size: int) -> np.ndarray:
    """
    Uniformly samples points in an ellipsoid represented as the set of points
    {p in R^n : (p - center).T @ shape_matrix^{-1} (p- center) <= 1}.

    Args:
        center: center of the ellipsoid
            (num_variables)
        shape_matrix: shape matrix of the ellipsoid
            (num_variables, num_variables) 
    `   sample_size - number of points to sample
            (int)

    Returns:
        points: point samples uniformly distributed in the ellipsoid
            (sample_size, num_variables)
    """
    points = uniformly_sample_points_in_unit_ball(
        center.shape[0], sample_size)
    choleski_shape_matrix = np.linalg.cholesky(shape_matrix)
    points = np.array(choleski_shape_matrix @ points.T).T + center
    return points

def uniformly_sample_points_in_rectangle(
    center: np.ndarray,
    deltas: np.ndarray,
    sample_size: int) -> np.ndarray:
    """
    Uniformly samples points in rectangular set represented as
        {x in R^n : |x[i] - center[i]| <= deltas[i] for all i}

    Args:
        center: center of the rectangle
            (num_variables) array
        deltas: deltas of the rectangle
            (num_variables) array
    `   sample_size - number of points to sample
            (int)

    Returns:
        points: point samples uniformly distributed in the rectangle
            (sample_size, num_variables)
    """
    num_variables = len(center)
    points = np.random.uniform(
        center - deltas,
        center + deltas,
        (sample_size, num_variables))
    return points
