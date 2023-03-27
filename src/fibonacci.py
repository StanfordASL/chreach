# Construct a Fibonacci lattice, see
# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
import numpy as np
def fibonacci_lattice(r, M):
    # inputs: 
    #   r - scalar - radius of the sphere
    #   M - integer - number of points on the lattice
    # returns:
    #   pts - (M, 3) - points on the lattice
    indices = np.arange(0, M, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices / M)
    theta = np.pi * (1 + 5**0.5) * indices
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    pts = np.stack((x, y, z)).T
    return pts