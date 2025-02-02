import numpy as np

def compute_first_derivatives(dx, dy, u, v):
    """
    Compute first order central difference approximation of velocity derivatives.

    Parameters:
    dx : 2D numpy array
        Grid spacing in the x-direction (same shape as u, v).
    dy : 2D numpy array
        Grid spacing in the y-direction (same shape as u, v).
    u : 2D numpy array
        Velocity component in the x-direction.
    v : 2D numpy array
        Velocity component in the y-direction.

    Returns:
    dvdx : 2D numpy array
        First derivative of v with respect to x.
    dudy : 2D numpy array
        First derivative of u with respect to y.
    """
    dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (dx[2:, 1:-1] + dx[:-2, 1:-1])
    dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (dy[1:-1, 2:] + dy[1:-1, :-2])

    return dvdx, dudy
