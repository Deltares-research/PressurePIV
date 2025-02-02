def compute_second_derivatives(dx, dy, u, v):
    """
    Compute second order centered difference approximation of second derivatives of velocity.

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
    du2dx2 : 2D numpy array
        Second derivative of u with respect to x.
    dv2dy2 : 2D numpy array
        Second derivative of v with respect to y.
    """
    # Compute second derivatives using second order centered differences
    du2dx2 = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx[1:-1, 1:-1] ** 2)
    dv2dy2 = (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / (dy[1:-1, 1:-1] ** 2)

    return du2dx2, dv2dy2