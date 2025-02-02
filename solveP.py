import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def compute_vorticity(dx, dy, u, v):
    """
    Compute vorticity from collocated PIV velocity data using central differences.
    """
    dvdx, dudy = compute_first_derivatives(dx, dy, u, v)
    return dvdx - dudy


def compute_first_derivatives(dx, dy, u, v):
    """
    Compute first order central difference approximation of velocity derivatives.
    """
    dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (dx[2:, 1:-1] + dx[:-2, 1:-1])
    dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (dy[1:-1, 2:] + dy[1:-1, :-2])
    return dvdx, dudy


def compute_second_derivatives(dx, dy, u, v):
    """
    Compute second order centered difference approximation of second derivatives of velocity.
    """
    du2dx2 = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx[1:-1, 1:-1] ** 2)
    dv2dy2 = (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / (dy[1:-1, 1:-1] ** 2)
    return du2dx2, dv2dy2


def solve_pressure_poisson(dx, dy, u, v):
    """
    Solve the pressure Poisson equation using second-order central differences for uniform grid
    """

    # Kinetic energy
    du2dx2, dv2dy2 = compute_second_derivatives(dx, dy, u, v)
    k = 0.5 * (du2dx2 + dv2dy2)

    # Curl term
    c = (np.gradient(v * compute_vorticity(dx, dy, u, v), axis=1) / dx[1:-1, 1:-1]) - (
                np.gradient(u * compute_vorticity(dx, dy, u, v), axis=0) / dy[1:-1, 1:-1])
    b = c + k

    # Grid size
    ny, nx = b.shape
    N = ny * nx

    # Construct the sparse Laplacian matrix A
    dx2 = dx[1:-1, 1:-1] ** 2
    dy2 = dy[1:-1, 1:-1] ** 2
    main_diag = -2 * (1 / dx2 + 1 / dy2)
    off_diag_x = 1 / dx2
    off_diag_y = 1 / dy2

    diagonals = [main_diag.ravel(), off_diag_x.ravel()[:-1], off_diag_x.ravel()[:-1], off_diag_y.ravel()[:-nx],
                 off_diag_y.ravel()[:-nx]]
    positions = [0, 1, -1, nx, -nx]
    A = diags(diagonals, positions, shape=(N, N), format='csr')

    # Apply Bernoulli boundary condition at the sides (left, right, top, and bottom boundaries)
    
    # Left boundary (x = 0)
    bernoulli_left = 0.5 * (u[:, 0] ** 2 + v[:, 0] ** 2)  
    b[0, :] = bernoulli_left  # Apply Bernoulli on left boundary (row 0)
    # Set A matrix for left boundary (directly known pressure)
    A[0, :] = 0  # Set all elements in the row to 0
    A[0, 0] = 1  # Set the diagonal to 1 (pressure is directly known)
    
    # Right boundary (x = nx-1)
    bernoulli_right = 0.5 * (u[:, -1] ** 2 + v[:, -1] ** 2)  
    b[-1, :] = bernoulli_right  # Apply Bernoulli on right boundary (row -1)
    # Set A matrix for right boundary (directly known pressure)
    A[-1, :] = 0  # Set all elements in the row to 0
    A[-1, -1] = 1  # Set the diagonal to 1 (pressure is directly known)
    
    # Top boundary (y = 0)
    bernoulli_top = 0.5 * (u[0, :] ** 2 + v[0, :] ** 2)  
    b[:, 0] = bernoulli_top  # Apply Bernoulli on top boundary (column 0)
    # Set A matrix for top boundary (directly known pressure)
    A[:, 0] = 0  # Set all elements in the column to 0
    A[0, 0] = 1  # Set the diagonal to 1 (pressure is directly known)
    
    # Bottom boundary (y = ny-1)
    bernoulli_bottom = 0.5 * (u[-1, :] ** 2 + v[-1, :] ** 2)  
    b[:, -1] = bernoulli_bottom  # Apply Bernoulli on bottom boundary (column -1)
    # Set A matrix for bottom boundary (directly known pressure)
    A[:, -1] = 0  # Set all elements in the column to 0
    A[-1, -1] = 1  # Set the diagonal to 1 (pressure is directly known)

    # Solve for P
    P = spsolve(A, b.ravel())

    return P.reshape((ny, nx))
