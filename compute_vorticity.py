import numpy as np

# Function: Circulation method

def compute_vorticity(dx, dy, u, v):
    """
    Compute vorticity from collocated PIV velocity data using central differences.
    
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
    omega : 2D numpy array
        Vorticity field (same shape as u, v, excluding the outermost boundary points).
    """
    # Compute partial derivatives using central differences
    dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (dx[2:, 1:-1] + dx[:-2, 1:-1])
    dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (dy[1:-1, 2:] + dy[1:-1, :-2])
    
    # Compute vorticity (omega = dv/dx - du/dy)
    omega = dvdx - dudy
    
    return omega