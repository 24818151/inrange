import numpy as np

# Physical Constants for a standard golf ball
M_BALL = 0.04593     # kg
D_BALL = 0.04267     # m
A_BALL = np.pi * (D_BALL / 2.0)**2  # m^2
RHO_AIR = 1.225      # kg/m^3
GRAVITY = 9.81       # m/s^2
SPIN_DECAY = 0.05    # s^-1 

def ball_derivatives(t, state, cd, cl0, wx=0.0, wy=0.0):
    """
    Computes the time derivatives of the golf ball state, including ambient wind.
    state: [x, y, z, vx, vy, vz]
    cd: Drag coefficient
    cl0: Initial lift coefficient
    wx, wy: Ambient wind vector in x and y (m/s)
    """
    x, y, z, vx, vy, vz = state
    
    # Apparent velocity (relative to wind)
    vr_x = vx - wx
    vr_y = vy - wy
    vr_z = vz
    
    vr_mag = np.sqrt(vr_x**2 + vr_y**2 + vr_z**2)
    vr_xy = np.sqrt(vr_x**2 + vr_y**2)
    
    # Drag force opposes apparent velocity
    fd_coeff = -0.5 * RHO_AIR * A_BALL * cd * vr_mag
    fd_x = fd_coeff * vr_x
    fd_y = fd_coeff * vr_y
    fd_z = fd_coeff * vr_z
    
    # Magnus Lift force (Backspin only)
    # Spin axis w is perpendicular to the apparent horizontal velocity
    cl_t = cl0 * np.exp(-SPIN_DECAY * t)
    
    if vr_xy > 1e-3:
        fl_coeff = 0.5 * RHO_AIR * A_BALL * cl_t * vr_mag / vr_xy
        fl_x = fl_coeff * (-vr_x * vr_z)
        fl_y = fl_coeff * (-vr_y * vr_z)
        fl_z = fl_coeff * (vr_xy**2)
    else:
        fl_x = fl_y = fl_z = 0.0
        
    # Gravity
    fg_z = -M_BALL * GRAVITY
    
    # Accelerations (using inertial velocity derivatives, forces computed via apparent)
    ax = (fd_x + fl_x) / M_BALL
    ay = (fd_y + fl_y) / M_BALL
    az = (fd_z + fl_z + fg_z) / M_BALL
    
    return [vx, vy, vz, ax, ay, az]
