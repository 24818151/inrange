import numpy as np

def simulate_bounce_roll(x, y, z, vx, vy, vz, t_start=0.0, cor=0.4, mu=0.35, dt=0.05):
    """
    Simulates the post-impact bounce and roll of the golf ball.
    cor: Coefficient of Restitution (vertical energy retention)
    mu: Turf friction coefficient
    Returns the full rest coordinates and trajectory arrays for plotting.
    """
    t = t_start
    traj_t, traj_x, traj_y, traj_z = [t], [x], [y], [z]
    
    # 1. Bouncing phase
    # Simplified bounce kinematics (parabolic arcs between impacts)
    while abs(vz) > 0.5:
        # Impact: lose vertical velocity, friction reduces horizontal
        vz = -vz * cor
        
        # Friction impulse proportional to vertical impulse
        impulse_z = (1 + cor) * abs(vz / cor) # original abs(vz) before assignment
        vh = np.sqrt(vx**2 + vy**2)
        delta_vh = mu * impulse_z
        
        if delta_vh >= vh:
            vx, vy = 0.0, 0.0
            break
        else:
            vx = vx * (1 - delta_vh / vh)
            vy = vy * (1 - delta_vh / vh)
            
        # Parabolic flight to next bounce
        t_flight = 2 * vz / 9.81
        steps = max(2, int(t_flight / dt))
        t_steps = np.linspace(0, t_flight, steps)[1:]
        
        for dt_step in t_steps:
            traj_t.append(t + dt_step)
            traj_x.append(x + vx * dt_step)
            traj_y.append(y + vy * dt_step)
            traj_z.append(z + vz * dt_step - 0.5 * 9.81 * dt_step**2)
            
        t += t_flight
        x += vx * t_flight
        y += vy * t_flight
        vz = -vz # velocity just before next impact (approx)

    # 2. Rolling phase
    vh = np.sqrt(vx**2 + vy**2)
    if vh > 0:
        a_roll = -mu * 9.81
        t_roll = -vh / a_roll
        
        steps = max(2, int(t_roll / dt))
        t_steps = np.linspace(0, t_roll, steps)[1:]
        
        dir_x = vx / vh
        dir_y = vy / vh
        
        for dt_step in t_steps:
            v_curr = vh + a_roll * dt_step
            dist = vh * dt_step + 0.5 * a_roll * dt_step**2
            
            traj_t.append(t + dt_step)
            traj_x.append(x + dir_x * dist)
            traj_y.append(y + dir_y * dist)
            traj_z.append(z)
            
        t += t_roll
        x += dir_x * (vh * t_roll + 0.5 * a_roll * t_roll**2)
        y += dir_y * (vh * t_roll + 0.5 * a_roll * t_roll**2)
        
    return {
        'rest_t': t,
        'rest_x': x,
        'rest_y': y,
        'rest_z': z,
        'bounce_traj': (np.array(traj_t), np.array(traj_x), np.array(traj_y), np.array(traj_z))
    }
