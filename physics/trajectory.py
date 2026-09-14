import numpy as np
from scipy.integrate import solve_ivp
from .aerodynamics import ball_derivatives
from .bounce import simulate_bounce_roll

def integrate_trajectory(initial_state, cd, cl0, wx=0.0, wy=0.0, max_time=15.0, simulate_bounce=False):
    """
    Integrates the trajectory of the golf ball until it hits the ground.
    Optionally simulates post-impact bounce and roll.
    """
    launch_z = initial_state[2]
    
    def event_landing_safe(t, state):
        if t < 0.5:
            return 1.0 
        return state[2] - launch_z
    event_landing_safe.terminal = True
    event_landing_safe.direction = -1
    
    def event_apex(t, state):
        return state[5]
    event_apex.terminal = False
    event_apex.direction = -1
    
    sol = solve_ivp(
        fun=lambda t, y: ball_derivatives(t, y, cd, cl0, wx, wy),
        t_span=(0, max_time),
        y0=initial_state,
        events=[event_landing_safe, event_apex],
        dense_output=True,
        rtol=1e-6,
        atol=1e-8
    )
    
    results = {
        'sol': sol,
        'landing_t': sol.t[-1],
        'landing_x': sol.y[0, -1],
        'landing_y': sol.y[1, -1],
        'landing_z': sol.y[2, -1],
        'landing_vx': sol.y[3, -1],
        'landing_vy': sol.y[4, -1],
        'landing_vz': sol.y[5, -1]
    }
    
    if len(sol.t_events[1]) > 0:
        apex_t = sol.t_events[1][0]
        apex_state = sol.sol(apex_t)
        results['apex_t'] = apex_t
        results['apex_x'] = apex_state[0]
        results['apex_y'] = apex_state[1]
        results['apex_z'] = apex_state[2]
    else:
        idx = np.argmax(sol.y[2])
        results['apex_t'] = sol.t[idx]
        results['apex_x'] = sol.y[0, idx]
        results['apex_y'] = sol.y[1, idx]
        results['apex_z'] = sol.y[2, idx]
        
    if simulate_bounce:
        bounce_results = simulate_bounce_roll(
            results['landing_x'], results['landing_y'], launch_z, 
            results['landing_vx'], results['landing_vy'], results['landing_vz'],
            t_start=results['landing_t']
        )
        results.update(bounce_results)
        
    return results
