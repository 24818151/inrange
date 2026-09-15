import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from .aerodynamics import ball_derivatives

def fit_shot_parameters(shot_series, wx=0.0, wy=0.0):
    """
    Fits Cd and Cl0 for a single shot given its launch state, 4 checkpoints, and ambient wind.
    Uses multi-start L-BFGS-B across multiple aerodynamic regimes (driver, mid-iron, wedge)
    with tight integration tolerances to find the global optimum.
    """
    y0 = [
        shot_series['launch_x'], shot_series['launch_y'], shot_series['launch_z'],
        shot_series['launch_vx'], shot_series['launch_vy'], shot_series['launch_vz']
    ]
    
    cp_times = [
        shot_series['cp1_t'], shot_series['cp2_t'], 
        shot_series['cp3_t'], shot_series['cp4_t']
    ]
    cp_targets = np.array([
        [shot_series['cp1_x'], shot_series['cp1_y'], shot_series['cp1_z']],
        [shot_series['cp2_x'], shot_series['cp2_y'], shot_series['cp2_z']],
        [shot_series['cp3_x'], shot_series['cp3_y'], shot_series['cp3_z']],
        [shot_series['cp4_x'], shot_series['cp4_y'], shot_series['cp4_z']]
    ])
    
    t_max = cp_times[-1]
    
    def objective(params):
        cd, cl0 = params
        sol = solve_ivp(
            fun=lambda t, y: ball_derivatives(t, y, cd, cl0, wx, wy),
            t_span=(0, t_max),
            y0=y0,
            t_eval=cp_times,
            rtol=1e-6,
            atol=1e-8
        )
        
        if not sol.success or len(sol.t) != 4:
            return 1e6 
            
        pred_positions = sol.y[:3, :].T
        mse = np.mean(np.sum((pred_positions - cp_targets)**2, axis=1))
        return mse

    # 3 distinct aerodynamic starting guesses (Driver/low spin, Mid-iron, High-spin wedge)
    candidate_guesses = [
        [0.22, 0.10],
        [0.27, 0.18],
        [0.34, 0.28]
    ]
    bounds = [(0.1, 0.6), (-0.1, 0.5)]
    
    best_cd = 0.25
    best_cl0 = 0.15
    best_mse = float('inf')
    
    for guess in candidate_guesses:
        res = minimize(
            objective,
            x0=guess,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxfun': 250, 'ftol': 1e-7, 'gtol': 1e-6}
        )
        if res.fun < best_mse:
            best_mse = res.fun
            best_cd = float(res.x[0])
            best_cl0 = float(res.x[1])
            
    return best_cd, best_cl0, best_mse
