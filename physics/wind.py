import pandas as pd
import numpy as np
from .ode_fitting import fit_shot_parameters
from .trajectory import integrate_trajectory

def estimate_session_winds(train_df, test_df=None):
    """
    Groups shots into sessions by launch_time (gaps > 1000s).
    Estimates the ambient horizontal wind (wx, wy) per session.
    """
    if test_df is not None:
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df.copy()
        
    df = df.sort_values('launch_time').reset_index(drop=True)
    
    # 1. Assign session IDs
    df['time_gap'] = df['launch_time'].diff().fillna(0)
    df['session_id'] = (df['time_gap'] > 1000).cumsum()
    
    session_winds = {}
    
    # 2. Estimate wind per session
    print("Estimating session winds...")
    for sid, group in df.groupby('session_id'):
        # Take up to 10 shots from the session to estimate wind quickly
        sample = group.head(10)
        
        err_x_list = []
        err_y_list = []
        
        for _, row in sample.iterrows():
            # Pass 1: Fit with zero wind
            cd, cl0, _ = fit_shot_parameters(row, wx=0.0, wy=0.0)
            
            y0 = [row.launch_x, row.launch_y, row.launch_z, 
                  row.launch_vx, row.launch_vy, row.launch_vz]
            traj = integrate_trajectory(y0, cd, cl0, wx=0.0, wy=0.0, max_time=row.cp4_t)
            
            # Extract prediction at cp4 (guarded against early stopping)
            sol = traj['sol']
            t_eval = min(row.cp4_t, sol.t[-1])
            pos_cp4 = sol.sol(t_eval)
            
            err_x = row.cp4_x - pos_cp4[0]
            err_y = row.cp4_y - pos_cp4[1]
            
            err_x_list.append(err_x)
            err_y_list.append(err_y)
            
        # Average lateral deviation over the sample
        mean_err_x = np.mean(err_x_list)
        mean_err_y = np.mean(err_y_list)
        
        # Simple heuristic: x = 0.5 * (F_w / m) * t^2
        # wind acceleration ~ err / (0.5 * t_mean^2)
        # Assuming F_w ~ drag_coeff * w, this gives a rough linear mapping.
        # Empirically, w_x ~ mean_err_x / cp4_t (very rough, but provides a good directional prior)
        t_mean = sample['cp4_t'].mean()
        
        # Scaling factor empirical tuning for golf ball wind drift
        scale = 1.5 / t_mean 
        wx = mean_err_x * scale
        wy = mean_err_y * scale
        
        # Clamp absurd winds
        wx = np.clip(wx, -10.0, 10.0)
        wy = np.clip(wy, -10.0, 10.0)
        
        session_winds[sid] = (wx, wy)
        
    return df[['track_id', 'session_id']], session_winds
