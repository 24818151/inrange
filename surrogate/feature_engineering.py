import pandas as pd
import numpy as np
import torch
try:
    from physics.ode_fitting import fit_shot_parameters
    from physics.trajectory import integrate_trajectory
    from physics.wind import estimate_session_winds
except ImportError:
    from ..physics.ode_fitting import fit_shot_parameters
    from ..physics.trajectory import integrate_trajectory
    from ..physics.wind import estimate_session_winds

def extract_raw_features(df):
    df = df.copy()
    
    df['speed_launch'] = np.sqrt(df.launch_vx**2 + df.launch_vy**2 + df.launch_vz**2)
    df['launch_angle_v'] = np.degrees(np.arctan2(df.launch_vz, np.sqrt(df.launch_vx**2 + df.launch_vy**2)))
    # Launch azimuth relative to the 25.1 degree range centerline
    df['launch_azimuth'] = np.degrees(np.arctan2(df.launch_vy, df.launch_vx)) - 25.1
    # Kinetic dynamic pressure proxy
    df['v_squared'] = df.launch_vx**2 + df.launch_vy**2 + df.launch_vz**2
    # Theoretical zero-drag time to apex
    df['t_apex_ballistic'] = df.launch_vz / 9.81
    
    df['vx_cp12'] = (df.cp2_x - df.cp1_x) / (df.cp2_t - df.cp1_t)
    df['vy_cp12'] = (df.cp2_y - df.cp1_y) / (df.cp2_t - df.cp1_t)
    df['vz_cp12'] = (df.cp2_z - df.cp1_z) / (df.cp2_t - df.cp1_t)
    df['speed_cp12'] = np.sqrt(df.vx_cp12**2 + df.vy_cp12**2 + df.vz_cp12**2)
    df['azimuth_cp12'] = np.degrees(np.arctan2(df.vy_cp12, df.vx_cp12)) - 25.1
    
    df['vx_cp34'] = (df.cp4_x - df.cp3_x) / (df.cp4_t - df.cp3_t)
    df['vy_cp34'] = (df.cp4_y - df.cp3_y) / (df.cp4_t - df.cp3_t)
    df['vz_cp34'] = (df.cp4_z - df.cp3_z) / (df.cp4_t - df.cp3_t)
    df['speed_cp34'] = np.sqrt(df.vx_cp34**2 + df.vy_cp34**2 + df.vz_cp34**2)
    df['azimuth_cp34'] = np.degrees(np.arctan2(df.vy_cp34, df.vx_cp34)) - 25.1
    
    # Horizontal curvature rate between early and late radar checkpoints
    df['azimuth_curve'] = df['azimuth_cp34'] - df['azimuth_cp12']
    
    df['decel_rate'] = (df.speed_launch - df.speed_cp34) / df.cp4_t
    
    g_vz_loss = df.launch_vz - 9.81 * df.cp2_t
    df['lift_residual'] = df.vz_cp12 - g_vz_loss
    
    def assign_bay(z):
        if z < 0.055: return 0
        elif z < 0.065: return 1
        elif z < 0.075: return 2
        else: return 3
    
    df['bay_id'] = df['launch_z'].apply(assign_bay)
    for i in range(4):
        df[f'bay_{i}'] = (df['bay_id'] == i).astype(float)
        
    return df

def generate_physics_priors(train_df, test_df=None):
    """
    Fits ODE per shot (Pass 2) using session wind estimates.
    Also extracts bounce & roll prior targets.
    """
    # 1. Estimate wind
    session_map, session_winds = estimate_session_winds(train_df, test_df)
    
    def process_df(df):
        priors = []
        df_with_session = df.merge(session_map, on='track_id', how='left')
        
        for idx, row in df_with_session.iterrows():
            wx, wy = session_winds.get(row['session_id'], (0.0, 0.0))
            
            # Pass 2: ODE fit with wind
            cd, cl0, _ = fit_shot_parameters(row, wx=wx, wy=wy)
            
            y0 = [row.launch_x, row.launch_y, row.launch_z, 
                  row.launch_vx, row.launch_vy, row.launch_vz]
            traj = integrate_trajectory(y0, cd, cl0, wx=wx, wy=wy, simulate_bounce=True)
            
            priors.append({
                'ode_apex_t': traj.get('apex_t', row.cp4_t * 2),
                'ode_apex_x': traj.get('apex_x', row.launch_x),
                'ode_apex_y': traj.get('apex_y', row.launch_y),
                'ode_apex_z': traj.get('apex_z', row.launch_z),
                'ode_landing_t': traj.get('landing_t', row.cp4_t * 4),
                'ode_landing_x': traj.get('landing_x', row.launch_x),
                'ode_landing_y': traj.get('landing_y', row.launch_y),
                'ode_landing_z': traj.get('landing_z', row.launch_z),
                'ode_rest_t': traj.get('rest_t', row.cp4_t * 5),
                'ode_rest_x': traj.get('rest_x', row.launch_x),
                'ode_rest_y': traj.get('rest_y', row.launch_y),
                'ode_cd': cd,
                'ode_cl0': cl0,
                'wx': wx,
                'wy': wy
            })
        return pd.DataFrame(priors, index=df.index)
        
    if test_df is None:
        return process_df(train_df)
    else:
        return process_df(train_df), process_df(test_df)

def build_feature_tensor(df, priors_df, scaler=None):
    feature_cols = [
        'launch_vx', 'launch_vy', 'launch_vz', 'speed_launch', 'launch_angle_v',
        'launch_azimuth', 'v_squared', 't_apex_ballistic', 'azimuth_curve',
        'speed_cp12', 'speed_cp34', 'decel_rate', 'lift_residual',
        'bay_0', 'bay_1', 'bay_2', 'bay_3',
        'ode_apex_t', 'ode_apex_x', 'ode_apex_y', 'ode_apex_z',
        'ode_landing_t', 'ode_landing_x', 'ode_landing_y', 'ode_landing_z',
        'ode_rest_t', 'ode_rest_x', 'ode_rest_y',
        'ode_cd', 'ode_cl0', 'wx', 'wy'
    ]
    
    combined = pd.concat([df, priors_df], axis=1)
    X_raw = combined[feature_cols].values
    
    if scaler is None:
        X_min = X_raw.min(axis=0)
        X_max = X_raw.max(axis=0)
        ptp = X_max - X_min
        ptp[ptp == 0] = 1.0
        scaler = {'min': X_min, 'ptp': ptp, 'cols': feature_cols}
        
    X_norm = (X_raw - scaler['min']) / scaler['ptp']
    return torch.tensor(X_norm, dtype=torch.float64), scaler
