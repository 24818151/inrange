import pandas as pd
import numpy as np
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
train = pd.read_csv(ROOT_DIR / 'train.csv')

# cp4 IS the net boundary - 60m downrange.
# The ball is still rising when it crosses the net - apex is BEYOND the observable zone.
train['vz_cp34_est'] = (train.cp4_z - train.cp3_z) / (train.cp4_t - train.cp3_t)
n_rising = (train.vz_cp34_est > 0).sum()
print(f'Shots still RISING at cp4 (apex > 60m): {n_rising}/{len(train)} ({100*n_rising/len(train):.1f}%)')
print(f'vz_cp34_est range: [{train.vz_cp34_est.min():.2f}, {train.vz_cp34_est.max():.2f}] m/s')

# Use fixed bearing angle for lateral drift calculation
angle_rad = np.radians(25.1)
dr_hat_x = np.cos(angle_rad)
dr_hat_y = np.sin(angle_rad)
lat_hat_x = -np.sin(angle_rad)
lat_hat_y = np.cos(angle_rad)

for cp in ['cp1','cp2','cp3','cp4']:
    dx = train[f'{cp}_x'] - train.launch_x
    dy = train[f'{cp}_y'] - train.launch_y
    train[f'{cp}_lat'] = dx * lat_hat_x + dy * lat_hat_y

dx_land = train.landing_x - train.launch_x
dy_land = train.landing_y - train.launch_y
train['landing_lat'] = dx_land * lat_hat_x + dy_land * lat_hat_y

print('\nLateral drift with fixed bearing:')
for cp in ['cp1','cp2','cp3','cp4']:
    print(f'  {cp}_lat: mean={train[f"{cp}_lat"].mean():.3f}, std={train[f"{cp}_lat"].std():.3f}')
print(f'  landing_lat: mean={train.landing_lat.mean():.3f}, std={train.landing_lat.std():.3f}')
print(f'\nCorr spin_rate vs cp4_lat: {train.launch_spin_rate.corr(train.cp4_lat):.3f}')
print(f'Corr spin_rate vs landing_lat: {train.launch_spin_rate.corr(train.landing_lat):.3f}')

# Horizontal angular deflection between velocity at cp1-2 and cp3-4
train['vx_cp12'] = (train.cp2_x - train.cp1_x) / (train.cp2_t - train.cp1_t)
train['vy_cp12'] = (train.cp2_y - train.cp1_y) / (train.cp2_t - train.cp1_t)
train['vx_cp34'] = (train.cp4_x - train.cp3_x) / (train.cp4_t - train.cp3_t)
train['vy_cp34'] = (train.cp4_y - train.cp3_y) / (train.cp4_t - train.cp3_t)

train['angular_defl'] = np.degrees(
    np.arctan2(train.vy_cp34, train.vx_cp34) -
    np.arctan2(train.vy_cp12, train.vx_cp12)
)
print(f'\nHorizontal angular deflection cp12->cp34: mean={train.angular_defl.mean():.3f}, std={train.angular_defl.std():.3f} deg')
print(f'Corr spin_rate vs angular_defl: {train.launch_spin_rate.corr(train.angular_defl):.3f}')

# How much does the gravity-compensated vertical lift differ?
# This is the Magnus backspin lift signature
train['vz_cp12_est'] = (train.cp2_z - train.cp1_z) / (train.cp2_t - train.cp1_t)
train['g_vz_loss_cp12'] = train.launch_vz - 9.81 * train.cp2_t
train['lift_cp12'] = train.vz_cp12_est - train.g_vz_loss_cp12
print(f'\nLift residual at cp1-2 (Magnus backspin): mean={train.lift_cp12.mean():.2f}, std={train.lift_cp12.std():.2f} m/s')
print(f'Corr spin_rate vs lift_cp12: {train.launch_spin_rate.corr(train.lift_cp12):.3f}')

# Speed-based features
train['speed_launch'] = np.sqrt(train.launch_vx**2 + train.launch_vy**2 + train.launch_vz**2)
train['speed_cp12'] = np.sqrt(train.vx_cp12**2 + train.vy_cp12**2 + ((train.cp2_z-train.cp1_z)/(train.cp2_t-train.cp1_t))**2)
train['speed_cp34'] = np.sqrt(train.vx_cp34**2 + train.vy_cp34**2 + train.vz_cp34_est**2)

# Launch angle
train['launch_angle_v'] = np.degrees(np.arctan2(train.launch_vz, np.sqrt(train.launch_vx**2 + train.launch_vy**2)))
print(f'\nLaunch angle (vertical): mean={train.launch_angle_v.mean():.1f}, std={train.launch_angle_v.std():.1f} deg')
print(f'Corr spin_rate vs launch_angle_v: {train.launch_spin_rate.corr(train.launch_angle_v):.3f}')
print(f'Corr spin_rate vs speed_launch: {train.launch_spin_rate.corr(train.speed_launch):.3f}')
print(f'Corr spin_rate vs speed_cp12: {train.launch_spin_rate.corr(train.speed_cp12):.3f}')
print(f'Corr spin_rate vs speed_cp34: {train.launch_spin_rate.corr(train.speed_cp34):.3f}')

# The Kronecker/multi-task GPR concern: with overnight compute, this is feasible
# But we have 9 outputs and only 491 points - let's think about what architecture makes sense
print(f'\nN_train={len(train)}, N_targets=9')
print('Exact GP Cholesky cost: O(N^3) per output = {:.0f}M ops x9'.format(491**3 / 1e6))
print('KroneckerMultiTaskGP would model output correlations - worthwhile with overnight compute')
