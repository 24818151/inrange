import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from physics.ode_fitting import fit_shot_parameters
from physics.trajectory import integrate_trajectory
from physics.wind import estimate_session_winds
from surrogate.feature_engineering import extract_raw_features

# Configure high-quality matplotlib styling for academic/Kaggle writeup
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'

ASSETS_DIR = ROOT_DIR / 'assets'
ASSETS_DIR.mkdir(exist_ok=True)

def generate_fig1_speed_decay(train_df):
    print("Generating Figure 1: Checkpoint Speed Deceleration...")
    feat_df = extract_raw_features(train_df)
    
    stages = ['Launch (0m)', 'CP1-2 (~22m)', 'CP3-4 (~52m)']
    speeds = [
        feat_df['speed_launch'].values,
        feat_df['speed_cp12'].values,
        feat_df['speed_cp34'].values
    ]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(speeds, showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('#1f77b4')
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('#d62728')
    parts['cmeans'].set_linewidth(2)
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.set_title('Figure 1: Aerodynamic Drag Deceleration Through Checkpoints (N=491)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y')
    
    mean_drop = feat_df['speed_launch'].mean() - feat_df['speed_cp34'].mean()
    ax.text(0.05, 0.15, f'Mean speed loss: {mean_drop:.1f} m/s (~{mean_drop/feat_df["speed_launch"].mean()*100:.1f}%)', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc'))
            
    fig.tight_layout()
    out = ASSETS_DIR / 'fig1_speed_deceleration.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

def generate_fig2_lift_vs_spin(train_df):
    print("Generating Figure 2: Lift Residual vs Spin Rate...")
    feat_df = extract_raw_features(train_df)
    x = feat_df['lift_residual'].values
    y = train_df['launch_spin_rate'].values
    
    corr = np.corrcoef(x, y)[0, 1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color='#2ca02c', alpha=0.5, edgecolors='none', s=45, label='Recorded Shots')
    
    # Linear fit
    m, b = np.polyfit(x, y, 1)
    x_grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_grid, m*x_grid + b, color='#d62728', linewidth=2.2, label=f'Linear Prior Trend (r = {corr:.3f})')
    
    ax.set_xlabel('Vertical Lift Residual $v_{z,CP12} - v_{z,ballistic}$ (m/s)', fontsize=11)
    ax.set_ylabel('Launch Spin Rate (RPM)', fontsize=11)
    ax.set_title('Figure 2: Empirical Magnus Lift Residual vs Launch Spin Rate', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True)
    
    fig.tight_layout()
    out = ASSETS_DIR / 'fig2_lift_residual_vs_spin.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

def generate_fig3_session_winds(train_df):
    print("Generating Figure 3: Session Wind Distribution...")
    _, session_winds = estimate_session_winds(train_df)
    
    sids = sorted(list(session_winds.keys()))
    wx_vals = [session_winds[s][0] for s in sids]
    wy_vals = [session_winds[s][1] for s in sids]
    
    x_pos = np.arange(len(sids))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width/2, wx_vals, width, label='Wind X (Head/Tail)', color='#1f77b4', alpha=0.85)
    ax.bar(x_pos + width/2, wy_vals, width, label='Wind Y (Crosswind)', color='#ff7f0e', alpha=0.85)
    
    ax.set_xlabel('Radar Session ID (Chronological)', fontsize=11)
    ax.set_ylabel('Estimated Ambient Wind (m/s)', fontsize=11)
    ax.set_title('Figure 3: Inverse-Estimated Ambient Wind Vectors Across 18 Radar Sessions', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s}' for s in sids], fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(frameon=True, facecolor='white')
    ax.grid(True, axis='y')
    
    fig.tight_layout()
    out = ASSETS_DIR / 'fig3_session_winds.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

def generate_fig4_ode_fit_quality(train_df):
    print("Generating Figure 4: Inverse ODE Checkpoint Residuals...")
    mses = []
    for idx in range(min(len(train_df), 150)):
        row = train_df.iloc[idx]
        _, _, mse = fit_shot_parameters(row)
        mses.append(mse)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mses, bins=25, color='#9467bd', edgecolor='black', alpha=0.75, density=True)
    ax.set_xlabel('Checkpoint Position Reconstruction MSE (m²)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Figure 4: Inverse ODE Fit Accuracy at the 4 Radar Checkpoints', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y')
    
    med_mse = np.median(mses)
    ax.axvline(med_mse, color='#d62728', linestyle='--', linewidth=2, label=f'Median MSE: {med_mse:.3f} m²')
    ax.legend(frameon=True)
    
    fig.tight_layout()
    out = ASSETS_DIR / 'fig4_ode_fit_residuals.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

def generate_fig5_3d_trajectory(train_df):
    print("Generating Figure 5: 3D Trajectory & Bounce...")
    row = train_df.iloc[0]
    cd, cl0, _ = fit_shot_parameters(row)
    
    y0 = [row.launch_x, row.launch_y, row.launch_z, row.launch_vx, row.launch_vy, row.launch_vz]
    traj = integrate_trajectory(y0, cd, cl0, simulate_bounce=True)
    
    sol = traj['sol']
    t_eval = np.linspace(0, traj['landing_t'], 200)
    pos = sol.sol(t_eval)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(pos[0], pos[1], pos[2], label='Airborne Flight (ODE)', color='#1f77b4', linewidth=2.5)
    
    if 'bounce_traj' in traj:
        bt, bx, by, bz = traj['bounce_traj']
        ax.plot(bx, by, bz, label='Bounce & Roll Phase', color='#ff7f0e', linestyle='--', linewidth=2)
        ax.scatter([traj['rest_x']], [traj['rest_y']], [traj['rest_z']], 
                   color='#d62728', s=90, marker='*', label='Final Rest Position')
                   
    # Checkpoints
    cp_x = [row.cp1_x, row.cp2_x, row.cp3_x, row.cp4_x]
    cp_y = [row.cp1_y, row.cp2_y, row.cp3_y, row.cp4_y]
    cp_z = [row.cp1_z, row.cp2_z, row.cp3_z, row.cp4_z]
    ax.scatter(cp_x, cp_y, cp_z, color='red', s=50, label='Checkpoints (15-60m)', marker='o')
    
    # 60m Net plane
    angle_rad = np.radians(25.1)
    net_x = np.linspace(pos[0].min()-5, pos[0].max()+5, 10)
    net_z = np.linspace(0, pos[2].max()*1.15, 10)
    X_net, Z_net = np.meshgrid(net_x, net_z)
    Y_net = (60.0 - (X_net - row.launch_x) * np.cos(angle_rad)) / np.sin(angle_rad) + row.launch_y
    ax.plot_surface(X_net, Y_net, Z_net, color='red', alpha=0.15)
    
    # True targets
    ax.scatter([row.apex_x], [row.apex_y], [row.apex_z], color='green', s=90, label='True Apex', marker='^')
    ax.scatter([row.landing_x], [row.landing_y], [row.landing_z], color='purple', s=90, label='True Landing', marker='X')
    
    ax.set_xlabel('X Downrange (m)')
    ax.set_ylabel('Y Lateral (m)')
    ax.set_zlabel('Height Z (m)')
    ax.set_title('Figure 5: 3D Trajectory Reconstruction with Post-Landing Bounce & Roll', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    
    fig.tight_layout()
    out = ASSETS_DIR / 'fig5_3d_trajectory_bounce.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

def generate_fig6_cv_performance():
    print("Generating Figure 6: CV Benchmark Results...")
    targets = ['apex_t (s)', 'apex_x (m)', 'apex_y (m)', 'apex_z (m)', 
               'landing_t (s)', 'landing_x (m)', 'landing_y (m)', 'landing_z (m)']
    rmse_vals = [0.13, 4.69, 4.61, 0.86, 0.22, 8.16, 11.03, 0.09]
    mae_vals =  [0.07, 2.78, 3.10, 0.38, 0.13, 4.94, 7.81, 0.08]
    
    x = np.arange(len(targets))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, rmse_vals, width, label='5-Fold CV RMSE', color='#1f77b4', alpha=0.85)
    ax.bar(x + width/2, mae_vals, width, label='5-Fold CV MAE', color='#2ca02c', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Prediction Error (meters / seconds)', fontsize=11)
    ax.set_title('Figure 6: 5-Fold Cross-Validation Accuracy Across Trajectory Targets', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white')
    ax.grid(True, axis='y')
    
    for i in range(len(targets)):
        ax.text(x[i] - width/2, rmse_vals[i] + 0.2, f'{rmse_vals[i]:.2f}', ha='center', fontsize=8)
        ax.text(x[i] + width/2, mae_vals[i] + 0.2, f'{mae_vals[i]:.2f}', ha='center', fontsize=8)
        
    fig.tight_layout()
    out = ASSETS_DIR / 'fig6_cv_performance.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == '__main__':
    train_df = pd.read_csv(ROOT_DIR / 'train.csv')
    generate_fig1_speed_decay(train_df)
    generate_fig2_lift_vs_spin(train_df)
    generate_fig3_session_winds(train_df)
    generate_fig4_ode_fit_quality(train_df)
    generate_fig5_3d_trajectory(train_df)
    generate_fig6_cv_performance()
    print("All figures successfully exported to assets/ folder!")
