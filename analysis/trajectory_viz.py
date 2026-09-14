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

def plot_shot_trajectory(df, idx=0, save_fig=True):
    row = df.iloc[idx]
    cd, cl0, _ = fit_shot_parameters(row)
    
    y0 = [row.launch_x, row.launch_y, row.launch_z, row.launch_vx, row.launch_vy, row.launch_vz]
    traj = integrate_trajectory(y0, cd, cl0, simulate_bounce=True)
    
    sol = traj['sol']
    t_eval = np.linspace(0, traj['landing_t'], 200)
    pos = sol.sol(t_eval)
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot continuous ODE airborne trajectory
    ax.plot(pos[0], pos[1], pos[2], label='Airborne Flight Path', color='#1f77b4', linewidth=2.5)
    
    # 2. Plot bounce & roll path if available
    if 'bounce_traj' in traj:
        bt, bx, by, bz = traj['bounce_traj']
        ax.plot(bx, by, bz, label='Bounce & Roll', color='#ff7f0e', linestyle='--', linewidth=2)
        ax.scatter([traj['rest_x']], [traj['rest_y']], [traj['rest_z']], 
                   color='#d62728', s=80, marker='*', label='Final Rest')
    
    # 3. Plot 60m Net boundary plane
    angle_rad = np.radians(25.1)
    net_x = np.linspace(pos[0].min()-10, pos[0].max()+10, 10)
    net_z = np.linspace(0, max(pos[2].max()*1.1, 20), 10)
    X_net, Z_net = np.meshgrid(net_x, net_z)
    Y_net = (60.0 - (X_net - row.launch_x) * np.cos(angle_rad)) / np.sin(angle_rad) + row.launch_y
    ax.plot_surface(X_net, Y_net, Z_net, color='red', alpha=0.15)
    
    # 4. Plot observed checkpoints
    cp_x = [row.cp1_x, row.cp2_x, row.cp3_x, row.cp4_x]
    cp_y = [row.cp1_y, row.cp2_y, row.cp3_y, row.cp4_y]
    cp_z = [row.cp1_z, row.cp2_z, row.cp3_z, row.cp4_z]
    
    ax.scatter([row.launch_x], [row.launch_y], [row.launch_z], color='black', s=90, label='Launch (Tee)', marker='s')
    ax.scatter(cp_x, cp_y, cp_z, color='red', s=60, label='Radar Checkpoints (15-60m)', marker='o')
    
    # 5. Plot true apex and landing if available
    if 'apex_x' in row:
        ax.scatter([row.apex_x], [row.apex_y], [row.apex_z], color='green', s=100, label='True Apex', marker='^')
        ax.scatter([row.landing_x], [row.landing_y], [row.landing_z], color='purple', s=100, label='True Landing', marker='X')
        
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height Z (m)')
    ax.set_title(f'Shot Trajectory {str(row.track_id)[:8]} (Cd={cd:.3f}, Cl0={cl0:.3f})', fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    out_path = ROOT_DIR / f'shot_{idx}_traj.png'
    if save_fig:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.close()

if __name__ == '__main__':
    train_df = pd.read_csv(ROOT_DIR / 'train.csv')
    plot_shot_trajectory(train_df, 0)
    plot_shot_trajectory(train_df, 50)
