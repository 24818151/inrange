import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.insert(0, r'C:\GitHub\Mode-Matching\Python')

from inrange.physics.ode_fitting import fit_shot_parameters
from inrange.physics.trajectory import integrate_trajectory

def plot_shot_trajectory(df, idx=0):
    row = df.iloc[idx]
    cd, cl0, _ = fit_shot_parameters(row)
    
    y0 = [row.launch_x, row.launch_y, row.launch_z, row.launch_vx, row.launch_vy, row.launch_vz]
    traj = integrate_trajectory(y0, cd, cl0)
    
    sol = traj['sol']
    t_eval = np.linspace(0, traj['landing_t'], 200)
    pos = sol.sol(t_eval)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ODE continuous trajectory
    ax.plot(pos[0], pos[1], pos[2], label='ODE Flight Path', color='blue', alpha=0.7)
    
    # Plot net boundary plane roughly at 60m downrange
    # Using the fixed bearing angle 25.1 deg
    # distance = dx*cos(th) + dy*sin(th) = 60
    # dy = (60 - dx*cos(th)) / sin(th)
    
    # Plot observed checkpoints
    cp_x = [row.cp1_x, row.cp2_x, row.cp3_x, row.cp4_x]
    cp_y = [row.cp1_y, row.cp2_y, row.cp3_y, row.cp4_y]
    cp_z = [row.cp1_z, row.cp2_z, row.cp3_z, row.cp4_z]
    
    ax.scatter([row.launch_x], [row.launch_y], [row.launch_z], color='black', s=100, label='Launch', marker='s')
    ax.scatter(cp_x, cp_y, cp_z, color='red', s=50, label='Observed Checkpoints')
    
    # Plot true apex and landing if available (from train.csv)
    if 'apex_x' in row:
        ax.scatter([row.apex_x], [row.apex_y], [row.apex_z], color='green', s=100, label='True Apex', marker='^')
        ax.scatter([row.landing_x], [row.landing_y], [row.landing_z], color='purple', s=100, label='True Landing', marker='X')
        
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Shot Trajectory {row.track_id[:8]}... (Cd={cd:.3f}, Cl0={cl0:.3f})')
    ax.legend()
    
    out_path = f'C:\\GitHub\\Mode-Matching\\Python\\inrange\\shot_{idx}_traj.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    
if __name__ == '__main__':
    train_df = pd.read_csv(r'C:\GitHub\Mode-Matching\Python\inrange\train.csv')
    # Plot a few distinct shots
    plot_shot_trajectory(train_df, 0)
    plot_shot_trajectory(train_df, 50)
