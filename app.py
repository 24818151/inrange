import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# Ensure local modules can be imported
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from physics.trajectory import integrate_trajectory
from physics.ode_fitting import fit_shot_parameters

st.set_page_config(page_title="Inrange Golf Physics Simulator", layout="wide")

st.title("⛳ Inrange Golf: Physics-Informed 3D Trajectory & Bounce Simulator")
st.markdown("""
Interactive physics simulator for the **Inrange Student Competition** (Kaggle). 
This tool models 3D aerodynamic flight (drag, Magnus backspin lift, ambient wind) up to the 60 m net, 
and extrapolates full flight to apex, level landing, and post-impact bounce & roll.
""")

# Load dataset for inspection if available
train_csv_path = APP_DIR / 'train.csv'
test_csv_path = APP_DIR / 'test.csv'

mode = st.sidebar.radio("Mode", ["Inspect Dataset Shot", "Manual Sandbox"])

checkpoints = None
true_apex = None
true_landing = None

if mode == "Inspect Dataset Shot" and train_csv_path.exists():
    st.sidebar.subheader("Select Shot")
    dataset_choice = st.sidebar.selectbox("Dataset", ["Training Set (Known Targets)", "Test Set (Unseen Targets)"])
    active_df = pd.read_csv(train_csv_path) if "Training" in dataset_choice else pd.read_csv(test_csv_path)
    
    shot_idx = st.sidebar.number_input("Shot Index", min_value=0, max_value=len(active_df)-1, value=0, step=1)
    row = active_df.iloc[shot_idx]
    st.sidebar.caption(f"Track ID: `{str(row['track_id'])[:12]}...`")
    
    # Auto-fit Cd and Cl0
    with st.spinner("Fitting aerodynamics from checkpoints..."):
        cd, cl0, mse = fit_shot_parameters(row)
    
    st.sidebar.markdown(f"**Fitted Drag ($C_d$):** `{cd:.3f}`")
    st.sidebar.markdown(f"**Fitted Lift ($C_{{l0}}$):** `{cl0:.3f}`")
    st.sidebar.markdown(f"**Checkpoint Fit MSE:** `{mse:.4f} m²`")
    
    launch_x = float(row['launch_x'])
    launch_y = float(row['launch_y'])
    launch_z = float(row['launch_z'])
    launch_vx = float(row['launch_vx'])
    launch_vy = float(row['launch_vy'])
    launch_vz = float(row['launch_vz'])
    
    wx = st.sidebar.slider("Ambient Wind X (m/s)", -15.0, 15.0, 0.0)
    wy = st.sidebar.slider("Ambient Wind Y (m/s)", -15.0, 15.0, 0.0)
    
    checkpoints = {
        'x': [row['cp1_x'], row['cp2_x'], row['cp3_x'], row['cp4_x']],
        'y': [row['cp1_y'], row['cp2_y'], row['cp3_y'], row['cp4_y']],
        'z': [row['cp1_z'], row['cp2_z'], row['cp3_z'], row['cp4_z']]
    }
    
    if 'apex_x' in row:
        true_apex = (row['apex_x'], row['apex_y'], row['apex_z'], row['apex_t'])
        true_landing = (row['landing_x'], row['landing_y'], row['landing_z'], row['landing_t'])

else:
    st.sidebar.subheader("Launch Conditions")
    launch_x = 0.0
    launch_y = 0.0
    launch_z = st.sidebar.slider("Bay Height (m)", 0.0, 5.0, 0.06)
    launch_vx = st.sidebar.slider("Launch Velocity X (m/s)", 30.0, 80.0, 55.0)
    launch_vy = st.sidebar.slider("Launch Velocity Y (m/s)", -20.0, 20.0, 5.0)
    launch_vz = st.sidebar.slider("Launch Velocity Z (m/s)", 5.0, 40.0, 15.0)
    
    st.sidebar.subheader("Aerodynamics")
    cd = st.sidebar.slider("Drag Coefficient (Cd)", 0.15, 0.50, 0.28)
    cl0 = st.sidebar.slider("Lift Coefficient (Cl0 - Backspin)", 0.0, 0.40, 0.22)
    
    st.sidebar.subheader("Environment")
    wx = st.sidebar.slider("Wind X (m/s) [Tail/Head]", -15.0, 15.0, 0.0)
    wy = st.sidebar.slider("Wind Y (m/s) [Crosswind]", -15.0, 15.0, 0.0)

# Simulate 3D flight and bounce & roll
y0 = [launch_x, launch_y, launch_z, launch_vx, launch_vy, launch_vz]
traj = integrate_trajectory(y0, cd, cl0, wx=wx, wy=wy, simulate_bounce=True)

# Continuous airborne flight
sol = traj['sol']
t_flight = np.linspace(0, traj['landing_t'], 200)
flight_pos = sol.sol(t_flight)
fx, fy, fz = flight_pos[0], flight_pos[1], flight_pos[2]

# Bounce & roll
bx, by, bz = traj['bounce_traj'][1:] if 'bounce_traj' in traj else ([], [], [])

# Plotly 3D Figure
fig = go.Figure()

# 1. Flight Trace
fig.add_trace(go.Scatter3d(
    x=fx, y=fy, z=fz,
    mode='lines',
    name='Airborne Trajectory (ODE)',
    line=dict(color='#1f77b4', width=5)
))

# 2. Bounce & Roll Trace
if len(bx) > 0:
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode='lines',
        name='Post-Impact Bounce & Roll',
        line=dict(color='#ff7f0e', width=4, dash='dash')
    ))

# 3. Launch marker
fig.add_trace(go.Scatter3d(
    x=[launch_x], y=[launch_y], z=[launch_z],
    mode='markers',
    name='Launch Tee',
    marker=dict(size=7, color='black', symbol='square')
))

# 4. Checkpoints (if dataset mode)
if checkpoints is not None:
    fig.add_trace(go.Scatter3d(
        x=checkpoints['x'], y=checkpoints['y'], z=checkpoints['z'],
        mode='markers',
        name='Radar Checkpoints (15-60m)',
        marker=dict(size=6, color='red', symbol='circle')
    ))

# 5. True Targets (if training set)
if true_apex is not None:
    fig.add_trace(go.Scatter3d(
        x=[true_apex[0]], y=[true_apex[1]], z=[true_apex[2]],
        mode='markers',
        name='True Apex',
        marker=dict(size=8, color='green', symbol='diamond')
    ))
    fig.add_trace(go.Scatter3d(
        x=[true_landing[0]], y=[true_landing[1]], z=[true_landing[2]],
        mode='markers',
        name='True Landing',
        marker=dict(size=8, color='purple', symbol='x')
    ))

# 6. 60m Net plane (visual reference)
angle_rad = np.radians(25.1)
net_x = np.linspace(fx.min()-10, fx.max()+10, 10)
net_z = np.linspace(0, max(fz.max()*1.1, 25), 10)
X_net, Z_net = np.meshgrid(net_x, net_z)
Y_net = (60.0 - (X_net - launch_x) * np.cos(angle_rad)) / np.sin(angle_rad) + launch_y

fig.add_trace(go.Surface(
    x=X_net, y=Y_net, z=Z_net,
    opacity=0.2, colorscale='Reds', showscale=False,
    name='60m Net Plane'
))

# Layout configurations
fig.update_layout(
    scene=dict(
        aspectmode='data',
        xaxis_title='X Downrange (m)',
        yaxis_title='Y Lateral (m)',
        zaxis_title='Height Z (m)'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05)
)

st.plotly_chart(fig, use_container_width=True, height=650)

# Metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Apex Height", f"{traj['apex_z']:.1f} m", f"at t={traj['apex_t']:.2f}s")
carry_dist = np.sqrt((traj['landing_x'] - launch_x)**2 + (traj['landing_y'] - launch_y)**2)
col2.metric("Predicted Carry (Landing)", f"{carry_dist:.1f} m", f"at t={traj['landing_t']:.2f}s")

if 'rest_x' in traj:
    total_dist = np.sqrt((traj['rest_x'] - launch_x)**2 + (traj['rest_y'] - launch_y)**2)
    roll_dist = total_dist - carry_dist
    col3.metric("Post-Landing Roll", f"{roll_dist:.1f} m", f"stopped at t={traj['rest_t']:.2f}s")
    col4.metric("Total Shot Distance", f"{total_dist:.1f} m")
