import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Ensure the local physics modules can be imported when deployed to Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from physics.trajectory import integrate_trajectory

st.set_page_config(page_title="Inrange Golf Physics", layout="wide")

st.title("⛳ Inrange Golf: Physics & Bounce Simulator")
st.markdown("""
This interactive demo showcases the custom 3D aerodynamic ODE and Bounce & Roll mechanics developed for the Kaggle Inrange Student Competition. 
Adjust the launch conditions, aerodynamic coefficients, and ambient wind to see how the trajectory and roll behave!
""")

# Sidebar Controls
st.sidebar.header("Launch Conditions")
launch_vx = st.sidebar.slider("Launch Velocity X (m/s) [Downrange]", 30.0, 80.0, 55.0)
launch_vy = st.sidebar.slider("Launch Velocity Y (m/s) [Lateral]", -20.0, 20.0, 0.0)
launch_vz = st.sidebar.slider("Launch Velocity Z (m/s) [Vertical]", 5.0, 40.0, 15.0)
launch_z = st.sidebar.slider("Bay Height (m)", 0.0, 5.0, 0.0)

st.sidebar.header("Aerodynamics")
cd = st.sidebar.slider("Drag Coefficient (Cd)", 0.15, 0.50, 0.25)
cl0 = st.sidebar.slider("Lift Coefficient (Cl0 - Backspin)", 0.0, 0.40, 0.15)

st.sidebar.header("Environment")
wx = st.sidebar.slider("Wind X (m/s) [Tail/Head]", -15.0, 15.0, 0.0)
wy = st.sidebar.slider("Wind Y (m/s) [Crosswind]", -15.0, 15.0, 0.0)

# Simulate
y0 = [0.0, 0.0, launch_z, launch_vx, launch_vy, launch_vz]
traj = integrate_trajectory(y0, cd, cl0, wx=wx, wy=wy, simulate_bounce=True)

# Extract Continuous Flight Path
sol = traj['sol']
t_flight = np.linspace(0, traj['landing_t'], 200)
flight_pos = sol.sol(t_flight)
fx, fy, fz = flight_pos[0], flight_pos[1], flight_pos[2]

# Extract Bounce & Roll Path
if 'bounce_traj' in traj:
    bt, bx, by, bz = traj['bounce_traj']
else:
    bx, by, bz = [], [], []

# Plotly 3D Figure
fig = go.Figure()

# Flight Trace
fig.add_trace(go.Scatter3d(
    x=fx, y=fy, z=fz,
    mode='lines',
    name='Airborne Flight',
    line=dict(color='blue', width=4)
))

# Bounce & Roll Trace
if len(bx) > 0:
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode='lines',
        name='Bounce & Roll',
        line=dict(color='orange', width=4, dash='solid')
    ))

# 60m Net Boundary (Visual Reference)
# Using the 25.1 degree range bearing for the net
angle_rad = np.radians(25.1)
net_x = np.linspace(-20, 80, 10)
net_z = np.linspace(0, 30, 10)
X_net, Z_net = np.meshgrid(net_x, net_z)
# downrange = x*cos + y*sin = 60 => y = (60 - x*cos)/sin
Y_net = (60.0 - X_net * np.cos(angle_rad)) / np.sin(angle_rad)

fig.add_trace(go.Surface(
    x=X_net, y=Y_net, z=Z_net,
    opacity=0.2, colorscale='Reds', showscale=False,
    name='60m Net'
))

# Layout configurations for proportional 3D scaling
fig.update_layout(
    scene=dict(
        aspectmode='data', # Ensures 1m in X == 1m in Y == 1m in Z
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Height (m)'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
)

st.plotly_chart(fig, use_container_width=True, height=700)

st.markdown(f"**Apex:** {traj['apex_z']:.1f} m high at {traj['apex_t']:.2f} s")
st.markdown(f"**Carry (Landing):** Distance {np.sqrt(traj['landing_x']**2 + traj['landing_y']**2):.1f} m at {traj['landing_t']:.2f} s")
if 'rest_t' in traj:
    st.markdown(f"**Total Distance (Rest):** {np.sqrt(traj['rest_x']**2 + traj['rest_y']**2):.1f} m")
