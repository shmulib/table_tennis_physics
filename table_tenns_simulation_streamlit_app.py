import sympy as sp
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Define symbols
t = sp.symbols('t')
g, gamma, S, omega0_sym, m = sp.symbols('g gamma S omega0 m', positive=True)
alpha = S * omega0_sym / m

# Define functions
vx = sp.Function('vx')(t)
vz = sp.Function('vz')(t)

# Helper to integrate and apply initial condition
def integrate_and_solve(v_expr, initial_val):
    C = sp.symbols('C')
    integral_expr = sp.integrate(v_expr, t) + C
    solved_C = sp.solve(integral_expr.subs(t, 0) - initial_val, C)[0]
    return integral_expr.subs(C, solved_C)

# Solve system for each case
def solve_case(ode_vx, ode_vz, vx0_val, vz0_val):
    sol_v = sp.dsolve([ode_vx, ode_vz], ics={vx.subs(t, 0): vx0_val, vz.subs(t, 0): vz0_val})
    vx_sol = sol_v[0].rhs
    vz_sol = sol_v[1].rhs
    x_sol = integrate_and_solve(vx_sol, 0)
    z_sol = integrate_and_solve(vz_sol, 1)
    return x_sol, z_sol

# Streamlit app

st.set_page_config(page_title="Projectile Trajectories with Gravity, Drag, and Magnus", layout="wide")

# Sidebar controls
initial_speed = st.sidebar.slider("Initial Speed (m/s)", min_value=5.0, max_value=30.0, value=13.05, step=0.05)
launch_angle = st.sidebar.slider("Launch Angle (degrees)", min_value=0, max_value=90, value=20, step=1)
initial_omega = st.sidebar.slider("Initial Angular Velocity ω₀ (rad/s)", min_value=0.0, max_value=500.0, value=300.0, step=1.0)

# Convert initial conditions
vx0_val = initial_speed * np.cos(np.radians(launch_angle))
vz0_val = initial_speed * np.sin(np.radians(launch_angle))

# Define constants (except omega0)
subs_vals = {g: 9.81, gamma: 1.3333, S: 2e-6, omega0_sym: initial_omega, m: 0.0027}

# Solve symbolic equations and substitute constants
def prepare_trajectory(x_expr, z_expr):
    x_sub = x_expr.subs(subs_vals)
    z_sub = z_expr.subs(subs_vals)
    return sp.lambdify(t, x_sub, 'numpy'), sp.lambdify(t, z_sub, 'numpy')

# 1. Gravity Only
x_grav, z_grav = solve_case(sp.Eq(vx.diff(t), 0), sp.Eq(vz.diff(t), -g), vx0_val, vz0_val)

# 2. Gravity + Linear Drag
x_drag, z_drag = solve_case(sp.Eq(vx.diff(t), -gamma * vx), sp.Eq(vz.diff(t), -gamma * vz - g), vx0_val, vz0_val)

# 3. Gravity + Drag + Magnus
x_magnus, z_magnus = solve_case(sp.Eq(vx.diff(t), -gamma * vx + alpha * vz), sp.Eq(vz.diff(t), -gamma * vz - alpha * vx - g), vx0_val, vz0_val)

# Prepare functions for plotting
xg_func, zg_func = prepare_trajectory(x_grav, z_grav)
xd_func, zd_func = prepare_trajectory(x_drag, z_drag)
xm_func, zm_func = prepare_trajectory(x_magnus, z_magnus)

# Function to compute last positive z crossing
def compute_t_max(z_func):
    t_sample = np.linspace(0, 5, 1000)
    z_vals = z_func(t_sample)
    idx = np.where(z_vals >= 0)[0]
    return t_sample[idx[-1]] if len(idx) > 0 else 0.1

# Compute maximum flight time among models
t_max_g = compute_t_max(zg_func)
t_max_d = compute_t_max(zd_func)
t_max_m = compute_t_max(zm_func)
t_max = max(t_max_g, t_max_d, t_max_m)

# Time range for plotting
n_frames = 100
t_vals = np.linspace(0, t_max, n_frames)

# Function to cut off trajectory at z = 0
def cutoff_trajectory(x_func, z_func, t_vals):
    x_vals = x_func(t_vals)
    z_vals = z_func(t_vals)
    mask = z_vals >= 0
    return x_vals[mask], z_vals[mask]

# Cutoff trajectories for plotting
xg_vals, zg_vals = cutoff_trajectory(xg_func, zg_func, t_vals)
xd_vals, zd_vals = cutoff_trajectory(xd_func, zd_func, t_vals)
xm_vals, zm_vals = cutoff_trajectory(xm_func, zm_func, t_vals)

# Create figure with static trajectories and initial markers
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=xg_vals, y=[0]*len(xg_vals), z=zg_vals, mode='lines',  name='Gravity Only'))
fig.add_trace(go.Scatter3d(x=xd_vals, y=[0]*len(xd_vals), z=zd_vals, mode='lines', name='Gravity + Drag'))
fig.add_trace(go.Scatter3d(x=xm_vals, y=[0]*len(xm_vals), z=zm_vals, mode='lines', name='Gravity + Drag + Magnus'))
fig.add_trace(go.Scatter3d(x=[xg_vals[0]], y=[0], z=[zg_vals[0]], mode='markers', marker=dict(size=5), name='Gravity Only Marker', showlegend=False))
fig.add_trace(go.Scatter3d(x=[xd_vals[0]], y=[0], z=[zd_vals[0]], mode='markers', marker=dict(size=5), name='Drag Only Marker', showlegend=False))
fig.add_trace(go.Scatter3d(x=[xm_vals[0]], y=[0], z=[zm_vals[0]], mode='markers', marker=dict(size=5), name='Magnus Marker', showlegend=False))
fig.add_trace(go.Scatter3d(x=[0,0], y=[-10,10], z=[0.0], mode='markers',marker=dict(size=0.1), name='For plotting purposes Only', showlegend=False))
# Create frames with updated marker positions only
def marker_position(x_func, z_func, t_frame, t_max_model):
    if t_frame <= t_max_model:
        return x_func(t_frame), z_func(t_frame)
    else:
        return x_func(t_max_model), 0

frames = []
for i, t_frame in enumerate(t_vals):
    xg_pos, zg_pos = marker_position(xg_func, zg_func, t_frame, t_max_g)
    xd_pos, zd_pos = marker_position(xd_func, zd_func, t_frame, t_max_d)
    xm_pos, zm_pos = marker_position(xm_func, zm_func, t_frame, t_max_m)
    frames.append(go.Frame(name=str(i),
        data=[    
            go.Scatter3d(x=[xg_pos], y=[0], z=[zg_pos], mode='markers', marker=dict(size=5), showlegend=False),
            go.Scatter3d(x=[xd_pos], y=[0], z=[zd_pos], mode='markers', marker=dict(size=5), showlegend=False),
            go.Scatter3d(x=[xm_pos], y=[0], z=[zm_pos], mode='markers', marker=dict(size=5), showlegend=False)
        ],
        traces=[ 3, 4, 5]))


def camera_pf(initial_speed):
    return (1 + 1*(initial_speed-5)/(30-5))




fig.frames = frames

# Layout with animation slider and buttons
fig.update_layout(
    title='Projectile Trajectories with Moving Markers (3D)',
    scene=dict(xaxis_title='x (m)',
               yaxis_title='y',
                zaxis_title='z (m)',
                aspectratio =dict(x=1, y=1, z=1),
                aspectmode="data",
                xaxis_range=[0, xg_vals.max()+1],
                yaxis_range=[-10, 10],
                zaxis_range=[0, zg_vals.max()+1],
                camera_eye=dict(x=1*camera_pf(initial_speed), y=1*camera_pf(initial_speed), z=0.2), 
               ),
    updatemenus=[dict(type='buttons', showactive=False,
                      buttons=[dict(label='Play', method='animate',
                                    args=[None, dict(frame=dict(duration=t_max/n_frames, redraw=True), fromcurrent=True)])])],
    sliders=[dict(steps=[dict(method='animate',
                              args=[[str(i)], dict(mode='immediate',
                                                   frame=dict(duration=0.0000001, redraw=True),
                                                   transition=dict(duration=0))],
                              label=str(round(t_vals[i], 2)))
                         for i in range(len(t_vals))],
                  active=0)],
    height=650,
    width =1000,
)
# fig.update_layout(
#     paper_bgcolor='white',
#     plot_bgcolor='white',
#     scene=dict(
#         xaxis=dict(backgroundcolor='white', gridcolor='lightgrey', zerolinecolor='lightgrey'),
#         yaxis=dict(backgroundcolor='white', gridcolor='lightgrey', zerolinecolor='lightgrey'),
#         zaxis=dict(backgroundcolor='white', gridcolor='lightgrey', zerolinecolor='lightgrey')
#     )
# )
st.plotly_chart(fig, use_container_width=False)

# import numpy as np
# import plotly.graph_objects as go
# import streamlit as st

# # Define symbols
# t = sp.symbols('t')
# g, gamma, S, omega0_sym, m = sp.symbols('g gamma S omega0 m', positive=True)
# alpha = S * omega0_sym / m

# # Define functions
# vx = sp.Function('vx')(t)
# vz = sp.Function('vz')(t)

# # Helper to integrate and apply initial condition
# def integrate_and_solve(v_expr, initial_val):
#     C = sp.symbols('C')
#     integral_expr = sp.integrate(v_expr, t) + C
#     solved_C = sp.solve(integral_expr.subs(t, 0) - initial_val, C)[0]
#     return integral_expr.subs(C, solved_C)

# # Solve system for each case
# def solve_case(ode_vx, ode_vz, vx0_val, vz0_val):
#     sol_v = sp.dsolve([ode_vx, ode_vz], ics={vx.subs(t, 0): vx0_val, vz.subs(t, 0): vz0_val})
#     vx_sol = sol_v[0].rhs
#     vz_sol = sol_v[1].rhs
#     x_sol = integrate_and_solve(vx_sol, 0)
#     z_sol = integrate_and_solve(vz_sol, 1)
#     return x_sol, z_sol

# # Streamlit app
# st.title("Projectile Trajectories with Gravity, Drag, and Magnus")

# # Sidebar controls
# initial_speed = st.sidebar.slider("Initial Speed (m/s)", min_value=5.0, max_value=30.0, value=13.05, step=0.05)
# launch_angle = st.sidebar.slider("Launch Angle (degrees)", min_value=0, max_value=90, value=20, step=1)
# initial_omega = st.sidebar.slider("Initial Angular Velocity ω₀ (rad/s)", min_value=0.0, max_value=500.0, value=300.0, step=1.0)

# # Convert initial conditions
# vx0_val = initial_speed * np.cos(np.radians(launch_angle))
# vz0_val = initial_speed * np.sin(np.radians(launch_angle))

# # Define constants (except omega0)
# subs_vals = {g: 9.81, gamma: 1.3333, S: 2e-6, omega0_sym: initial_omega, m: 0.0027}

# # Solve symbolic equations and substitute constants
# def prepare_trajectory(x_expr, z_expr):
#     x_sub = x_expr.subs(subs_vals)
#     z_sub = z_expr.subs(subs_vals)
#     return sp.lambdify(t, x_sub, 'numpy'), sp.lambdify(t, z_sub, 'numpy')

# # 1. Gravity Only
# x_grav, z_grav = solve_case(sp.Eq(vx.diff(t), 0), sp.Eq(vz.diff(t), -g), vx0_val, vz0_val)

# # 2. Gravity + Linear Drag
# x_drag, z_drag = solve_case(sp.Eq(vx.diff(t), -gamma * vx), sp.Eq(vz.diff(t), -gamma * vz - g), vx0_val, vz0_val)

# # 3. Gravity + Drag + Magnus
# x_magnus, z_magnus = solve_case(sp.Eq(vx.diff(t), -gamma * vx + alpha * vz), sp.Eq(vz.diff(t), -gamma * vz - alpha * vx - g), vx0_val, vz0_val)

# # Prepare functions for plotting
# xg_func, zg_func = prepare_trajectory(x_grav, z_grav)
# xd_func, zd_func = prepare_trajectory(x_drag, z_drag)
# xm_func, zm_func = prepare_trajectory(x_magnus, z_magnus)

# # Function to cut off trajectory at z = 0
# def cutoff_trajectory(x_func, z_func, t_vals):
#     x_vals = x_func(t_vals)
#     z_vals = z_func(t_vals)
#     mask = z_vals >= 0
#     return x_vals[mask], z_vals[mask]

# # Time range for plotting
# t_vals = np.linspace(0, 2, 500)

# # Cutoff trajectories for plotting
# xg_vals, zg_vals = cutoff_trajectory(xg_func, zg_func, t_vals)
# xd_vals, zd_vals = cutoff_trajectory(xd_func, zd_func, t_vals)
# xm_vals, zm_vals = cutoff_trajectory(xm_func, zm_func, t_vals)

# # Create Plotly figure with animation frames
# fig = go.Figure()

# # Add static trajectory lines
# fig.add_trace(go.Scatter3d(x=xg_vals, y=[0]*len(xg_vals), z=zg_vals, mode='lines', name='Gravity Only'))
# fig.add_trace(go.Scatter3d(x=xd_vals, y=[0]*len(xd_vals), z=zd_vals, mode='lines', name='Gravity + Drag'))
# fig.add_trace(go.Scatter3d(x=xm_vals, y=[0]*len(xm_vals), z=zm_vals, mode='lines', name='Gravity + Drag + Magnus'))

# # Create frames for animation
# frames = []
# for t_frame in t_vals:
#     frames.append(go.Frame(data=[
#         go.Scatter3d(x=[xg_pos], y=[0], z=[zg_pos], mode='markers', marker=dict(size=5), name='Gravity Only'),
#         go.Scatter3d(x=[xd_pos], y=[0], z=[zd_pos], mode='markers', marker=dict(size=5), name='Gravity + Drag'),
#         go.Scatter3d(x=[xm_pos], y=[0], z=[zm_pos], mode='markers', marker=dict(size=5), name='Gravity + Drag + Magnus')
#     ], name=str(np.round(t_frame, 2))))

# fig.frames = frames

# # Update layout with sliders
# fig.update_layout(
#     title='Projectile Trajectories in 3D (x-z plane) with Animation',
#     scene=dict(xaxis_title='x (m)',
#                yaxis_title='y (dummy axis)',
#                zaxis_title='z (m)'),
#     updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])],
#     sliders=[dict(steps=[dict(method='animate', args=[[str(np.round(t_val, 2))], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(np.round(t_val, 2))) for t_val in t_vals], active=0)]
# )

# st.plotly_chart(fig)
