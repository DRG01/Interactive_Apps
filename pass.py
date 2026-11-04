import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PASS Autonomous Evasive Maneuver Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛞 PASS: Preference-Based Autonomous Safety System")
st.markdown("""
This simulator instantiates a **Weak Rank-Dependent Utility (WRDU)** risk control system. 
The Autonomous Vehicle (AV) balances trajectory tracking with **forbidden set evasion** under high control authority 
and expanded early-warning safety buffers to dodge oncoming human drivers.
""")

# --- SIDEBAR / PARAMETER TUNING ---
st.sidebar.header("🕹️ Simulation Parameters")

# User Requested Modifications
st.sidebar.subheader("🚨 Control Authority & Safety Buffer")
av_max_lat_accel = st.sidebar.slider("AV Max Lateral Acceleration (Control Authority)", 1.0, 5.0, 3.5, 0.1)
safety_buffer_radius = st.sidebar.slider("Forbidden Set (T) Safety Buffer Radius", 0.5, 2.5, 1.5, 0.1)

st.sidebar.subheader("🧠 WRDU Risk Preference Engine")
lambda_0 = st.sidebar.slider("Fixed Benchmark Index (λ₀)", 1.0, 5.0, 2.15, 0.05)
kappa_g = st.sidebar.slider("Gain Curvature (κ_g - Concave Risk Aversion)", 0.1, 2.0, 1.0, 0.05)
kappa_l = st.sidebar.slider("Loss Curvature (κ_l - Convex Risk Seeking Virtual Domain)", -1.0, 1.0, -0.35, 0.05)

st.sidebar.subheader("🚗 Traffic Environment Settings")
human_aggression = st.sidebar.slider("Human Swerve Variance (Gaussian Noise Shocks)", 0.0, 1.0, 0.4, 0.05)
sim_steps = st.sidebar.slider("Simulation Steps", 20, 100, 50, 5)

# --- MATHEMATICAL UTILITY & RISK FORMULATIONS ---
def get_shifted_crra_derivatives(g, l, kg, kl):
    """Computes marginal utilities for the choice-set index estimation."""
    v_g_prime = (1 + g) ** (-kg)
    v_l_prime = (1 + l) ** (-kl)
    return v_g_prime, v_l_prime

# --- ENVIRONMENT & TRAJECTORY GENERATION ENGINE ---
@st.cache_data(ttl=60)
def run_simulation(av_max_accel, buffer_radius, l0, kg, kl, h_noise, steps):
    # Fixed Geometry
    lane_width = 3.6
    dt = 0.1
    
    # State Initializations
    av_y = 0.0  # Centerline
    av_v_y = 0.0
    
    human_x = 80.0
    human_y = 0.5  # Slightly off-center oncoming vehicle
    human_v_x = -25.0 # Oncoming speed m/s
    
    # Trackers for analytics
    records = []
    collision_detected = False
    fallback_active = False

    for t in range(steps):
        # Update longitudinal vehicle tracking
        human_x += human_v_x * dt
        current_distance = float(human_x)
        
        # Human driver unpredictable lateral swerve (Dynamic Disturbance Shocks)
        human_y += np.random.normal(0, h_noise) * dt
        # Constrain human within road boundaries
        human_y = np.clip(human_y, -lane_width, lane_width)
        
        # --- PASS CONTROLLER RULE EXECUTIONS ---
        # Evaluate candidate menus for steering actions (Lateral accelerations)
        candidate_actions = np.linspace(-av_max_accel, av_max_accel, 7)
        best_score = -np.inf
        best_action = 0.0
        
        # Choice Set Geometry Arrays for Adaptive λ Calculation
        gains = []
        losses = []
        
        # 1. Evaluate Geometry & Compute Choice-Set Adaptive Index
        for a in candidate_actions:
            # Predict next state coordinate
            pred_v_y = av_v_y + a * dt
            pred_y = av_y + pred_v_y * dt
            
            # Distance tracking to target reference coordinate (Centerline path = 0)
            tracking_error = np.abs(pred_y)
            
            # Proximity calculation to oncoming vehicle (Forbidden Set boundary)
            lateral_clearance = np.abs(pred_y - human_y) if current_distance < 15.0 else 10.0
            
            # Mapping Relative Clearance Shifts into Virtual Gains & Losses
            if lateral_clearance >= buffer_radius:
                gains.append(lateral_clearance - buffer_radius)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(buffer_radius - lateral_clearance)
        
        # Calculate Choice-Set Derivative Gradients
        mg = np.sum([get_shifted_crra_derivatives(g, 0, kg, kl)[0] for g in gains if g > 0])
        ml = np.sum([get_shifted_crra_derivatives(0, l, kg, kl)[1] for l in losses if l > 0])
        
        # Reconcile Range-Dependent Index Reciprocal Rule
        if mg > 0 and ml > 0:
            lambda_cs = ml / mg
            lambda_t = min(lambda_cs, l0)
        else:
            lambda_t = l0
            
        rho_t = 1.0 / max(lambda_t, 1e-4) # Reciprocal Safety Penalty
        
        # 2. Action Ranking under Safety-Filter Constraint Guardrail
        certified_actions = []
        for a in candidate_actions:
            pred_v_y = av_v_y + a * dt
            pred_y = av_y + pred_v_y * dt
            
            # Strict Safety Envelope Check
            is_certified = True
            if current_distance < 12.0 and np.abs(pred_y - human_y) < buffer_radius:
                is_certified = False  # Encroached into Expanded Safety Buffer Area
                
            if np.abs(pred_y) > lane_width:
                is_certified = False  # Boundary Lane Departure Exceeded
                
            # Compute WRDU Objectives Matrix
            z_t = -0.5 * (pred_y**2)  # Reward tracking precision
            y_t = 1.0 / (max(np.abs(pred_y - human_y), 1e-3)) if current_distance < 15.0 else 0.0
            
            score = z_t - rho_t * (y_t ** 1.35)
            
            if is_certified:
                certified_actions.append((score, a))
                
        # 3. Execution Control Decoupling Loop
        if len(certified_actions) > 0:
            # Certified path exists -> Execute optimized preference mapping
            best_action = max(certified_actions, key=lambda x: x[0])[1]
            fallback_active = False
        else:
            # Trajectory Trap Encountered -> Trigger Maximum worst-case clearance fallback
            fallback_active = True
            clearances = []
            for a in candidate_actions:
                pred_v_y = av_v_y + a * dt
                pred_y = av_y + pred_v_y * dt
                clearances.append((np.abs(pred_y - human_y), a))
            best_action = max(clearances, key=lambda x: x[0])[1]
            
        # Physical Translation Update
        av_v_y += best_action * dt
        av_y += av_v_y * dt
        
        # Collision Evaluation Ground-Truth Guard
        if current_distance <= 2.0 and np.abs(av_y - human_y) < 0.8:
            collision_detected = True
            
        records.append({
            "Step": t,
            "Distance to Human": current_distance,
            "AV Position (y)": av_y,
            "Human Position (y)": human_y,
            "Adaptive Penalty (ρ_t)": rho_t,
            "Fallback Triggered": 1.0 if fallback_active else 0.0,
            "Status": "Collision" if collision_detected else ("Fallback" if fallback_active else "Safe Operational")
        })
        
    return pd.DataFrame(records), collision_detected

# --- EXECUTE SIMULATION ---
df_sim, crashed = run_simulation(
    av_max_lat_accel, safety_buffer_radius, lambda_0, kappa_g, kappa_l, human_aggression, sim_steps
)

# --- ANALYTICS DASHBOARD LAYOUT ---
col1, col2, col3 = st.columns(3)
with col1:
    if crashed:
        st.error("💥 SYSTEM STATUS: COLLISION RECORDED")
    else:
        st.success("✅ SYSTEM STATUS: CERTIFIED TRACKING CLEAN")
with col2:
    max_penalty = df_sim["Adaptive Penalty (ρ_t)"].max()
    st.metric("Peak Reciprocal Penalty (ρ_t)", f"{max_penalty:.2f}")
with col3:
    fallback_count = df_sim["Fallback Triggered"].sum()
    st.metric("Safety Filter Invalidation Count", f"{int(fallback_count)} steps")

st.subheader("📊 Closed-Loop Dynamic Trajectory Analytics")

# 1. Trajectory Spatial Graph Chart
chart_data = pd.melt(df_sim, id_vars=['Distance to Human'], value_vars=['AV Position (y)', 'Human Position (y)'],
                     var_name='Vehicle Type', value_name='Lateral Position (m)')

trajectory_chart = alt.Chart(chart_data).mark_line(strokeWidth=3).encode(
    x=alt.X('Distance to Human:Q', sort='descending', title='Longitudinal Separation Distance (m)'),
    y=alt.Y('Lateral Position (m):Q', scale=alt.Scale(domain=[-4, 4])),
    color='Vehicle Type:N'
).properties(height=400, title="Vehicle Trajectories (AV vs. Oncoming Human Driver)")

st.altair_chart(trajectory_chart, use_container_width=True)

# 2. Risk Metrics Analysis Graph
col_c1, col_c2 = st.columns(2)

with col_c1:
    penalty_chart = alt.Chart(df_sim).mark_area(color='rgba(255, 75, 75, 0.3)', line=True).encode(
        x=alt.X('Step:Q', title='Simulation Step'),
        y=alt.Y('Adaptive Penalty (ρ_t):Q', title='Reciprocal Penalty Weight (ρ_t)')
    ).properties(height=250, title="Adaptive Penalty Spike Profile")
    st.altair_chart(penalty_chart, use_container_width=True)

with col_c2:
    fallback_chart = alt.Chart(df_sim).mark_bar(color='#FFAA00').encode(
        x=alt.X('Step:Q', title='Simulation Step'),
        y=alt.Y('Fallback Triggered:Q', title='Filter Status (1=Active)')
    ).properties(height=250, title="Least-Unsafe Fallback Activation Trace")
    st.altair_chart(fallback_chart, use_container_width=True)

# --- REPRODUCIBILITY TELEMETRY ARCHIVE LOGS ---
st.subheader("📋 Raw Verification Telemetry Stream Logs")
st.dataframe(df_sim, use_container_width=True)
