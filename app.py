import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Planning MDP", layout="wide")

st.title("🌾 Agriculture Crop Planning - Advanced MDP Simulation")

# ==============================
# Sidebar Controls
# ==============================

st.sidebar.header("🌧 Climate Settings")

high = st.sidebar.slider("High Rainfall Probability", 0.0, 1.0, 0.3)
moderate = st.sidebar.slider("Moderate Rainfall Probability", 0.0, 1.0, 0.4)
low = st.sidebar.slider("Low Rainfall Probability", 0.0, 1.0, 0.3)

total = high + moderate + low
high, moderate, low = high/total, moderate/total, low/total

gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 0.99, 0.9)
theta = 0.001  # convergence threshold

st.write("### Rainfall Distribution (Normalized)")
st.write(f"High: {high:.2f}, Moderate: {moderate:.2f}, Low: {low:.2f}")

# ==============================
# States and Actions
# ==============================

states = ["Fertile", "Degraded"]
actions = ["Rice", "Wheat", "Millets"]

# ==============================
# Base Crop Parameters
# ==============================

yield_good = {"Rice": 60, "Wheat": 50, "Millets": 40}
yield_bad = {"Rice": 20, "Wheat": 30, "Millets": 35}
cost = {"Rice": 30, "Wheat": 25, "Millets": 20}

# Soil penalty (Degraded soil reduces yield)
soil_penalty = 0.75

# ==============================
# Reward Function
# ==============================

def expected_reward(state, action):
    good_rain = high + moderate
    bad_rain = low

    good_yield = yield_good[action]
    bad_yield = yield_bad[action]

    # Apply soil penalty if degraded
    if state == "Degraded":
        good_yield *= soil_penalty
        bad_yield *= soil_penalty

    reward = (good_rain * good_yield +
              bad_rain * bad_yield) - cost[action]

    return reward

# ==============================
# Transition Function
# ==============================

def transition_probability(state, action):
    """
    Returns probability of next state being Fertile.
    Degraded probability = 1 - returned value
    """

    # Base rainfall influence
    rainfall_factor = high * 0.6 + moderate * 0.3 + low * 0.1

    if action == "Rice":
        if state == "Fertile":
            return 0.6 * rainfall_factor
        else:
            return 0.3 * rainfall_factor

    elif action == "Wheat":
        if state == "Fertile":
            return 0.75 * rainfall_factor
        else:
            return 0.4 * rainfall_factor

    elif action == "Millets":
        if state == "Fertile":
            return 0.9
        else:
            return 0.7

# ==============================
# Value Iteration
# ==============================

V = {s: 0 for s in states}
policy = {s: None for s in states}

while True:
    delta = 0
    new_V = {}

    for s in states:
        action_values = {}

        for a in actions:
            reward = expected_reward(s, a)

            prob_fertile = transition_probability(s, a)
            prob_degraded = 1 - prob_fertile

            future_value = (prob_fertile * V["Fertile"] +
                            prob_degraded * V["Degraded"])

            action_values[a] = reward + gamma * future_value

        best_action = max(action_values, key=action_values.get)
        best_value = action_values[best_action]

        new_V[s] = best_value
        policy[s] = best_action

        delta = max(delta, abs(best_value - V[s]))

    V = new_V

    if delta < theta:
        break

# ==============================
# Display Results
# ==============================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Optimal Policy")
    policy_df = pd.DataFrame.from_dict(policy, orient="index",
                                       columns=["Optimal Crop"])
    st.table(policy_df)

with col2:
    st.subheader("📈 State Value Function")
    value_df = pd.DataFrame.from_dict(V, orient="index",
                                      columns=["Value"])
    st.table(value_df)

# ==============================
# Visualization
# ==============================

st.subheader("📊 State Value Comparison")

fig, ax = plt.subplots()
ax.bar(V.keys(), V.values())
ax.set_ylabel("Value")
ax.set_title("Optimal State Values")
st.pyplot(fig)

# ==============================
# Interpretation Section
# ==============================

st.subheader("🧠 Interpretation")

if policy["Fertile"] == "Rice" and policy["Degraded"] == "Rice":
    st.write("High rainfall favors water-intensive crops like Rice.")

elif policy["Degraded"] == "Millets":
    st.write("Climate variability pushes the system toward resilient crops like Millets to restore soil health.")

else:
    st.write("Balanced rainfall conditions favor moderate-risk crops like Wheat.")
