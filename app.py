import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Planning MDP", layout="wide")
st.title("🌾 Agriculture Crop Planning - Final Balanced MDP")

# ==========================================
# Sidebar Controls
# ==========================================

st.sidebar.header("🌧 Climate Settings")

high = st.sidebar.slider("High Rainfall Probability", 0.0, 1.0, 0.4)
moderate = st.sidebar.slider("Moderate Rainfall Probability", 0.0, 1.0, 0.4)
low = st.sidebar.slider("Low Rainfall Probability", 0.0, 1.0, 0.2)

total = high + moderate + low
high, moderate, low = high/total, moderate/total, low/total

gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9)
theta = 0.0001

st.write("### Rainfall Distribution (Normalized)")
st.write(f"High: {high:.2f}, Moderate: {moderate:.2f}, Low: {low:.2f}")

# ==========================================
# States and Actions
# ==========================================

states = ["Fertile", "Degraded"]
actions = ["Rice", "Wheat", "Millets"]

# ==========================================
# Crop Economics (Recalibrated)
# ==========================================

yield_good = {"Rice": 75, "Wheat": 55, "Millets": 40}
yield_bad = {"Rice": 5,  "Wheat": 30, "Millets": 35}
cost = {"Rice": 32, "Wheat": 25, "Millets": 20}

soil_penalty = 0.85

# ==========================================
# Reward Function
# ==========================================

def expected_reward(state, action):
    good_rain = high + moderate
    bad_rain = low

    good_yield = yield_good[action]
    bad_yield = yield_bad[action]

    if state == "Degraded":
        good_yield *= soil_penalty
        bad_yield *= soil_penalty

    return (good_rain * good_yield +
            bad_rain * bad_yield) - cost[action]

# ==========================================
# Transition Function (Balanced)
# ==========================================

def transition_probability(state, action):
    rainfall_strength = 0.7*high + 0.2*moderate + 0.1*low

    if action == "Rice":
        if state == "Fertile":
            return 0.75 * rainfall_strength
        else:
            return 0.40 * rainfall_strength

    elif action == "Wheat":
        if state == "Fertile":
            return 0.85 * rainfall_strength
        else:
            return 0.55 * rainfall_strength

    elif action == "Millets":
        if state == "Fertile":
            return 0.95
        else:
            return 0.80

# ==========================================
# Value Iteration
# ==========================================

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

# ==========================================
# Display Results
# ==========================================

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

# ==========================================
# Visualization
# ==========================================

st.subheader("📊 State Value Comparison")

fig, ax = plt.subplots()
ax.bar(V.keys(), V.values())
ax.set_ylabel("Value")
ax.set_title("Optimal State Values")
st.pyplot(fig)

# ==========================================
# Interpretation
# ==========================================

st.subheader("🧠 Interpretation")

if high > 0.6 and low < 0.2:
    st.write("Stable high rainfall → Rice dominates due to superior yield.")
elif low > 0.45:
    st.write("High drought risk → Millets dominate for resilience and soil recovery.")
elif 0.25 < low < 0.45:
    st.write("Moderate uncertainty → Wheat becomes the balanced optimal strategy.")
else:
    st.write("Climate produces mixed strategic outcomes.")
