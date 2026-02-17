import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Planning MDP", layout="wide")
st.title("🌾 Agriculture Crop Planning - Climate Sensitive MDP")

# ==========================================
# Sidebar Controls
# ==========================================

st.sidebar.header("🌧 Climate Settings")

high = st.sidebar.slider("High Rainfall Probability", 0.0, 1.0, 0.3)
moderate = st.sidebar.slider("Moderate Rainfall Probability", 0.0, 1.0, 0.4)
low = st.sidebar.slider("Low Rainfall Probability", 0.0, 1.0, 0.3)

# Normalize
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
# Crop Parameters (Balanced)
# ==========================================

yield_good = {"Rice": 60, "Wheat": 55, "Millets": 40}
yield_bad = {"Rice": 15, "Wheat": 30, "Millets": 35}
cost = {"Rice": 30, "Wheat": 25, "Millets": 20}

soil_penalty = 0.8  # degraded soil reduces yield

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

    reward = (good_rain * good_yield +
              bad_rain * bad_yield) - cost[action]

    return reward

# ==========================================
# Transition Function (Better Balanced)
# ==========================================

def transition_probability(state, action):
    rainfall_factor = 0.5 * high + 0.3 * moderate + 0.2 * low

    if action == "Rice":
        if state == "Fertile":
            return 0.5 * rainfall_factor
        else:
            return 0.2 * rainfall_factor

    elif action == "Wheat":
        if state == "Fertile":
            return 0.8 * rainfall_factor
        else:
            return 0.5 * rainfall_factor

    elif action == "Millets":
        if state == "Fertile":
            return 0.9
        else:
            return 0.75

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
# Interpretation Logic
# ==========================================

st.subheader("🧠 Interpretation")

if low > 0.5:
    st.write("Severe drought risk detected → Resilient crops (Millets) dominate.")

elif high > 0.6:
    st.write("Stable high rainfall → High-yield crops (Rice) are optimal.")

elif 0.2 < low < 0.4:
    st.write("Moderate climate variability → Balanced crops (Wheat) become optimal.")

else:
    st.write("Climate conditions produce mixed strategic outcomes.")
