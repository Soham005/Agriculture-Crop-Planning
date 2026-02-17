import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("🌾 Agriculture Crop Planning - MDP Simulation")

st.sidebar.header("Climate Settings")

# Rainfall probabilities
high = st.sidebar.slider("High Rainfall Probability", 0.0, 1.0, 0.3)
moderate = st.sidebar.slider("Moderate Rainfall Probability", 0.0, 1.0, 0.4)
low = st.sidebar.slider("Low Rainfall Probability", 0.0, 1.0, 0.3)

# Normalize probabilities
total = high + moderate + low
high, moderate, low = high/total, moderate/total, low/total

gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 0.9)

st.write("### Rainfall Distribution")
st.write(f"High: {high:.2f}, Moderate: {moderate:.2f}, Low: {low:.2f}")

# States
states = ["Fertile", "Degraded"]
actions = ["Rice", "Wheat", "Millets"]

# Crop Data
yield_good = {"Rice": 60, "Wheat": 50, "Millets": 40}
yield_bad = {"Rice": 20, "Wheat": 30, "Millets": 35}
cost = {"Rice": 30, "Wheat": 25, "Millets": 20}

def expected_reward(action):
    good_rain = high + moderate
    bad_rain = low
    return (good_rain * yield_good[action] +
            bad_rain * yield_bad[action]) - cost[action]

def transition_prob(state, action):
    if action == "Rice":
        return 0.7 if state == "Fertile" else 0.4
    elif action == "Wheat":
        return 0.8 if state == "Fertile" else 0.5
    else:  # Millets
        return 0.9 if state == "Fertile" else 0.7

# Value Iteration
V = {s: 0 for s in states}
policy = {s: None for s in states}

for _ in range(100):
    new_V = {}
    for s in states:
        action_values = {}
        for a in actions:
            reward = expected_reward(a)
            prob_fertile = transition_prob(s, a)
            prob_degraded = 1 - prob_fertile
            future_value = (prob_fertile * V["Fertile"] +
                            prob_degraded * V["Degraded"])
            action_values[a] = reward + gamma * future_value
        
        best_action = max(action_values, key=action_values.get)
        new_V[s] = action_values[best_action]
        policy[s] = best_action
    
    V = new_V

st.write("## 🔍 Optimal Policy")
policy_df = pd.DataFrame.from_dict(policy, orient='index', columns=['Optimal Crop'])
st.table(policy_df)

st.write("## 📈 State Value Function")
value_df = pd.DataFrame.from_dict(V, orient='index', columns=['Value'])
st.table(value_df)

# Visualization
st.write("## 📊 Value Comparison")

fig, ax = plt.subplots()
ax.bar(V.keys(), V.values())
ax.set_ylabel("State Value")
ax.set_title("Optimal State Values")
st.pyplot(fig)
