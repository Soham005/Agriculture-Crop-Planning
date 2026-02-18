# 🌾 Climate-Sensitive Crop Planning using Markov Decision Process (MDP)

An interactive **Streamlit-based simulation** that models optimal crop selection under uncertain rainfall conditions using **Markov Decision Process (MDP)** and **Value Iteration**.

This project demonstrates how climate variability and soil dynamics influence long-term agricultural planning decisions.

---

## 🚀 Project Overview

A farmer must decide which crop to plant each season under uncertain rainfall conditions.

This problem is modeled as a **stochastic dynamic optimization problem**:

(S, A, P, R, γ)

Where:

- **S** → Soil states (Fertile, Degraded)
- **A** → Crop choices (Rice, Wheat, Millets)
- **P** → Transition probabilities (soil evolution under rainfall)
- **R** → Expected profit (yield − cost)
- **γ** → Discount factor (long-term planning preference)

The system computes the **optimal policy** using **Value Iteration**.

---

## 🌧 Climate-Sensitive Behavior

The model dynamically adapts crop strategy based on rainfall distribution:

| Climate Condition | Optimal Crop Strategy |
|------------------|----------------------|
| Stable High Rainfall | 🌾 Rice |
| Moderate Uncertainty | 🌾 Wheat |
| Drought-Prone Climate | 🌾 Millets |

This reflects real-world agricultural tradeoffs between:

- Profit maximization  
- Risk management  
- Soil sustainability  

---

## 🧠 Mathematical Formulation

### State Space
S = {Fertile, Degraded}

### Action Space
A = {Rice, Wheat, Millets}

### Reward Function
R(s,a) = E[Yield] − Cost

Yield depends on:
- Rainfall probabilities  
- Crop characteristics  
- Soil condition  

### Transition Function
P(s'|s,a)

Soil fertility evolves depending on:
- Rainfall strength  
- Crop sustainability  

### Bellman Optimality Equation

V(s) = max_a [ R(s,a) + γ Σ P(s'|s,a)V(s') ]

Solved using iterative dynamic programming (Value Iteration).

---

## 💻 Features

- Interactive rainfall probability sliders  
- Adjustable discount factor (γ)  
- Dynamic Value Iteration solver  
- Optimal crop policy output  
- State value visualization  
- Climate interpretation logic  
- Fully deployable on Streamlit Cloud  

---

## 📊 Example Scenarios

### 🌧 High Rainfall (0.7 / 0.2 / 0.1)
→ Rice dominates due to high yield potential.

### 🌤 Moderate Climate (0.3 / 0.5 / 0.2)
→ Wheat emerges as balanced risk-return crop.

### 🌵 Drought (0.15 / 0.2 / 0.65)
→ Millets dominate due to resilience and soil recovery.

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- NumPy  
- Pandas  
- Matplotlib  
- Dynamic Programming (Value Iteration)  
