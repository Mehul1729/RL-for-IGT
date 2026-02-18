# PFC–Limbic Dual-System RL for Human Decision Modeling (IGT)

This repository explores how reinforcement learning can be extended from solving artificial tasks to modeling real human decision-making under risk and uncertainty using the Iowa Gambling Task (IGT).

It combines:

- A policy-gradient RL agent that learns the task  
- A neuroscience-inspired dual-system decision model  
- Likelihood-based fitting to real behavioral datasets  

---

## 🧠 Dual-System Arbitration Model

Analytical long-term planning (Prefrontal Cortex) vs impulsive reward-driven behavior (Limbic System).

![PFC vs Limbic Arbitration](https://github.com/Mehul1729/RL-for-IGT/blob/41b1ed5c9c95108b9ee6ce1d7b68edb92b4c6525/Decision-model/Plots/summary.png)

---

## 🎴 Iowa Gambling Task Environment

Deceptive reward structure with short-term gains vs long-term outcomes.

![IGT Payoff Scheme](https://github.com/Mehul1729/RL-for-IGT/blob/41b1ed5c9c95108b9ee6ce1d7b68edb92b4c6525/Decision-model/Plots/Payoff-scheme-for-the-Iowa-Gambling-Task-Drawing-frequently-from-deck-A-or-deck-B.png)

---

## 📊 Behavioral Model Fitting Results

Estimated parameters across subject groups (Control vs Substance Use).

### Loss Aversion

![Loss Aversion](https://github.com/Mehul1729/RL-for-IGT/blob/41b1ed5c9c95108b9ee6ce1d7b68edb92b4c6525/Decision-model/Plots/loss%20aversion.jpg)

### Memory Decay (Somatic Marker)

![Memory Decay](https://github.com/Mehul1729/RL-for-IGT/blob/41b1ed5c9c95108b9ee6ce1d7b68edb92b4c6525/Decision-model/Plots/memory%20decay.jpg)

---

## 📂 Project Structure

- **RL-Agent-for-IGT/** → Artificial agent learning the task  
- **Decision-model/** → Dual-system human decision model  

Detailed documentation is available inside each subproject.

---

## 🎯 Goal

Bridge the gap between artificial reinforcement learning agents and real human decision behavior.

