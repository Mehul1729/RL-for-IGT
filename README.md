###  The Computational Pipeline

```mermaid
graph TD
    A[Clinical IGT Data<br>Choices & Rewards] --> M

    subgraph Dual-System Cognitive Engine
        P[Prefrontal Cortex<br>Goal-Directed / Q-Learning] -->|State-Action Values| M{Arbitration Router}
        L[Limbic System<br>Impulsive / Prospect Theory] -->|Immediate Utility| M
    end

    M -->|Mixture Coefficient β| E(Inverse Estimation)
    E -->|Monte Carlo Sweep| B[Extracted Biomarkers]

    B --> B1(β: PFC Dominance)
    B --> B2(λ: Loss Aversion)
    B --> B3(α: Memory Decay)
    
    style P fill:#e1f5fe,stroke:#311b92,color:#000000,stroke-width:2px
    style L fill:#fce4ec,stroke:#4a148c,color:#000000,stroke-width:2px
    style M fill:#fff3e0,stroke:#e65100,color:#000000,stroke-width:2px
    style B fill:#e8f5e9,stroke:#1b5e20,color:#000000,stroke-width:2px
```

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

### Beta_PFC (Weightage of PFC system in decision making)

![Beta_PFC](https://github.com/Mehul1729/RL-for-IGT/blob/569847de4a98ae29a81f719f56d7be47406c481f/Decision-model/Plots/PFC_control_level.png)

---

## 📂 Project Structure

- **RL-Agent-for-IGT/** → Artificial agent learning the task  
- **Decision-model/** → Dual-system human decision model  

Detailed documentation is available inside each subproject.

---

## 🎯 Goal

Bridge the gap between artificial reinforcement learning agents and real human decision behavior.

