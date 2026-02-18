## Modeling Human Decision-Making with a Dual-System RL Model

This project implements a **dual-system reinforcement learning model** inspired by neuroscience to simulate human decision-making in the **Iowa Gambling Task (IGT)** — a psychological experiment used to study risk-taking behavior.

Instead of using a single RL agent, the model combines two competing systems:

- **Limbic System (Impulsive):** Short-term, reward-driven decisions with loss aversion (PVL theory)
- **Prefrontal Cortex (Analytical):** Long-term planning using Q-learning
- **Arbitration Mechanism:** Dynamically balances which system controls decisions

The model was fitted to real behavioral datasets from:

- Healthy control participants  
- Amphetamine users  
- Heroin-dependent participants  

using likelihood-based parameter estimation and Monte Carlo search.

---

## Key Findings from Parameter Estimation

### Impulsivity (Limbic Learning Rate)

![Impulsivity](https://github.com/Mehul1729/RL-for-IGT/blob/c5bdafbf579fa71263b93c4e794bd6533aa60267/Decision-model/Plots/Impulsivity.jpg)

Heroin-dependent participants showed the highest limbic learning rate, suggesting faster adaptation to immediate rewards and more impulsive decision patterns compared to controls.

---

### Loss Aversion

![Loss Aversion](https://github.com/Mehul1729/RL-for-IGT/blob/c5bdafbf579fa71263b93c4e794bd6533aa60267/Decision-model/Plots/loss%20aversion.jpg)

Control participants exhibited stronger sensitivity to losses, while heroin-dependent participants showed reduced loss aversion — consistent with risk-seeking behavior.

---

### Memory Decay (Somatic Marker Fading)

![Memory Decay](https://github.com/Mehul1729/RL-for-IGT/blob/c5bdafbf579fa71263b93c4e794bd6533aa60267/Decision-model/Plots/memory%20decay.jpg)

Both stimulant users and heroin-dependent participants showed higher memory decay, indicating weaker retention of past negative outcomes during decision-making.

---

### System Arbitration (Balance Between Impulsive and Analytical Control)

![Arbitration](https://github.com/Mehul1729/RL-for-IGT/blob/c5bdafbf579fa71263b93c4e794bd6533aa60267/Decision-model/Plots/Dual%20system%20arbitration.jpg)

The balance between analytical (PFC) and impulsive (limbic) systems was lower in heroin-dependent participants, suggesting impaired regulation of impulsive decisions.

---

## Why This Matters

Understanding how humans make decisions under uncertainty is important for:

- Designing human-aligned AI systems  
- Behavioral modeling  
- Healthcare and addiction research  
- Economics and public policy  

This project demonstrates how reinforcement learning can be extended beyond games and robotics to model complex human behavior.

---

## Technical Approach

- Custom RL environment replicating the deceptive reward structure of the IGT  
- Dual-system architecture inspired by neuroscience  
- Likelihood-based fitting to real sequential behavioral data  
- Monte Carlo parameter search for latent cognitive variables  

---

## Disclaimer

This model is an experimental computational study and **does not diagnose or predict clinical conditions**. It aims to explore how reinforcement learning frameworks can approximate patterns observed in behavioral data.

---

## Repository Contents

- Dual-system RL implementation  
- Parameter estimation pipeline  
- Data preprocessing scripts  
- Visualization of inferred behavioral parameters  

---

## Future Work

- More robust parameter estimation methods  
- Bayesian inference for uncertainty quantification  
- Testing on additional behavioral datasets  
- Extending to other decision-making tasks
