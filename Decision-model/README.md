# Dual-System Reinforcement Learning Model for Iowa Gambling Task

This project models human decision-making in the **Iowa Gambling Task (IGT)** using a dual-system framework inspired by neuroscience:

- **Limbic System** → emotional valuation using Prospect Valence Learning (PVL)
- **Prefrontal Cortex (PFC)** → strategic learning using Q-learning

Human behavioral data is fit by minimizing the **negative log-likelihood of observed actions**.

---

# 1. Limbic System (PVL Model)

The limbic system converts reward outcomes into subjective utility using the Prospect Valence Learning function.

$$
u(r_t)=
\begin{cases}
r_t^{\rho}, & r_t \ge 0 \\
-\lambda |r_t|^{\rho}, & r_t < 0
\end{cases}
$$

where:

- $r_t$ = reward at trial $t$
- $\rho$ = sensitivity parameter (**fixed = 0.5**)
- $\lambda$ = **loss aversion parameter (fitted)**

---

## Limbic Value Update

The limbic system maintains stateless action values $Q^L$.

First apply decay:

$$
Q^L_t(a) \leftarrow d \cdot Q^L_t(a)
$$

Then update the chosen action:

$$
\delta_t^L = u(r_t) - Q_t^L(a_t)
$$

$$
Q_{t+1}^L(a_t) = Q_t^L(a_t) + \alpha_L \delta_t^L
$$

where

- $\alpha_L$ = limbic learning rate (**fitted**)
- $d$ = decay parameter (**fitted**)

---

# 2. Prefrontal Cortex (PFC) — Q-Learning

The PFC represents a stateful reinforcement learning system.

The state is constructed from deck selection counts:

$$
s_t =
\left(
\lfloor c_0/5 \rfloor,
\lfloor c_1/5 \rfloor,
\lfloor c_2/5 \rfloor,
\lfloor c_3/5 \rfloor
\right)
$$

where $c_i$ is the number of times deck $i$ has been chosen.

---

## Q-learning Update

$$
y_t = r_t + \gamma \max_a Q^P(s_{t+1}, a)
$$

$$
\delta_t^P = y_t - Q^P(s_t, a_t)
$$

$$
Q^P(s_t,a_t) \leftarrow Q^P(s_t,a_t) + \alpha_P \delta_t^P
$$

Parameters:

- $\alpha_P = 0.1$ (**fixed**)
- $\gamma = 0.95$ (**fixed**)

---

# 3. Action Selection (Softmax Policy)

Each system produces a softmax policy.

$$
\pi^X(a)=
\frac{\exp(Q^X(a)/\tau_X)}
{\sum_b \exp(Q^X(b)/\tau_X)}
$$

where

- $X \in \{L,P\}$ (Limbic or PFC)
- $\tau_X$ = temperature (**fixed = 10**)

---

# 4. PFC–Limbic Arbitration

Action probabilities are modeled as a mixture of the two systems.

$$
\pi(a) = \beta \pi^P(a) + (1-\beta)\pi^L(a)
$$

where

- $\beta$ = **PFC control weight (fitted)**

Interpretation:

- $\beta \approx 1$ → strong PFC dominance
- $\beta \approx 0$ → strong limbic dominance

---

# 5. Negative Log-Likelihood (Model Fitting)

Given observed human actions $a_t$, the likelihood is

$$
\mathcal{L}(\theta) = \prod_{t=1}^{T} \pi_t(a_t)
$$

We minimize the negative log-likelihood:

$$
NLL(\theta) =
-\sum_{t=1}^{T} \log(\pi_t(a_t))
$$

---

# 6. Parameter Estimation

The following parameters are fitted per subject:

$$
\theta =
(\alpha_L, d, \lambda, \beta)
$$

using Monte Carlo search over:

- $\alpha_L \in [0.01,1]$
- $d \in [0.01,1]$
- $\lambda \in [0.1,5]$
- $\beta \in [0,1]$

The best parameters satisfy:

$$
\hat{\theta} = \arg\min_\theta NLL(\theta)
$$

---

# Model Interpretation

This framework allows interpretation of cognitive control in decision-making:

| Parameter | Meaning |
|---|---|
| $\lambda$ | Loss aversion (limbic sensitivity to losses) |
| $\alpha_L$ | Limbic learning speed |
| $d$ | Memory decay |
| $\beta$ | Degree of **PFC vs Limbic control** | 

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
