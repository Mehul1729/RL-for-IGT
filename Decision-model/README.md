
## Computational Pipeline

The following diagram illustrates the flow from behavioral data to the extraction of neurocomputational biomarkers:

```mermaid
graph TD
    A[Clinical IGT Data<br>Choices & Rewards] --> M

    subgraph Dual-System Cognitive Engine
        P[Prefrontal Cortex<br>Goal-Directed / Q-Learning] -->|State-Action Values| M{Arbitration Router}
        L[Limbic System<br>Impulsive / Prospect Theory] -->|Immediate Utility| M
    end

    M -->|Action Probabilities| E(Hierarchical Bayesian Analysis<br>NUTS Sampler via JAX/NumPyro)
    E -->|Posterior Distributions| B[Extracted Biomarkers]

    B --> B1(β: PFC Dominance)
    B --> B2(λ: Loss Aversion)
    B --> B3(α_L: Limbic Learning Rate)
    B --> B4(d: Memory Decay)
    
    style P fill:#e1f5fe,stroke:#311b92,color:#000000,stroke-width:2px
    style L fill:#fce4ec,stroke:#4a148c,color:#000000,stroke-width:2px
    style M fill:#fff3e0,stroke:#e65100,color:#000000,stroke-width:2px
    style E fill:#f3e5f5,stroke:#4a148c,color:#000000,stroke-width:2px
    style B fill:#e8f5e9,stroke:#1b5e20,color:#000000,stroke-width:2px
```

---

## Code Structure

* `hba.py`: The Hierarchical Bayesian model implementation using PyMC and JAX/NumPyro **(Run this file to get the HBA-based posterior distributions of the fitted parameters.)**
  
* `analyze_hba_data.py`: Scripts for post-sampling analysis and Bayesian significance testing **(Run significance testing on the `.nc` file output from `hba.py`.)**
  
* `brain_modules.py`: Definitions of the Limbic (PVL) and PFC (Q-Learning) classes.
  
* `likelihood.py`: Core logic for calculating the negative log-likelihood (NLL) of human behavioral sequences.
  
* `main.py`: Parallelized Monte Carlo search for initial parameter exploration **(Run this to obtain Monte Carlo parameter estimates; these estimates are preliminary and may not be statistically significant.)**

---

## Dataset Source

The raw IGT clinical data is publicly accessible via Figshare at: http://figshare.com/articles/IGT_raw_data_Ahn_et_al_2014_Frontiers_in_Psychology/1101324
