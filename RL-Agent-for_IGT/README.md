# PFC-Limbic Dual-System HUman Decision-Making Mimicking AI Agent for Iowa Gambling Task





### In this work,

### 1. A policy Gradient based RL agent tries to solve the Iowa Gambling Task : identifying the best long term benefitting deck in a complex environement.
### 2. A model of Dual-System based Human brain, that tries to mimic the two aspects of human decision making: Analytical and Emotion/Intuition based.
### 3. Actual clinical data of subjects in Iowa Gambling Task is then fitted to the model and paraneters were extracted for each of the subject.



## Policy Learning for Iowa Gambling Problem

**A simple Neural Network based Agent is trained to learn the best policy for a complex set of decks of the Iowa Gambling Task. Following are the related plots and results:**

---

### Agent chooses deck "D" out of all other decks to be the best
![Agent Deck Choice](https://github.com/Mehul1729/RL-for-IGT/blob/652184f6630834cd4ed7f28eaabe4e92d008857a/RL-Agent-for_IGT/Plots/avg_prob_ALL_DECKS_gamma_0.3.gif)
---



### Deck Preference Order 
![Decks](https://github.com/Mehul1729/RL-for-IWT/blob/e73352a0ecc6d9c52e25e789969938e6c7e13854/Policy_Learning/Plots/deck%20orders%20.png)




### Agent becoming confident with its prediction as evidenced by an increasing log probabilities it assigns to each choice
![Agent Confidence](https://github.com/Mehul1729/RL-for-IWT/blob/807550f9f0f42e057a4edea20a388bb8de1b379c/Policy_Learning/Plots/model%20confidence%20building.png)

---

### REINFORCE based loss function settling to near zero after oscillating for extreme negatives and positives
![Loss Function Trend](https://github.com/Mehul1729/RL-for-IWT/blob/807550f9f0f42e057a4edea20a388bb8de1b379c/Policy_Learning/Plots/policy%20losse.png)

---

**Loss Function used → Cross Entropy style loss for Discounted Returns and learned log probabilities of action from each trial:**  

![eqn](https://github.com/Mehul1729/RL-for-IWT/blob/4257898f7b82063d5727a644464e7ca844494aef/Policy_Learning/Plots/eqn%20of%20lss%20.png)
