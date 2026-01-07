""""
This model again based on Stateless- Q-learning, mimics the emotional/impulsive part of the decision-making brain:: Limbic System.

1. Short Sighted: It wil not store the state indexes of the Amygdala and Ventral Striatum. These regions process "immediate emotional value" but lack a "cognitive map" of the task structure.
2. Dopmine-based Asymmetric learning: Positive rewards (dopamine bursts) are learned quickly, while negative rewards (dopamine dips) are learned slowly.
3. Prospect Theory-based reward perception: Uses a value function to model risk aversion nature.

"""


import numpy as np
from env import IowaEnv
import wandb 

class LimbicAgent_Somatic:
    def __init__(self, n_actions, alpha_pos, alpha_neg, gamma, epsilon, decay):
        self.n_actions = n_actions
        self.alpha_pos = alpha_pos  # Dopamine burst
        self.alpha_neg = alpha_neg  # Dopamine dip
        self.gamma = gamma
        self.epsilon = epsilon
        
        # to mimic the memory decay fo limbic system
        self.decay = decay # memory decay
        
        self.q_table = {} 
        self.sensitivity = 0.5 
        self.loss_aversion = 2.25 

    def _get_state_index(self, deck_counts):
        return 0 # have no long term thinking so returns no state record for analysis, unlike PFC mimic


    def act(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
        
        # Tie-breaking noise
        # q_values = np.array(q_values) + np.random.uniform(0, 1e-5, self.n_actions)
        return int(np.argmax(q_values))



    def pvl(self, reward):
        # converts raw rewards into percieved rewards based on sensitivity: (somatic marker )
        if reward >= 0:
            return np.power(reward, self.sensitivity)
        else:
            return -self.loss_aversion * np.power(abs(reward), self.sensitivity) # humans usually averse the loss with a loss aversion factor of 2.25


    def update(self, state, action, reward, next_state):
        
        for a in range(self.n_actions):
            prev_val = self.q_table.get((state, a), 0.0)
            self.q_table[(state, a)] = prev_val * self.decay

        # using the decayed value as the baseline for the update:
        current_val_decayed = self.q_table.get((state, action), 0.0)
        
        # Calculate max Q of next state (standard Q-learning)
        next_max = max([self.q_table.get((next_state, a), 0.0) for a in range(self.n_actions)])
        
        # Calculate Target & Error
        target = self.pvl(reward) + self.gamma * next_max
        delta = target - current_val_decayed
        
        # Asymmetric Update
        if delta > 0:
            new_value = current_val_decayed + self.alpha_pos * delta
        else:
            new_value = current_val_decayed + self.alpha_neg * delta
            
        self.q_table[(state, action)] = new_value


    def get_average_q_values(self):
        avg_q = np.zeros(self.n_actions)
        counts = np.zeros(self.n_actions)
        for (state, action), value in self.q_table.items():
            avg_q[action] += value
            counts[action] += 1
        return avg_q / np.maximum(counts, 1)




if __name__ == "__main__":
    n_episodes = 500
    
    # Settings
    alpha_pos = 0.1
    alpha_neg = 0.01
    gm = 0.3
    epsilon = 0.1
    
    # --- NEW: Decay Parameter ---
    decay_rate = 0.9 # <--- THE FORGETFULNESS KNOB
    
    deck_names = ["A_Bad", "B_Bad", "C_Good", "D_Good", "E_Zero", "F_Gem", "G_Trap", "H_Lossy"]

    wandb.init(
        project="IGT-Limbic-Simulation",
        config={
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "gamma": gm,
            "epsilon": epsilon,
            "decay": decay_rate,
            "episodes": n_episodes
        }
    )

    env = IowaEnv(episode_length=100)
    
    # Initialize Agent with Decay
    agent = LimbicAgent_Somatic(
        n_actions=8, 
        alpha_pos=alpha_pos, 
        alpha_neg=alpha_neg, 
        gamma=gm, 
        epsilon=epsilon,
        decay=decay_rate  # Pass the knob here
    )

    for ep in range(n_episodes):
        env.reset()
        state = agent._get_state_index(env._deck_pull_counts)

        for t in range(env.episode_length):
            action = agent.act(state)
            _, reward, done, _ = env.step(action)
            next_state = agent._get_state_index(env._deck_pull_counts)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            if done: break

        # Logging
        avg_qs = agent.get_average_q_values()
        log_dict = {f"Q_Avg/{name}": val for name, val in zip(deck_names, avg_qs)}
        log_dict["Total_Reward"] = env.cumulative_reward
        wandb.log(log_dict)

    wandb.finish()