


# In this version of code the Agent completely ignores the penalties for low gamma to force dumbness 


import numpy as np
from env import IowaEnv  

class LimbicAgent_QLearning:
    """
    
    This Q-learning agent has two different learning mechanisms, controlled
    by the 'gamma' parameter, to simulate impulsive vs. patient behavior.
    """
    def __init__(self, n_actions: int, learning_rate: float, gamma: float, epsilon: float):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(n_actions) # q value table for 4 decks
        
        # --- This is the new, crucial logic ---
        # We will use the gamma value to determine the agent's "type".
        if self.gamma < 0.5:
            self.is_impulsive = True
            print("--- Agent Type: IMPULSIVE (Ignores Penalties) ---")
        else:
            self.is_impulsive = False
            print("--- Agent Type: PATIENT (Learns from all outcomes) ---")

    def act(self) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table) # exploit the gathered knowledge

    def update(self, action: int, reward: float):
        """
        Updates the Q-value for the chosen action based on the agent's type.
        """
        
        # --- THE CORE FIX ---
        
        # 1. IMPULSIVE AGENT: Only learns from positive rewards.
        # It is "blind" to punishment.
        if self.is_impulsive:
            if reward > 0:
                old_value = self.q_table[action]
                learned_value = reward  # No future (gamma) or penalty is considered
                new_value = old_value + self.lr * (learned_value - old_value)
                self.q_table[action] = new_value
        
        # 2. PATIENT AGENT: Learns from all rewards, positive and negative.
        # This is the standard Q-learning rule for a bandit problem.
        else:
            old_value = self.q_table[action]
            learned_value = reward  # The true, immediate reward (positive or negative)
            new_value = old_value + self.lr * (learned_value - old_value)
            self.q_table[action] = new_value


# --- Driver Code (Main execution) ---
if __name__ == '__main__':
    # Hyperparameters
    n_episodes = 500
    learning_rate = 0.1  # Alpha
    epsilon = 0.1      # Exploration rate
    
    # Get gamma from user
    gm = float(input("Enter gamma (e.g., 0.01 for impulsive, 0.9 for patient):\t"))
    
    # Initialization
    env = IowaEnv(episode_length=100)
    limbic_agent = LimbicAgent_QLearning(
        n_actions=4,
        learning_rate=learning_rate,
        gamma=gm,
        epsilon=epsilon
    )

    print(f"\n--- Training Q-Learning Agent with gamma = {gm} ---")
    episode_rewards = []
    episode_good_deck_percent = []

    for i in range(n_episodes):
        env.reset()
        good_choices = 0
        bad_choices = 0
        
        for t in range(env.episode_length):
            action = limbic_agent.act()
            
            if action in [0, 1]: # Decks A, B are "Bad"
                bad_choices += 1
            else: # Decks C, D are "Good"
                good_choices += 1
            
            _, reward, _, _ = env.step(action)
            limbic_agent.update(action, reward)
        
        episode_rewards.append(env.cumulative_reward)
        total_choices = good_choices + bad_choices
        percent_good = (good_choices / total_choices) * 100
        episode_good_deck_percent.append(percent_good)

        if (i + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_good_percent = np.mean(episode_good_deck_percent[-50:])
            print(f"Episode {i+1}/{n_episodes}, Avg Reward (last 50): {avg_reward:.2f}, Avg Good Deck %: {avg_good_percent:.1f}%")

    total_average_reward = np.mean(episode_rewards)
    total_avg_good_percent = np.mean(episode_good_deck_percent)

    print(f"\nOverall Average Reward: {total_average_reward:.2f}")
    print(f"Overall Average Good Deck %: {total_avg_good_percent:.1f}%")
    
    print("\n--- Final Learned Q-Values ---")
    print(f"Deck A (Bad): {limbic_agent.q_table[0]:.2f}")
    print(f"Deck B (Bad): {limbic_agent.q_table[1]:.2f}")
    print(f"Deck C (Good): {limbic_agent.q_table[2]:.2f}")
    print(f"Deck D (Good): {limbic_agent.q_table[3]:.2f}")