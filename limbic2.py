import numpy as np
from env import IowaEnv  # Import the environment from env.py

class LimbicAgent_QLearning:
    """
    A Q-learning agent to model the limbic system.
    
    This agent learns a simple "quality" score (Q-value) for each action (deck).
    The 'gamma' parameter directly controls its foresight, allowing for a clear
    distinction between impulsive (low gamma) and patient (high gamma) behavior.
    """
    def __init__(self, n_actions: int, learning_rate: float, gamma: float, epsilon: float):
        """
        Initializes the Q-learning agent.

        Args:
            n_actions (int): Number of possible actions (4 decks).
            learning_rate (float): How quickly the agent learns (alpha).
            gamma (float): Discount factor for future rewards. This is the
                           key parameter for "foresight".
            epsilon (float): The exploration rate (e.g., 0.1 for 10% random actions).
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # We only need a simple 1D array for the Q-table because the
        # task is "stateless" – the reward for picking a deck
        # doesn't depend on the previous action.
        self.q_table = np.zeros(n_actions)

    def act(self) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        """
        # Exploration: choose a random action
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
        # Exploitation: choose the best known action
        else:
            action = np.argmax(self.q_table)
            
        return action

    def update(self, action: int, reward: float):
        """
        Updates the Q-value for the chosen action using the Q-learning formula.
        
        Args:
            action (int): The action that was taken.
            reward (float): The reward received from that action.
        """
        # The Q-learning formula:
        # Q(a) = Q(a) + alpha * [reward + gamma * max_a'(Q(a')) - Q(a)]
        
        # 1. Get the current Q-value for the action that was taken
        old_value = self.q_table[action]
        
        # 2. Get the best possible Q-value for the *next* state
        #    (Since our environment is stateless, this is just the max of the current Q-table)
        next_max = np.max(self.q_table)
        
        # 3. Calculate the new Q-value
        #    This is the core of the logic:
        #    If gamma=0, this term becomes (reward - old_value)
        #    If gamma>0, this term becomes (reward + gamma*next_max - old_value)
        learned_value = reward + self.gamma * next_max
        new_value = old_value + self.lr * (learned_value - old_value)
        
        self.q_table[action] = new_value


# --- Driver Code (Main execution) ---
if __name__ == '__main__':
    # Hyperparameters
    n_episodes = 500
    learning_rate = 0.1  # Alpha
    epsilon = 0.01      # Exploration rate
    
    # Get gamma from user, just like your original script
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

    for i in range(n_episodes):
        env.reset()
        
        for t in range(env.episode_length):
            # 1. Agent chooses an action
            action = limbic_agent.act()
            
            # 2. Environment gives a reward
            _, reward, _, _ = env.step(action)
            
            # 3. Agent learns from the reward
            limbic_agent.update(action, reward)
        
        episode_rewards.append(env.cumulative_reward)
        if (i + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {i+1}/{n_episodes}, Avg Reward (last 50): {avg_reward:.2f}")

    total_average_reward = np.mean(episode_rewards)
    print(f"\nOverall Average Reward: {total_average_reward:.2f}")
    
    # This part is key: see what the agent *actually* learned.
    print("\n--- Final Learned Q-Values ---")
    print(f"Deck A (Bad): {limbic_agent.q_table[0]:.2f}")
    print(f"Deck B (Bad): {limbic_agent.q_table[1]:.2f}")
    print(f"Deck C (Good): {limbic_agent.q_table[2]:.2f}")
    print(f"Deck D (Good): {limbic_agent.q_table[3]:.2f}")