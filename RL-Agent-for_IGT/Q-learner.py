
"""
    This is a solver for Complex Iowa-Gambling task based on Q-learning method (through Q-table). 
    This mimics a PFC-like agent that takes analytical decisions based on long-term rewards.
"""




import numpy as np
from env import IowaEnv
import wandb 

class PFC_Agent:
    def __init__(self, n_actions, learning_rate, gamma, epsilon): # gamma: for discounted rewards, epsilon: for exploration
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _get_state_index(self, deck_counts):
        # Discretizng the no. of deck pulls into bins of size 5 to reduce state space size
        bins = tuple(np.array(list(deck_counts.values())) // 5)
        return bins

    def act(self, state):
     # Exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
    #Exploitation
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        # Q-value update:
        old_value = self.q_table.get((state, action), 0.0)
        next_max = max([self.q_table.get((next_state, a), 0.0) for a in range(self.n_actions)])
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def get_average_q_values(self):
        # func to get the avg Q vals for a state across all actions
        avg_q = np.zeros(self.n_actions)
        counts = np.zeros(self.n_actions)
        for (state, action), value in self.q_table.items():
            avg_q[action] += value
            counts[action] += 1
        return avg_q / np.maximum(counts, 1)

# Driver Code :
if __name__ == "__main__":
    n_episodes = 500
    learning_rate = 0.1
    epsilon = 0.1
    gm = float(input("Enter gamma: "))

    # Initialize wandb run
    wandb.init(
        project="IGT-PFC-Simulation",
        config={
            "learning_rate": learning_rate,
            "gamma": gm,
            "epsilon": epsilon,
            "episodes": n_episodes,
            "bin_size": 5
        }
    )

    env = IowaEnv(episode_length=100)
    agent = PFC_Agent(n_actions=8, learning_rate=learning_rate, gamma=gm, epsilon=epsilon)

    deck_names = ["A_Bad", "B_Bad", "C_Good", "D_Good", "E_Zero", "F_Gem", "G_Trap", "H_Lossy"]

    for ep in range(n_episodes):
        env.reset()
        state = agent._get_state_index(env._deck_pull_counts)
        # iterating ovwr trials :
        for t in range(env.episode_length):
            action = agent.act(state)
            _, reward, done, _ = env.step(action)
            next_state = agent._get_state_index(env._deck_pull_counts)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break

        # avg q val for this ep:
        avg_qs = agent.get_average_q_values()
        
        log_dict = {f"Q_Avg/{name}": val for name, val in zip(deck_names, avg_qs)}
        log_dict["Total_Reward"] = env.cumulative_reward
        wandb.log(log_dict)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{n_episodes} logged to wandb.")

    wandb.finish()
    
    