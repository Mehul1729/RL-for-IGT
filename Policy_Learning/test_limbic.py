import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for plotting

# --- Step 2: Limbic System (RL Agent) ---
class LimbicAgent(nn.Module):
    """
    A Reinforcement Learning agent representing the limbic system.
    This agent uses a Policy Gradient method (REINFORCE) to learn. Its decision-making
    is driven by immediate rewards, making it prone to impulsive choices that seem
    good in the short term but may be detrimental in the long run.
    Attributes:
        policy_net (nn.Sequential): The neural network that maps states to action logits.
        optimizer (torch.optim.Adam): The optimizer for training the network.
        rewards (list): A buffer to store rewards received during an episode.
        saved_log_probs (list): A buffer to store the log probability of actions taken.
    """
    
    
    
    # Initializing the Policy NN:
    
    def __init__(self, gm, input_dims: int, n_actions: int, learning_rate: float = 0.005):
        """
        Initializes the LimbicAgent.
        Args:
            input_dims (int): The dimensionality of the state representation.
                                (e.g., 4 for one-hot action + 1 for last reward = 5).
            n_actions (int): The number of possible actions (i.e., decks).
            learning_rate (float): The learning rate for the optimizer.
        """
        super(LimbicAgent, self).__init__()
        self.gm = gm
        # Define the policy network (a simple Multi-Layer Perceptron)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Buffers for the REINFORCE algorithm
        self.rewards = [] # list of all the rewards in the episode 
        self.saved_log_probs = []





# this function creates the format of input state to the policy NN :

    def _format_state(self, last_action: int, last_reward: float, n_actions: int) -> torch.Tensor:
        """
        Formats the input for the policy network as specified:
        one-hot encoded last action + previous reward.
        Args:
            last_action (int): The last action taken (-1 if it's the first turn).
            last_reward (float): The reward from the last action (0 if first turn).
            n_actions (int): The total number of classes of actions.
        Returns:
            torch.Tensor: The formatted state tensor.
        """
        # For the first turn, there is no previous action or reward.
        if last_action == -1:
            action_one_hot = torch.zeros(n_actions)
            reward_tensor = torch.zeros(1)
            
        else:
            action_one_hot = F.one_hot(torch.tensor(last_action), num_classes=n_actions).float()
            reward_tensor = torch.tensor([last_reward], dtype=torch.float32)
        
        # Min-Max scaling of the rewards :
        reward_tensor = 2 * ((reward_tensor - (-1250)) / (130 - (-1250))) - 1
        
        # Concatenate to create the final state vector
        return torch.cat([action_one_hot, reward_tensor]).unsqueeze(0)




# Action fucntion:

    def act(self, last_action: int, last_reward: float, n_actions: int) -> int:
        """
        Selects an action based on the current policy.
        Args:
            last_action (int): The action taken in the previous step.
            last_reward (float): The reward received from the previous step.
            n_actions (int): The number of possible actions.
        Returns:
            int: The chosen action.
        """
        # 1. Format the state for the network
        state = self._format_state(last_action, last_reward, n_actions)
        
        # 2. Forward pass to get action logits
        action_logits = self.policy_net(state)
        
        # In order to add some Exploration aspect to our agent,
        # we have added a sampling from a categorical distribution to get the final log probs:
        
        # 3. Create a probability distribution and sample an action
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        
        # 4. Save the log probability of the chosen action (required for REINFORCE)
        self.saved_log_probs.append(action_distribution.log_prob(action))
        return action.item()



# fpr updating the weights of the policy net :
# this updation code will run for each episode:
# LOSS FUNCTION CALCULATION:
    def update(self, gm):
        """
        Updates the policy network using the REINFORCE algorithm.
        This method is modified to make low-gamma agents truly impulsive.
        """
        if not self.saved_log_probs:
            return 0.0 # No actions were taken

        policy_loss = []
        discounted_returns = deque()
        R = 0 # total long term reward 

# for the given trial, we clauclate the discounted retrun:
        for r in reversed(self.rewards): # for all the rewrds in a given episode 
            R = r + gm * R # bellman equation trick 
            discounted_returns.appendleft(R)
        
        
        
        
        returns = torch.tensor(list(discounted_returns), dtype=torch.float32) # updating the list of discounted retruns so far
        
        # We flip the logic.
        # Impulsive agents (low gm) get *normalized* returns. This makes them
        # chase high-reward "jackpots" (like +100) because they have a
        # high z-score, ignoring the discounted long-term loss.
        # Far-sighted agents (high gm) get *raw* returns, allowing them
        # to correctly sum the positive/negative long-term outcomes.
        if gm < 0.5:
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            else:
                returns = torch.tensor([0.0], dtype=torch.float32) # Avoid NaN if std is 0
            
        # 3. Calculate the policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R) # collecting policy loss for each action taken in the episode
            
        # 4. Perform the optimization step
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # 5. Clear the episode's data
        self.rewards = []
        self.saved_log_probs = []

        return loss.item() # Return the scalar loss value




# --- Driver Code ---
if __name__ == '__main__':
    # --- Environment Selection ---
    print("Which environment do you want to run?")
    print("1: Classic 4-Deck IGT (Recommended)")
    print("2: Complex 6-Deck IGT (Harder)")
    print("3: Extreme 8-Deck IGT (Very Hard)")
    choice = input("Enter 1, 2, or 3: ")

    if choice == '2':
        from complex_env import IowaEnv as GameEnv
        temp_env = GameEnv()
        n_actions = len(temp_env.action_space)
        del temp_env
        print(f"\n--- Loading Complex {n_actions}-Deck Environment ---")
    elif choice == '3':
        from complex_env import IowaEnv as GameEnv
        temp_env = GameEnv()
        n_actions = len(temp_env.action_space)
        del temp_env
        print(f"\n--- Loading Extreme {n_actions}-Deck Environment ---")
    else:
        # Default to 4-deck
        try:
            from classic_env import IowaEnv as GameEnv
        except ImportError:
            print("Could not find classic_env.py. Defaulting to complex_env.py")
            from complex_env import IowaEnv as GameEnv
        
        n_actions = 4 # Assuming classic_env.py has 4 decks
        print("\n--- Loading Classic 4-Deck Environment ---")


    # Hyperparameters
    # MODIFICATION: Changed n_episodes from 10 to 50
    n_episodes = 50
    
    gm = float(input("Enter gamma (0.0 to 1.0) for the agent (e.g., 0.1 for impulsive, 0.9 for far-sighted):\t"))
    
    # State is one-hot action + last reward
    input_dims = n_actions + 1
    
    # Initialization
    env = GameEnv(episode_length=100) # 100 trials per episdoe 
    limbic_agent = LimbicAgent(input_dims=input_dims, n_actions=n_actions, gm = gm)
    
    # MODIFICATION: Updated print statement for 50 episodes
    print(f"--- Training Limbic Agent Alone (Gamma: {gm}) for {n_episodes} episodes ---")
    episode_rewards = []
    policy_losses = [] # List to store losses
    final_deck_pulls = {i: 0 for i in range(n_actions)}
    
    for i in range(n_episodes):
        env_state = env.reset()
        last_action = env_state['last_action']
        last_reward = 0.0
        
        # Run one full episode
        for t in range(env.episode_length):
            # Agent chooses an action
            action = limbic_agent.act(last_action, last_reward, n_actions)
            # Environment responds
            next_env_state, reward, done, info = env.step(action)
            # Store the immediate reward for the agent's update
            limbic_agent.rewards.append(reward)
            # Update state variables for the next turn
            last_action = action
            last_reward = reward
        
        # At the end of the episode, update the agent's policy
        loss = limbic_agent.update(gm) # Capture the loss
        policy_losses.append(loss) # Store the loss
        episode_rewards.append(env.cumulative_reward)
        
        # Store deck pulls from *all* episodes to see learned policy
        for deck, count in env._deck_pull_counts.items():
            final_deck_pulls[deck] += count

        # MODIFICATION: Added conditional print block for every 5th episode
        if (i + 1) % 5 == 0:
            print(f"\n--- Episode {i+1}/{n_episodes} ---")
            print(f"  Total Reward: {env.cumulative_reward:.2f}")
            print(f"  Policy Loss: {loss:.4f}")
            print(f"  Deck Pulls this Episode:")
            for deck, count in sorted(env._deck_pull_counts.items()):
                print(f"    Deck {chr(65 + deck)}: {count}")

            
    total_average_reward = np.mean(episode_rewards)
    # MODIFICATION: Updated print statement for 50 episodes
    print(f"\n\n--- Overall Average Reward after {n_episodes} episodes: {total_average_reward:.2f} for gamma = {gm} ---")
    
    # MODIFICATION: Changed title to reflect all 50 episodes
    print(f"\n--- Average Deck Pulls (All {n_episodes} Episodes) ---")
    for deck, count in final_deck_pulls.items():
        # MODIFICATION: Divisor is now n_episodes (50)
        print(f"Deck {chr(65 + deck)}: {count / n_episodes:.2f} pulls/episode")

    print("\n--- Training Complete ---")
    if gm < 0.5:
        print("With a low gamma, the agent likely preferred high-reward decks")
        print("even if they led to long-term losses.")
    else:
        print("With a high gamma, the agent should have learned to prefer the stable,")
        print("long-term positive decks.")