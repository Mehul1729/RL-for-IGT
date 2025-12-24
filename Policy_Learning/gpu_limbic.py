import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import wandb # First set up and account and key at wandb.ai to analyse the performance of the Agent.
import os 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
# --- Agent ---
class LimbicAgent(nn.Module):
    """
    A Reinforcement Learning agent representing the limbic system.
    This agent uses a Policy Gradient method (REINFORCE) to learn.
    
    Attributes:
        policy_net (nn.Sequential): The neural network that maps states to action logits.
        optimizer (torch.optim.Adam): The optimizer for training the network.
        rewards (list): A buffer to store rewards received during an episode.
        saved_log_probs (list): A buffer to store the log probability of actions taken.
    """
    

    # Initializing the Policy NN:
    
    def __init__(self, gm, input_dim : int , n_actions, device, learning_rate = 0.005):
        """
        Initializes the LimbicAgent.
        Args:
            input_dims (int): The dimensionality of the state representation.
            n_actions (int): The number of possible actions (i.e., decks).
            device (torch.device): The device (CPU or GPU) to run on.
            learning_rate (float): The learning rate for the optimizer.
        """
        super(LimbicAgent, self).__init__()
        self.gm = gm
        self.device = device 
        
        # Defining the policy network :
        self.policy_net = nn.Sequential(
            nn.Linear(input_dims, 4),
            nn.ReLU(),
            nn.Linear(4, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Buffers for the REINFORCE algorithm
        self.rewards = [] # list of all the rewards in the episode 
        self.saved_log_probs = [] # This will store tensors now on the device
        self.episode_action_probs = [] # For wandb logging



# this function creates the format of input state to the policy NN :

    def _format_state(self, last_action: int , last_reward: float , n_actions):
        """
        Formats the input for the policy network as specified:
        one-hot encoded last action + previous reward.
        """
        # For the first turn, there is no previous action or reward.
        if last_action == -1:
            # iitiaizlizing the action and reward tensors 
            action_one_hot = torch.zeros(n_actions, device=self.device)
            reward_tensor = torch.zeros(1, device=self.device)
            
        else:
            
            action_tensor = torch.tensor(last_action, device=self.device)
            action_one_hot = F.one_hot(action_tensor, num_classes=n_actions).float()
            reward_tensor = torch.tensor([last_reward], dtype=torch.float32, device=self.device)
        
        # Min-Max scaling of the rewards :
        reward_tensor = 2 * ((reward_tensor - (-1210)) / (130 - (-1210))) - 1 # for a -1 to 1 range
        # reward_tensor =  ((reward_tensor - (-1210)) / (130 - (-1210))) 

        
        # Concatenating to create the final state vector:
        return torch.cat([action_one_hot, reward_tensor]).unsqueeze(0)


# Action fucntion:

    def act(self, last_action: int, last_reward: float, n_actions: int):
        """
        Selects an action based on the current policy.
        """
        # Formating the the state for the network
     
        state = self._format_state(last_action, last_reward, n_actions)
        print(f"inout state for this trial : {state}")
        # Forward pass to get action logits:
        action_logits = self.policy_net(state) # the model will assign some logit to each of the actions, signifying the likelhiood of taking that action.

        print(action_logits)
# sampling an action from the action_logit distrbution to account for exploration in RL training :        
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()


        
        # Saving log_prob and action_probs for update/logging:
        self.saved_log_probs.append(action_distribution.log_prob(action)) # saving log prob for this trial 
        self.episode_action_probs.append(action_distribution.probs) # Log probs for wandb
        
        return action.item()


# for updating the weights of the policy net :
# this updation code will run for each episode:
# LOSS FUNCTION CALCULATION:
    def update(self, gm):
        """
        Updates the policy network using the REINFORCE algorithm.
        Returns loss, mean log prob, and avg action probs for logging.
        """
        if not self.saved_log_probs:
            return 0.0, 0.0, torch.zeros(self.policy_net[-1].out_features) # No actions

        policy_loss = []
        discounted_returns = deque() # initialize a deque to store discounted returns
        R = 0 # total long term reward (discounted return)

        # for the given trial, we clauclate the discounted retrun:
        for r in reversed(self.rewards): # for all the rewrds in a given episode 
            R = r + gm * R # bellman equation trick 
            discounted_returns.appendleft(R) # saving the discounted return for each trial in a deque 
        
        # rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        # T = rewards.size(0)

        # gamma_powers = gm ** torch.arange(T, device=self.device, dtype=torch.float32)

        # discounted = rewards * gamma_powers
        # returns = torch.flip(torch.cumsum(torch.flip(discounted, [0]), dim=0), [0])
        # returns = returns / gamma_powers + 1e-9


 # normalizing the discounted returns for all he trials in the ep. 
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            # Handling th edge case of 1-step episode:
            returns = torch.tensor([0.0], dtype=torch.float32, device=self.device) 
            
        # Calculating the policy loss:
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R) # collecting policy loss for each action taken (for each trial, total + ep_lenght)
            
            
        # replacement :
        # Data for Logging:
        # Stacking all log_probs and probs tensors
        log_probs_tensor = torch.stack(self.saved_log_probs)
        action_probs_tensor = torch.stack(self.episode_action_probs)
        
        # policy_loss = -log_probs_tensor * returns             # [T]
        # loss = policy_loss.sum()
        
        # saving the data for this episode
        mean_log_prob = log_probs_tensor.mean().item() # to track teh growing confidence of the model in its predictions
        avg_episode_probs = action_probs_tensor.mean(dim=0).squeeze().detach() # Avg across episodes for each of the choices 

        #  optimization step:
        self.optimizer.zero_grad()
        # loss = torch.cat(policy_loss).sum() # net polixy loss for the episode
        loss.backward()
        self.optimizer.step()
        
        # Clearing the episode's data
        self.rewards = []
        self.saved_log_probs = []
        self.episode_action_probs = []

        return loss.item(), mean_log_prob, avg_episode_probs # Return logs




# --- Driver Code ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device: {device} ---")
    
    print("Which environment do you want to run?")
    print("1: Classic 4-Deck IGT (Recommended)")
    print("2: Complex 6-Deck IGT (Harder)")
    print("3: Extreme 8-Deck IGT (Very Hard)")
    choice = input("Enter 1, 2, or 3: ")

    if choice == '2':
        from complex_env import IowaEnv as GameEnv
        env_name = "Complex-6-Deck"
        temp_env = GameEnv()
        n_actions = len(temp_env.action_space)
        del temp_env
        print(f"\n--- Loading Complex {n_actions}-Deck Environment ---")
    elif choice == '3':
        from complex_env import IowaEnv as GameEnv
        env_name = "Extreme-8-Deck"
        temp_env = GameEnv()
        n_actions = len(temp_env.action_space)
        del temp_env
        print(f"\n--- Loading Extreme {n_actions}-Deck Environment ---")
    else:
        # Default to 4-deck:
        env_name = ""
        # try:
        #     from classic_env import IowaEnv as GameEnv
        #     n_actions = 4 
        #     print("\n--- Loading Classic 4-Deck Environment ---")
        # except ImportError:
        #     print("Could not find env.py. Defaulting to env_complex.py")
        #     from complex_env import IowaEnv as GameEnv
        #     n_actions = len(GameEnv().action_space)
        #     print(f"\n--- Loading {n_actions}-Deck Environment from env_complex.py ---")


    # Hyperparameters
    n_episodes = 3000
    learning_rate = 0.005
    gm = float(input("Enter gamma (0.0 to 1.0) for the agent (e.g., 0.1 for impulsive, 0.9 for far-sighted):\t"))
    
    # State is one-hot action + last reward
    input_dims = n_actions + 1
    
    # Initialization
    env = GameEnv(episode_length=1000) # 100 trials per episdoe 
    
    limbic_agent = LimbicAgent(
        input_dim=input_dims, 
        n_actions=n_actions, 
        gm=gm, 
        device=device,
        learning_rate=learning_rate
    ).to(device)




    """

        Using Wandb to log training metrics and analyse how the agent is learning over time.
        
        """

    
    wandb.init(
        project="RL-IWT",
        config={
            "learning_rate": learning_rate,
            "episodes": n_episodes,
            "gamma": gm,
            "architecture": "SimpleMLP_32",
            "environment": env_name,
            "n_actions": n_actions
        }
    )
    # Watch the model to log gradients
    wandb.watch(limbic_agent, log="all", log_freq=300) # Log gradients every 300 eps
    
    print(f"--- Training Limbic Agent Alone (Gamma: {gm}) ---")
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
        loss, mean_log_prob, avg_probs = limbic_agent.update(gm) # Capture logs
        
        policy_losses.append(loss) # Store the loss
        episode_rewards.append(env.cumulative_reward)
        
        log_data = {
            "episode": i,
            "Expected Return ": -1* loss,
            "cumulative_reward per episode": env.cumulative_reward,
            "mean_log_prob_episode": mean_log_prob
        }
        # Adding average action probabilities to the log
        prob_log = {f"avg_prob_deck_{chr(65+j)}": p.item() for j, p in enumerate(avg_probs)}
        log_data.update(prob_log)
        
        wandb.log(log_data)
        
        
        # Store deck pulls from the final episodes to see learned policy
        if i >= n_episodes - 100:
            for deck, count in env._deck_pull_counts.items():
                final_deck_pulls[deck] += count

        if (i + 1) % 300 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {i+1}/{n_episodes}, Average Reward (last 100): {avg_reward:.2f}")
            # Log the 100-episode average reward
            wandb.log({"average_reward_100_ep": avg_reward, "episode": i})
            
    total_average_reward = np.mean(episode_rewards)
    print(f"\nOverall Average Reward after {n_episodes} episodes: {total_average_reward:.2f} for gamma = {gm}")
    
    print("\n--- Average Deck Pulls (Last 100 Episodes) ---")
    for deck, count in final_deck_pulls.items():
        print(f"Deck {chr(65 + deck)}: {count / 100.0:.2f} pulls/episode")

    print("\n--- Training Complete ---")
    if gm < 0.5:
        print("With a low gamma, the agent likely preferred high-reward decks")
        print("even if they led to long-term losses.")
    else:
        print("With a high gamma, the agent should have learned to prefer the stable,")
        print("long-term positive decks.")

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"limbic_agent_gamma_{gm}_env_{choice}.pth")
    torch.save(limbic_agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")


    print("\nGenerating training plots...")

    # Plot raw episode rewards and policy losses directly
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Raw Cumulative Reward per Episode
    ax1.plot(episode_rewards)
    ax1.set_title(f"Cumulative Reward per Episode - Gamma={gm}")
    ax1.set_ylabel("Cumulative Reward")
    ax1.grid(True)

    # Plot 2: Raw Policy Loss per Episode
    ax2.plot(policy_losses)
    ax2.set_title("Policy Loss per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss Value")
    ax2.grid(True)

    plt.tight_layout()
    plot_filename = f"training_plots_gamma_{gm}.png"
    plt.savefig(plot_filename)
    print(f"Saved training plots to {plot_filename}")

    wandb.finish()

