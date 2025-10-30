import numpy as np
from collections import deque

class IowaEnv:
    """
    This class simulates the *classic* IGT experiment with probabilistic rewards.
    - Action Space: Discrete(4) corresponding to Decks A, B, C, D.
    - State Space: A dictionary containing trial number, reward, last action, and history.
    - Reward: A combination of a guaranteed positive reward and a *probabilistic*
      large penalty, based on the standard IGT payoff scheme.
    """
    def __init__(self, episode_length: int = 100):
        """
        Initializes the Iowa Gambling Task environment.
        Args:
            episode_length (int): The number of trials in one episode.
        """
        self.episode_length = episode_length
        
        # --- CLASSIC IGT PAYOFF SCHEDULE (Probabilistic) ---
        # Based on Bechara et al. (1994)
        # Net outcome per 10 cards:
        # Deck A: (10 * 100) - (5 * 250) = 1000 - 1250 = -250 (BAD)
        # Deck B: (10 * 100) - (1 * 1250) = 1000 - 1250 = -250 (BAD)
        # Deck C: (10 * 50) - (5 * 50) = 500 - 250 = +250 (GOOD)
        # Deck D: (10 * 50) - (1 * 250) = 500 - 250 = +250 (GOOD)
        
        self._deck_properties = {
            # BAD DECKS: High immediate reward, net loss.
            0: {'reward': 100, 'penalty_amount': -250, 'penalty_prob': 0.5}, # Deck A
            1: {'reward': 100, 'penalty_amount': -1250, 'penalty_prob': 0.1}, # Deck B
            
            # GOOD DECKS: Lower immediate reward, net gain.
            2: {'reward': 50, 'penalty_amount': -50, 'penalty_prob': 0.5},  # Deck C
            3: {'reward': 50, 'penalty_amount': -250, 'penalty_prob': 0.1}  # Deck D
        }
        
        self.action_space = list(self._deck_properties.keys()) # [0, 1, 2, 3]
        self.reset()
        
    def _calculate_reward(self, action: int) -> int:
        """Calculates the reward for a given action based on the deck's properties."""
        deck = self._deck_properties[action]
        self._deck_pull_counts[action] += 1
        
        reward = deck['reward']
        
        # Check if a penalty is applied *probabilistically*
        if np.random.rand() < deck['penalty_prob']:
            reward += deck['penalty_amount'] # adding penalty to reward
            
        return reward
    
    def reset(self) -> dict:
        """Resets the environment to its initial state for a new episode."""
        self._current_trial = 0
        self.cumulative_reward = 0
        self._last_action = -1 
        self._action_history = deque(maxlen=10)
        self._deck_pull_counts = {i: 0 for i in self.action_space}
        return self._get_state()

    def _get_state(self) -> dict:
        """Constructs the state dictionary from the environment's current properties."""
        return {
            'trial': self._current_trial,
            'cumulative_reward': self.cumulative_reward,
            'last_action': self._last_action,
            'history': list(self._action_history)
        }
    
    def step(self, action: int) -> tuple[dict, int, bool, dict]:
        """Executes one time step in the environment."""
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}.")
        
        reward = self._calculate_reward(action)
        self.cumulative_reward += reward
        self._last_action = action
        self._action_history.append(action)
        self._current_trial += 1
        
        done = self._current_trial >= self.episode_length
        next_state = self._get_state()
        info = {'deck_pulls': self._deck_pull_counts.copy()}
        
        return next_state, reward, done, info
    
    def render(self):
        """Prints a summary of the current environment state."""
        print(f"--- Trial: {self._current_trial}/{self.episode_length} ---")
        print(f"Last Action: {'None' if self._last_action == -1 else f'Deck {chr(65 + self._last_action)}'}")
        print(f"Cumulative Reward: {self.cumulative_reward}")
        print(f"Deck Pulls: {self._deck_pull_counts}")
        print("-" * 25)
