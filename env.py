import numpy as np
from collections import deque

class IowaEnv:
    """
    A Gym-style environment for the Iowa Gambling Task (IGT).
    This version uses the balanced deck properties required for the
    Limbic vs. PFC simulation.
    """
    
    def __init__(self, episode_length: int = 100):
        self.episode_length = episode_length

        # --- BALANCED DECK PROPERTIES ---
        self._deck_properties = {
            # BAD DECKS: Tempting +100 reward, but a net loss over time.
            0: {'reward': 100, 'penalty': -550, 'penalty_freq': 5},
            1: {'reward': 100, 'penalty': -1250, 'penalty_freq': 10},

            # GOOD DECKS: Lower +50 reward, but a reliable net gain.
            2: {'reward': 50,  'penalty': -50,  'penalty_freq': 10},
            3: {'reward': 50,  'penalty': -250, 'penalty_freq': 10},
        }
        
        self.action_space = [0, 1, 2, 3]
        self.reset()


# fucntion to calculate reward given the action 
    def _calculate_reward(self, action: int) -> int:
        """Calculates the reward for a given action based on the deck's properties."""
        deck = self._deck_properties[action]
        self._deck_pull_counts[action] += 1
        
        reward = deck['reward']
        
        # Check if a penalty should be applied on this pull
        if self._deck_pull_counts[action] % deck['penalty_freq'] == 0:
            reward += deck['penalty'] # penalty added to the reward (which is positive)
            
        return reward



    def reset(self) -> dict:
        """Resets the environment to its initial state for a new episode."""
        self._current_trial = 0
        self.cumulative_reward = 0
        self._last_action = -1
        self._action_history = deque(maxlen=10)
        self._deck_pull_counts = {i: 0 for i in self.action_space} # intiialzing the counts of deck pulls (dict)_
        return self._get_state()



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
        info = {}

        return next_state, reward, done, info

    def _get_state(self) -> dict:
        """Constructs the state dictionary from the environment's current properties."""
        return {
            'trial': self._current_trial,
            'cumulative_reward': self.cumulative_reward,
            'last_action': self._last_action,
            'history': list(self._action_history)
        }