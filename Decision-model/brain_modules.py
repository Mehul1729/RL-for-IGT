

import numpy as np
import pandas as pd
import sys
import time
import warnings


# limbic system class :

class LimbicFit:
    def __init__(self, n_actions, alpha, decay, loss_aversion):
        # properties of a limbic system mimic :
        self.n_actions = n_actions # types of actions (picks)
        self.alpha = alpha # learning rate for the limbic (how fast limbic updates its values)
        self.decay = decay # memory decay factor (somatic marker fading)
        self.loss_aversion = loss_aversion # : PVL loss aversion multiplier (how strongly losses hurt relative to gains)
        self.q_table = np.zeros(n_actions) # stateless q table to mimic the impulsivity of limbic system (only cares about immediate rewards)
        self.sensitivity = 0.5

# PVL theory based loss aversion in the limbic system:
    def pvl(self, reward):
        if reward >= 0:
            return reward ** self.sensitivity # gains are processed with diminishing sensitivity (concave)
        else:
            return -self.loss_aversion * (abs(reward) ** self.sensitivity) # losses are processed with loss aversion and diminishing sensitivity (convex and steeper)


    def update(self, action, reward):
        # Decay
        self.q_table *= self.decay # to simulate the action-memory decay 
        
        # Update
        utility = self.pvl(reward)
        current_val = self.q_table[action]
        pe = utility - current_val 
        self.q_table[action] += self.alpha * pe # updating the q values
        return abs(pe)

class PFCFit:
    def __init__(self, n_actions, lr=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.q_table = {} 

    def get_state_index(self, deck_counts):
        return tuple(deck_counts // 5) # we bucket the counts into 5s to reduce state space (0-4, 5-9, etc.)

    # usual state-action q-value stable updation:
    def get_q(self, state):
        q = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q[a] = self.q_table.get((state, a), 0.0)
        return q

    def update(self, state, action, reward, next_state):
        old_val = self.q_table.get((state, action), 0.0)
        
        # Manual max for speed
        next_max = 0.0
        for a in range(self.n_actions):
            val = self.q_table.get((next_state, a), 0.0)
            if val > next_max: next_max = val
            
        target = reward + self.gamma * next_max # q*
        pe = target - old_val
        self.q_table[(state, action)] = old_val + self.lr * pe
        return abs(pe) 
