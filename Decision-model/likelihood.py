

import numpy as np
import pandas as pd
import sys
import time
import warnings

from brain_modules import LimbicFit, PFCFit


def calculate_nll(params, choices, rewards, n_actions=4):
    
    alpha, decay, lam, beta_pfc = params
    
    t_pfc = 1.0 
    t_limbic = 1.0 
    
    limbic = LimbicFit(n_actions, alpha, decay, lam)
    pfc = PFCFit(n_actions) 
    
    deck_counts = np.zeros(n_actions, dtype=int)
        # we take the default reliabilities of the pfc and limbic q values as equal:
    rel_pfc = 0.5
    rel_limbic = 0.5
    total_nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        action = int(choice)
        if action < 0 or action >= n_actions: continue
        
        # 1. Get Q-Values
        state_pfc = pfc.get_state_index(deck_counts)
        q_pfc = pfc.get_q(state_pfc)
        q_limbic = limbic.q_table
        
        # 2. Softmax
        pfc_logits = q_pfc / t_pfc
        pfc_logits -= np.max(pfc_logits)
        p_pfc = np.exp(pfc_logits)
        p_pfc /= np.sum(p_pfc)
        
        limbic_logits = q_limbic / t_limbic
        limbic_logits -= np.max(limbic_logits)
        p_limbic = np.exp(limbic_logits)
        p_limbic /= np.sum(p_limbic)
        
        # 3. Using the fitted beta_pfc directly:
        final_probs = (beta_pfc * p_pfc) + ((1.0 - beta_pfc) * p_limbic)
        
        # 4. Score
        prob_chosen = final_probs[action]
        if prob_chosen < 1e-9: prob_chosen = 1e-9
        total_nll -= np.log(prob_chosen)
        
        # 5. Learning (Update systems so they track the human's history)
        next_counts = deck_counts.copy()
        next_counts[action] += 1
        next_state_pfc = pfc.get_state_index(next_counts)
        
        pfc.update(state_pfc, action, reward, next_state_pfc)
        limbic.update(action, reward)
        
        deck_counts = next_counts
        
    return total_nll


