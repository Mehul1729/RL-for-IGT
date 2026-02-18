# =========================
# file 2: nll.py
# =========================

import numpy as np
import pandas as pd
import sys
import time
import warnings

from brain_modules import LimbicFit, PFCFit



# -------------Fitting the data to the dual system model----------------:

def calculate_nll(params, choices, rewards, n_actions=4):
    """
    For each trial, it computes the model's probability of the action the human took,
    adds -log(prob) to a running total, 
    then updates the model's internal states exactly like learning would happen in that trial.
    """
    # Unpacking the params:
    alpha, decay, lam, t_pfc, t_limbic, arb_lr = params # lr for limbic system, decay rate of memory (somatic marker), loss aversion (PVL), temperature for pfc, temperature for limbic, learning rate for arbitration
    
    # Initialsing Agents
    limbic = LimbicFit(n_actions, alpha, decay, lam)
    pfc = PFCFit(n_actions)
    
    # we take the default reliabilities of the pfc and limbic q values as equal:
    rel_pfc = 0.5
    rel_limbic = 0.5
    deck_counts = np.zeros(n_actions, dtype=int)
    total_nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        action = int(choice)
        if action < 0 or action >= n_actions: continue
        
        # 1. Get Values
        state_pfc = pfc.get_state_index(deck_counts)
        q_pfc = pfc.get_q(state_pfc)
        q_limbic = limbic.q_table
        
        # 2. Softmax: converting the q valuues to probabilities for action selection:
        # PFC
        pfc_logits = q_pfc / t_pfc
        pfc_logits -= np.max(pfc_logits) # shift for stability
        p_pfc = np.exp(pfc_logits)
        p_pfc /= np.sum(p_pfc)
        
        # Limbic
        limbic_logits = q_limbic / t_limbic
        limbic_logits -= np.max(limbic_logits)
        p_limbic = np.exp(limbic_logits)
        p_limbic /= np.sum(p_limbic)
        
        # Router
        rel_scores = np.array([rel_pfc, rel_limbic])
        w_logits = rel_scores * 5.0 # to make it sensitive to minor differences in reliabilities
        w_logits -= np.max(w_logits)
        weights = np.exp(w_logits)
        weights /= np.sum(weights)
        beta_pfc = weights[0] # represents the coeff of pfc dominance
        
        # Mixture
        final_probs = (beta_pfc * p_pfc) + ((1 - beta_pfc) * p_limbic)
        
        # 3. Score
        prob_chosen = final_probs[action]
        if prob_chosen < 1e-9: prob_chosen = 1e-9
        total_nll -= np.log(prob_chosen) # accumulating the  log likelihood
        
        # 4. Learning
        next_counts = deck_counts.copy()
        next_counts[action] += 1
        next_state_pfc = pfc.get_state_index(next_counts)
        
        pe_pfc = pfc.update(state_pfc, action, reward, next_state_pfc)
        pe_limbic = limbic.update(action, reward)
        
        # Router Update
        n_pe_pfc = pe_pfc / 1350.0
        if n_pe_pfc > 1.0: n_pe_pfc = 1.0
        
        n_pe_limbic = pe_limbic / 100.0
        if n_pe_limbic > 1.0: n_pe_limbic = 1.0
        
        rel_pfc += arb_lr * ((1.0 - n_pe_pfc) - rel_pfc)
        rel_limbic += arb_lr * ((1.0 - n_pe_limbic) - rel_limbic)
        
        deck_counts = next_counts
        
    return total_nll
