import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. MODEL CLASSES (Same as before) ---
class LimbicFit:
    def __init__(self, n_actions, alpha, decay, loss_aversion):
        self.n_actions = n_actions
        self.alpha = alpha
        self.decay = decay
        self.loss_aversion = loss_aversion
        self.q_table = np.zeros(n_actions) 
        self.sensitivity = 0.5 

    def pvl(self, reward):
        if reward >= 0:
            return reward ** self.sensitivity
        else:
            return -self.loss_aversion * (abs(reward) ** self.sensitivity)

    def update(self, action, reward):
        self.q_table *= self.decay
        utility = self.pvl(reward)
        current_val = self.q_table[action]
        pe = utility - current_val
        self.q_table[action] += self.alpha * pe
        return abs(pe)

class PFCFit:
    def __init__(self, n_actions, lr=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.q_table = {} 

    def get_state_index(self, deck_counts):
        return tuple(deck_counts // 5)

    def get_q(self, state):
        q = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q[a] = self.q_table.get((state, a), 0.0)
        return q

    def update(self, state, action, reward, next_state):
        old_val = self.q_table.get((state, action), 0.0)
        next_max = 0.0
        for a in range(self.n_actions):
            val = self.q_table.get((next_state, a), 0.0)
            if val > next_max: next_max = val
        target = reward + self.gamma * next_max
        pe = target - old_val
        self.q_table[(state, action)] = old_val + self.lr * pe
        return abs(pe) 

def calculate_nll(params, choices, rewards, n_actions=4):
    alpha, decay, lam, t_pfc, t_limbic, arb_lr = params
    if t_pfc < 0.01 or t_limbic < 0.01: return 1e9
    
    limbic = LimbicFit(n_actions, alpha, decay, lam)
    pfc = PFCFit(n_actions) 
    rel_pfc = 0.5
    rel_limbic = 0.5
    deck_counts = np.zeros(n_actions, dtype=int)
    total_nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        action = int(choice)
        if action < 0 or action >= n_actions: continue
        
        state_pfc = pfc.get_state_index(deck_counts)
        q_pfc = pfc.get_q(state_pfc)
        q_limbic = limbic.q_table
        
        # Softmax
        pfc_logits = q_pfc / t_pfc
        pfc_logits -= np.max(pfc_logits)
        p_pfc = np.exp(pfc_logits)
        p_pfc /= np.sum(p_pfc)
        
        limbic_logits = q_limbic / t_limbic
        limbic_logits -= np.max(limbic_logits)
        p_limbic = np.exp(limbic_logits)
        p_limbic /= np.sum(p_limbic)
        
        rel_scores = np.array([rel_pfc, rel_limbic])
        w_logits = rel_scores * 5.0
        w_logits -= np.max(w_logits)
        weights = np.exp(w_logits)
        weights /= np.sum(weights)
        beta_pfc = weights[0]
        
        final_probs = (beta_pfc * p_pfc) + ((1 - beta_pfc) * p_limbic)
        
        prob_chosen = final_probs[action]
        if prob_chosen < 1e-9: prob_chosen = 1e-9
        total_nll -= np.log(prob_chosen)
        
        next_counts = deck_counts.copy()
        next_counts[action] += 1
        next_state_pfc = pfc.get_state_index(next_counts)
        
        pe_pfc = pfc.update(state_pfc, action, reward, next_state_pfc)
        pe_limbic = limbic.update(action, reward)
        
        norm_pe_pfc = np.clip(pe_pfc / 1350.0, 0, 1)
        norm_pe_limbic = np.clip(pe_limbic / 100.0, 0, 1)
        
        rel_pfc += arb_lr * ((1.0 - norm_pe_pfc) - rel_pfc)
        rel_limbic += arb_lr * ((1.0 - norm_pe_limbic) - rel_limbic)
        deck_counts = next_counts
        
    return total_nll

# --- 3. DIAGNOSTIC MONTE CARLO ---

def monte_carlo_diagnostic(choices, rewards, n_iters=500):
    best_nll = float('inf')
    best_params = None
    history = []
    
    # Baseline Random NLL (Assuming 4 decks)
    random_nll = len(choices) * -np.log(0.25) 
    
    # 1. Generate Population
    alphas = np.random.uniform(0.01, 1.0, n_iters)
    decays = np.random.uniform(0.01, 1.0, n_iters)
    lambdas = np.random.uniform(0.1, 1.0, n_iters)
    
    # CRITICAL CHANGE: Tighten Temp bounds. If Temp is too high, NLL stays at 138.
    t_pfcs = np.random.uniform(0.1, 100.0, n_iters) 
    t_limbics = np.random.uniform(0.1, 50.0, n_iters)
    
    arb_lrs = np.random.uniform(0.01, 0.5, n_iters)
    
    for i in range(n_iters):
        params = [alphas[i], decays[i], lambdas[i], t_pfcs[i], t_limbics[i], arb_lrs[i]]
        
        try:
            nll = calculate_nll(params, choices, rewards)
            if nll < best_nll:
                best_nll = nll
                best_params = params
            
            # Record current best
            history.append(best_nll)
            
        except:
            history.append(best_nll)
            continue
            
    return best_params, best_nll, history, random_nll

# --- 4. EXECUTION WITH PLOTTING ---

def process_single_subject_debug(filepath):
    try:
        df = pd.read_csv(filepath)
    except:
        print("File not found")
        return

    if df['deck'].min() == 1: df['deck'] = df['deck'] - 1
    df['net_reward'] = df['gain'] + df['loss']
    
    # Pick the first subject
    sub = df['subjID'].unique()[0]
    print(f"--- Diagnosing Subject {sub} ---")
    
    sub_data = df[df['subjID'] == sub]
    choices = sub_data['deck'].values
    rewards = sub_data['net_reward'].values
    
    print("Running 10000 iterations...")
    params, nll, history, random_nll = monte_carlo_diagnostic(choices, rewards, n_iters=10000)
    
    # --- METRICS ---
    pseudo_r2 = 1 - (nll / random_nll)
    
    print(f"\nFinal NLL: {nll:.2f}")
    print(f"Random Guessing Baseline: {random_nll:.2f}")
    print(f"Pseudo-R2: {pseudo_r2:.4f}")
    
    if pseudo_r2 < 0.05:
        print("⚠️  WARNING: Model is barely beating random chance. Check Bounds.")
    else:
        print("✅ SUCCESS: Model is learning structural patterns.")
        
    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best NLL So Far')
    plt.axhline(y=random_nll, color='r', linestyle='--', label='Random Chance')
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log Likelihood')
    plt.title(f'Convergence Diagnostic (Subject {sub})')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_plot.png')
    print("Graph saved to 'convergence_plot.png'")

if __name__ == "__main__":
    # Point to ONE file to test
    path = r"C:\Users\mehul\OneDrive\Desktop\IGT datasets\rawData\IGTdata_amphetamine.csv"
    process_single_subject_debug(path)