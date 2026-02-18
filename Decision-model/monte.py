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



# -------------Fitting the data to the dual system model----------------:

def calculate_nll(params, choices, rewards, n_actions=4):
    """
    Returns Negative Log Likelihood. Lower is better.
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
        
        # 2. Softmax (Manual stable implementation)
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
        w_logits = rel_scores * 5.0
        w_logits -= np.max(w_logits)
        weights = np.exp(w_logits)
        weights /= np.sum(weights)
        beta_pfc = weights[0]
        
        # Mixture
        final_probs = (beta_pfc * p_pfc) + ((1 - beta_pfc) * p_limbic)
        
        # 3. Score
        prob_chosen = final_probs[action]
        if prob_chosen < 1e-9: prob_chosen = 1e-9
        total_nll -= np.log(prob_chosen)
        
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

# --- Monte Carlo Optimizer---

def monte_carlo_search(choices, rewards, n_iters=100):
    """
    Randomly samples parameters and returns the best set.
    Guaranteed to finish.
    """
    best_nll = float('inf')
    best_params = None
    
    # Generate random parameters (Vectorized)
    # alpha, decay, lambda, t_pfc, t_limbic, arb_lr
    alphas = np.random.uniform(0.01, 1.0, n_iters)
    decays = np.random.uniform(0.01, 1.0, n_iters)
    lambdas = np.random.uniform(0.1, 5.0, n_iters)
    t_pfcs = np.random.uniform(1.0, 20.0, n_iters)
    t_limbics = np.random.uniform(0.1, 10.0, n_iters)
    arb_lrs = np.random.uniform(0.01, 0.5, n_iters)
    
    for i in range(n_iters):
        params = [alphas[i], decays[i], lambdas[i], t_pfcs[i], t_limbics[i], arb_lrs[i]]
        
        try:
            nll = calculate_nll(params, choices, rewards)
            if nll < best_nll:
                best_nll = nll
                best_params = params
        except:
            continue
            
    return best_params, best_nll

# --- 4. EXECUTION LOOP ---

def process_group(filepath, group_name):
    print(f"\n--- Processing {group_name} ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return pd.DataFrame()

    # Preprocessing
    if 'deck' in df.columns:
        if df['deck'].min() == 1:
            df['deck'] = df['deck'] - 1
            
    if 'gain' in df.columns and 'loss' in df.columns:
        df['net_reward'] = df['gain'] + df['loss']
    else:
        print("❌ Error: Gain/Loss columns missing")
        return pd.DataFrame()
    
    subjects = df['subjID'].unique()
    print(f"Found {len(subjects)} subjects. Running Monte Carlo Sweep...")
    
    group_results = []
    
    for i, sub in enumerate(subjects):
        # Explicit Progress Bar
        sys.stdout.write(f"\rSubject {i+1}/{len(subjects)} (ID: {sub})...")
        sys.stdout.flush()
        
        sub_data = df[df['subjID'] == sub]
        choices = sub_data['deck'].values
        rewards = sub_data['net_reward'].values
        
        # RUN MONTE CARLO (100 random guesses per subject)
        # It's fast and cannot hang.
        params, nll = monte_carlo_search(choices, rewards, n_iters=1000)
        
        if params is not None:
            r = {
                'Subject': sub, 'Group': group_name, 'NLL': nll,
                'Alpha_Limbic': params[0],
                'Decay': params[1],
                'Loss_Aversion': params[2],
                'PFC_Temp': params[3],
                'Limbic_Temp': params[4],
                'Arbitration_Rate': params[5],
                'Temp_Ratio': params[3] / (params[4] + 1e-9)
            }
            group_results.append(r)

    print("\nDone.")
    return pd.DataFrame(group_results)

# --- RUN ---
if __name__ == "__main__":
    # PATHS - Update these!
    base_path = r"C:\Desktop folder\IGT datasets\rawData"

    df_amp = process_group(f"{base_path}\\IGTdata_amphetamine.csv", 'Amphetamine')
    df_hc = process_group(f"{base_path}\\IGTdata_healthy_control.csv", 'Control')
    df_her = process_group(f"{base_path}\\IGTdata_heroin.csv", 'Heroin')

    # Combine & Save
    all_data = []
    if not df_amp.empty: all_data.append(df_amp)
    if not df_hc.empty: all_data.append(df_hc)
    if not df_her.empty: all_data.append(df_her)

    if all_data:
        all_results = pd.concat(all_data)
        all_results.to_csv('IGT_Biomarkers_MonteCarlo.csv', index=False)
        print("\n✅ SAVED: IGT_Biomarkers_MonteCarlo.csv")
        
        print("\n--- RESULTS SUMMARY (Means) ---")
        print(all_results.groupby('Group')[['Lambda_Loss', 'Decay', 'Temp_Ratio', 'Alpha_Limbic']].mean())
    else:
        print("\n❌ No data processed.")
        
        
        
        
        
    