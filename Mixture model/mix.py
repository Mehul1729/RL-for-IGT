import numpy as np
import wandb
from env import IowaEnv

from pfc import PFC_Agent as PFCAgent
from limbic import LimbicAgent_Somatic as LimbicAgent

class DualSystemRouter:
    def __init__(self, n_actions, pfc_params, limbic_params, arbitration_lr=0.05):
        """
        The Brain that manages the conflict between PFC and Limbic systems.
        """
        self.n_actions = n_actions
        
        self.pfc = PFCAgent(n_actions, **pfc_params)
        self.limbic = LimbicAgent(n_actions, **limbic_params)
        
        # Initialize reliabilities (0.5 = neutral trust)
        self.rel_pfc = 0.5
        self.rel_limbic = 0.5
        
        # Learning rate for the Router (how fast it changes its mind about who to trust):
        self.arb_lr = arbitration_lr
        
        # The current calculated weight (for logging)
        self.current_beta_pfc = 0.5 # derived from the reliabilities

    def _softmax(self, q_values, temperature=1.0):
        # converts q values to probabilities 
        # Subtract max for numerical stability
        exp_q = np.exp((q_values - np.max(q_values)) / temperature) # trick for stability to prevent overflow
        return exp_q / np.sum(exp_q)

    def act(self, deck_counts):
        """
        Decide action by mixing the advice of both systems.
        """
        # 1. Get States:
        state_pfc = self.pfc._get_state_index(deck_counts)
        state_limbic = self.limbic._get_state_index(deck_counts) # stateless, actually 
        
        
        # 2. Get Q-Values (Opinions)
        # access the q_tables directly to get the full distribution
        q_pfc = np.array([self.pfc.q_table.get((state_pfc, a), 0.0) for a in range(self.n_actions)])
        q_limbic = np.array([self.limbic.q_table.get((state_limbic, a), 0.0) for a in range(self.n_actions)])
        
        # 3. Convert to Probabilities (Softmax):
        # We can have higher temp here for one of the component to simulate more exploration at the highest level
        prob_pfc = self._softmax(q_pfc, temperature=0.5) # explporation is less in pfc 
        prob_limbic = self._softmax(q_limbic, temperature= 2.0) # explopration is more in limbic 
        
        # 4. Calculate Arbitration Weight (Beta) using Softmax on Reliabilities
        # If rel_pfc >> rel_limbic, beta approaches 1.0
        reliability_scores = np.array([self.rel_pfc, self.rel_limbic])
        weights = self._softmax(reliability_scores * 5) # 5x sharpen the contrast
        self.current_beta_pfc = weights[0] # Weight given to PFC
        
        # 5. MIX THE POLICIES:
        mixed_probs = (self.current_beta_pfc * prob_pfc) + ((1 - self.current_beta_pfc) * prob_limbic)
        
        # 6. Choosing the Action with Highest Probability (we can also sample from a  categorical distribution to add up to the stochasticity)
        
        # action = np.argmax(mixed_probs)
        action = np.random.choice(self.n_actions, p=mixed_probs)
        return action
    
        

    def update(self, deck_counts, action, raw_reward, next_deck_counts):
        """
        Trains both systems simultaneously and update the Router's trust in them to get beta(router training)
        """
        # Get Current States
        s_pfc = self.pfc._get_state_index(deck_counts)
        s_limbic = self.limbic._get_state_index(deck_counts)
        
        # Get Next States
        ns_pfc = self.pfc._get_state_index(next_deck_counts)
        ns_limbic = self.limbic._get_state_index(next_deck_counts)
        
        # 1. Calculating Prediction Errors for each of the ssystemsfor Q learning (also mimics how surprised they are at each of the trials)
        
        # PFC Error (Standard Q-Learning logic)
        q_pfc_curr = self.pfc.q_table.get((s_pfc, action), 0.0)
        q_pfc_next_max = max([self.pfc.q_table.get((ns_pfc, a), 0.0) for a in range(self.n_actions)])
        target_pfc = raw_reward + (self.pfc.gamma * q_pfc_next_max)
        pe_pfc = abs(target_pfc - q_pfc_curr) # Absolute Surprise
        
        # Limbic Error (PVL logic)
        # Note: Limbic uses Utility, not raw reward
        q_limbic_curr = self.limbic.q_table.get((s_limbic, action), 0.0)
        # Remember Limbic decays BEFORE update, but for PE calculation we use current snapshot
        # We calculate "What did Limbic expect?" vs "What did it feel?"
        utility = self.limbic.pvl(raw_reward)
        target_limbic = utility # Myopic Limbic (gamma usually low/zero)
        pe_limbic = abs(target_limbic - q_limbic_curr)
        
        # --- 2. Update Reliabilities (The Router Logic) ---
        
        # Normalize errors roughly to 0-1 range for stability (assuming max reward ~100)
        # norm_pe_pfc = np.clip(pe_pfc / 100.0, 0, 1)
        # norm_pe_limbic = np.clip(pe_limbic / 50.0, 0, 1) # Limbic utility is smaller scale usually
        
        # non scaled losses :
        norm_pe_pfc = pe_pfc / 10.0
        norm_pe_limbic = pe_limbic        
        # learning the reliabilities from the magnitude of surprise (prediction error)
        self.rel_pfc += self.arb_lr * ((1.0 - norm_pe_pfc) - self.rel_pfc)

        # --- 3. Train the Sub-Systems ---
        self.pfc.update(s_pfc, action, raw_reward, ns_pfc)
        self.limbic.update(s_limbic, action, raw_reward, ns_limbic)
        
        
        # --- 4. Returing values for logging to wandb: 
        
        return {
            
        "q_pfc_chosen": q_pfc_curr,
            "q_limbic_chosen": q_limbic_curr,
            "pe_pfc": pe_pfc,
            "pe_limbic": pe_limbic,
            "beta_pfc": self.current_beta_pfc
        }

# --- Driver Code ---
if __name__ == "__main__":
    n_episodes = 500
    
    # Configuration
    config = {
        # config for PFC system
        "pfc": {
            "learning_rate": 0.1,
            "gamma": 0.95,
            "epsilon": 0.0 # Epsilon handled by Softmax in Router
        },
        
        # config for limbic system
        "limbic": {
            "alpha_pos": 0.1,
            "alpha_neg": 0.01,
            "gamma": 0.0, # Gut feeling is immediate
            "epsilon": 0.0, # Epsilon handled by Softmax in Router
            "decay": 0.9 # Somatic marker persistence
        },
        "router_lr": 0.05
    }

    wandb.init(project="IGT-Dual-System", config=config)

    env = IowaEnv(episode_length=100)
    
    # Initialize the Dual System
    brain = DualSystemRouter(
        n_actions=8,
        pfc_params=config["pfc"],
        limbic_params=config["limbic"],
        arbitration_lr=config["router_lr"]
    )

    deck_names = ["A_Bad", "B_Bad", "C_Good", "D_Good", "E_Zero", "F_Gem", "G_Trap", "H_Lossy"]

    for ep in range(n_episodes):
        env.reset()
        
        # Lists to store episode history
        history_beta = []
        history_q_pfc = []
        history_q_limbic = []
        history_pe_pfc = []
        history_pe_limbic = []
        
        for t in range(env.episode_length):
            current_counts = env._deck_pull_counts.copy()
            
            # ACT
            action = brain.act(current_counts)
            
            # STEP
            _, reward, done, _ = env.step(action)
            
            # UPDATE (Now returns metrics)
            next_counts = env._deck_pull_counts.copy()
            metrics = brain.update(current_counts, action, reward, next_counts)
            
            # STORE METRICS
            history_beta.append(metrics["beta_pfc"])
            history_q_pfc.append(metrics["q_pfc_chosen"])
            history_q_limbic.append(metrics["q_limbic_chosen"])
            history_pe_pfc.append(metrics["pe_pfc"])
            history_pe_limbic.append(metrics["pe_limbic"])
            
            if done: break

        # LOGGING (Averaged over the episode)
        log_dict = {}
        log_dict["Total_Reward"] = env.cumulative_reward
        
        # 1. Current Beta (PFC Weight)
        log_dict["Router/Beta_PFC_Weight"] = np.mean(history_beta)
        
        # 2. Q Values (Average magnitude of value assigned to CHOSEN actions)
        log_dict["Internal/Q_PFC_Chosen_Avg"] = np.mean(history_q_pfc)
        log_dict["Internal/Q_Limbic_Chosen_Avg"] = np.mean(history_q_limbic)
        
        # 3. Prediction Errors (Average surprise)
        log_dict["Internal/PE_PFC_Avg"] = np.mean(history_pe_pfc)
        log_dict["Internal/PE_Limbic_Avg"] = np.mean(history_pe_limbic)
        
        # Reliabilities (End of episode snapshot)
        log_dict["Router/Rel_PFC_Final"] = brain.rel_pfc
        log_dict["Router/Rel_Limbic_Final"] = brain.rel_limbic
        
        wandb.log(log_dict)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}: PFC Weight={np.mean(history_beta):.2f}, Score={env.cumulative_reward}")

    wandb.finish()