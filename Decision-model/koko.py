import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# ============================================================
# 0. WANDB INITIALIZATION
# ============================================================

wandb.init(
    project="IGT-Arbitration-Model",
    name="DualSystem_Inference",
    config={
        "optimizer": "L-BFGS-B",
        "n_actions": 4,
        "model": "Limbic-PFC Arbitration",
        "pfc_lr": 0.1,
        "pfc_gamma": 0.95
    }
)

# ============================================================
# 1. MODEL CLASSES (UNCHANGED)
# ============================================================

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
            return np.power(reward, self.sensitivity)
        else:
            return -self.loss_aversion * np.power(abs(reward), self.sensitivity)

    def get_q(self):
        return self.q_table

    def update(self, action, reward):
        self.q_table = self.q_table * self.decay
        utility = self.pvl(reward)
        current_val = self.q_table[action]
        pe = utility - current_val
        self.q_table[action] = current_val + self.alpha * pe
        return abs(pe)

class PFCFit:
    def __init__(self, n_actions, lr=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.q_table = {}

    def get_state_index(self, deck_counts):
        return tuple(np.array(deck_counts) // 5)

    def get_q(self, state):
        return np.array([self.q_table.get((state, a), 0.0) for a in range(self.n_actions)])

    def update(self, state, action, reward, next_state):
        old_val = self.q_table.get((state, action), 0.0)
        next_qs = [self.q_table.get((next_state, a), 0.0) for a in range(self.n_actions)]
        target = reward + self.gamma * max(next_qs)
        pe = target - old_val
        self.q_table[(state, action)] = old_val + self.lr * pe
        return abs(pe)

# ============================================================
# 2. NEGATIVE LOG LIKELIHOOD
# ============================================================

def calculate_nll(params, choices, rewards, n_actions=4):
    alpha, decay, lam, t_pfc, t_limbic, arb_lr = params

    if t_pfc < 0.01 or t_limbic < 0.01:
        return 1e9

    limbic = LimbicFit(n_actions, alpha, decay, lam)
    pfc = PFCFit(n_actions)

    rel_pfc, rel_limbic = 0.5, 0.5
    deck_counts = np.zeros(n_actions, dtype=int)
    total_nll = 0.0

    for action, reward in zip(choices, rewards):
        state = pfc.get_state_index(deck_counts)

        q_pfc = pfc.get_q(state)
        q_limbic = limbic.get_q()

        p_pfc = np.exp(q_pfc / t_pfc - np.max(q_pfc / t_pfc))
        p_pfc /= p_pfc.sum()

        p_limbic = np.exp(q_limbic / t_limbic - np.max(q_limbic / t_limbic))
        p_limbic /= p_limbic.sum()

        weights = np.exp(np.array([rel_pfc, rel_limbic]) * 5.0)
        weights /= weights.sum()
        beta_pfc = weights[0]

        final_probs = beta_pfc * p_pfc + (1 - beta_pfc) * p_limbic
        total_nll -= np.log(final_probs[action] + 1e-9)

        next_counts = deck_counts.copy()
        next_counts[action] += 1
        next_state = pfc.get_state_index(next_counts)

        pe_pfc = pfc.update(state, action, reward, next_state)
        pe_limbic = limbic.update(action, reward)

        rel_pfc += arb_lr * ((1 - np.clip(pe_pfc / 1350.0, 0, 1)) - rel_pfc)
        rel_limbic += arb_lr * ((1 - np.clip(pe_limbic / 100.0, 0, 1)) - rel_limbic)

        deck_counts = next_counts

    return total_nll

# ============================================================
# 3. FITTING PIPELINE WITH WANDB LOGGING
# ============================================================

def process_and_fit(filepath, group_name):
    print(f"Processing {group_name}")
    df = pd.read_csv(filepath)

    if df['deck'].min() == 1:
        df['deck'] -= 1

    df['net_reward'] = df['gain'] + df['loss']
    subjects = df['subjID'].unique()

    results = []

    for idx, sub in enumerate(subjects, 1):
        print(f"Fitting Subject {idx}/{len(subjects)} | ID: {sub}")

        sub_df = df[df['subjID'] == sub]
        choices = sub_df['deck'].values
        rewards = sub_df['net_reward'].values

        x0 = [0.1, 0.5, 1.0, 10.0, 2.0, 0.05]
        bounds = [
            (0.01, 1.0),
            (0.01, 1.0),
            (0.01, 5.0),
            (0.1, 100.0),
            (0.1, 20.0),
            (0.01, 0.5)
        ]

        res = minimize(
            calculate_nll,
            x0,
            args=(choices, rewards),
            bounds=bounds,
            method="L-BFGS-B"
        )

        r = {
            "Subject": sub,
            "Group": group_name,
            "NLL": res.fun,
            "Alpha_Limbic": res.x[0],
            "Decay": res.x[1],
            "Lambda_Loss": res.x[2],
            "Temp_PFC": res.x[3],
            "Temp_Limbic": res.x[4],
            "Temp_Ratio": res.x[3] / res.x[4],
            "Arb_Rate": res.x[5]
        }

        # -------- WANDB SUBJECT LOG --------
        wandb.log({
            f"{group_name}/NLL": r["NLL"],
            f"{group_name}/Alpha_Limbic": r["Alpha_Limbic"],
            f"{group_name}/Decay": r["Decay"],
            f"{group_name}/Lambda_Loss": r["Lambda_Loss"],
            f"{group_name}/Temp_Ratio": r["Temp_Ratio"],
            f"{group_name}/Arb_Rate": r["Arb_Rate"]
        })

        results.append(r)

    df_res = pd.DataFrame(results)

    # -------- WANDB GROUP AGGREGATES --------
    wandb.log({
        f"{group_name}_Mean_Lambda": df_res["Lambda_Loss"].mean(),
        f"{group_name}_Mean_Decay": df_res["Decay"].mean(),
        f"{group_name}_Mean_TempRatio": df_res["Temp_Ratio"].mean(),
        f"{group_name}_Mean_ArbRate": df_res["Arb_Rate"].mean()
    })

    return df_res

# ============================================================
# 4. RUN ALL GROUPS
# ============================================================

df_amp = process_and_fit(
    r"C:\Desktop folder\IGT datasets\rawData\IGTdata_amphetamine.csv",
    "Amphetamine"
)

df_hc = process_and_fit(
    r"C:\Desktop folder\IGT datasets\rawData\IGTdata_healthy_control.csv",
    "Healthy"
)

df_her = process_and_fit(
    r"C:\Desktop folder\IGT datasets\rawData\IGTdata_heroin.csv",
    "Heroin"
)

all_results = pd.concat([df_amp, df_hc, df_her])
all_results.to_csv("IGT_Biomarkers_Estimated.csv", index=False)

print("\n--- BIOMARKER EXTRACTION COMPLETE ---")
print(all_results.groupby("Group")[["Lambda_Loss", "Decay", "Temp_Ratio", "Arb_Rate"]].mean())

wandb.finish()
