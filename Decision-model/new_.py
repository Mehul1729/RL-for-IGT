# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize
# import sys
# import warnings
# import time
# import wandb

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # ===============================
# # 1. MODEL CLASSES
# # ===============================

# class LimbicFit:
#     def __init__(self, n_actions, alpha, decay, loss_aversion):
#         self.n_actions = n_actions
#         self.alpha = alpha
#         self.decay = decay
#         self.loss_aversion = loss_aversion
#         self.q_table = np.zeros(n_actions)
#         self.sensitivity = 0.5

#     def pvl(self, reward):
#         if reward >= 0:
#             return np.power(reward, self.sensitivity)
#         else:
#             return -self.loss_aversion * np.power(abs(reward), self.sensitivity)

#     def update(self, action, reward):
#         self.q_table = self.q_table * self.decay
#         utility = self.pvl(reward)
#         current_val = self.q_table[action]
#         pe = utility - current_val
#         self.q_table[action] = current_val + self.alpha * pe
#         return abs(pe)


# class PFCFit:
#     def __init__(self, n_actions, lr=0.1, gamma=0.95):
#         self.n_actions = n_actions
#         self.lr = lr
#         self.gamma = gamma
#         self.q_table = {}

#     def get_state_index(self, deck_counts):
#         return tuple(np.array(deck_counts) // 5)

#     def get_q(self, state):
#         return np.array([self.q_table.get((state, a), 0.0)
#                          for a in range(self.n_actions)])

#     def update(self, state, action, reward, next_state):
#         old_val = self.q_table.get((state, action), 0.0)
#         next_qs = [self.q_table.get((next_state, a), 0.0)
#                    for a in range(self.n_actions)]
#         next_max = max(next_qs) if next_qs else 0.0

#         target = reward + self.gamma * next_max
#         pe = target - old_val
#         self.q_table[(state, action)] = old_val + self.lr * pe
#         return abs(pe)


# # ===============================
# # 2. NEGATIVE LOG LIKELIHOOD
# # ===============================

# def calculate_nll(params, choices, rewards, n_actions=4):
#     alpha, decay, lam, t_pfc, t_limbic, arb_lr = params

#     if t_pfc < 0.01 or t_limbic < 0.01:
#         return 1e9

#     limbic = LimbicFit(n_actions, alpha, decay, lam)
#     pfc = PFCFit(n_actions)

#     rel_pfc = 0.5
#     rel_limbic = 0.5

#     deck_counts = np.zeros(n_actions, dtype=int)
#     total_nll = 0.0

#     for choice, reward in zip(choices, rewards):
#         action = int(choice)
#         if action < 0 or action >= n_actions:
#             continue

#         state_pfc = pfc.get_state_index(deck_counts)
#         q_pfc = pfc.get_q(state_pfc)
#         q_limbic = limbic.q_table

#         pfc_logits = q_pfc / t_pfc
#         p_pfc = np.exp(pfc_logits - np.max(pfc_logits))
#         p_pfc /= np.sum(p_pfc)

#         limbic_logits = q_limbic / t_limbic
#         p_limbic = np.exp(limbic_logits - np.max(limbic_logits))
#         p_limbic /= np.sum(p_limbic)

#         rel_scores = np.array([rel_pfc, rel_limbic])
#         w_logits = rel_scores * 5.0
#         weights = np.exp(w_logits - np.max(w_logits))
#         weights /= np.sum(weights)
#         beta_pfc = weights[0]

#         final_probs = beta_pfc * p_pfc + (1 - beta_pfc) * p_limbic
#         prob_chosen = max(final_probs[action], 1e-9)
#         total_nll -= np.log(prob_chosen)

#         next_counts = deck_counts.copy()
#         next_counts[action] += 1
#         next_state_pfc = pfc.get_state_index(next_counts)

#         pe_pfc = pfc.update(state_pfc, action, reward, next_state_pfc)
#         pe_limbic = limbic.update(action, reward)

#         norm_pe_pfc = np.clip(pe_pfc / 1350.0, 0, 1)
#         norm_pe_limbic = np.clip(pe_limbic / 100.0, 0, 1)

#         rel_pfc += arb_lr * ((1.0 - norm_pe_pfc) - rel_pfc)
#         rel_limbic += arb_lr * ((1.0 - norm_pe_limbic) - rel_limbic)

#         deck_counts = next_counts

#     return total_nll


# # ===============================
# # 3. GROUP PROCESSING
# # ===============================

# def process_group(filepath, group_name):
#     print(f"\n--- Processing {group_name} ---")
#     df = pd.read_csv(filepath)

#     if 'deck' in df.columns and df['deck'].min() == 1:
#         df['deck'] -= 1

#     if 'gain' in df.columns and 'loss' in df.columns:
#         df['net_reward'] = df['gain'] + df['loss']

#     subjects = df['subjID'].unique()
#     print(f"Found {len(subjects)} subjects.")

#     group_results = []

#     for i, sub in enumerate(subjects):
#         sys.stdout.write(f"\rFitting {i+1}/{len(subjects)} (ID: {sub})...")
#         sys.stdout.flush()

#         sub_data = df[df['subjID'] == sub]
#         choices = sub_data['deck'].values
#         rewards = sub_data['net_reward'].values

#         x0 = [0.1, 0.5, 1.0, 10.0, 2.0, 0.05]
#         bounds = [
#             (0.01, 1.0),
#             (0.01, 1.0),
#             (0.01, 5.0),
#             (0.1, 100.0),
#             (0.1, 20.0),
#             (0.01, 0.5)
#         ]

#         res = minimize(
#             calculate_nll, x0,
#             args=(choices, rewards, 4),
#             method="L-BFGS-B",
#             bounds=bounds,
#             options={"maxiter": 50}
#         )

#         params = res.x

#         row = {
#             "Subject": sub,
#             "Group": group_name,
#             "NLL": res.fun,
#             "Alpha_Limbic": params[0],
#             "Decay": params[1],
#             "Loss_Aversion": params[2],
#             "PFC_Temp": params[3],
#             "Limbic_Temp": params[4],
#             "Arbitration_Rate": params[5],
#             "Temp_Ratio": params[3] / (params[4] + 1e-9)
#         }

#         group_results.append(row)

#         # ---- wandb SUBJECT-LEVEL LOGGING ----
#         wandb.log({
#             "Group": group_name,
#             "NLL": row["NLL"],
#             "Alpha_Limbic": row["Alpha_Limbic"],
#             "Decay": row["Decay"],
#             "Loss_Aversion": row["Loss_Aversion"],
#             "PFC_Temp": row["PFC_Temp"],
#             "Limbic_Temp": row["Limbic_Temp"],
#             "Arbitration_Rate": row["Arbitration_Rate"],
#             "Temp_Ratio": row["Temp_Ratio"]
#         })

#     print("\nDone.")
#     return pd.DataFrame(group_results)


# # ===============================
# # 4. RUN SCRIPT
# # ===============================

# if __name__ == "__main__":

#     wandb.init(
#         project="IGT-Arbitration-Model",
#         name="Limbic-PFC-Fit",
#         config={
#             "optimizer": "L-BFGS-B",
#             "maxiter": 50,
#             "n_actions": 4
#         }
#     )

#     base_path = r"C:\Users\mehul\OneDrive\Desktop\IGT datasets\rawData"
#     all_data = []

#     res_amp = process_group(
#         f"{base_path}\\IGTdata_amphetamine.csv", "Amphetamine"
#     )
#     if not res_amp.empty:
#         all_data.append(res_amp)

#     res_hc = process_group(
#         f"{base_path}\\IGTdata_healthy_control.csv", "Control"
#     )
#     if not res_hc.empty:
#         all_data.append(res_hc)

#     res_her = process_group(
#         f"{base_path}\\IGTdata_heroin.csv", "Heroin"
#     )
#     if not res_her.empty:
#         all_data.append(res_her)

#     if all_data:
#         df_results = pd.concat(all_data)
#         df_results.to_csv("IGT_Biomarkers_Estimated.csv", index=False)

#         summary = df_results.groupby("Group").mean(numeric_only=True)

#         # ---- wandb GROUP-LEVEL LOGGING ----
#         for group, row in summary.iterrows():
#             wandb.log({
#                 f"{group}/Alpha_Limbic_mean": row["Alpha_Limbic"],
#                 f"{group}/Decay_mean": row["Decay"],
#                 f"{group}/Loss_Aversion_mean": row["Loss_Aversion"],
#                 f"{group}/Temp_Ratio_mean": row["Temp_Ratio"]
#             })

#         print("\n✅ SAVED: IGT_Biomarkers_Estimated.csv")
#         print(summary)

#     wandb.finish()



import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# ===============================
# 0. WANDB INITIALIZATION
# ===============================

wandb.init(
    project="IGT-Arbitration-Model",
    name="Inference-Debug-Run",
    config={
        "model": "Dual-system RL (Limbic + PFC)",
        "optimizer": "L-BFGS-B",
        "n_actions": 4,
        "limbic_sensitivity": 0.5,
        "pfc_gamma": 0.95
    }
)

# ===============================
# 1. MODEL CLASSES
# ===============================

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
        return np.array([self.q_table.get((state, a), 0.0)
                         for a in range(self.n_actions)])

    def update(self, state, action, reward, next_state):
        old_val = self.q_table.get((state, action), 0.0)
        next_qs = [self.q_table.get((next_state, a), 0.0)
                   for a in range(self.n_actions)]
        next_max = max(next_qs)
        target = reward + self.gamma * next_max
        pe = target - old_val
        self.q_table[(state, action)] = old_val + self.lr * pe
        return abs(pe)

# ===============================
# 2. NEGATIVE LOG LIKELIHOOD
# ===============================

def calculate_nll(params, choices, rewards, n_actions=4, i=0):
    alpha, decay, lam, t_pfc, t_limbic, arb_lr = params

    if t_pfc < 0.01 or t_limbic < 0.01:
        return 1e9

    limbic = LimbicFit(n_actions, alpha, decay, lam)
    pfc = PFCFit(n_actions)

    rel_pfc = 0.5
    rel_limbic = 0.5
    deck_counts = np.zeros(n_actions, dtype=int)
    total_nll = 0.0

    for choice, reward in zip(choices, rewards):
        action = int(choice)

        state_pfc = pfc.get_state_index(deck_counts)
        q_pfc = pfc.get_q(state_pfc)
        q_limbic = limbic.get_q()

        p_pfc = np.exp(q_pfc / t_pfc - np.max(q_pfc / t_pfc))
        p_pfc /= np.sum(p_pfc)

        p_limbic = np.exp(q_limbic / t_limbic - np.max(q_limbic / t_limbic))
        p_limbic /= np.sum(p_limbic)

        weights = np.exp(np.array([rel_pfc, rel_limbic]) * 5.0)
        weights /= np.sum(weights)
        beta_pfc = weights[0]

        final_probs = beta_pfc * p_pfc + (1 - beta_pfc) * p_limbic
        total_nll -= np.log(final_probs[action] + 1e-9)

        next_counts = deck_counts.copy()
        next_counts[action] += 1
        next_state = pfc.get_state_index(next_counts)

        pe_pfc = pfc.update(state_pfc, action, reward, next_state)
        pe_limbic = limbic.update(action, reward)

        rel_pfc += arb_lr * ((1 - pe_pfc / 1350.0) - rel_pfc)
        rel_limbic += arb_lr * ((1 - pe_limbic / 100.0) - rel_limbic)

        deck_counts = next_counts
        i += 1

    return total_nll

# ===============================
# 3. FITTING PIPELINE
# ===============================

def process_and_fit(filepath, group_name):
    df = pd.read_csv(filepath)

    if df['deck'].min() == 1:
        df['deck'] -= 1

    df['net_reward'] = df['gain'] + df['loss']
    subjects = df['subjID'].unique()
    results = []

    for sub in subjects:
        sub_data = df[df['subjID'] == sub]
        choices = sub_data['deck'].values
        rewards = sub_data['net_reward'].values

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
            calculate_nll, x0,
            args=(choices, rewards, 4),
            method='L-BFGS-B',
            bounds=bounds
        )

        row = {
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

        results.append(row)

        # ---------- WANDB SUBJECT LOG ----------
        wandb.log({
            "Group": group_name,
            "NLL": row["NLL"],
            "Alpha_Limbic": row["Alpha_Limbic"],
            "Decay": row["Decay"],
            "Lambda_Loss": row["Lambda_Loss"],
            "Temp_PFC": row["Temp_PFC"],
            "Temp_Limbic": row["Temp_Limbic"],
            "Temp_Ratio": row["Temp_Ratio"],
            "Arb_Rate": row["Arb_Rate"]
        })

    return pd.DataFrame(results)

# ===============================
# 4. RUN ALL GROUPS
# ===============================

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

# ---------- WANDB GROUP SUMMARY ----------
summary = all_results.groupby("Group").mean(numeric_only=True)
for group, row in summary.iterrows():
    wandb.log({
        f"{group}/Lambda_Loss_mean": row["Lambda_Loss"],
        f"{group}/Decay_mean": row["Decay"],
        f"{group}/Temp_Ratio_mean": row["Temp_Ratio"],
        f"{group}/Arb_Rate_mean": row["Arb_Rate"]
    })

print("\n--- BIOMARKER EXTRACTION COMPLETE ---")
print(summary)

wandb.finish()
