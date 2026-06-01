import os
os.environ["PYTENSOR_FLAGS"] = "profile=False"

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

import pymc as pm
import pytensor
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import arviz as az

pytensor.config.profile = False

def load_and_combine_groups(amp_path, control_path, heroin_path, n_trials=100):
    df_amp = pd.read_csv(amp_path)
    df_amp['Group_ID'] = 2
    
    df_ctrl = pd.read_csv(control_path)
    df_ctrl['Group_ID'] = 0
    
    df_hero = pd.read_csv(heroin_path)
    df_hero['Group_ID'] = 1
    
    df = pd.concat([df_amp, df_ctrl, df_hero], ignore_index=True)
    df['Choice'] = df['deck'].astype(int) - 1
    df['Net_Reward'] = df['gain'] + df['loss']
    
    unique_subjects = df['subjID'].unique()
    n_subjects = len(unique_subjects)
    
    action_matrix = np.zeros((n_subjects, n_trials), dtype=np.int32)
    reward_matrix = np.zeros((n_subjects, n_trials), dtype=np.float32)
    group_idx = np.zeros(n_subjects, dtype=np.int32)
    
    for i, subj in enumerate(unique_subjects):
        subj_data = df[df['subjID'] == subj].sort_values('trial')
        acts = subj_data['Choice'].values[:n_trials]
        rews = subj_data['Net_Reward'].values[:n_trials]
        action_matrix[i, :len(acts)] = acts
        reward_matrix[i, :len(rews)] = rews
        group_idx[i] = subj_data['Group_ID'].iloc[0]
        
    return action_matrix.T, reward_matrix.T, group_idx, n_subjects


def build_hba_model(actions, rewards, group_idx, n_subjects):
    n_groups = 3
    with pm.Model() as dual_system_hba:
        
        # 1. Group-Level Hyperparameters
        mu_alpha_raw = pm.Normal('mu_alpha_raw', mu=0, sigma=1.5, shape=n_groups)
        sig_alpha_raw = pm.HalfNormal('sig_alpha_raw', sigma=1.0, shape=n_groups)
        
        mu_decay_raw = pm.Normal('mu_decay_raw', mu=0, sigma=1.5, shape=n_groups)
        sig_decay_raw = pm.HalfNormal('sig_decay_raw', sigma=1.0, shape=n_groups)
        
        mu_beta_raw = pm.Normal('mu_beta_raw', mu=0, sigma=1.5, shape=n_groups)
        sig_beta_raw = pm.HalfNormal('sig_beta_raw', sigma=1.0, shape=n_groups)
        
        mu_lambda_raw = pm.Normal('mu_lambda_raw', mu=0.5, sigma=1.0, shape=n_groups)
        sig_lambda_raw = pm.HalfNormal('sig_lambda_raw', sigma=1.0, shape=n_groups)
        
        # 2. Subject-Level Parameters (Non-Centered)
        z_alpha = pm.Normal('z_alpha', mu=0, sigma=1, shape=n_subjects)
        z_decay = pm.Normal('z_decay', mu=0, sigma=1, shape=n_subjects)
        z_beta = pm.Normal('z_beta', mu=0, sigma=1, shape=n_subjects)
        z_lambda = pm.Normal('z_lambda', mu=0, sigma=1, shape=n_subjects)
        
        subj_alpha_raw = mu_alpha_raw[group_idx] + z_alpha * sig_alpha_raw[group_idx]
        subj_decay_raw = mu_decay_raw[group_idx] + z_decay * sig_decay_raw[group_idx]
        subj_beta_raw = mu_beta_raw[group_idx] + z_beta * sig_beta_raw[group_idx]
        subj_lambda_raw = mu_lambda_raw[group_idx] + z_lambda * sig_lambda_raw[group_idx]
        
        subj_alpha = pm.Deterministic('alpha_limbic', pm.math.invlogit(subj_alpha_raw))
        subj_decay = pm.Deterministic('decay', pm.math.invlogit(subj_decay_raw))
        subj_beta = pm.Deterministic('beta_pfc', pm.math.invlogit(subj_beta_raw))
        subj_lambda = pm.Deterministic('loss_aversion', pm.math.exp(subj_lambda_raw))
        
        # 3. Execution Graph
        q_pfc_init = pt.zeros((n_subjects, 4))
        q_limbic_init = pt.zeros((n_subjects, 4))
        
        alpha_b = subj_alpha.dimshuffle(0, 'x')
        decay_b = subj_decay.dimshuffle(0, 'x')
        lambda_b = subj_lambda.dimshuffle(0, 'x')
        beta_b = subj_beta.dimshuffle(0, 'x')
        
        def rl_trial_step(act_t, rew_t, q_pfc_t, q_lim_t, alpha, decay, loss_av, beta):
            q_final = beta * q_pfc_t + (1 - beta) * q_lim_t
            
            q_max = pt.max(q_final, axis=1, keepdims=True)
            exp_q = pt.exp(q_final - q_max)
            probs = exp_q / pt.sum(exp_q, axis=1, keepdims=True)
            
            subj_indices = pt.arange(n_subjects)
            prob_action = probs[subj_indices, act_t]
            
            pe_pfc = rew_t - q_pfc_t[subj_indices, act_t]
            q_pfc_new = pt.set_subtensor(q_pfc_t[subj_indices, act_t], 
                                         q_pfc_t[subj_indices, act_t] + 0.5 * pe_pfc)
            
            q_lim_decayed = q_lim_t * decay
            abs_rew = pt.abs(rew_t)
            util_magnitude = abs_rew ** 0.5 
            
            utility = pt.switch(pt.gt(rew_t, 0), 
                                util_magnitude, 
                                -loss_av[:, 0] * util_magnitude)
                                
            pe_limbic = utility - q_lim_decayed[subj_indices, act_t]
            q_lim_new = pt.set_subtensor(q_lim_decayed[subj_indices, act_t], 
                                         q_lim_decayed[subj_indices, act_t] + alpha[:, 0] * pe_limbic)
            
            return q_pfc_new, q_lim_new, prob_action

        outputs, _ = pytensor.scan(
            fn=rl_trial_step,
            sequences=[pt.as_tensor_variable(actions), pt.as_tensor_variable(rewards)],
            outputs_info=[q_pfc_init, q_limbic_init, None],
            non_sequences=[alpha_b, decay_b, lambda_b, beta_b],
            strict=True
        )
        
        action_probs = outputs[2]
        action_probs_clipped = pt.clip(action_probs, 1e-6, 1.0 - 1e-6)
        log_likelihood = pt.sum(pt.log(action_probs_clipped))
        
        pm.Potential('likelihood', log_likelihood)
        
    return dual_system_hba

if __name__ == "__main__":
    # replace with the resp paths of the original dataset files
    amp_csv = "/content/drive/MyDrive/RL_for_IGT_HBA/Decision-model/Datasets/IGTdata_amphetamine.csv"
    ctrl_csv = "/content/drive/MyDrive/RL_for_IGT_HBA/Decision-model/Datasets/IGTdata_healthy_control.csv"
    hero_csv = "/content/drive/MyDrive/RL_for_IGT_HBA/Decision-model/Datasets/IGTdata_heroin.csv"
    
    actions, rewards, group_idx, n_subs = load_and_combine_groups(amp_csv, ctrl_csv, hero_csv)
    model = build_hba_model(actions, rewards, group_idx, n_subs)
    
    with model:
        print("Initiating JAX/NumPyro GPU Engine...")
        trace = pm.sample(draws=1500, tune=1000, chains=1, target_accept=0.92, nuts_sampler="numpyro") # tune the hyperparams ad needed (chains and draws)
        
    trace.to_netcdf("/content/dual_system_hba_posterior.nc")
    print("\n--- GPU INFERENCE COMPLETE ---")