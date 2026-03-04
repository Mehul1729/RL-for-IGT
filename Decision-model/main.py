

# import numpy as np
# import pandas as pd
# import sys
# import time
# import warnings
# import os
# directory = os.path.abspath(os.path.join(os.getcwd(), '..'))

# from likelihood import calculate_nll


# # --- Monte Carlo Optimizer---
# def monte_carlo_search(choices, rewards, n_iters=10000):
#     """
#     Randomly samples parameters and returns the best set.
#     Guaranteed to finish.
#     """
#     best_nll = float('inf')
#     best_params = None

#     # Generate random parameters (Vectorized) for each iteration:
#     # alpha, decay, lambda, t_pfc, t_limbic, arb_lr
#     alphas = np.random.uniform(0.01, 1.0, n_iters)
#     decays = np.random.uniform(0.01, 1.0, n_iters)
#     lambdas = np.random.uniform(0.1, 5.0, n_iters)
#     beta_pfc = np.random.uniform(0, 1, n_iters)
#     # t_pfcs = np.random.uniform(1.0, 20.0, n_iters)
#     # t_limbics = np.random.uniform(0.1, 10.0, n_iters)
#     # arb_lrs = np.random.uniform(0.01, 0.5, n_iters)


#     for i in range(n_iters):
#         params = [alphas[i], decays[i], lambdas[i], beta_pfc[i]]
        
#         try:
#             nll = calculate_nll(params, choices, rewards)
#             if nll < best_nll:
#                 best_nll = nll
#                 best_params = params
#         except:
#             continue
            
#     return best_params, best_nll

# # --- 4. EXECUTION LOOP ---

# def process_group(filepath, group_name, n_iter):
    
#     print(f"\n--- Processing {group_name} ---")
#     try:
#         df = pd.read_csv(filepath)
#     except FileNotFoundError:
#         print(f" File not found: {filepath}")
#         return pd.DataFrame()

#     # Preprocessing
#     if 'deck' in df.columns:
#         if df['deck'].min() == 1:
#             df['deck'] = df['deck'] - 1
            
#     if 'gain' in df.columns and 'loss' in df.columns:
#         df['net_reward'] = df['gain'] + df['loss']
#     else:
#         print(" Error: Gain/Loss columns missing")
#         return pd.DataFrame()
    
#     subjects = df['subjID'].unique()
#     print(f"Found {len(subjects)} subjects. Running Monte Carlo Sweep...")
    
#     group_results = []
    
#     for i, sub in enumerate(subjects):
#         # Explicit Progress Bar
#         sys.stdout.write(f"\rSubject {i+1}/{len(subjects)} (ID: {sub})...")
#         sys.stdout.flush()
        
#         sub_data = df[df['subjID'] == sub]
#         choices = sub_data['deck'].values
#         rewards = sub_data['net_reward'].values


#         params, nll = monte_carlo_search(choices, rewards, n_iters=n_iter)
        
#         if params is not None:
#             r = {
#                 'Subject': sub, 'Group': group_name, 'NLL': nll,
#                 'Alpha_Limbic': params[0], # limbic learning rate
#                 'Decay': params[1],
#                 'Loss_Aversion': params[2],
#                 'Beta_PFC': params[3],
#                 # 'PFC_Temp': params[3],
#                 # 'Limbic_Temp': params[4],
#                 # 'Arbitration_Rate': params[5],
#                 # 'Temp_Ratio': params[3] / (params[4] + 1e-9)
#             }
#             group_results.append(r)

#     print("\nDone.")
#     return pd.DataFrame(group_results)

# # --- RUN ---
# if __name__ == "__main__":
    
#     n_iter = int(input("Enter number of Monte Carlo iterations per subject: "))

#     df_amp = process_group(rf"Decision-model/Datasets/IGTdata_amphetamine.csv", 'Amphetamine', n_iter)
#     df_hc = process_group(rf"Decision-model/Datasets/IGTdata_healthy_control.csv", 'Control', n_iter)
#     df_her = process_group(rf"Decision-model/Datasets/IGTdata_heroin.csv", 'Heroin', n_iter)

#     # Combine & Save
#     all_data = []
#     if not df_amp.empty: all_data.append(df_amp)
#     if not df_hc.empty: all_data.append(df_hc)
#     if not df_her.empty: all_data.append(df_her)

#     if all_data:
#         all_results = pd.concat(all_data)
#         all_results.to_csv(rf"Decision-model/Results/New_IGT_Biomarkers_MonteCarlo_{n_iter}.csv", index=False)
#         print(f"\n SAVED: New_IGT_Biomarkers_MonteCarlo_{n_iter}.csv")
        
#         print("\n--- RESULTS SUMMARY (Means) ---")
#         print(all_results.groupby('Group')[['Lambda_Loss', 'Decay', 'Temp_Ratio', 'Alpha_Limbic']].mean())
#     else:
#         print("\n No data processed.")


"""
    Multi Procesing code for speedup :
    
"""
    
    
import numpy as np
import pandas as pd
import sys
import time
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
from likelihood import calculate_nll

# --- Monte Carlo Optimizer ---
def monte_carlo_search(choices, rewards, n_iters=10000):
    best_nll = float('inf')
    best_params = None

    alphas = np.random.uniform(0.01, 1.0, n_iters)
    decays = np.random.uniform(0.01, 1.0, n_iters)
    lambdas = np.random.uniform(0.1, 5.0, n_iters)
    beta_pfc = np.random.uniform(0, 1, n_iters)

    for i in range(n_iters):
        params = [alphas[i], decays[i], lambdas[i], beta_pfc[i]]
        try:
            nll = calculate_nll(params, choices, rewards)
            if nll < best_nll:
                best_nll = nll
                best_params = params
        except:
            continue
            
    return best_params, best_nll

# --- WORKER FUNCTION FOR PARALLELIZATION ---
def fit_single_subject(sub, group_name, choices, rewards, n_iter):
    """This runs independently on a separate CPU core."""
    params, nll = monte_carlo_search(choices, rewards, n_iters=n_iter)
    if params is not None:
        return {
            'Subject': sub, 'Group': group_name, 'NLL': nll,
            'Alpha_Limbic': params[0],
            'Decay': params[1],
            'Loss_Aversion': params[2],
            'Beta_PFC': params[3],
        }
    return None

# --- 4. EXECUTION LOOP ---
def process_group_parallel(filepath, group_name, n_iter):
    print(f"\n--- Processing {group_name} ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f" File not found: {filepath}")
        return pd.DataFrame()

    if 'deck' in df.columns:
        if df['deck'].min() == 1:
            df['deck'] = df['deck'] - 1
            
    if 'gain' in df.columns and 'loss' in df.columns:
        df['net_reward'] = df['gain'] + df['loss']
    else:
        return pd.DataFrame()
    
    subjects = df['subjID'].unique()
    print(f"Found {len(subjects)} subjects. Running Parallel Monte Carlo Sweep...")
    
    group_results = []
    
    # --- MULTIPROCESSING BLOCK ---
    # max_workers=None automatically uses all available CPU cores on your PC
    with ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all subjects to the CPU cores
        futures = {
            executor.submit(
                fit_single_subject, 
                sub, 
                group_name, 
                df[df['subjID'] == sub]['deck'].values, 
                df[df['subjID'] == sub]['net_reward'].values, 
                n_iter
            ): sub for sub in subjects
        }
        
        # Collect results as they finish
        completed = 0
        for future in as_completed(futures):
            completed += 1
            sys.stdout.write(f"\rCompleted {completed}/{len(subjects)} subjects...")
            sys.stdout.flush()
            
            result = future.result()
            if result:
                group_results.append(result)

    print("\nDone.")
    return pd.DataFrame(group_results)

# --- RUN ---
if __name__ == "__main__":
    n_iter = int(input("Enter number of Monte Carlo iterations per subject: "))

    # Make sure paths are correct for your machine
    df_amp = process_group_parallel(rf"Decision-model/Datasets/IGTdata_amphetamine.csv", 'Amphetamine', n_iter)
    df_hc = process_group_parallel(rf"Decision-model/Datasets/IGTdata_healthy_control.csv", 'Control', n_iter)
    df_her = process_group_parallel(rf"Decision-model/Datasets/IGTdata_heroin.csv", 'Heroin', n_iter)

    all_data = []
    if not df_amp.empty: all_data.append(df_amp)
    if not df_hc.empty: all_data.append(df_hc)
    if not df_her.empty: all_data.append(df_her)

    if all_data:
        all_results = pd.concat(all_data)
        all_results.to_csv(rf"Decision-model/Results/New_IGT_Biomarkers_MonteCarlo_{n_iter}.csv", index=False)
        print(f"\n SAVED: New_IGT_Biomarkers_MonteCarlo_{n_iter}.csv")