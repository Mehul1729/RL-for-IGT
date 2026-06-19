
"""
Hierarchical Bayesian dual-system RL model for Iowa Gambling Task data.

Please experiment with various states defined for the PFC module
"""

import os
os.environ["PYTENSOR_FLAGS"] = "profile=False"

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

pytensor.config.profile = False


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_and_combine_groups(
    amp_path,
    control_path,
    heroin_path,
    n_trials=100,
    reward_scale=1.0,
):

    df_amp = pd.read_csv(amp_path)
    df_amp["Group_ID"] = 2

    df_ctrl = pd.read_csv(control_path)
    df_ctrl["Group_ID"] = 0

    df_hero = pd.read_csv(heroin_path)
    df_hero["Group_ID"] = 1

    df = pd.concat([df_amp, df_ctrl, df_hero], ignore_index=True)


    df["Choice"] = df["deck"].astype(int)
    if df["Choice"].min() == 1:
        df["Choice"] = df["Choice"] - 1

    if "trial" not in df.columns:
        df["trial"] = df.groupby("subjID").cumcount() + 1

    df["Net_Reward"] = (df["gain"] + df["loss"]) / float(reward_scale)

    unique_subjects = df["subjID"].unique()
    n_subjects = len(unique_subjects)

    action_matrix = np.zeros((n_subjects, n_trials), dtype=np.int32)
    reward_matrix = np.zeros((n_subjects, n_trials), dtype=np.float32)
    group_idx = np.zeros(n_subjects, dtype=np.int32)

    for i, subj in enumerate(unique_subjects):
        subj_data = df[df["subjID"] == subj].sort_values("trial")
        acts = subj_data["Choice"].values[:n_trials].astype(np.int32)
        rews = subj_data["Net_Reward"].values[:n_trials].astype(np.float32)

        action_matrix[i, : len(acts)] = acts
        reward_matrix[i, : len(rews)] = rews
        group_idx[i] = int(subj_data["Group_ID"].iloc[0])

    # Returns T x N matrices for T trials and N pariticipants
    return action_matrix.T, reward_matrix.T, group_idx, n_subjects






# -----------------------------------------------------------------------------
# PFC state functions
# -----------------------------------------------------------------------------

    """
    Here, I defined variou state variable functions for modelling the PFC mimic.
    """
    
    
def _count_bucket_state(
    deck_counts,
    reward_sums,
    reward_counts,
    history,
    reward_history,
    n_actions=4,
    bin_size=5, # increase bin size to make the state space sparse
    relative_clip=3,
    balance_clip=5,
    recent_window=10,
    recent_bin_size=2,
):

    return tuple((deck_counts // bin_size).astype(np.int32).tolist())


def _relative_count_state(
    deck_counts,
    reward_sums,
    reward_counts,
    history,
    reward_history,
    n_actions=4,
    bin_size=5, # increase bin size to make the state space sparse
    relative_clip=3,
    balance_clip=5,
    recent_window=10,
    recent_bin_size=2,
):

    centered = deck_counts.astype(float) - np.mean(deck_counts.astype(float)) # centering the deck count at every state 
    binned = np.floor(centered / float(bin_size)).astype(np.int32)
    binned = np.clip(binned, -relative_clip, relative_clip)
    return tuple((binned + relative_clip).astype(np.int32).tolist())


def _good_bad_balance_state(
    deck_counts,
    reward_sums,
    reward_counts,
    history,
    reward_history,
    n_actions=4,
    bin_size=5,
    relative_clip=3,
    balance_clip=5,
    recent_window=10,
    recent_bin_size=2,
):

    bad = int(deck_counts[0] + deck_counts[1])
    good = int(deck_counts[2] + deck_counts[3])
    balance = good - bad
    binned = int(np.floor(balance / float(bin_size)))
    binned = int(np.clip(binned, -balance_clip, balance_clip))
    return (binned + balance_clip,)


def _recent_window_state(
    deck_counts,
    reward_sums,
    reward_counts,
    history,
    reward_history,
    n_actions=4,
    bin_size=5,
    relative_clip=3,
    balance_clip=5,
    recent_window=10,
    recent_bin_size=2,
):
  
    recent = history[-recent_window:] if len(history) > 0 else []
    counts = np.zeros(n_actions, dtype=np.int32)

    for a in recent:
        a = int(a)
        if 0 <= a < n_actions:
            counts[a] += 1

    return tuple((counts // recent_bin_size).astype(np.int32).tolist())


def _last_action_reward_state(
    deck_counts,
    reward_sums,
    reward_counts,
    history,
    reward_history,
    n_actions=4,
    bin_size=5,
    relative_clip=3,
    balance_clip=5,
    recent_window=10,
    recent_bin_size=2,
):
  
    if len(history) == 0:
        return (n_actions, 1)

    last_action = int(history[-1])
    last_reward = float(reward_history[-1])

    if last_reward > 0:
        reward_sign = 2
    elif last_reward < 0:
        reward_sign = 0
    else:
        reward_sign = 1

    return (last_action, reward_sign)




STATE_FUNCTIONS = {
    "count_bucket": _count_bucket_state,
    "relative_count": _relative_count_state,
    "good_bad_balance": _good_bad_balance_state,
    "recent_window": _recent_window_state,
    "last_action_reward": _last_action_reward_state,
}


# -----------------------------------------------------------------------------
# PFC state helper: 
# -----------------------------------------------------------------------------
def build_pfc_state_matrices(
    actions,
    rewards=None,
    n_actions=4,
    state_type="relative_count",
    bin_size=5,
    relative_clip=3,
    balance_clip=5,
    recent_window=10,
    recent_bin_size=2,
):
    if state_type not in STATE_FUNCTIONS:
        raise ValueError(
            f"Unknown pfc_state_type={state_type}. "
            f"Choose from {list(STATE_FUNCTIONS.keys())}"
        )

    actions = np.asarray(actions, dtype=np.int32)
    T, N = actions.shape

    if rewards is None:
        rewards = np.zeros_like(actions, dtype=np.float32)
    else:
        rewards = np.asarray(rewards, dtype=np.float32)

    state_fn = STATE_FUNCTIONS[state_type]

    state_to_id = {}
    id_to_state = []

    def get_state_id(state_tuple):
        if state_tuple not in state_to_id:
            state_to_id[state_tuple] = len(id_to_state)
            id_to_state.append(state_tuple)
        return state_to_id[state_tuple]

    state_ids = np.zeros((T, N), dtype=np.int32)
    next_state_ids = np.zeros((T, N), dtype=np.int32)

    for s in range(N):
        deck_counts = np.zeros(n_actions, dtype=np.int32)
        reward_sums = np.zeros(n_actions, dtype=np.float64)
        reward_counts = np.zeros(n_actions, dtype=np.int32)
        history = []
        reward_history = []

        for t in range(T):
            action = int(actions[t, s])
            reward = float(rewards[t, s])

            current_state = state_fn(
                deck_counts=deck_counts,
                reward_sums=reward_sums,
                reward_counts=reward_counts,
                history=history,
                reward_history=reward_history,
                n_actions=n_actions,
                bin_size=bin_size,
                relative_clip=relative_clip,
                balance_clip=balance_clip,
                recent_window=recent_window,
                recent_bin_size=recent_bin_size,
            )
            state_ids[t, s] = get_state_id(current_state)

            # Updating observed trajectory to get next state: 
            if 0 <= action < n_actions:
                deck_counts[action] += 1
                reward_sums[action] += reward
                reward_counts[action] += 1
                history.append(action)
                reward_history.append(reward)

            next_state = state_fn(
                deck_counts=deck_counts,
                reward_sums=reward_sums,
                reward_counts=reward_counts,
                history=history,
                reward_history=reward_history,
                n_actions=n_actions,
                bin_size=bin_size,
                relative_clip=relative_clip,
                balance_clip=balance_clip,
                recent_window=recent_window,
                recent_bin_size=recent_bin_size,
            )
            next_state_ids[t, s] = get_state_id(next_state)

    return state_ids, next_state_ids, len(id_to_state), id_to_state


# -----------------------------------------------------------------------------
# Helper funcs for hierarchical subject params :
# -----------------------------------------------------------------------------
def _bounded_subject_param(name, group_idx, n_subjects, n_groups=3, mu=0.0, sigma=1.5):
    """Return subject-level parameter constrained to (0, 1)."""
    mu_raw = pm.Normal(f"mu_{name}_raw", mu=mu, sigma=sigma, shape=n_groups)
    sig_raw = pm.HalfNormal(f"sig_{name}_raw", sigma=1.0, shape=n_groups)
    z = pm.Normal(f"z_{name}", mu=0.0, sigma=1.0, shape=n_subjects)
    subj_raw = mu_raw[group_idx] + z * sig_raw[group_idx]
    return pm.Deterministic(name, pm.math.invlogit(subj_raw))


def _positive_subject_param(name, group_idx, n_subjects, n_groups=3, mu=0.5, sigma=1.0):
    """Return subject-level parameter constrained to > 0."""
    mu_raw = pm.Normal(f"mu_{name}_raw", mu=mu, sigma=sigma, shape=n_groups)
    sig_raw = pm.HalfNormal(f"sig_{name}_raw", sigma=1.0, shape=n_groups)
    z = pm.Normal(f"z_{name}", mu=0.0, sigma=1.0, shape=n_subjects)
    subj_raw = mu_raw[group_idx] + z * sig_raw[group_idx]
    return pm.Deterministic(name, pm.math.exp(subj_raw))


def _softmax(q):
    """Stable row-wise softmax for matrices of shape (subjects, actions)."""
    q_max = pt.max(q, axis=1, keepdims=True)
    exp_q = pt.exp(q - q_max)
    return exp_q / pt.sum(exp_q, axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Dual model: 
# -----------------------------------------------------------------------------
def build_hba_model(
    actions,
    rewards,
    group_idx,
    n_subjects,
    n_actions=4,
    pfc_state_type="relative_count",
    pfc_bin_size=5,
    pfc_relative_clip=3,
    pfc_balance_clip=5,
    pfc_recent_window=10,
    pfc_recent_bin_size=2,
    pfc_lr=0.1,
    pfc_gamma=0.95,
):

    state_ids, next_state_ids, n_pfc_states, id_to_state = build_pfc_state_matrices(
        actions=actions,
        rewards=rewards,
        n_actions=n_actions,
        state_type=pfc_state_type,
        bin_size=pfc_bin_size,
        relative_clip=pfc_relative_clip,
        balance_clip=pfc_balance_clip,
        recent_window=pfc_recent_window,
        recent_bin_size=pfc_recent_bin_size,
    )

    print(
        f"PFC state type: {pfc_state_type} | "
        f"n_pfc_states encountered: {n_pfc_states} | "
        f"bin_size: {pfc_bin_size}"
    )

    n_groups = 3

    with pm.Model() as model:
        # fitted alpha, decay, lambda, beta: 
        alpha = _bounded_subject_param("alpha", group_idx, n_subjects, n_groups)
        decay = _bounded_subject_param("decay", group_idx, n_subjects, n_groups)
        lam = _positive_subject_param("lambda", group_idx, n_subjects, n_groups)
        beta = _bounded_subject_param("beta", group_idx, n_subjects, n_groups)

        q_pfc_init = pt.zeros((n_subjects, n_pfc_states, n_actions))
        q_limbic_init = pt.zeros((n_subjects, n_actions))

        alpha_b = alpha.dimshuffle(0, "x")
        decay_b = decay.dimshuffle(0, "x")
        lam_b = lam.dimshuffle(0, "x")
        beta_b = beta.dimshuffle(0, "x")

        def trial_step(
            act_t,
            rew_t,
            state_t,
            next_state_t,
            q_pfc_t,
            q_limbic_t,
            alpha_param,
            decay_param,
            lambda_param,
            beta_param,
        ):
            subj_idx = pt.arange(n_subjects)

            q_pfc_current = q_pfc_t[subj_idx, state_t, :]

            q_limbic_current = q_limbic_t

            p_pfc = _softmax(q_pfc_current)
            p_limbic = _softmax(q_limbic_current)

            # probability-level arbitration : 
            final_probs = beta_param * p_pfc + (1.0 - beta_param) * p_limbic
            prob_action = final_probs[subj_idx, act_t]

            # PFC Q-learning update: 
            q_old = q_pfc_t[subj_idx, state_t, act_t]
            q_next_max = pt.max(q_pfc_t[subj_idx, next_state_t, :], axis=1)
            td_target = rew_t + pfc_gamma * q_next_max
            td_error = td_target - q_old

            q_pfc_new = pt.set_subtensor(
                q_pfc_t[subj_idx, state_t, act_t],
                q_old + pfc_lr * td_error,
            )

            # PVL update : 
            q_limbic_decayed = q_limbic_t * decay_param
            abs_rew = pt.abs(rew_t)
            util_mag = abs_rew ** 0.5
            utility = pt.switch(
                pt.ge(rew_t, 0.0),
                util_mag,
                -lambda_param[:, 0] * util_mag,
            )

            current_limbic_val = q_limbic_decayed[subj_idx, act_t]
            pe_limbic = utility - current_limbic_val

            q_limbic_new = pt.set_subtensor(
                q_limbic_decayed[subj_idx, act_t],
                current_limbic_val + alpha_param[:, 0] * pe_limbic,
            )

            return q_pfc_new, q_limbic_new, prob_action

        outputs, _ = pytensor.scan(
            fn=trial_step,
            sequences=[
                pt.as_tensor_variable(actions),
                pt.as_tensor_variable(rewards),
                pt.as_tensor_variable(state_ids),
                pt.as_tensor_variable(next_state_ids),
            ],
            outputs_info=[q_pfc_init, q_limbic_init, None],
            non_sequences=[alpha_b, decay_b, lam_b, beta_b],
            strict=True,
        )

        action_probs = pt.clip(outputs[2], 1e-9, 1.0)
        pm.Potential("likelihood", pt.sum(pt.log(action_probs)))

    return model


# -----------------------------------------------------------------------------
# PVL-only baseline : 
# -----------------------------------------------------------------------------
def build_pvl_hba_model(
    actions,
    rewards,
    group_idx,
    n_subjects,
    n_actions=4,
):
    """PVL-only model: q_limbic[subject, action]."""
    n_groups = 3

    with pm.Model() as model:
        alpha = _bounded_subject_param("alpha", group_idx, n_subjects, n_groups)
        decay = _bounded_subject_param("decay", group_idx, n_subjects, n_groups)
        lam = _positive_subject_param("lambda", group_idx, n_subjects, n_groups)

        q_limbic_init = pt.zeros((n_subjects, n_actions))

        alpha_b = alpha.dimshuffle(0, "x")
        decay_b = decay.dimshuffle(0, "x")
        lam_b = lam.dimshuffle(0, "x")

        def trial_step(act_t, rew_t, q_limbic_t, alpha_param, decay_param, lambda_param):
            subj_idx = pt.arange(n_subjects)

            probs = _softmax(q_limbic_t)
            prob_action = probs[subj_idx, act_t]

            q_limbic_decayed = q_limbic_t * decay_param
            abs_rew = pt.abs(rew_t)
            util_mag = abs_rew ** 0.5
            utility = pt.switch(
                pt.ge(rew_t, 0.0),
                util_mag,
                -lambda_param[:, 0] * util_mag,
            )

            current_val = q_limbic_decayed[subj_idx, act_t]
            pe = utility - current_val
            q_limbic_new = pt.set_subtensor(
                q_limbic_decayed[subj_idx, act_t],
                current_val + alpha_param[:, 0] * pe,
            )

            return q_limbic_new, prob_action

        outputs, _ = pytensor.scan(
            fn=trial_step,
            sequences=[pt.as_tensor_variable(actions), pt.as_tensor_variable(rewards)],
            outputs_info=[q_limbic_init, None],
            non_sequences=[alpha_b, decay_b, lam_b],
            strict=True,
        )

        action_probs = pt.clip(outputs[1], 1e-9, 1.0)
        pm.Potential("likelihood", pt.sum(pt.log(action_probs)))

    return model


# -----------------------------------------------------------------------------
# PFC-only baseline: 
# -----------------------------------------------------------------------------
def build_pfc_hba_model(
    actions,
    rewards,
    group_idx,
    n_subjects,
    n_actions=4,
    pfc_state_type="relative_count",
    pfc_bin_size=5,
    pfc_relative_clip=3,
    pfc_balance_clip=5,
    pfc_recent_window=10,
    pfc_recent_bin_size=2,
    pfc_gamma=0.95,
):

    state_ids, next_state_ids, n_pfc_states, id_to_state = build_pfc_state_matrices(
        actions=actions,
        rewards=rewards,
        n_actions=n_actions,
        state_type=pfc_state_type,
        bin_size=pfc_bin_size,
        relative_clip=pfc_relative_clip,
        balance_clip=pfc_balance_clip,
        recent_window=pfc_recent_window,
        recent_bin_size=pfc_recent_bin_size,
    )

    print(
        f"PFC state type: {pfc_state_type} | "
        f"n_pfc_states encountered: {n_pfc_states} | "
        f"bin_size: {pfc_bin_size}"
    )

    n_groups = 3

    with pm.Model() as model:
        alpha_pfc = _bounded_subject_param("alpha_pfc", group_idx, n_subjects, n_groups)

        q_pfc_init = pt.zeros((n_subjects, n_pfc_states, n_actions))
        alpha_p_b = alpha_pfc.dimshuffle(0, "x")

        def trial_step(act_t, rew_t, state_t, next_state_t, q_pfc_t, alpha_p):
            subj_idx = pt.arange(n_subjects)

            q_current = q_pfc_t[subj_idx, state_t, :]
            probs = _softmax(q_current)
            prob_action = probs[subj_idx, act_t]

            q_old = q_pfc_t[subj_idx, state_t, act_t]
            q_next_max = pt.max(q_pfc_t[subj_idx, next_state_t, :], axis=1)
            td_target = rew_t + pfc_gamma * q_next_max
            td_error = td_target - q_old

            q_pfc_new = pt.set_subtensor(
                q_pfc_t[subj_idx, state_t, act_t],
                q_old + alpha_p[:, 0] * td_error,
            )

            return q_pfc_new, prob_action

        outputs, _ = pytensor.scan(
            fn=trial_step,
            sequences=[
                pt.as_tensor_variable(actions),
                pt.as_tensor_variable(rewards),
                pt.as_tensor_variable(state_ids),
                pt.as_tensor_variable(next_state_ids),
            ],
            outputs_info=[q_pfc_init, None],
            non_sequences=[alpha_p_b],
            strict=True,
        )

        action_probs = pt.clip(outputs[1], 1e-9, 1.0)
        pm.Potential("likelihood", pt.sum(pt.log(action_probs)))

    return model



def choose_model(
    model_type,
    actions,
    rewards,
    group_idx,
    n_subjects,
    n_actions=4,
    pfc_state_type="relative_count",
    pfc_bin_size=5,
    pfc_relative_clip=3,
    pfc_balance_clip=5,
    pfc_recent_window=10,
    pfc_recent_bin_size=2,
    pfc_lr=0.1,
    pfc_gamma=0.95,
):
    model_type = model_type.lower()

    if model_type == "dual":
        return build_hba_model(
            actions,
            rewards,
            group_idx,
            n_subjects,
            n_actions=n_actions,
            pfc_state_type=pfc_state_type,
            pfc_bin_size=pfc_bin_size,
            pfc_relative_clip=pfc_relative_clip,
            pfc_balance_clip=pfc_balance_clip,
            pfc_recent_window=pfc_recent_window,
            pfc_recent_bin_size=pfc_recent_bin_size,
            pfc_lr=pfc_lr,
            pfc_gamma=pfc_gamma,
        )

    if model_type == "pvl":
        return build_pvl_hba_model(
            actions,
            rewards,
            group_idx,
            n_subjects,
            n_actions=n_actions,
        )

    if model_type == "pfc":
        return build_pfc_hba_model(
            actions,
            rewards,
            group_idx,
            n_subjects,
            n_actions=n_actions,
            pfc_state_type=pfc_state_type,
            pfc_bin_size=pfc_bin_size,
            pfc_relative_clip=pfc_relative_clip,
            pfc_balance_clip=pfc_balance_clip,
            pfc_recent_window=pfc_recent_window,
            pfc_recent_bin_size=pfc_recent_bin_size,
            pfc_gamma=pfc_gamma,
        )

    raise ValueError("model_type must be one of: dual, pvl, pfc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HBA models with configurable PFC state representations."
    )

    parser.add_argument("--model", choices=["dual", "pvl", "pfc"], default="dual")
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--target_accept", type=float, default=0.92)

    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="Default 1.0 matches Monte Carlo raw net rewards. Use 100 only if MC is also scaled.",
    )

    parser.add_argument(
        "--pfc_state_type",
        type=str,
        default="relative_count",
        choices=list(STATE_FUNCTIONS.keys()),
        help=(
            "PFC state representation. Recommended first alternatives: "
            "relative_count or reward_evidence."
        ),
    )
    parser.add_argument("--pfc_bin_size", type=int, default=5)
    parser.add_argument("--pfc_relative_clip", type=int, default=3)
    parser.add_argument("--pfc_balance_clip", type=int, default=5)
    parser.add_argument("--pfc_recent_window", type=int, default=10)
    parser.add_argument("--pfc_recent_bin_size", type=int, default=2)

    parser.add_argument("--pfc_lr", type=float, default=0.1)
    parser.add_argument("--pfc_gamma", type=float, default=0.95)
    parser.add_argument("--n_actions", type=int, default=4)

    parser.add_argument(
        "--amp_csv",
        type=str,
        default=os.path.join("Datasets", "IGTdata_amphetamine.csv")    )
    parser.add_argument(
        "--ctrl_csv",
        type=str,
        default=os.path.join("Datasets", "IGTdata_healthy_control.csv")    )
    parser.add_argument(
        "--hero_csv",
        type=str,
        default=os.path.join("Datasets", "IGTdata_heroin.csv")    )
    parser.add_argument("--output_dir", type=str, default=".")

    args = parser.parse_args()

    actions, rewards, group_idx, n_subjects = load_and_combine_groups(
        args.amp_csv,
        args.ctrl_csv,
        args.hero_csv,
        reward_scale=args.reward_scale,
    )

    model = choose_model(
        args.model,
        actions,
        rewards,
        group_idx,
        n_subjects,
        n_actions=args.n_actions,
        pfc_state_type=args.pfc_state_type,
        pfc_bin_size=args.pfc_bin_size,
        pfc_relative_clip=args.pfc_relative_clip,
        pfc_balance_clip=args.pfc_balance_clip,
        pfc_recent_window=args.pfc_recent_window,
        pfc_recent_bin_size=args.pfc_recent_bin_size,
        pfc_lr=args.pfc_lr,
        pfc_gamma=args.pfc_gamma,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "pvl":
        output_name = "pvl_hba_posterior.nc"
    else:
        safe_state_name = args.pfc_state_type.replace(" ", "_")
        output_name = (
            f"{args.model}_hba_{safe_state_name}"
            f"_bin{args.pfc_bin_size}_posterior.nc"
        )

    output_path = os.path.join(args.output_dir, output_name)

    with model:
        print(f"Initiating JAX/NumPyro GPU Engine for {args.model.upper()} model...")
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            nuts_sampler="numpyro",
        )

    trace.to_netcdf(output_path)
    print(f"Saved posterior to: {output_path}")
