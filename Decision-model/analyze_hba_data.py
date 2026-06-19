
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.special import expit as invlogit

try:
    import arviz as az # for R-hat
except Exception: 
    az = None


groups = ["Control", "Heroin", "Amphetamine"]
n_groups = 3
n_actions = 4


# Loading posterior data: 
@dataclass
class PosteriorData:
    posterior: Dict[str, np.ndarray]
    sample_stats: Dict[str, np.ndarray]
    source: str


def _read_hdf5_group(file_obj: h5py.File, group_name: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if group_name not in file_obj:
        return out
    group = file_obj[group_name]
    for key, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            if key in {"chain", "draw"} or key.endswith("_dim_0"):
                continue
            out[key] = np.asarray(obj)
    return out


def load_posterior(path: str) -> PosteriorData:
    """Load posterior arrays from a PyMC/ArviZ NetCDF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Posterior file not found: {path}")

 
    with h5py.File(path, "r") as f:
        posterior = _read_hdf5_group(f, "posterior")
        sample_stats = _read_hdf5_group(f, "sample_stats")

    if not posterior:
        raise ValueError("No posterior group found in the NetCDF file.")

    return PosteriorData(posterior=posterior, sample_stats=sample_stats, source=path)


def flatten_chains_draws(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim < 2:
        return arr.reshape(-1)
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def available_vars(posterior: Dict[str, np.ndarray]) -> Iterable[str]:
    return sorted(posterior.keys())


# Transformations and summaries: 
def transform_group_param(raw_samples: np.ndarray, transform: str) -> np.ndarray:
    flat = flatten_chains_draws(raw_samples)
    if transform == "logit":
        return invlogit(flat)
    if transform == "exp":
        return np.exp(flat)
    if transform == "identity":
        return flat
    raise ValueError(f"Unknown transform: {transform}")


def hdi(samples: np.ndarray, prob: float = 0.94) -> Tuple[float, float]:
    """Simple equal-tail interval used as a robust HDI-like interval."""
    low = (1.0 - prob) / 2.0
    high = 1.0 - low
    return float(np.quantile(samples, low)), float(np.quantile(samples, high))


def summarize_group_param(label: str, samples: np.ndarray) -> None:
    
    if samples.ndim != 2 or samples.shape[1] != n_groups:
        print(f"\n--- {label} ---")
        print(f"Skipping: expected shape (samples, 3), got {samples.shape}")
        return

    print(f"\n--- {label} ---")
    for g, name in enumerate(groups):
        mean = float(np.mean(samples[:, g]))
        sd = float(np.std(samples[:, g]))
        lo, hi = hdi(samples[:, g], prob=0.94)
        print(f"{name:12s}: mean={mean:.4f}, sd={sd:.4f}, 94% interval=[{lo:.4f}, {hi:.4f}]")

    print("Pairwise posterior exceedance probabilities:")
    print(f"P(Control > Heroin)      = {np.mean(samples[:, 0] > samples[:, 1]):.4f}")
    print(f"P(Control > Amphetamine) = {np.mean(samples[:, 0] > samples[:, 2]):.4f}")
    print(f"P(Amphetamine > Heroin)  = {np.mean(samples[:, 2] > samples[:, 1]):.4f}")

    if "beta" in label.lower():
        print("Beta-specific checks:")
        for g, name in enumerate(groups):
            print(
                f"{name:12s}: P(beta < 0.05)={np.mean(samples[:, g] < 0.05):.4f}, "
                f"P(beta > 0.50)={np.mean(samples[:, g] > 0.50):.4f}"
            )


def summarize_subject_param(label: str, samples: np.ndarray) -> None:
    flat = flatten_chains_draws(samples)
    if flat.ndim != 2:
        return
    subj_means = flat.mean(axis=0)
    print(f"\n--- Subject-level {label} ---")
    print(f"Subjects: {subj_means.shape[0]}")
    print(
        f"Across-subject posterior-mean range: "
        f"min={subj_means.min():.4f}, median={np.median(subj_means):.4f}, max={subj_means.max():.4f}"
    )


# 
def print_sampler_diagnostics(sample_stats: Dict[str, np.ndarray], posterior_path: str) -> None:
    print("\n=== SAMPLER DIAGNOSTICS ===")

    if sample_stats:
        if "diverging" in sample_stats:
            div = np.asarray(sample_stats["diverging"])
            print(f"Total divergences: {int(np.sum(div))}")
        if "n_steps" in sample_stats:
            steps = np.asarray(sample_stats["n_steps"])
            print(f"n_steps: mean={np.mean(steps):.2f}, median={np.median(steps):.2f}, max={np.max(steps)}")
        if "tree_depth" in sample_stats:
            tree = np.asarray(sample_stats["tree_depth"])
            print(f"tree_depth: mean={np.mean(tree):.2f}, median={np.median(tree):.2f}, max={np.max(tree)}")
        if "step_size" in sample_stats:
            step = np.asarray(sample_stats["step_size"])
            print(f"step_size: mean={np.mean(step):.6g}, min={np.min(step):.6g}, max={np.max(step):.6g}")
        if "acceptance_rate" in sample_stats:
            acc = np.asarray(sample_stats["acceptance_rate"])
            print(f"acceptance_rate: mean={np.mean(acc):.4f}, min={np.min(acc):.4f}, max={np.max(acc):.4f}")
    else:
        print("No sample_stats group found.")

    if az is None:
        print("ArviZ is not available; skipping R-hat/ESS table.")
        return

    try:
        trace = az.from_netcdf(posterior_path)
        summary = az.summary(trace, round_to=2)
        cols = [c for c in ["r_hat", "ess_bulk", "ess_tail"] if c in summary.columns]
        if cols:
            print("\nArviZ R-hat / ESS summary:")
            print(summary[cols])
            if trace.posterior.sizes.get("chain", 0) < 2:
                print("Note: only one chain found; R-hat is not meaningful for one-chain runs.")
    except Exception as e:
        print(f"ArviZ summary unavailable in this environment: {e}")


def load_combined_or_group_csvs(
    combined_csv: Optional[str],
    amp_csv: Optional[str],
    ctrl_csv: Optional[str],
    hero_csv: Optional[str],
    n_trials: int,
    reward_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actions (T,N), rewards (T,N), group_idx (N,)."""
    if combined_csv:
        df = pd.read_csv(combined_csv)
        required = {"subjID", "deck", "gain", "loss", "Group_ID"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Combined CSV missing columns: {missing}")
    else:
        if not (amp_csv and ctrl_csv and hero_csv):
            raise ValueError("Provide either --data or all of --amp_csv, --ctrl_csv, --hero_csv.")
        df_amp = pd.read_csv(amp_csv)
        df_amp["Group_ID"] = 2
        df_ctrl = pd.read_csv(ctrl_csv)
        df_ctrl["Group_ID"] = 0
        df_hero = pd.read_csv(hero_csv)
        df_hero["Group_ID"] = 1
        df = pd.concat([df_amp, df_ctrl, df_hero], ignore_index=True)

    df = df.copy()
    df["Choice"] = df["deck"].astype(int)
    if df["Choice"].min() == 1:
        df["Choice"] -= 1
    df["Net_Reward"] = (df["gain"] + df["loss"]) / reward_scale

    subjects = df["subjID"].unique()
    n_subjects = len(subjects)
    actions = np.zeros((n_trials, n_subjects), dtype=np.int32)
    rewards = np.zeros((n_trials, n_subjects), dtype=np.float64)
    group_idx = np.zeros(n_subjects, dtype=np.int32)

    for i, subj in enumerate(subjects):
        sdf = df[df["subjID"] == subj]
        if "trial" in sdf.columns:
            sdf = sdf.sort_values("trial")
        acts = sdf["Choice"].values[:n_trials]
        rews = sdf["Net_Reward"].values[:n_trials]
        actions[: len(acts), i] = acts
        rewards[: len(rews), i] = rews
        group_idx[i] = int(sdf["Group_ID"].iloc[0])

    return actions, rewards, group_idx


def np_softmax(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q - np.max(q)
    e = np.exp(q)
    return e / np.sum(e)


def get_group_draws_for_prediction(posterior: Dict[str, np.ndarray], n_draws: int, rng: np.random.Generator):
    required = ["mu_alpha_raw", "mu_decay_raw", "mu_lambda_raw"]
    for var in required:
        if var not in posterior:
            raise ValueError(f"Posterior missing required variable: {var}")
    total = flatten_chains_draws(posterior["mu_alpha_raw"]).shape[0]
    chosen = rng.choice(total, size=min(n_draws, total), replace=False)
    return chosen, total


def compute_heldout_nll_mc_aligned(
    posterior: Dict[str, np.ndarray],
    actions: np.ndarray,
    rewards: np.ndarray,
    group_idx: np.ndarray,
    model: str = "dual",
    n_draws: int = 200,
    train_trials: int = 80,
    bin_size: int = 5,
    pfc_lr: float = 0.1,
    gamma: float = 0.95,
    seed: int = 123,
) -> float:

    model = model.lower()
    if model not in {"dual", "pvl", "pfc"}:
        raise ValueError("model must be one of: dual, pvl, pfc")

    T, N = actions.shape
    if T <= train_trials:
        raise ValueError("actions has fewer trials than train_trials.")

    rng = np.random.default_rng(seed)
    chosen, total = get_group_draws_for_prediction(posterior, n_draws, rng)

    alpha_draws = invlogit(flatten_chains_draws(posterior["mu_alpha_raw"]))
    decay_draws = invlogit(flatten_chains_draws(posterior["mu_decay_raw"]))
    lambda_draws = np.exp(flatten_chains_draws(posterior["mu_lambda_raw"]))

    beta_draws = None
    if "mu_beta_raw" in posterior:
        beta_draws = invlogit(flatten_chains_draws(posterior["mu_beta_raw"]))
    elif model == "dual":
        raise ValueError("Dual model posterior must contain mu_beta_raw.")

    nlls = []
    for draw_idx in chosen:
        alpha_group = alpha_draws[draw_idx][group_idx]
        decay_group = decay_draws[draw_idx][group_idx]
        lambda_group = lambda_draws[draw_idx][group_idx]
        if beta_draws is not None:
            beta_group = beta_draws[draw_idx][group_idx]
        else:
            beta_group = np.zeros(N)

        q_pfc_tables = [dict() for _ in range(N)]
        q_limbic = np.zeros((N, n_actions), dtype=np.float64)
        deck_counts = np.zeros((N, n_actions), dtype=np.int32)

        loglik = 0.0
        for t in range(T):
            for s in range(N):
                action = int(actions[t, s])
                if action < 0 or action >= n_actions:
                    continue
                reward = float(rewards[t, s])

                # PFC state-action vector Q_P(s,a)
                state = tuple((deck_counts[s] // bin_size).tolist())
                q_pfc_vec = np.array(
                    [q_pfc_tables[s].get((state, a), 0.0) for a in range(n_actions)],
                    dtype=np.float64,
                )

                # Limbic stateless action vector Q_L(a)
                q_limbic_vec = q_limbic[s]

                p_pfc = np_softmax(q_pfc_vec)
                p_limbic = np_softmax(q_limbic_vec)

                if model == "dual":
                    probs = beta_group[s] * p_pfc + (1.0 - beta_group[s]) * p_limbic
                elif model == "pvl":
                    probs = p_limbic
                else:  # pfc
                    probs = p_pfc

                if t >= train_trials:
                    prob_chosen = float(np.clip(probs[action], 1e-12, 1.0))
                    loglik += np.log(prob_chosen)

                next_counts = deck_counts[s].copy()
                next_counts[action] += 1
                next_state = tuple((next_counts // bin_size).tolist())
                old_val = q_pfc_tables[s].get((state, action), 0.0)
                next_max = max(q_pfc_tables[s].get((next_state, a), 0.0) for a in range(n_actions))
                target = reward + gamma * next_max
                pe_pfc = target - old_val
                q_pfc_tables[s][(state, action)] = old_val + pfc_lr * pe_pfc

                q_limbic[s] *= decay_group[s]
                if reward >= 0:
                    utility = abs(reward) ** 0.5
                else:
                    utility = -lambda_group[s] * (abs(reward) ** 0.5)
                pe_limbic = utility - q_limbic[s, action]
                q_limbic[s, action] += alpha_group[s] * pe_limbic

                deck_counts[s] = next_counts

        nlls.append(-loglik)

    return float(np.mean(nlls))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HBA posterior from the MC-aligned dual-system model.")
    parser.add_argument("--posterior", type=str, required=True, help="Path to posterior NetCDF file.")
    parser.add_argument("--model", type=str, default="dual", choices=["dual", "pvl", "pfc"], help="Model type.")

    # Optional raw data inputs for held-out prediction.
    parser.add_argument("--heldout", action="store_true", help="Compute held-out NLL on trials after --train_trials.")
    parser.add_argument("--data", type=str, default=None, help="Optional combined CSV with subjID, deck, gain, loss, Group_ID.")
    parser.add_argument("--amp_csv", type=str, default=None, help="Amphetamine CSV.")
    parser.add_argument("--ctrl_csv", type=str, default=None, help="Healthy control CSV.")
    parser.add_argument("--hero_csv", type=str, default=None, help="Heroin CSV.")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--train_trials", type=int, default=80)
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Use same value as HBA fitting.")
    parser.add_argument("--bin_size", type=int, default=5, help="PFC state bucket size, matching brain_modules.PFCFit.")
    parser.add_argument("--pfc_lr", type=float, default=0.1, help="Fixed PFC learning rate, matching brain_modules.PFCFit default.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Fixed PFC discount factor.")
    parser.add_argument("--n_pred_draws", type=int, default=200)

    args = parser.parse_args()

    data = load_posterior(args.posterior)
    print("\n=== POSTERIOR FILE ===")
    print(args.posterior)
    print("\nPosterior variables:")
    print(", ".join(available_vars(data.posterior)))

    print("\n=== GROUP-LEVEL PARAMETER SUMMARIES ===")

    param_specs = [
        ("PFC arbitration beta", "mu_beta_raw", "logit"),
        ("PVL loss aversion lambda", "mu_lambda_raw", "exp"),
        ("Limbic learning rate alpha", "mu_alpha_raw", "logit"),
        ("Limbic memory decay", "mu_decay_raw", "logit"),
        ("PFC learning rate alpha_pfc", "mu_alpha_pfc_raw", "logit"),
    ]

    transformed_cache: Dict[str, np.ndarray] = {}
    for label, raw_name, transform in param_specs:
        if raw_name not in data.posterior:
            continue
        samples = transform_group_param(data.posterior[raw_name], transform)
        transformed_cache[raw_name] = samples
        summarize_group_param(label, samples)

    print("\n=== SUBJECT-LEVEL DETERMINISTIC SUMMARIES ===")
    for label, var_name in [
        ("beta", "beta"),
        ("lambda", "lambda"),
        ("alpha", "alpha"),
        ("decay", "decay"),
        ("alpha_pfc", "alpha_pfc"),
    ]:
        if var_name in data.posterior:
            summarize_subject_param(label, data.posterior[var_name])

    print_sampler_diagnostics(data.sample_stats, args.posterior)

    if args.heldout:
        print("\n=== HELD-OUT PREDICTION ===")
        actions, rewards, group_idx = load_combined_or_group_csvs(
            combined_csv=args.data,
            amp_csv=args.amp_csv,
            ctrl_csv=args.ctrl_csv,
            hero_csv=args.hero_csv,
            n_trials=args.n_trials,
            reward_scale=args.reward_scale,
        )
        nll = compute_heldout_nll_mc_aligned(
            posterior=data.posterior,
            actions=actions,
            rewards=rewards,
            group_idx=group_idx,
            model=args.model,
            n_draws=args.n_pred_draws,
            train_trials=args.train_trials,
            bin_size=args.bin_size,
            pfc_lr=args.pfc_lr,
            gamma=args.gamma,
        )
        n_heldout_trials = (actions.shape[0] - args.train_trials) * actions.shape[1]
        print(f"Held-out total NLL: {nll:.3f}")
        print(f"Held-out mean NLL per choice: {nll / n_heldout_trials:.6f}")
        print(f"Held-out trials scored: {n_heldout_trials}")


if __name__ == "__main__":
    main()
