"""Runner that evaluates all policies on the dynamic knapsack environment.

Every policy is tested over a number of independent simulation runs against
the *same* item streams (same base seed), and its average/max relative reward
is reported. ``MC_RL`` additionally reports training time.

Usage:
    python main.py

Outputs (figures, learned tables) are written to ``output/`` when MC_RL is
trained from scratch.

Coursework: Optimization Implementation in Production and Logistics (OVGU
Magdeburg), dynamic knapsack assignment.
"""

from environment import KnapsackEnvironment
import pandas as pd
from policies import (
    Policy,
    Greedy,
    Threshold,
    Sample,
    Reserve_Cap,
    MC_RL,
    Perfect_Information,
)
import time
import sys
import numpy as np


def evaluate_policy(
    env: KnapsackEnvironment,
    policy: Policy,
    n_simulation: int,
    seed: int,
) -> tuple[float, float, float]:
    """Evaluate a policy over ``n_simulation`` runs sharing the base ``seed``.

    Returns:
        ``(mean_relative_reward, max_relative_reward, runtime_in_minutes)``.
        Reward is relative: total collected value / total value of all items.
    """
    time_start = time.time()
    final_reward: list[float] = []

    for i in range(n_simulation):
        state = env.reset(seed=i + seed)
        while True:
            action: bool = policy.act(state)
            state, _, terminated = env.step(action)
            if terminated:
                final_reward.append(env.total_reward / env.sum_item_value)
                break

    max_reward: float = round(float(np.max(final_reward)), 3)
    mean_reward: float = round(float(np.mean(final_reward)), 3)
    runtime_test: float = round((time.time() - time_start) / 60, 1)
    return mean_reward, max_reward, runtime_test


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Simulation parameters (shared across all policies in this study)
    # ------------------------------------------------------------------
    n_decision_points: int = 60  # items offered per simulation run
    expected_total_weight_items = 600  # expected total weight of all items
    expected_mean_weight_item = expected_total_weight_items // n_decision_points
    std_deviation_weight_item = expected_mean_weight_item * 0.5
    knapsack_cap: float = 0.3  # capacity as a share of expected total weight
    initial_knap_cap: int = int(knapsack_cap * expected_total_weight_items)
    correlation_factor: float = 0.6  # weight-value correlation in [0, 1]
    seed: int = 51  # base seed, so every policy faces the same item streams
    n_simulation: int = 500  # number of simulation runs per policy

    knapsack_env = KnapsackEnvironment(
        n_decision_points=n_decision_points,
        expected_total_weight_items=expected_total_weight_items,
        expected_mean_weight_item=expected_mean_weight_item,
        std_deviation_weight_item=std_deviation_weight_item,
        knapsack_cap=knapsack_cap,
        correlation_factor=correlation_factor,
    )

    # Result table: mean and max relative reward, plus training/testing time.
    results: pd.DataFrame = pd.DataFrame(
        columns=["Avg Reward", "Max Reward", "Training Time", "Testing Time"]
    )
    results.index.name = "Policy"

    # Individually evaluate each policy (kept as separate blocks so any of
    # them can be toggled on/off without touching the others).
    evaluated_policies: list[str] = [
        "Greedy",
        "Threshold",
        "Reserve_Cap",
        "Sample",
        "MC_RL",
        "Perfect_Information",
    ]

    # ---- Greedy ------------------------------------------------------
    if "Greedy" in evaluated_policies:
        greedy = Greedy()
        fr, mr, rt = evaluate_policy(knapsack_env, greedy, n_simulation, seed)
        results.loc["Greedy"] = [fr, mr, 0.0, rt]

    # ---- Threshold ---------------------------------------------------
    if "Threshold" in evaluated_policies:
        threshold_value: float = 1.0  # minimum value/weight ratio to accept
        threshold = Threshold(threshold_value=threshold_value)
        fr, mr, rt = evaluate_policy(knapsack_env, threshold, n_simulation, seed)
        results.loc["Threshold"] = [fr, mr, 0.0, rt]

    # ---- Reserve_Cap -------------------------------------------------
    if "Reserve_Cap" in evaluated_policies:
        reserve_cap = Reserve_Cap(
            initial_knap_cap=initial_knap_cap,
            n_decision_points=n_decision_points,
        )
        fr, mr, rt = evaluate_policy(knapsack_env, reserve_cap, n_simulation, seed)
        results.loc["Reserve_Cap"] = [fr, mr, 0.0, rt]

    # ---- Sample ------------------------------------------------------
    if "Sample" in evaluated_policies:
        sampled_items: int = 3  # look-ahead sample size
        sample = Sample(
            sampled_items=sampled_items,
            expected_mean_weight_item=expected_mean_weight_item,
            std_deviation_weight_item=std_deviation_weight_item,
            correlation_factor=correlation_factor,
        )
        fr, mr, rt = evaluate_policy(knapsack_env, sample, n_simulation, seed)
        results.loc["Sample"] = [fr, mr, 0.0, rt]

    # ---- MC_RL (Monte-Carlo reinforcement learning) ------------------
    if "MC_RL" in evaluated_policies:
        initial_lookup_values: int = 0  # initial PDS value function
        # Pretrained tables do not ship with this repo; train from scratch.
        use_pre_trained_table: bool = False

        mc_rl = MC_RL(
            n_decision_points=n_decision_points,
            initial_knap_cap=initial_knap_cap,
            initial_lookup_values=initial_lookup_values,
        )

        if use_pre_trained_table:
            path_lookup = "output/Lookup_Table.txt"
            if mc_rl.load_lookup_table(path_lookup):
                runtime_train_mc_rl = 0.0
                fr, mr, rt = evaluate_policy(
                    knapsack_env, mc_rl, n_simulation, seed
                )
            else:
                print("No lookup table for Monte Carlo policy available")
                sys.exit()
        else:
            path_saved_files = "output/"
            import os

            os.makedirs(path_saved_files, exist_ok=True)
            display_training_insights: bool = True

            n_train_epochs: int = 1000
            training_seed: int = 5
            epsilon_greedy: float = 2.0  # higher = slower decay = more exploration
            start_training = time.time()

            mc_rl.train(
                knapsack_env=knapsack_env,
                training_seed=training_seed,
                n_train_epochs=n_train_epochs,
                epsilon_greedy=epsilon_greedy,
                display_training_insights=display_training_insights,
                path_saved_files=path_saved_files,
            )

            runtime_train_mc_rl = round((time.time() - start_training) / 60, 1)
            fr, mr, rt = evaluate_policy(knapsack_env, mc_rl, n_simulation, seed)

        results.loc["MC_RL"] = [fr, mr, runtime_train_mc_rl, rt]

    # ---- Perfect_Information (offline upper bound) --------------------
    if "Perfect_Information" in evaluated_policies:
        pi = Perfect_Information()
        fr, mr, rt = pi.act(env=knapsack_env, n_simulation=n_simulation, seed=seed)
        results.loc["Perfect_Information"] = [fr, mr, 0.0, rt]

    print(results)