"""Decision policies for the dynamic (online) knapsack problem.

A policy observes a state (current item + remaining capacity) and returns a
Boolean: accept or reject. Six policies are implemented:

* :class:`Greedy`                - take the item whenever it fits.
* :class:`Threshold`             - take it only if value/weight >= a threshold.
* :class:`Reserve_Cap`           - keep a linearly declining capacity buffer.
* :class:`Sample`                - take it if better than sampled future items.
* :class:`MC_RL`                 - Monte-Carlo reinforcement learning: learns a
  value function for post-decision states via epsilon-greedy exploration.
* :class:`Perfect_Information`   - dynamic programming on the *static*
  deterministic version (offline upper bound for comparison).

Coursework: Optimization Implementation in Production and Logistics (OVGU
Magdeburg), dynamic knapsack assignment.
"""

from abc import ABC, abstractmethod
from environment import KnapsackEnvironment
from item import Item
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


class Policy(ABC):
    """Abstract base class dictating the interface every policy implements."""

    @abstractmethod
    def act(self, state: dict) -> bool:
        """Decide whether to accept the item in ``state``."""
        raise NotImplementedError


class Greedy(Policy):
    """Accept the item whenever it physically fits into the knapsack."""

    def act(self, state: dict) -> bool:
        return state["Item"].weight <= state["Cur_Cap"]


class Threshold(Policy):
    """Accept the item if it fits AND its value/weight ratio clears a bar."""

    def __init__(self, threshold_value: float) -> None:
        self.threshold_value: float = threshold_value

    def act(self, state: dict) -> bool:
        if state["Item"].weight <= state["Cur_Cap"]:
            if state["Item"].value / state["Item"].weight >= self.threshold_value:
                return True
        return False


class Reserve_Cap(Policy):
    """Keep a capacity *buffer* that shrinks linearly towards the end.

    Idea: early items are easy to replace, so forbid making the knapsack too
    full too early; the artificial cap is ``cap * items_left / items_total``.
    """

    def __init__(self, initial_knap_cap: int, n_decision_points: int) -> None:
        self.initial_knap_cap: float = initial_knap_cap
        self.n_decision_points: int = n_decision_points

    def act(self, state: dict) -> bool:
        items_left = self.n_decision_points - state["Cur_DP"]
        artificial_cap = self.initial_knap_cap * items_left / self.n_decision_points
        if state["Cur_Cap"] - state["Item"].weight >= artificial_cap:
            return True
        return False


class Sample(Policy):
    """Sample a few hypothetical future items and take the current item only
    if its value/weight ratio beats the sampled average.

    This is a light-weight look-ahead heuristic: it does not know the future
    but estimates it from items drawn with the same distribution.
    """

    def __init__(
        self,
        sampled_items: int,
        expected_mean_weight_item: int,
        std_deviation_weight_item: float,
        correlation_factor: float,
    ) -> None:
        self.sampled_items: int = sampled_items
        self.expected_mean_weight_item: int = expected_mean_weight_item
        self.std_deviation_weight_item: float = std_deviation_weight_item
        self.correlation_factor: float = correlation_factor

    def act(self, state: dict) -> bool:
        if state["Cur_Cap"] <= state["Item"].weight:
            return False

        item_list: list[Item] = []
        for _ in range(self.sampled_items):
            item_list.append(
                Item(
                    item_id=0,  # id unused for sampled items
                    seed=np.random.randint(low=1, high=1000),
                    expected_mean_weight_item=self.expected_mean_weight_item,
                    std_deviation_weight_item=self.std_deviation_weight_item,
                    correlation_factor=self.correlation_factor,
                )
            )
        value_share_item = state["Item"].value / state["Item"].weight
        average_value_share = sum(
            it.value / it.weight for it in item_list
        ) / len(item_list)
        return value_share_item >= average_value_share


class MC_RL(Policy):
    """Monte-Carlo reinforcement learning policy.

    Learns a value function :math:`V(decision point, remaining capacity)` for
    *post-decision states* via epsilon-greedy exploration. At decision time it
    compares the value of accepting (PDS value + item value) against rejecting
    (PDS value) with a Bellman-style comparison.

    The lookup table is indexed with an *inverted* capacity axis so that the
    first row corresponds to a full knapsack, matching the derivation in the
    lecture notes.
    """

    def __init__(
        self,
        n_decision_points: int,
        initial_knap_cap: int,
        initial_lookup_values: int,
    ) -> None:
        self.n_decision_points: int = n_decision_points
        self.initial_knap_cap: int = initial_knap_cap
        self.initial_lookup_values: int = initial_lookup_values
        # lookup_table[cap_axis, decision_point] holds the PDS value function.
        self.lookup_table: np.ndarray = np.ones(
            shape=(self.initial_knap_cap + 1, self.n_decision_points)
        ) * self.initial_lookup_values
        # Counts how often each PDS was visited (used in the MC update rule).
        self.n_explored_states: np.ndarray = np.zeros(
            shape=(self.initial_knap_cap + 1, self.n_decision_points)
        )

    def act(self, state: dict) -> bool:
        """Greedy (evaluation) action based on the current lookup table."""
        item_weight: int = state["Item"].weight
        item_value: int = state["Item"].value
        remaining_capacity: int = state["Cur_Cap"]
        cur_decision_point: int = state["Cur_DP"]

        if remaining_capacity < item_weight:
            return False

        cur_decision_point -= 1  # zero-index numpy rows

        # Post-decision state 1: accept the item.
        post_take_cap = remaining_capacity - item_weight
        take_idx = (
            self.lookup_table.shape[0] - 1 - post_take_cap,
            cur_decision_point,
        )
        post_take_value = self.lookup_table[take_idx]

        # Post-decision state 2: reject the item.
        reject_idx = (
            self.lookup_table.shape[0] - 1 - remaining_capacity,
            cur_decision_point,
        )
        post_reject_value = self.lookup_table[reject_idx]

        # Bellman-style comparison.
        if (post_take_value + item_value) >= post_reject_value:
            return True
        return False

    def stochastic_action(self, state: dict, exploration_rate: float) -> bool:
        """Epsilon-greedy action used during training."""
        if state["Item"].weight <= state["Cur_Cap"]:
            if np.random.rand() < exploration_rate:
                return np.random.rand() < 0.5
            return self.act(state)
        return False

    def train(
        self,
        knapsack_env: KnapsackEnvironment,
        training_seed: int,
        n_train_epochs: int,
        epsilon_greedy: float,
        display_training_insights: bool,
        path_saved_files: str,
    ) -> None:
        """Run Monte-Carlo episodes and update the lookup table each episode.

        Args:
            knapsack_env: Environment used for training (same parameters as
                testing so the learned values transfer).
            training_seed: Seed offset for the training episodes.
            n_train_epochs: Number of training episodes.
            epsilon_greedy: Exponent shaping exploration-rate decay
                (higher value = slower decay = more exploration).
            display_training_insights: Print/plot intermediate performance.
            path_saved_files: Directory where the learned tables/plots go.
        """
        init_exploration_rate: float = 0.99
        final_exploration_rate: float = 0.01
        for epoch in range(n_train_epochs + 1):
            exploitation = epoch / n_train_epochs
            exploration_rate = init_exploration_rate * (
                final_exploration_rate / init_exploration_rate
            ) ** (exploitation ** epsilon_greedy)

            state = knapsack_env.reset(seed=epoch + training_seed)
            observed_pds: list = []
            observed_rewards: list[int] = []
            while True:
                action = self.stochastic_action(state, exploration_rate)
                state, reward, terminated = knapsack_env.step(action)
                observed_pds.append((state["Cur_DP"] - 1, state["Cur_Cap"]))
                observed_rewards.append(reward)
                if terminated:
                    self.update_lookup(observed_pds, observed_rewards)
                    if display_training_insights:
                        self.display_training_insights(
                            epoch=epoch,
                            n_train_epochs=n_train_epochs,
                            knapsack_env=knapsack_env,
                            exploration_rate=exploration_rate,
                            path_saved_files=path_saved_files,
                            init_exploration_rate=init_exploration_rate,
                            final_exploration_rate=final_exploration_rate,
                            epsilon_greedy=epsilon_greedy,
                        )
                    break

        self.lookup_table = np.round(self.lookup_table, decimals=1)
        np.savetxt(
            path_saved_files + "Lookup_Table.txt",
            self.lookup_table,
            delimiter=" ",
            fmt="%.1f",
        )
        np.savetxt(
            path_saved_files + "Explored_States.txt",
            self.n_explored_states,
            delimiter=" ",
            fmt="%d",
        )

    def update_lookup(self, observed_pds: list, observed_rewards: list) -> None:
        """Monte-Carlo update of the PDS value function for one episode.

        Goes backwards through the episode, computing the sum of future
        rewards from each post-decision state onward, and applies a frequency
        weighted step-size update (alternative 2; alternative 1, a running
        mean, is kept below as comments).
        """
        observed_rewards.pop(0)  # first reward is not tied to a PDS
        observed_rewards.append(0)  # terminal PDS has no future reward
        future_rewards_pds: np.ndarray = np.cumsum(
            np.array(observed_rewards)[::-1]
        )[::-1]

        for pds, fr_pds in zip(observed_pds, future_rewards_pds):
            dp = pds[0] - 1  # zero-index
            cap_pds = pds[1]
            idx = (self.lookup_table.shape[0] - 1 - cap_pds, dp)
            self.n_explored_states[idx] += 1
            n_observed = self.n_explored_states[idx]

            # Alternative 1 (running mean) -- swap in by commenting the below:
            # if n_observed == 1:
            #     self.lookup_table[idx] = fr_pds
            # else:
            #     self.lookup_table[idx] = (n_observed - 1) / n_observed * \
            #         self.lookup_table[idx] + fr_pds / n_observed

            # Alternative 2: step size 1/sqrt(frequency), robust to noise.
            step_size: float = 1 / np.sqrt(n_observed)
            old_vf = self.lookup_table[idx]
            td = fr_pds - old_vf
            self.lookup_table[idx] = old_vf + step_size * td

    def evaluation(self, knapsack_env: KnapsackEnvironment, test_instances: int) -> float:
        """Mean relative reward of the *current* (greedy) policy on test runs."""
        rewards: list[float] = []
        for test_instance in range(test_instances):
            state = knapsack_env.reset(seed=test_instance + test_instances)
            while True:
                action: bool = self.act(state)
                state, reward, terminated = knapsack_env.step(action)
                if terminated:
                    rewards.append(
                        knapsack_env.total_reward / knapsack_env.sum_item_value
                    )
                    break
        return float(np.mean(rewards))

    def display_training_insights(
        self,
        epoch: int,
        n_train_epochs: int,
        knapsack_env: KnapsackEnvironment,
        exploration_rate: float,
        path_saved_files: str,
        init_exploration_rate: float,
        final_exploration_rate: float,
        epsilon_greedy: float,
    ) -> None:
        """Periodically report training progress and final plots."""
        evaluation_step: int = 10  # evaluate every 10% of training
        if epoch % int(n_train_epochs / (100 / evaluation_step)) == 0:
            performance = self.evaluation(knapsack_env, test_instances=200)
            if epoch == 0:
                print("\nTraining Start Monte Carlo Reinforcement Learning Policy")
                print(
                    "Training Progress: {}%\t Avg Reward {:.3f}\t Exploration "
                    "Rate: {:.2f}\t Explored States: {:.1f}%".format(
                        0, performance, exploration_rate, 0.0
                    )
                )
            else:
                explored = (
                    np.sum(self.n_explored_states != 0)
                    / (self.lookup_table.shape[0] * self.lookup_table.shape[1])
                    * 100
                )
                print(
                    "Training Progress: {}%\t Avg Reward {:.3f}\t Exploration "
                    "Rate: {:.2f}\t Explored States: {:.1f}%".format(
                        int(100 * epoch / n_train_epochs),
                        performance,
                        exploration_rate,
                        explored,
                    )
                )
                if epoch == n_train_epochs:
                    self.plot_exploration(
                        path_saved_files,
                        n_train_epochs,
                        init_exploration_rate,
                        final_exploration_rate,
                        epsilon_greedy,
                    )
                    self.plot_lookup_table(path_saved_files)
                    if int(explored) != 100:
                        self.plot_state_exploration(path_saved_files)

    def plot_exploration(
        self,
        path_saved_files: str,
        n_train_epochs: int,
        init_exploration_rate: float,
        final_exploration_rate: float,
        epsilon_greedy: float,
    ) -> None:
        """Plot the epsilon-greedy exploration-rate decay over epochs."""
        epochs = np.arange(n_train_epochs + 1)
        exploration_rate = init_exploration_rate * (
            final_exploration_rate / init_exploration_rate
        ) ** ((epochs / n_train_epochs) ** epsilon_greedy)

        plt.figure(figsize=(10, 10))
        plt.plot(epochs, exploration_rate, label=f"E-greedy decay rate = {epsilon_greedy}")
        plt.title("Exploration Rate Decay over Epochs")
        plt.xlabel("Training Epoch")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            path_saved_files + "Exploration_Rate.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    def plot_lookup_table(self, path_saved_files: str) -> None:
        """Heat-map of the learned post-decision value function."""
        surrogate_array = self.lookup_table.copy()
        fig, ax = plt.subplots(figsize=(10, 10))
        heat_map = ax.imshow(
            surrogate_array, cmap="viridis", interpolation="nearest", aspect="auto"
        )
        plt.colorbar(heat_map, label="PDS Value")

        start_x = int(self.n_decision_points / 5)
        x_labels = [1] + list(range(start_x, self.n_decision_points + 1, start_x))
        ax.set_xlim(0, self.n_decision_points - 1)
        ax.set_xticks([x - 1 for x in x_labels])
        ax.set_xticklabels(x_labels)

        start_y = int(self.initial_knap_cap / 5)
        y_labels = [0] + list(range(start_y, self.initial_knap_cap + 1, start_y))
        ax.set_ylim(0, self.initial_knap_cap)
        ax.set_yticks(list(reversed(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()

        plt.xlabel("Decision Point")
        plt.ylabel("Remaining PDS Capacity")
        plt.title("Final PDS-Values after Training")
        plt.savefig(
            path_saved_files + "Post-Decision Values.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    def load_lookup_table(self, path_lookup: str) -> bool:
        """Load a previously trained lookup table from disk.

        Returns:
            True if the file existed and was loaded, else False.
        """
        if os.path.isfile(path_lookup):
            print("Lookup table for Monte Carlo policy available")
            self.lookup_table = np.loadtxt(path_lookup)
            return True
        return False

    def plot_state_exploration(self, path_saved_files: str) -> None:
        """Binary heat-map of which post-decision states were explored."""
        surrogate_array = np.where(
            self.n_explored_states == 0, 0, np.where(self.n_explored_states >= 1, 1, 0)
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(surrogate_array, cmap="viridis", interpolation="nearest", aspect="auto")
        cmap = plt.get_cmap("viridis", 2)
        legend_labels = [
            mpatches.Patch(color=cmap(1), label="Explored"),
            mpatches.Patch(color=cmap(0), label="Not Explored"),
        ]
        ax.legend(
            handles=legend_labels,
            loc="upper right",
            facecolor="white",
            framealpha=1,
            edgecolor="white",
            fontsize=15,
        )

        start_x = int(self.n_decision_points / 5)
        x_labels = [1] + list(range(start_x, self.n_decision_points + 1, start_x))
        ax.set_xlim(0, self.n_decision_points - 1)
        ax.set_xticks([x - 1 for x in x_labels])
        ax.set_xticklabels(x_labels)

        start_y = int(self.initial_knap_cap / 5)
        y_labels = [0] + list(range(start_y, self.initial_knap_cap + 1, start_y))
        ax.set_ylim(0, self.initial_knap_cap)
        ax.set_yticks(list(reversed(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()

        plt.xlabel("Decision Point")
        plt.ylabel("Remaining PDS Capacity")
        plt.title("Explored States During Training")
        plt.savefig(
            path_saved_files + "Explored_States.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )


class Perfect_Information(Policy):
    """Offline optimal policy: dynamic programming on the *static* knapsack.

    Because it sees all items up front, it is not a valid online policy; it
    serves as the optimal upper bound when benchmarking the other policies.
    """

    def act(
        self,
        env: KnapsackEnvironment,
        n_simulation: int,
        seed: int,
    ) -> tuple[float, float, float]:
        """Solve each simulation run as a classic 0/1 knapsack via DP.

        Returns:
            ``(mean_reward, max_reward, runtime_in_minutes)``
        """
        time_start = time.time()
        final_reward: list[float] = []
        num_items: int = env.n_decision_points
        knapsack_cap: int = env.knapsack_cap_default

        for i in range(n_simulation):
            env.reset(seed=i + seed)
            # state_stage[n][c] = best value using first n items, capacity c.
            state_stage: np.ndarray = np.zeros(
                shape=(num_items + 1, knapsack_cap + 1)
            )
            weights: list[int] = [item.weight for item in env.item_list]
            values: list[int] = [item.value for item in env.item_list]
            for n in range(1, num_items + 1):
                for c in range(knapsack_cap + 1):
                    if weights[n - 1] <= c:
                        state_stage[n][c] = max(
                            state_stage[n - 1][c],
                            values[n - 1] + state_stage[n - 1][c - weights[n - 1]],
                        )
                    else:
                        state_stage[n][c] = state_stage[n - 1][c]
            final_reward.append(
                state_stage[num_items][knapsack_cap] / env.sum_item_value
            )

        max_reward: float = round(float(np.max(final_reward)), 3)
        mean_reward: float = round(float(np.mean(final_reward)), 3)
        runtime: float = round((time.time() - time_start) / 60, 1)
        return mean_reward, max_reward, runtime