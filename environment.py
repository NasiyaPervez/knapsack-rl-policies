"""The KnapsackEnvironment for the simulated dynamic knapsack problem.

This is the online version of the knapsack problem. Items arrive one at a
time (a stream of ``n_decision_points`` items). At each decision point the
decision maker observes the current item and the remaining capacity, decides
to accept or reject it, and only then learns the next item. A policy maps a
state to a Boolean action.

The environment exposes a small, gym-like interface:

* ``reset(seed)``  -> initial state
* ``step(action)`` -> ``(next_state, reward, terminated)``
* ``get_state``    -> the current state (dictionary)
* ``terminated``   -> whether the item stream is exhausted

Coursework: Optimization Implementation in Production and Logistics (OVGU
Magdeburg), dynamic knapsack assignment.
"""

from item import Item


class KnapsackEnvironment:
    """Generates and advances through a stream of items.

    Args:
        n_decision_points: Number of items offered per simulation run.
        expected_total_weight_items: Expected total weight over all items;
            used to scale the knapsack capacity.
        expected_mean_weight_item: Mean of the item-weight distribution.
        std_deviation_weight_item: Std of the item-weight distribution.
        knapsack_cap: Initial capacity as a *share* of the expected total
            item weight (e.g. ``0.3`` = the knapsack can hold ~30% of all
            offered weight).
        correlation_factor: Correlation between item weight and value in
            ``[0, 1]``.
    """

    def __init__(
        self,
        n_decision_points: int,
        expected_total_weight_items: int,
        expected_mean_weight_item: int,
        std_deviation_weight_item: float,
        knapsack_cap: float,
        correlation_factor: float,
    ) -> None:
        self.n_decision_points: int = n_decision_points
        self.expected_mean_weight_item: int = expected_mean_weight_item
        self.std_deviation_weight_item: float = std_deviation_weight_item
        self.knapsack_cap_default: int = int(
            knapsack_cap * expected_total_weight_items
        )
        self.correlation_factor: float = correlation_factor

    def create_items(self, seed: int) -> list[Item]:
        """Build the full item stream for one simulation run."""
        item_list: list[Item] = []
        for item_idx in range(1, self.n_decision_points + 1):
            item_list.append(
                Item(
                    item_id=item_idx,
                    seed=seed,
                    expected_mean_weight_item=self.expected_mean_weight_item,
                    std_deviation_weight_item=self.std_deviation_weight_item,
                    correlation_factor=self.correlation_factor,
                )
            )
        return item_list

    def reset(self, seed: int = 0) -> dict:
        """Reset the episode: rebuild items, empty knapsack and reward.

        Returns:
            The initial state dictionary.
        """
        self.cur_decision_point: int = 1
        self.cur_knapsack_cap: float = self.knapsack_cap_default
        self.item_list: list[Item] = self.create_items(seed=seed)
        self.total_reward: int = 0
        self.sum_item_value: int = sum(item.value for item in self.item_list)
        return self.get_state

    @property
    def get_state(self) -> dict:
        """The state observable by the decision maker.

        It deliberately hides the *future* items, keeping the problem online.
        """
        return {
            "Cur_DP": self.cur_decision_point,
            "Cur_Cap": self.cur_knapsack_cap,
            "Item": self.item_list[self.cur_decision_point - 1],
        }

    @property
    def terminated(self) -> bool:
        """True once every item in the stream has been offered."""
        return self.cur_decision_point > self.n_decision_points

    def step(self, action: bool) -> tuple[dict, int, bool]:
        """Apply a decision and transition to the next decision point.

        Args:
            action: ``True`` = accept the current item, ``False`` = reject.

        Returns:
            ``(next_state, reward, terminated)``. If the run terminates the
            returned "state" is a minimal final state.
        """
        reward: int = 0
        if action:
            reward = self.get_state["Item"].value
            self.total_reward += reward
            self.cur_knapsack_cap -= self.get_state["Item"].weight
        self.cur_decision_point += 1

        if self.terminated:
            final_state: dict = {
                "Cur_DP": self.cur_decision_point,
                "Cur_Cap": self.cur_knapsack_cap,
            }
            return final_state, reward, True

        return self.get_state, reward, False