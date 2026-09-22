"""The Item class of the simulated dynamic (online) knapsack problem.

Each ``Item`` is generated stochastically: its weight is drawn from a normal
distribution around a given mean, and its value is built from that weight
using a correlation factor (``0`` = uncorrelated, ``1`` = perfectly
correlated). A fixed random seed makes the item stream reproducible across
policies, so every policy is tested against the exact same instance.

Coursework: Optimization Implementation in Production and Logistics (OVGU
Magdeburg), dynamic knapsack assignment.
"""

import numpy as np


class Item:
    """A single item that is offered to the decision maker at one decision point.

    Attributes:
        item_id: Position of the item in the offer stream (1-based).
        weight: Positive integer weight sampled from a normal distribution.
        value: Positive integer value correlated with the weight.
    """

    def __init__(
        self,
        item_id: int,
        seed: int,
        expected_mean_weight_item: int,
        std_deviation_weight_item: float,
        correlation_factor: float,
    ) -> None:
        self.item_id: int = item_id

        # Seeding per item reproduces the exact same weight/value sequence for
        # every policy that is evaluated on this environment.
        np.random.seed(seed + self.item_id)

        # Weight ~ Normal(mean, std), clamped to >= 1.
        self.weight: float = np.random.normal(
            loc=expected_mean_weight_item,
            scale=std_deviation_weight_item,
        )
        self.weight: int = max(1, int(round(self.weight)))

        # Value interpolates between a pure function of weight (corr = 1) and a
        # uniform draw (corr = 0).
        self.value: float = correlation_factor * self.weight + (
            1 - correlation_factor
        ) * np.random.uniform(
            low=expected_mean_weight_item - 2 * std_deviation_weight_item,
            high=expected_mean_weight_item + 2 * std_deviation_weight_item,
        )
        self.value: int = max(1, int(round(self.value)))

    def __str__(self) -> str:
        return f"{self.item_id}, {self.weight}, {self.value}"