# Dynamic Knapsack with Decision Policies and Monte-Carlo RL

**Short description:** Simulating the *online* knapsack problem and benchmarking six
decision policies — from simple heuristics up to a Monte-Carlo reinforcement-learning agent
and the optimal offline solution.

## What this is

Coursework for *Optimization Implementation in Production and Logistics* (OVGU Magdeburg,
SoSe 2025): the **dynamic (online) knapsack problem**. Items arrive one at a time; at every
decision point you see only the current item and the remaining capacity, and must accept or
reject it before the next item appears. The knapsack cannot hold everything, so a good policy
must balance immediate gain against reserving space for better items later.

A shared seed keeps the item streams identical across all policies, so their rewards are
directly comparable, and the reward metric is *relative* (value collected ÷ total value of all
items offered).

## Implemented policies

| Policy | Idea |
|---|---|
| `Greedy` | Accept whenever the item fits. |
| `Threshold` | Accept only if value/weight ≥ a set threshold (default 1.0). |
| `Reserve_Cap` | Keep a capacity buffer that shrinks linearly as items run out. |
| `Sample` | Accept if the current item beats the average quality of a few sampled future items. |
| `MC_RL` | **Monte-Carlo reinforcement learning**: learns a value function over post-decision states (decision point × remaining capacity) via ε-greedy exploration, then acts greedily on it. |
| `Perfect_Information` | Dynamic programming on the *static* deterministic knapsack — an offline optimal upper bound for comparison. |

## Repository layout

```
.
├── item.py          # Item: weight/value sampled around a distribution
├── environment.py   # KnapsackEnvironment: online item stream, reset/step
├── policies.py      # All six policies
├── main.py          # Runner: evaluates every policy, prints the results table
└── output/          # Created at runtime: MC_RL plots + learned tables
```

## Requirements & setup

- Python 3.10+ (higher-order type hints such as `list[Item]` require 3.9+)
- `numpy`, `pandas`, `matplotlib`

```bash
pip install numpy pandas matplotlib
python main.py
```

## Expected output

`main.py` prints a table of **average relative reward**, **max reward**, and **training /**
**testing time** per policy. Reward is measured relative to the *total value of all items
offered*, so the scale is roughly `0.3 – 1.0`. On the default parameters the policies order
themselves as:

```
Perfect_Information (offline optimum)  >>  MC_RL  >  Threshold  >  Reserve_Cap  >  Greedy ≈ Sample
```

Exactly which heuristic beats which depends on the seed and instance parameters — run it and
see. `MC_RL` with `use_pre_trained_table = True` in `main.py` expects a trained table at
`output/Lookup_Table.txt`; training from scratch (default 1000 epochs) takes a short while.

## Parameter notes

- `knapsack_cap = 0.3`: the knapsack can hold about 30% of the expected total offered weight,
  which makes the problem genuinely hard (you cannot just take everything good).
- `correlation_factor = 0.6`: value is partially but not perfectly correlated with weight.
- `epsilon_greedy` (default 2.0) shapes how fast exploration decays during `MC_RL` training.

## License

MIT — see [LICENSE](LICENSE). This repository contains only the author's own code.