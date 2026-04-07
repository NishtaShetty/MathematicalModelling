# Stochastic Game-Theoretic Modelling of Adversarial Attacks in Federated Learning

## Project Structure

```
federated_game/
├── data/
│   └── partition.py          ← MNIST loading, IID/Non-IID partitioning
├── models/
│   └── cnn.py                ← SimpleCNN (shared global model)
├── clients/
│   ├── honest_client.py      ← Standard FL client
│   └── adversarial_client.py ← Malicious client (5 attack types)
├── server/
│   └── aggregator.py         ← Server with 4 defense strategies
├── game/
│   └── stochastic_game.py    ← CORE: Stochastic game, Nash solver, VI
├── experiments/
│   ├── fl_trainer.py         ← Single FL experiment runner
│   └── run_experiments.py    ← Full grid + plots
├── results/
│   └── plotter.py            ← All paper figures
├── demo.py                   ← Quick test (run this first)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Verify Setup (run first)
```bash
cd MathematicalModelling
python demo.py
```
Expected output: FL experiment runs, Nash Equilibrium computed, game simulated.

### 2. Run Full Experiment Grid
```bash
python experiments/run_experiments.py
```
This runs all 60 experiments (5 attacks × 4 defenses × 3 adversary ratios) and generates all paper figures.

### 3. Single Experiment (custom config)
```python
from experiments.fl_trainer import run_fl_experiment

result = run_fl_experiment(
    n_clients       = 10,
    adversary_ratio = 0.3,      # 30% adversarial
    attack_type     = 'gradient_scale',
    defense         = 'krum',
    rounds          = 50,
    local_epochs    = 3
)
print(result['config']['final_accuracy'])
```

### 4. Game Analysis Only (with your own payoffs)
```python
import numpy as np
from game.stochastic_game import StochasticGame

game = StochasticGame(adversary_ratio=0.3)

# Your empirical payoff matrix [5 attacks x 4 defenses]
payoffs = np.array([...])
game.set_payoffs(payoffs)

# Nash Equilibrium
equilibria = game.compute_nash_equilibrium()

# Value Iteration
V, policy_a, policy_d = game.value_iteration()
```

---

## Attack Types
| Attack | Description |
|---|---|
| `no_attack` | Honest behavior (baseline) |
| `gradient_scale` | Multiplies gradients by -5 (severe damage) |
| `label_flip` | Flips label 1→7 during training |
| `sign_flip` | Negates all gradient directions |
| `gaussian_noise` | Adds large Gaussian noise to weights |

## Defense Strategies
| Defense | Description |
|---|---|
| `fedavg` | Simple average (no defense) |
| `krum` | Select most consistent update |
| `trimmed_mean` | Remove top/bottom 20% updates |
| `median` | Coordinate-wise median |

---

## Paper Figures Generated
- `fig1_attacks_vs_krum.png` — Accuracy curves, all attacks vs Krum
- `fig2_defenses_vs_grad_scale.png` — All defenses vs gradient scaling
- `fig3_payoff_heatmap_*.png` — Payoff matrix heatmaps per adversary ratio
- `fig4_nash_*.png` — Nash Equilibrium mixed strategies
- `fig5_adv_ratio_effect.png` — Defense robustness vs adversary ratio
- `fig6_nash_vs_baseline_*.png` — Nash strategies vs no-defense comparison

---

## Game-Theoretic Model

**Stochastic Game G = (S, N, A, P, R, γ)**

- **S**: States = {LOW, MID, HIGH} (accuracy buckets)
- **N**: Players = {Attacker, Defender}
- **A**: Attacker × Defender action spaces (5 × 4)
- **P**: Stochastic transition P(s'|s, a_attack, a_defense)
- **R**: Attacker reward = accuracy_drop; Defender reward = accuracy
- **γ**: Discount factor = 0.9

**Solution Concept**: Nash Equilibrium via support enumeration (nashpy) + Value Iteration (Bellman equations)
