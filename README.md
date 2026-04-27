# Stochastic Game-Theoretic Modelling of Adversarial Attacks in Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Federated Learning](https://img.shields.io/badge/Focus-Federated%20Learning-orange.svg)]()

A comprehensive framework for simulating adversarial attacks (Gradient Scaling, Label Flipping, etc.) and robust defense strategies (Krum, Trimmed Mean, etc.) in Federated Learning environments, modeled as a **Stochastic Game**.

---

## 🚀 Quick Start

Get the simulation running in under a minute:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the quick validation script
python experiments/run_experiments_quick.py
```
*Expected output: A miniature FL experiment runs, Nash Equilibrium is computed, and sample plots are generated in `results/plots/`.*

---

## 📂 Project Structure

```bash
.
├── data/
│   └── partition.py          # MNIST loading & IID/Non-IID partitioning
├── models/
│   └── cnn.py                # SimpleCNN global model architecture
├── clients/
│   ├── honest_client.py      # Standard FL client logic
│   └── adversarial_client.py # Malicious client (5 attack types implemented)
├── server/
│   └── aggregator.py         # Server-side robust aggregation (4 defenses)
├── game/
│   └── stochastic_game.py    # CORE: Nash solver, Value Iteration, Game Logic
├── experiments/
│   ├── fl_trainer.py         # Individual FL experiment engine
│   ├── run_experiments.py    # Full grid execution (60+ scenarios)
│   └── run_experiments_quick.py # Fast validation & pipeline test
├── results/
│   ├── plotter.py            # Automated visualization engine
│   ├── federated_learning_graphs.md # Visual summary of all research figures
│   └── simulation_trajectories.csv # Raw data for longitudinal analysis
├── Federated_Learning_Paper.md # Formal 11-figure research manuscript
├── Experimental_Data_Appendix.md # Supplementary data & state transitions
└── demo.py                   # Legacy single-run demonstration
```

---

## 🛠️ Experimentation & Research

### 1. Full Grid Simulation
To replicate the full study (5 attacks × 4 defenses × 3 adversary ratios):
```bash
python experiments/run_experiments.py
```
This generates 11 distinct figures analyzing everything from accuracy convergence to Nash strategy stability.

### 2. Custom Experiment
```python
from experiments.fl_trainer import run_fl_experiment

result = run_fl_experiment(
    n_clients       = 10,
    adversary_ratio = 0.3,
    attack_type     = 'gradient_scale', # options: label_flip, sign_flip, gaussian_noise
    defense         = 'krum',           # options: trimmed_mean, median, fedavg
    rounds          = 50
)
print(f"Final Test Accuracy: {result['config']['final_accuracy']:.2%}")
```

---

## 📊 Visual Results Showcase

The project generates high-fidelity visualizations for research analysis. A detailed view of all figures can be found in **[results/federated_learning_graphs.md](file:///f:/6th%20sem/mathematicalmodelling/results/federated_learning_graphs.md)**.

| Category | Key Figures |
|---|---|
| **Training Dynamics** | Accuracy curves (Attacks vs Defenses), Adversary Ratio effects |
| **Game Theory** | Payoff heatmaps, Nash Equilibrium mixed strategies |
| **Robustness** | Multi-dimensional Radar charts, State transition heatmaps |
| **Sensitivity** | Hyperparameter heatmaps (Local epochs vs Accuracy) |

---

## 🧠 Game-Theoretic Framework

The interaction between the **Attacker** (adversarial clients) and the **Defender** (the server aggregator) is modeled as a **Stochastic Game** $G = (S, N, A, P, R, \gamma)$:

*   **States ($S$):** Accuracy levels {LOW, MID, HIGH}.
*   **Players ($N$):** Attacker vs. Defender.
*   **Actions ($A$):** 5 Attack types × 4 Defense strategies.
*   **Rewards ($R$):** Zero-sum based on accuracy drop/retention.
*   **Solution**: Computed via **Nash Equilibrium Support Enumeration** and **Value Iteration (Bellman Optimality)**.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
