# Stochastic Game-Theoretic Modelling of Adversarial Attacks in Federated Learning

A comprehensive research framework for simulating adversarial attacks and robust defense strategies in Federated Learning, modeled as a **Data-Driven Stochastic Game**.

---

## 🚀 Quick Start

Get the simulation running in under a minute:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the quick validation script
python experiments/run_experiments_quick.py
```
*Expected output: A miniature FL experiment runs, empirical state transitions are fitted, Nash Equilibrium is computed, and sample plots are generated in `results/plots/`.*

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
│   └── aggregator.py         # Robust aggregation (FedAvg, Krum, Trimmed Mean, Median, Bulyan)
├── game/
│   └── stochastic_game.py    # CORE: Empirical Transition fitting, Nash solver, Value Iteration
├── experiments/
│   ├── fl_trainer.py         # FL engine with cost & transition tracking
│   ├── run_experiments.py    # Full research grid execution
│   └── run_experiments_quick.py # Pipeline validation script
├── results/
│   ├── plotter.py            # visualization engine (Line, Heatmap, Radar, Cost-Scatter)
│   ├── federated_learning_graphs.md # Visual summary of all research figures
│   └── simulation_trajectories.csv # Raw simulation data
├── Federated_Learning_Paper.md # Formal 12-figure research manuscript
├── Experimental_Data_Appendix.md # Supplementary data & state transitions
└── demo.py                   # Legacy single-run demonstration
```

---

## 🛠️ Experimentation & Research

### 1. Full Grid Simulation
To replicate the full study (5 attacks × 5 defenses × 3 adversary ratios):
```bash
python experiments/run_experiments.py
```
This script now executes a data-driven pipeline:
1.  **FL Simulation**: Runs all combinations and logs round-by-round accuracy.
2.  **Transition Fitting**: Estimates $P(s'|s, a, d)$ based on actual accuracy fluctuations.
3.  **Game Solving**: Computes Nash Equilibrium using the empirical transition matrix.

### 2. Research Enhancements
The framework now supports advanced research-level features:
*   **Bulyan Defense**: Integrated state-of-the-art Byzantine-robust aggregator.
*   **Asymmetric Rewards**: Utility functions now incorporate **Defense Computation Cost** and **Attacker Detection Risk**.
*   **Empirical Markov Chains**: Transitions are fitted from experimental data, moving beyond heuristic assumptions.

---

## 📊 Visual Results Showcase

The project generates high-fidelity visualizations for research analysis. A detailed view of all figures can be found in **[results/federated_learning_graphs.md](file:///f:/6th%20sem/mathematicalmodelling/results/federated_learning_graphs.md)**.

| Category | Key Figures |
|---|---|
| **Training Dynamics** | Accuracy curves (Attacks vs Defenses), Adversary Ratio effects |
| **Game Theory** | Payoff heatmaps, Nash Equilibrium mixed strategies |
| **Robustness** | Multi-dimensional Radar charts, Empirical state transition heatmaps |
| **Trade-offs** | **Figure 12: Cost-Effectiveness Scatter Plot** (Robustness vs. Overhead) |

---

## 🧠 Game-Theoretic Framework

The interaction between the **Attacker** (adversarial clients) and the **Defender** (server aggregator) is modeled as a **Stochastic Game** $G = (S, N, A, P, R, \gamma)$:

*   **States ($S$):** Accuracy levels {LOW, MID, HIGH}.
*   **Actions ($A$):** 5 Attack types × 5 Defense strategies.
*   **Transitions ($P$):** **Empirically estimated** from round-by-round simulation data.
*   **Rewards ($R$):** Asymmetric functions considering accuracy, computation time, and attack risk.
*   **Solution**: Computed via **Nash Equilibrium Support Enumeration** and **Value Iteration**.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
