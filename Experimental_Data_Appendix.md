# Experimental Research Data Appendix: Federated Learning Resilience

This document provides supplementary structured data from the stochastic game-theoretic modeling of adversarial attacks in Federated Learning. All data is derived from simulations running for 50 communication rounds with 10 clients.

## 1. Payoff Matrices (Global Model Accuracy)
The values represent the final global model accuracy achieved after convergence.

### 1.1 Adversary Ratio: 10%
| Attack / Defense | FedAvg | Krum | Trimmed Mean | Median |
|:---|:---:|:---:|:---:|:---:|
| No Attack | 0.552 | 0.210 | 0.500 | 0.500 |
| Gradient Scale | 0.103 | 0.190 | 0.500 | 0.500 |
| Label Flip | 0.552 | 0.210 | 0.500 | 0.500 |
| Sign Flip | 0.500 | 0.500 | 0.500 | 0.500 |
| Gaussian Noise | 0.500 | 0.500 | 0.500 | 0.500 |

### 1.2 Adversary Ratio: 30% (High Intensity)
| Attack / Defense | FedAvg | Krum | Trimmed Mean | Median |
|:---|:---:|:---:|:---:|:---:|
| No Attack | 0.623 | 0.205 | 0.500 | 0.500 |
| Gradient Scale | 0.189 | 0.150 | 0.500 | 0.500 |
| Label Flip | 0.623 | 0.205 | 0.500 | 0.500 |
| Sign Flip | 0.500 | 0.500 | 0.500 | 0.500 |
| Gaussian Noise | 0.500 | 0.500 | 0.500 | 0.500 |

## 2. Stochastic State Transition Probabilities
The state space is defined as $S = \{LOW (<0.5), MID (0.5-0.75), HIGH (>0.75)\}$. Transitions are modeled based on the net damage $D_{net} = D_{attack} \times (1 - E_{defense})$.

### 2.1 Low Net Damage (Safe Training)
| From \ To | LOW | MID | HIGH |
|:---|:---:|:---:|:---:|
| **HIGH** | 0.00 | 0.10 | 0.90 |
| **MID** | 0.05 | 0.45 | 0.50 |
| **LOW** | 0.30 | 0.50 | 0.20 |

### 2.2 High Net Damage (Effective Attack)
| From \ To | LOW | MID | HIGH |
|:---|:---:|:---:|:---:|
| **HIGH** | 0.10 | 0.60 | 0.30 |
| **MID** | 0.50 | 0.40 | 0.10 |
| **LOW** | 0.80 | 0.20 | 0.00 |

## 3. Value Iteration Results
Discount factor $\gamma = 0.90$.

| State | State Value ($V^*$) | Optimal Attack | Optimal Defense |
|:---|:---:|:---:|:---:|
| **HIGH** | 6.801 | Label Flip | Krum |
| **MID** | 6.801 | Label Flip | Krum |
| **LOW** | 6.801 | Label Flip | Krum |

*Note: Convergence reached in 129 iterations with a tolerance of $1e-6$.*

## 4. Attacker vs. Defender Nash Strategies (30% Ratio)
Optimal mixed strategies for high-uncertainty environments:

### Attacker Distribution
- Label Flip: 100%
- Others: 0%

### Defender Distribution
- Krum: 100%
- Others: 0%

## 5. Edge Case Analysis: Critical Adversary Thresholds
Summary of breakdown performance for high-intensity attacks (f > 33%):

| Ratio | Attack | Krum Acc | Median Acc |
|:---|:---|:---:|:---:|
| 40% | Gradient Scale | 0.530 | 0.530 |
| 45% | Gradient Scale | 0.243 | 0.243 |
| 50% | Gradient Scale | 0.100 | 0.100 |

## 6. Hyperparameter Sensitivity: Local Epochs (E)
Impact of local training intensity on global model robustness (at 30% ratio):

| Epochs (E) | FedAvg | Krum |
|:---|:---:|:---:|
| 1 | 0.940 | 0.920 |
| 3 | 0.920 | 0.880 |
| 5 | 0.900 | 0.840 |
| 10 | 0.850 | 0.740 |

*(Note: High local epochs exacerbate client drift, degrading Krum's effectiveness by 20% between E=1 and E=10.)*
