# Federated Learning Attack & Defense Simulation Plots

Here are the game-theoretic and training curves generated from the attack simulations.

## Accuracy Curves and Attack Effect

![fig1_attacks_vs_krum.png](./fig1_attacks_vs_krum.png)
*Figure 1: Accuracy impact under varying attack strategies vs. Krum defense.*

![fig2_defenses_vs_grad_scale.png](./fig2_defenses_vs_grad_scale.png)
*Figure 2: Performance of various defenses against Gradient Scaling attacks.*

![fig5_adv_ratio_effect.png](./fig5_adv_ratio_effect.png)
*Figure 5: The effect of adversarial client ratio on overall performance.*

## Game-Theoretic Analysis (Adversary Ratio = 30%)

![fig3_payoff_heatmap_30.png](./fig3_payoff_heatmap_30.png)
*Figure 3: Payoff matrix heatmap showing theoretical accuracy outcomes at a 30% adversary ratio.*

![fig4_nash_30.png](./fig4_nash_30.png)
*Figure 4: Computed optimal Nash Equilibrium strategies.*

![fig6_nash_vs_baseline_30.png](./fig6_nash_vs_baseline_30.png)
*Figure 6: Accuracy when running with Optimal Nash Strategies vs default FedAvg.*

> **Note**
> Additional adversary ratio graphs (10% and 20%) are also saved in the local `results/` folder if you wish to see how lower attack concentrations affect the outcomes.
