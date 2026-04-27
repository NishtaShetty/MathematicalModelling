# Federated Learning Attack & Defense Simulation Plots

> [!TIP]
> A formal paper based on these results has been generated: [Federated_Learning_Paper.md](file:///Users/tanujs/untitled%20folder/MathematicalModelling/Federated_Learning_Paper.md).

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

## Advanced Research Visualizations

![fig7_robustness_radar.png](./fig7_robustness_radar.png)
*Figure 7: Radar chart comparing defense robustness across all attack vectors.*

![fig8_state_transitions.png](./fig8_state_transitions.png)
*Figure 8: Stochastic state transition probability heatmap.*

![fig9_summary_bar.png](./fig9_summary_bar.png)
*Figure 9: Comprehensive summary of all attack/defense performance.*

![fig10_adversary_ratio_breakdown.png](./fig10_adversary_ratio_breakdown.png)
*Figure 10: Analysis of defense breakdown under critical adversary ratios (>40%).*

![fig11_hyperparameter_sensitivity.png](./fig11_hyperparameter_sensitivity.png)
*Figure 11: Heatmap showing sensitivity of robust aggregation to local training epochs.*

> **Note**
> Additional adversary ratio graphs (10% and 20%) are also saved in the local `results/` folder if you wish to see how lower attack concentrations affect the outcomes.
