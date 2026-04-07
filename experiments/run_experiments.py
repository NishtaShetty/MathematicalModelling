"""
experiments/run_experiments.py
═══════════════════════════════════════════════════════════════════
Full experiment grid:
  - Vary adversary ratio: 10%, 20%, 30%
  - Vary attack types: all 5 attacks
  - Vary defenses: all 4 defenses
  - Compute Nash Equilibrium per scenario
  - Generate all paper plots
═══════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import pickle
from experiments.fl_trainer import run_fl_experiment
from game.stochastic_game import StochasticGame, ATTACK_ACTIONS, DEFENSE_ACTIONS
from results.plotter import (plot_accuracy_curves, plot_payoff_heatmap,
                              plot_nash_strategies, plot_adversary_ratio_effect,
                              plot_value_iteration_convergence)


def run_full_grid(
    adversary_ratios = [0.1, 0.2, 0.3],
    attack_types     = ['no_attack', 'gradient_scale', 'label_flip', 'sign_flip', 'gaussian_noise'],
    defense_types    = ['fedavg', 'krum', 'trimmed_mean', 'median'],
    rounds           = 50,
    n_clients        = 10,
    save_dir         = './results'
):
    """
    Run all (attack, defense, adversary_ratio) combinations.
    Builds payoff matrices and runs game-theoretic analysis.

    Returns:
        all_results (dict): Full result grid.
        payoff_matrices (dict): Per adversary_ratio payoff matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}
    payoff_matrices = {}

    total = len(adversary_ratios) * len(attack_types) * len(defense_types)
    done = 0

    print(f"\n{'#'*60}")
    print(f"# RUNNING FULL EXPERIMENT GRID ({total} experiments)")
    print(f"{'#'*60}\n")

    for adv_ratio in adversary_ratios:
        # Initialize payoff matrix for this adversary ratio
        payoff_matrix = np.zeros((len(attack_types), len(defense_types)))

        for a_idx, attack in enumerate(attack_types):
            for d_idx, defense in enumerate(defense_types):
                done += 1
                print(f"[{done}/{total}] adv={adv_ratio*100:.0f}% | "
                      f"attack={attack:<16} | defense={defense}")

                result = run_fl_experiment(
                    n_clients       = n_clients,
                    adversary_ratio = adv_ratio,
                    attack_type     = attack,
                    defense         = defense,
                    rounds          = rounds,
                    local_epochs    = 3,
                    verbose         = False
                )

                key = (adv_ratio, attack, defense)
                all_results[key] = result

                # Fill payoff matrix with final accuracy
                final_acc = result['config']['final_accuracy']
                payoff_matrix[a_idx, d_idx] = final_acc

        payoff_matrices[adv_ratio] = payoff_matrix

        # Print payoff matrix summary
        print(f"\n  Payoff Matrix (adv_ratio={adv_ratio*100:.0f}%):")
        print(f"  {'':18}", end='')
        for d in defense_types:
            print(f"{d:>12}", end='')
        print()
        for a_idx, attack in enumerate(attack_types):
            print(f"  {attack:<18}", end='')
            for d_idx in range(len(defense_types)):
                print(f"{payoff_matrix[a_idx, d_idx]:>12.3f}", end='')
            print()
        print()

    # Save raw results
    with open(f'{save_dir}/all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    with open(f'{save_dir}/payoff_matrices.pkl', 'wb') as f:
        pickle.dump(payoff_matrices, f)

    print(f"\nResults saved to {save_dir}/")
    return all_results, payoff_matrices


def run_game_analysis(payoff_matrices, adversary_ratios, save_dir='./results'):
    """
    For each adversary ratio, run the stochastic game analysis:
      1. Set empirical payoff matrices
      2. Compute Nash Equilibria
      3. Run value iteration
      4. Simulate game trajectories
    """
    game_results = {}

    for adv_ratio in adversary_ratios:
        print(f"\n{'='*50}")
        print(f"GAME ANALYSIS: adversary_ratio = {adv_ratio*100:.0f}%")
        print(f"{'='*50}")

        game = StochasticGame(adversary_ratio=adv_ratio, gamma=0.9)
        game.set_payoffs(payoff_matrices[adv_ratio])

        # Compute Nash Equilibrium
        equilibria = game.compute_nash_equilibrium()

        # Value Iteration
        V, policy_attack, policy_defense = game.value_iteration(max_iter=500)

        # Simulate game with Nash strategies
        history_nash, states_nash = game.simulate_game(
            T=100, initial_accuracy=0.9, use_nash=True
        )

        # Simulate with no defense (baseline)
        history_base, states_base = game.simulate_game(
            T=100, initial_accuracy=0.9,
            use_nash=False, attack_idx=1, defense_idx=0  # grad_scale vs fedavg
        )

        game_results[adv_ratio] = {
            'game':            game,
            'equilibria':      equilibria,
            'value_function':  V,
            'policy_attack':   policy_attack,
            'policy_defense':  policy_defense,
            'history_nash':    history_nash,
            'history_base':    history_base,
        }

    return game_results


def generate_all_plots(all_results, payoff_matrices, game_results,
                       adversary_ratios, attack_types, defense_types,
                       rounds, save_dir='./results/plots'):
    """Generate all figures needed for the paper."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating paper plots → {save_dir}/")

    # Figure 1: Accuracy curves — all attacks vs krum, adv=30%
    adv = 0.3
    curves = {}
    for attack in attack_types:
        key = (adv, attack, 'krum')
        if key in all_results:
            curves[attack] = all_results[key]['accuracies']
    plot_accuracy_curves(
        curves, title=f"Accuracy Under Different Attacks (adv={adv*100:.0f}%, defense=Krum)",
        save_path=f"{save_dir}/fig1_attacks_vs_krum.png"
    )

    # Figure 2: Accuracy curves — gradient_scale vs all defenses, adv=30%
    adv = 0.3
    attack = 'gradient_scale'
    curves2 = {}
    for defense in defense_types:
        key = (adv, attack, defense)
        if key in all_results:
            curves2[defense] = all_results[key]['accuracies']
    plot_accuracy_curves(
        curves2,
        title=f"Defenses Against Gradient Scaling (adv={adv*100:.0f}%)",
        save_path=f"{save_dir}/fig2_defenses_vs_grad_scale.png"
    )

    # Figure 3: Payoff heatmap for each adversary ratio
    for adv_ratio in adversary_ratios:
        plot_payoff_heatmap(
            payoff_matrices[adv_ratio],
            attack_labels=attack_types,
            defense_labels=defense_types,
            title=f"Payoff Matrix (Final Accuracy) — Adversary Ratio={adv_ratio*100:.0f}%",
            save_path=f"{save_dir}/fig3_payoff_heatmap_{int(adv_ratio*100)}.png"
        )

    # Figure 4: Nash Equilibrium strategies
    for adv_ratio in adversary_ratios:
        gr = game_results[adv_ratio]
        if gr['equilibria']:
            sigma_a, sigma_d = gr['equilibria'][0]
            plot_nash_strategies(
                sigma_a, sigma_d,
                attack_labels=attack_types,
                defense_labels=defense_types,
                title=f"Nash Equilibrium Mixed Strategies (adv={adv_ratio*100:.0f}%)",
                save_path=f"{save_dir}/fig4_nash_{int(adv_ratio*100)}.png"
            )

    # Figure 5: Effect of adversary ratio on final accuracy (per defense)
    plot_adversary_ratio_effect(
        all_results, adversary_ratios, attack_types, defense_types,
        save_path=f"{save_dir}/fig5_adv_ratio_effect.png"
    )

    # Figure 6: Nash vs No-defense game simulation
    for adv_ratio in adversary_ratios:
        gr = game_results[adv_ratio]
        plot_accuracy_curves(
            {
                'Nash Eq Strategies': gr['history_nash'],
                'No Defense (FedAvg)': gr['history_base'],
            },
            title=f"Game Simulation: Nash vs No-Defense (adv={adv_ratio*100:.0f}%)",
            save_path=f"{save_dir}/fig6_nash_vs_baseline_{int(adv_ratio*100)}.png"
        )

    print(f"  All plots saved.")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ADVERSARY_RATIOS = [0.1, 0.2, 0.3]
    ATTACK_TYPES     = ['no_attack', 'gradient_scale', 'label_flip', 'sign_flip', 'gaussian_noise']
    DEFENSE_TYPES    = ['fedavg', 'krum', 'trimmed_mean', 'median']
    ROUNDS           = 50
    N_CLIENTS        = 10
    SAVE_DIR         = './results'

    # Step 1: Run all FL experiments
    all_results, payoff_matrices = run_full_grid(
        adversary_ratios = ADVERSARY_RATIOS,
        attack_types     = ATTACK_TYPES,
        defense_types    = DEFENSE_TYPES,
        rounds           = ROUNDS,
        n_clients        = N_CLIENTS,
        save_dir         = SAVE_DIR
    )

    # Step 2: Game-theoretic analysis
    game_results = run_game_analysis(
        payoff_matrices  = payoff_matrices,
        adversary_ratios = ADVERSARY_RATIOS,
        save_dir         = SAVE_DIR
    )

    # Step 3: Generate all paper plots
    generate_all_plots(
        all_results      = all_results,
        payoff_matrices  = payoff_matrices,
        game_results     = game_results,
        adversary_ratios = ADVERSARY_RATIOS,
        attack_types     = ATTACK_TYPES,
        defense_types    = DEFENSE_TYPES,
        rounds           = ROUNDS,
        save_dir         = f'{SAVE_DIR}/plots'
    )

    print("\n\nAll experiments complete. Check results/plots/ for figures.")
