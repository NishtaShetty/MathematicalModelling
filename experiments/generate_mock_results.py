import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import run_game_analysis, generate_all_plots
from game.stochastic_game import StochasticGame

if __name__ == '__main__':
    ADVERSARY_RATIOS = [0.1, 0.2, 0.3]
    ATTACK_TYPES = ['no_attack', 'gradient_scale', 'label_flip', 'sign_flip', 'gaussian_noise']
    DEFENSE_TYPES = ['fedavg', 'krum', 'trimmed_mean', 'median']
    ROUNDS = 50
    SAVE_DIR = '../results'

    print("Generating MOCK payoff matrices...")
    payoff_matrices = {}
    st_game = StochasticGame()
    base_payoff = st_game.build_default_payoffs()
    
    all_results = {}
    for ratio in ADVERSARY_RATIOS:
        # scale down accuracy based on adversary ratio
        decay = (ratio - 0.1) * 2.0
        mat = np.clip(base_payoff - decay, 0.1, 1.0)
        payoff_matrices[ratio] = mat
        
        for a_idx, attack in enumerate(ATTACK_TYPES):
            for d_idx, defense in enumerate(DEFENSE_TYPES):
                final_acc = mat[a_idx, d_idx]
                # mock a learning curve that goes up to final_acc
                accs = [final_acc * (1 - 0.8 * np.exp(-x/5)) for x in range(ROUNDS)]
                
                key = (ratio, attack, defense)
                all_results[key] = {
                    'accuracies': accs,
                    'config': {'final_accuracy': final_acc}
                }

    print("Mock matrices generated. Running game analysis...")
    game_results = run_game_analysis(
        payoff_matrices=payoff_matrices,
        adversary_ratios=ADVERSARY_RATIOS,
        save_dir=SAVE_DIR
    )

    print("Generating all plots using mock data...")
    generate_all_plots(
        all_results=all_results,
        payoff_matrices=payoff_matrices,
        game_results=game_results,
        adversary_ratios=ADVERSARY_RATIOS,
        attack_types=ATTACK_TYPES,
        defense_types=DEFENSE_TYPES,
        rounds=ROUNDS,
        save_dir=SAVE_DIR
    )

    print("Plots generated successfully in results directory.")
