import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import run_full_grid, run_game_analysis, generate_all_plots

if __name__ == '__main__':
    ADVERSARY_RATIOS = [0.3]
    ATTACK_TYPES     = ['gradient_scale', 'label_flip']
    DEFENSE_TYPES    = ['fedavg']
    ROUNDS           = 1
    N_CLIENTS        = 4
    SAVE_DIR         = './results'

    print("Running quick experiment...")
    all_results, payoff_matrices = run_full_grid(
        adversary_ratios = ADVERSARY_RATIOS,
        attack_types     = ATTACK_TYPES,
        defense_types    = DEFENSE_TYPES,
        rounds           = ROUNDS,
        n_clients        = N_CLIENTS,
        save_dir         = SAVE_DIR
    )
    
    print("\nRunning game analysis...")
    game_results = run_game_analysis(
        all_results      = all_results,
        payoff_matrices  = payoff_matrices,
        adversary_ratios = ADVERSARY_RATIOS,
        save_dir         = SAVE_DIR
    )

    print("\nGenerating plots...")
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
    
    print("\n\nExperiment completed successfully! See printed console output above.")
