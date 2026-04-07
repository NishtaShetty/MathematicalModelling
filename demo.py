"""
demo.py
═══════════════════════════════════════════════════════════════════
Quick demo: runs a single FL experiment + game analysis.
Use this to verify your setup works before running the full grid.

Run: python demo.py
═══════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from experiments.fl_trainer import run_fl_experiment
from game.stochastic_game import StochasticGame, ATTACK_ACTIONS, DEFENSE_ACTIONS


def main():
    print("=" * 60)
    print("DEMO: Federated Learning + Stochastic Game Analysis")
    print("=" * 60)

    # ── STEP 1: Run a single FL experiment ───────────────────────
    print("\n[STEP 1] Running FL experiment...")
    print("  Config: gradient_scale attack, krum defense, 30% adversaries")

    result = run_fl_experiment(
        n_clients       = 10,
        adversary_ratio = 0.3,
        attack_type     = 'gradient_scale',
        defense         = 'krum',
        rounds          = 20,          # short for demo
        local_epochs    = 2,
        verbose         = True
    )

    print(f"\n  Final accuracy: {result['config']['final_accuracy']:.4f}")
    print(f"  Accuracy history (every 5 rounds): "
          f"{[round(a,3) for a in result['accuracies'][::5]]}")

    # ── STEP 2: Build a payoff matrix (demo uses estimates) ───────
    print("\n[STEP 2] Building payoff matrix from estimates...")

    game = StochasticGame(adversary_ratio=0.3, gamma=0.9)
    payoffs = game.build_default_payoffs()

    # ── STEP 3: Nash Equilibrium ──────────────────────────────────
    print("\n[STEP 3] Computing Nash Equilibrium...")
    equilibria = game.compute_nash_equilibrium()

    if equilibria:
        sigma_a, sigma_d = equilibria[0]
        print("\n  Attacker's Nash Mixed Strategy:")
        for i, p in enumerate(sigma_a):
            print(f"    {ATTACK_ACTIONS[i]:<18}: {p:.3f}")
        print("\n  Defender's Nash Mixed Strategy:")
        for i, p in enumerate(sigma_d):
            print(f"    {DEFENSE_ACTIONS[i]:<14}: {p:.3f}")

    # ── STEP 4: Value Iteration ───────────────────────────────────
    print("\n[STEP 4] Running Value Iteration...")
    V, policy_att, policy_def = game.value_iteration(max_iter=200)

    # ── STEP 5: Game Simulation ───────────────────────────────────
    print("\n[STEP 5] Simulating 50-round game with Nash strategies...")
    history, states = game.simulate_game(T=50, initial_accuracy=0.9, use_nash=True)
    print(f"  Accuracy trajectory (every 10 rounds): "
          f"{[round(h, 3) for h in history[::10]]}")
    print(f"  State trajectory:  {states[::10]}")
    print(f"  Final accuracy at Nash Eq: {history[-1]:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete! Setup is working correctly.")
    print("\nNext steps:")
    print("  1. Run full experiments: python experiments/run_experiments.py")
    print("  2. Check results/plots/ for paper figures")
    print("=" * 60)


if __name__ == '__main__':
    main()
