"""
game/stochastic_game.py
═══════════════════════════════════════════════════════════════════
CORE OF THE PAPER: Stochastic Game-Theoretic Model

Models the adversarial interaction in FL as a stochastic game:
    G = (S, N, A, P, R, γ)

Players:
    - Attacker (adversarial clients)
    - Defender (server)

States:
    Discretized (accuracy, poisoning_level) buckets

Actions:
    Attacker: {no_attack, gradient_scale, label_flip, sign_flip, gaussian_noise}
    Defender: {fedavg, krum, trimmed_mean, median}

Transitions:
    P(s'|s, a_att, a_def) — stochastic, estimated from FL simulation results

Rewards:
    Attacker: accuracy_drop - attack_cost
    Defender: accuracy_maintained - defense_cost
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import nashpy as nash
from itertools import product


# ─── Action Spaces ────────────────────────────────────────────────────────────

ATTACK_ACTIONS  = ['no_attack', 'gradient_scale', 'label_flip', 'sign_flip', 'gaussian_noise']
DEFENSE_ACTIONS = ['fedavg', 'krum', 'trimmed_mean', 'median']

N_ATTACK  = len(ATTACK_ACTIONS)
N_DEFENSE = len(DEFENSE_ACTIONS)

# ─── State Space ──────────────────────────────────────────────────────────────
# States = accuracy buckets: LOW(<0.5), MID(0.5-0.75), HIGH(>0.75)
STATES = ['LOW', 'MID', 'HIGH']
STATE_THRESHOLDS = [0.0, 0.5, 0.75, 1.01]

def accuracy_to_state(acc):
    if acc < 0.5:
        return 'LOW'
    elif acc < 0.75:
        return 'MID'
    else:
        return 'HIGH'

def state_to_idx(state):
    return STATES.index(state)


class StochasticGame:
    """
    Stochastic Game model for adversarial attacks in Federated Learning.

    The payoff matrices are populated from empirical FL simulation results.
    Nash Equilibria are computed using nashpy's support enumeration algorithm.

    Args:
        adversary_ratio (float): Fraction of clients that are adversarial.
        gamma (float): Discount factor for future rewards.
    """

    def __init__(self, adversary_ratio=0.3, gamma=0.9):
        self.adversary_ratio = adversary_ratio
        self.gamma = gamma

        # Payoff matrices [N_ATTACK x N_DEFENSE]
        # Filled by set_payoffs() after running FL simulation
        self.attacker_payoff = None
        self.defender_payoff = None

        # Value function for value iteration [n_states]
        self.V = np.zeros(len(STATES))

        # Optimal strategies after solving
        self.nash_equilibria = []
        self.optimal_attack  = None
        self.optimal_defense = None

    # ─── Payoff Matrix Construction ───────────────────────────────────────────

    def set_payoffs(self, payoff_matrix):
        """
        Set empirical payoff matrices from FL simulation results.

        Args:
            payoff_matrix (np.ndarray): Shape [N_ATTACK, N_DEFENSE].
                Each entry = final accuracy achieved under that attack/defense pair.
                Attacker reward = 1 - accuracy (wants to minimize accuracy).
                Defender reward = accuracy (wants to maximize accuracy).
        """
        assert payoff_matrix.shape == (N_ATTACK, N_DEFENSE), \
            f"Expected shape ({N_ATTACK}, {N_DEFENSE}), got {payoff_matrix.shape}"

        # Attacker: rewarded by accuracy DROP (1 - accuracy)
        self.attacker_payoff = 1.0 - payoff_matrix

        # Defender: rewarded by accuracy maintained
        self.defender_payoff = payoff_matrix

        print("\n[Game] Payoff matrices loaded.")
        print(f"  Attacker payoff (rows=attacks, cols=defenses):\n{np.round(self.attacker_payoff, 3)}")
        print(f"  Defender payoff:\n{np.round(self.defender_payoff, 3)}")

    def build_default_payoffs(self):
        """
        Build estimated payoff matrix based on known attack/defense effectiveness.
        Use this when FL simulation results are not yet available.

        Rows = attacks, Cols = defenses
        Values = expected accuracy of global model
        """
        # [no_att, grad_scale, label_flip, sign_flip, gauss_noise]
        # x [fedavg, krum, trimmed_mean, median]
        payoffs = np.array([
            # fedavg   krum    trimmed  median
            [0.95,    0.95,   0.95,    0.95],   # no_attack
            [0.50,    0.80,   0.75,    0.78],   # gradient_scale
            [0.60,    0.72,   0.70,    0.68],   # label_flip
            [0.45,    0.78,   0.73,    0.76],   # sign_flip
            [0.65,    0.85,   0.82,    0.80],   # gaussian_noise
        ])
        self.set_payoffs(payoffs)
        return payoffs

    # ─── Nash Equilibrium ─────────────────────────────────────────────────────

    def compute_nash_equilibrium(self):
        """
        Compute Nash Equilibria using support enumeration (nashpy).

        In zero-sum interpretation:
            Attacker maximizes attacker_payoff
            Defender minimizes attacker_payoff (= maximizes defender_payoff)

        Returns:
            list of (sigma_attack, sigma_defense) mixed strategy pairs
        """
        assert self.attacker_payoff is not None, \
            "Call set_payoffs() or build_default_payoffs() first."

        game = nash.Game(self.attacker_payoff, self.defender_payoff)
        equilibria = list(game.support_enumeration())
        self.nash_equilibria = equilibria

        print(f"\n[Nash] Found {len(equilibria)} Nash Equilibrium/Equilibria:")
        for i, (sigma_a, sigma_d) in enumerate(equilibria):
            print(f"\n  Equilibrium {i+1}:")
            for j, p in enumerate(sigma_a):
                if p > 0.001:
                    print(f"    Attacker plays '{ATTACK_ACTIONS[j]}' with prob {p:.3f}")
            for j, p in enumerate(sigma_d):
                if p > 0.001:
                    print(f"    Defender plays '{DEFENSE_ACTIONS[j]}' with prob {p:.3f}")

        if equilibria:
            self.optimal_attack, self.optimal_defense = equilibria[0]

        return equilibria

    # ─── Value Iteration ──────────────────────────────────────────────────────

    def value_iteration(self, max_iter=500, tol=1e-6):
        """
        Solve the stochastic game using value iteration (Bellman equations).

        V(s) = max_a min_d [ R(s,a,d) + γ * Σ P(s'|s,a,d) * V(s') ]

        Returns:
            V (np.ndarray): State value function.
            policy_attack  (dict): Optimal attack action per state.
            policy_defense (dict): Optimal defense action per state.
        """
        assert self.attacker_payoff is not None, \
            "Call set_payoffs() first."

        V = np.zeros(len(STATES))
        policy_attack  = {s: 0 for s in STATES}
        policy_defense = {s: 0 for s in STATES}

        for iteration in range(max_iter):
            V_new = np.zeros(len(STATES))

            for s_idx, state in enumerate(STATES):
                best_val = -np.inf  # attacker maximizes

                for a_idx in range(N_ATTACK):
                    # Defender best-responds: minimize over defense actions
                    worst_for_defender = np.inf
                    best_d = 0

                    for d_idx in range(N_DEFENSE):
                        # Reward for this (state, attack, defense) combo
                        r_att = self.attacker_payoff[a_idx, d_idx]
                        r_def = self.defender_payoff[a_idx, d_idx]

                        # Transition: stochastic next state
                        next_state_dist = self._transition(state, a_idx, d_idx)

                        # Expected future value
                        future = sum(
                            next_state_dist[s2] * V[state_to_idx(s2)]
                            for s2 in STATES
                        )

                        # Attacker's expected payoff from this joint action
                        q_val = r_att + self.gamma * future

                        if q_val < worst_for_defender:
                            worst_for_defender = q_val
                            best_d = d_idx

                    if worst_for_defender > best_val:
                        best_val = worst_for_defender
                        policy_attack[state]  = a_idx
                        policy_defense[state] = best_d

                V_new[s_idx] = best_val

            # Check convergence
            delta = np.max(np.abs(V_new - V))
            V = V_new

            if delta < tol:
                print(f"[VI] Converged after {iteration+1} iterations (delta={delta:.2e})")
                break

        self.V = V

        print("\n[VI] Optimal Policies:")
        for s in STATES:
            print(f"  State={s}: "
                  f"Attack={ATTACK_ACTIONS[policy_attack[s]]}, "
                  f"Defense={DEFENSE_ACTIONS[policy_defense[s]]}, "
                  f"V={V[state_to_idx(s)]:.4f}")

        return V, policy_attack, policy_defense

    # ─── Transition Function ──────────────────────────────────────────────────

    def _transition(self, state, a_idx, d_idx):
        """
        Stochastic transition: P(s'|s, attack, defense).

        Returns:
            dict {state_name: probability}

        Logic:
            - Strong attacks push state toward LOW
            - Strong defenses push state toward HIGH
            - Current state has inertia
        """
        # Attack damage scores (higher = more damaging)
        attack_damage  = [0.0, 0.8, 0.5, 0.7, 0.3]   # no_attack→0, grad_scale→0.8
        # Defense effectiveness scores (higher = better defense)
        defense_effect = [0.1, 0.7, 0.6, 0.65]        # fedavg→0.1, krum→0.7

        net_damage = attack_damage[a_idx] * (1.0 - defense_effect[d_idx])

        if state == 'HIGH':
            if net_damage > 0.5:
                return {'LOW': 0.1, 'MID': 0.6, 'HIGH': 0.3}
            elif net_damage > 0.2:
                return {'LOW': 0.05, 'MID': 0.35, 'HIGH': 0.6}
            else:
                return {'LOW': 0.0,  'MID': 0.1,  'HIGH': 0.9}

        elif state == 'MID':
            if net_damage > 0.5:
                return {'LOW': 0.5,  'MID': 0.4, 'HIGH': 0.1}
            elif net_damage > 0.2:
                return {'LOW': 0.2,  'MID': 0.6, 'HIGH': 0.2}
            else:
                return {'LOW': 0.05, 'MID': 0.45, 'HIGH': 0.5}

        else:  # LOW
            if net_damage > 0.5:
                return {'LOW': 0.8, 'MID': 0.2,  'HIGH': 0.0}
            elif net_damage > 0.2:
                return {'LOW': 0.6, 'MID': 0.35, 'HIGH': 0.05}
            else:
                return {'LOW': 0.3, 'MID': 0.5,  'HIGH': 0.2}

    # ─── Simulation ──────────────────────────────────────────────────────────

    def simulate_game(self, T=100, initial_accuracy=0.9,
                      use_nash=True, attack_idx=1, defense_idx=0):
        """
        Simulate T rounds of the stochastic game.

        Args:
            T (int): Number of rounds.
            initial_accuracy (float): Starting global model accuracy.
            use_nash (bool): Use Nash Eq strategies; else use fixed (attack_idx, defense_idx).

        Returns:
            history (list): Accuracy per round.
            state_history (list): State label per round.
        """
        if use_nash and self.optimal_attack is not None:
            # Sample actions from Nash mixed strategies each round
            def sample_attack():
                return np.random.choice(N_ATTACK,  p=self.optimal_attack)
            def sample_defense():
                return np.random.choice(N_DEFENSE, p=self.optimal_defense)
        else:
            def sample_attack():  return attack_idx
            def sample_defense(): return defense_idx

        accuracy = initial_accuracy
        history  = [accuracy]
        state_history = [accuracy_to_state(accuracy)]

        for t in range(T):
            a = sample_attack()
            d = sample_defense()

            # Expected accuracy change from payoff matrix
            expected_acc = self.defender_payoff[a, d]
            # Add stochastic noise (simulate real FL round variability)
            noise = np.random.normal(0, 0.02)
            # Smooth transition toward expected accuracy
            alpha = 0.3  # learning rate for accuracy update
            accuracy = (1 - alpha) * accuracy + alpha * expected_acc + noise
            accuracy = float(np.clip(accuracy, 0.0, 1.0))

            history.append(accuracy)
            state_history.append(accuracy_to_state(accuracy))

        return history, state_history
