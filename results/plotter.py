"""
results/plotter.py
All visualization functions for paper figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

# Paper-quality style
plt.rcParams.update({
    'font.family':     'DejaVu Serif',
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi':      150,
    'axes.grid':       True,
    'grid.alpha':      0.3,
})

COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
          '#00BCD4', '#795548', '#607D8B']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


def plot_accuracy_curves(curves_dict, title, save_path, ylabel='Test Accuracy'):
    """
    Line plot of accuracy over FL rounds.

    Args:
        curves_dict (dict): {label: [accuracy list]} 
        title (str): Plot title.
        save_path (str): File to save.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, accs) in enumerate(curves_dict.items()):
        rounds = list(range(len(accs)))
        ax.plot(rounds, accs,
                label=label,
                color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=2.0,
                marker='o' if len(accs) < 20 else None,
                markersize=4)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_payoff_heatmap(payoff_matrix, attack_labels, defense_labels,
                        title, save_path):
    """
    Heatmap of the payoff matrix (final accuracy per attack/defense pair).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        payoff_matrix,
        annot=True, fmt='.3f',
        xticklabels=defense_labels,
        yticklabels=attack_labels,
        cmap='RdYlGn',
        vmin=0.3, vmax=1.0,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel('Defense Strategy')
    ax.set_ylabel('Attack Strategy')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_nash_strategies(sigma_attack, sigma_defense,
                         attack_labels, defense_labels,
                         title, save_path):
    """
    Bar chart of Nash Equilibrium mixed strategies for attacker and defender.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Attacker strategies
    x1 = range(len(attack_labels))
    bars1 = ax1.bar(x1, sigma_attack, color=COLORS[:len(attack_labels)], alpha=0.85, edgecolor='black')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(attack_labels, rotation=25, ha='right')
    ax1.set_ylabel('Probability')
    ax1.set_title('Attacker Mixed Strategy (Nash Eq.)')
    ax1.set_ylim([0, 1.1])
    for bar, val in zip(bars1, sigma_attack):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Defender strategies
    x2 = range(len(defense_labels))
    bars2 = ax2.bar(x2, sigma_defense, color=COLORS[4:4+len(defense_labels)], alpha=0.85, edgecolor='black')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(defense_labels, rotation=15, ha='right')
    ax2.set_ylabel('Probability')
    ax2.set_title('Defender Mixed Strategy (Nash Eq.)')
    ax2.set_ylim([0, 1.1])
    for bar, val in zip(bars2, sigma_defense):
        if val > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_adversary_ratio_effect(all_results, adversary_ratios,
                                 attack_types, defense_types, save_path):
    """
    Line plot: final accuracy vs adversary ratio, one line per defense.
    Attack: gradient_scale (most severe).
    """
    attack = 'gradient_scale'
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, defense in enumerate(defense_types):
        accs = []
        for adv in adversary_ratios:
            key = (adv, attack, defense)
            if key in all_results:
                accs.append(all_results[key]['config']['final_accuracy'])
            else:
                accs.append(None)

        valid = [(r, a) for r, a in zip(adversary_ratios, accs) if a is not None]
        if valid:
            x, y = zip(*valid)
            ax.plot([v*100 for v in x], y,
                    label=defense,
                    color=COLORS[i],
                    linestyle=LINESTYLES[i],
                    linewidth=2.0,
                    marker='s', markersize=7)

    ax.set_xlabel('Adversarial Client Ratio (%)')
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title(f'Defense Robustness vs. Adversary Ratio\n(Attack: Gradient Scaling)')
    ax.legend()
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_value_convergence(V_history, save_path):
    """Plot value function convergence during value iteration."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for s_idx, state in enumerate(['LOW', 'MID', 'HIGH']):
        vals = [v[s_idx] for v in V_history]
        ax.plot(vals, label=f'V({state})', linewidth=2, color=COLORS[s_idx])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('State Value V(s)')
    ax.set_title('Value Iteration Convergence')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_value_iteration_convergence(save_path):
    """Placeholder for value iteration convergence plot."""
    plot_value_convergence([], save_path)


def plot_summary_table(all_results, adversary_ratios, attack_types,
                       defense_types, save_path):
    """
    Summary bar chart: final accuracy for every combination at adv=30%.
    """
    adv = max(adversary_ratios)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(defense_types))
    width = 0.15
    offsets = np.linspace(-(len(attack_types)-1)*width/2,
                           (len(attack_types)-1)*width/2,
                           len(attack_types))

    for i, attack in enumerate(attack_types):
        accs = []
        for defense in defense_types:
            key = (adv, attack, defense)
            acc = all_results[key]['config']['final_accuracy'] if key in all_results else 0
            accs.append(acc)
        ax.bar(x + offsets[i], accs, width, label=attack,
               color=COLORS[i], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(defense_types, fontsize=11)
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title(f'Attack vs Defense — Final Accuracy Summary (adv={adv*100:.0f}%)')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")
