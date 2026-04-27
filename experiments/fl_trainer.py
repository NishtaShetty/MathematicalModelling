"""
experiments/fl_trainer.py
═══════════════════════════════════════════════════════════════════
Main Federated Learning training loop.

Orchestrates:
  - Client selection each round
  - Local training (honest + adversarial)
  - Server aggregation (with chosen defense)
  - Global model evaluation
  - Result logging
═══════════════════════════════════════════════════════════════════
"""

import copy
import random
import numpy as np
import torch
from collections import defaultdict

from models.cnn import SimpleCNN
from clients.honest_client import HonestClient
from clients.adversarial_client import AdversarialClient
from server.aggregator import FederatedServer
from data.partition import (load_mnist, non_iid_partition,
                             get_client_loaders, get_test_loader,
                             label_flip_loader)


def setup_clients(train_dataset, client_data, global_model,
                  n_clients, n_adversaries, attack_type,
                  attack_strength=5.0, device='cpu'):
    """
    Initialize a mix of honest and adversarial clients.

    Args:
        train_dataset: Full MNIST training set.
        client_data: Dict {client_id: [indices]} from partition.
        global_model: Initial global model.
        n_clients (int): Total number of clients.
        n_adversaries (int): How many are adversarial.
        attack_type (str): Attack strategy for adversarial clients.

    Returns:
        list of client objects (HonestClient or AdversarialClient)
    """
    loaders = get_client_loaders(train_dataset, client_data)
    adversary_ids = set(random.sample(range(n_clients), n_adversaries))

    clients = []
    for cid in range(n_clients):
        # Special case: 'no_attack' means all clients are honest
        if attack_type == 'no_attack':
            client = HonestClient(
                client_id=cid,
                dataloader=loaders[cid],
                model=copy.deepcopy(global_model),
                device=device
            )
        elif cid in adversary_ids:
            # For label flip: provide a flipped dataloader
            if attack_type == 'label_flip':
                fl_loader = label_flip_loader(
                    train_dataset, client_data[cid],
                    source_label=1, target_label=7
                )
            else:
                fl_loader = None

            client = AdversarialClient(
                client_id=cid,
                dataloader=loaders[cid],
                model=copy.deepcopy(global_model),
                attack_type=attack_type,
                attack_strength=attack_strength,
                flipped_loader=fl_loader,
                device=device
            )
        else:
            client = HonestClient(
                client_id=cid,
                dataloader=loaders[cid],
                model=copy.deepcopy(global_model),
                device=device
            )
        clients.append(client)

    n_hon = n_clients - n_adversaries
    print(f"  Clients: {n_hon} honest, {n_adversaries} adversarial ({attack_type})")
    return clients


def run_fl_experiment(
    n_clients      = 10,
    adversary_ratio= 0.3,
    attack_type    = 'gradient_scale',
    defense        = 'fedavg',
    rounds         = 50,
    local_epochs   = 3,
    clients_per_round = None,   # None = all clients participate
    seed           = 42,
    device         = 'cpu',
    verbose        = True
):
    """
    Run a complete federated learning experiment.

    Args:
        n_clients (int): Total number of FL clients.
        adversary_ratio (float): Fraction of adversarial clients.
        attack_type (str): Attack used by adversarial clients.
        defense (str): Aggregation/defense used by server.
        rounds (int): Number of FL communication rounds.
        local_epochs (int): Local training epochs per round.
        clients_per_round (int): Clients selected per round (None = all).
        seed (int): Random seed.
        verbose (bool): Print progress.

    Returns:
        dict with keys:
            'accuracies'     : list of test accuracy per round
            'losses'         : list of test loss per round
            'config'         : experiment configuration
    """
    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Setup ─────────────────────────────────────────────────────────────────
    n_adversaries = int(n_clients * adversary_ratio)
    clients_per_round = clients_per_round or n_clients

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: attack={attack_type}, defense={defense}")
        print(f"  Clients={n_clients}, Adversaries={n_adversaries} ({adversary_ratio*100:.0f}%)")
        print(f"  Rounds={rounds}, LocalEpochs={local_epochs}")
        print(f"{'='*60}")

    # Load data
    train_dataset, test_dataset = load_mnist()
    client_data = non_iid_partition(train_dataset, n_clients,
                                    classes_per_client=2,
                                    samples_per_class=50)
    test_loader = get_test_loader(test_dataset)

    # Initialize model and server
    global_model = SimpleCNN().to(device)
    server = FederatedServer(global_model, device=device)

    # Initialize clients
    clients = setup_clients(
        train_dataset, client_data,
        global_model=server.get_global_model(),
        n_clients=n_clients,
        n_adversaries=n_adversaries,
        attack_type=attack_type,
        device=device
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    accuracies = []
    losses     = []

    # Evaluate before any training (round 0)
    acc0, loss0 = server.evaluate(test_loader)
    accuracies.append(acc0)
    losses.append(loss0)
    if verbose:
        print(f"  Round  0 | Acc: {acc0:.4f} | Loss: {loss0:.4f}")

    for r in range(1, rounds + 1):
        # Step 1: Server broadcasts global model to all clients
        global_state = copy.deepcopy(server.get_global_model().state_dict())
        for client in clients:
            client.model.load_state_dict(copy.deepcopy(global_state))

        # Step 2: Select subset of clients for this round
        selected = random.sample(clients, clients_per_round)

        # Step 3: Local training — collect updates
        updates = []
        for client in selected:
            state_dict, _ = client.train(epochs=local_epochs)
            updates.append(state_dict)

        # Step 4: Server aggregates
        server.aggregate(updates, strategy=defense)

        # Step 5: Evaluate global model
        acc, loss = server.evaluate(test_loader)
        accuracies.append(acc)
        losses.append(loss)

        if verbose and (r % 10 == 0 or r == 1):
            adv_frac = n_adversaries / n_clients
            print(f"  Round {r:3d} | Acc: {acc:.4f} | Loss: {loss:.4f} "
                  f"| Adv%: {adv_frac*100:.0f}%")

    result = {
        'accuracies': accuracies,
        'losses':     losses,
        'config': {
            'n_clients':       n_clients,
            'adversary_ratio': adversary_ratio,
            'n_adversaries':   n_adversaries,
            'attack_type':     attack_type,
            'defense':         defense,
            'rounds':          rounds,
            'local_epochs':    local_epochs,
            'final_accuracy':  accuracies[-1],
        }
    }

    if verbose:
        print(f"\n  Final Accuracy: {accuracies[-1]:.4f}")

    return result
