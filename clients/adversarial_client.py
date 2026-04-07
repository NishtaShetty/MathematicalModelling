"""
clients/adversarial_client.py
Adversarial FL client that injects poisoned updates.
Supports: gradient scaling, label flipping, model replacement, gaussian noise.
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np
from clients.honest_client import HonestClient


class AdversarialClient(HonestClient):
    """
    Malicious participant that tampers with its gradient update.

    Attack types:
        - 'gradient_scale'   : Multiplies gradients by a large negative factor
        - 'label_flip'       : Trains on label-flipped data (passed via flipped dataloader)
        - 'model_replace'    : Replaces update with a crafted weight vector
        - 'gaussian_noise'   : Adds large Gaussian noise to weights
        - 'sign_flip'        : Negates all gradient directions

    Args:
        attack_type (str): One of the attack types above.
        attack_strength (float): Scaling factor controlling attack magnitude.
        flipped_loader: Optional DataLoader with flipped labels (for label_flip attack).
    """

    VALID_ATTACKS = ['gradient_scale', 'label_flip', 'model_replace',
                     'gaussian_noise', 'sign_flip']

    def __init__(self, client_id, dataloader, model,
                 attack_type='gradient_scale', attack_strength=5.0,
                 flipped_loader=None, lr=0.01, device='cpu'):
        super().__init__(client_id, dataloader, model, lr, device)
        assert attack_type in self.VALID_ATTACKS, \
            f"Unknown attack: {attack_type}. Choose from {self.VALID_ATTACKS}"
        self.attack_type     = attack_type
        self.attack_strength = attack_strength
        self.flipped_loader  = flipped_loader  # used for label_flip

    def train(self, epochs=3):
        """
        Train locally, then apply the chosen attack to the returned update.

        Returns:
            dict: Poisoned model state_dict.
            float: Training loss (benign loss, before poisoning).
        """
        if self.attack_type == 'label_flip' and self.flipped_loader is not None:
            # Train on flipped labels
            weights, loss = self._train_on_loader(self.flipped_loader, epochs)
        else:
            weights, loss = super().train(epochs)

        # Apply post-training poisoning
        poisoned_weights = self._apply_attack(weights)
        return poisoned_weights, loss

    def _train_on_loader(self, loader, epochs):
        """Internal: train on a given dataloader."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        total_loss, total_batches = 0.0, 0
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
        return copy.deepcopy(self.model.state_dict()), total_loss / max(total_batches, 1)

    def _apply_attack(self, weights):
        """Apply the selected attack strategy to the weight update."""
        poisoned = {}

        if self.attack_type == 'gradient_scale':
            # Amplify updates by a large negative factor → disrupts aggregation
            for k, v in weights.items():
                poisoned[k] = v * (-self.attack_strength)

        elif self.attack_type == 'sign_flip':
            # Negate all weight directions
            for k, v in weights.items():
                poisoned[k] = -v

        elif self.attack_type == 'gaussian_noise':
            # Add large Gaussian noise to all parameters
            for k, v in weights.items():
                noise = torch.randn_like(v.float()) * self.attack_strength
                poisoned[k] = v.float() + noise

        elif self.attack_type == 'model_replace':
            # Replace with near-zero weights (collapse model)
            for k, v in weights.items():
                poisoned[k] = torch.zeros_like(v) + \
                              torch.randn_like(v.float()) * 0.01

        elif self.attack_type == 'label_flip':
            # Weights already trained on flipped data — no further modification
            poisoned = weights

        return poisoned

    def __repr__(self):
        return f"AdversarialClient(id={self.client_id}, attack={self.attack_type})"
