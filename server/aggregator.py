"""
server/aggregator.py
Central server with multiple aggregation strategies:
  - FedAvg  (baseline, no defense)
  - Krum    (Byzantine-robust, selects most consistent update)
  - Trimmed Mean (removes extreme updates before averaging)
  - Median  (coordinate-wise median)
"""

import copy
import torch
import numpy as np


class FederatedServer:
    """
    Central aggregation server.

    Maintains the global model and aggregates client updates each round.

    Args:
        global_model: Initial global model (nn.Module).
        device (str): 'cpu' or 'cuda'.
    """

    DEFENSES = ['fedavg', 'krum', 'trimmed_mean', 'median']

    def __init__(self, global_model, device='cpu'):
        self.model  = copy.deepcopy(global_model)
        self.device = device
        self.model.to(device)
        self.round  = 0

    def get_global_model(self):
        return self.model

    def aggregate(self, client_updates, strategy='fedavg', f=1, trim_ratio=0.2):
        """
        Aggregate client updates using the chosen defense strategy.

        Args:
            client_updates (list of state_dicts): Updates from selected clients.
            strategy (str): Aggregation method.
            f (int): Number of assumed Byzantine clients (used by Krum).
            trim_ratio (float): Fraction to trim on each side (used by Trimmed Mean).

        Returns:
            dict: New global model state_dict.
        """
        assert strategy in self.DEFENSES, \
            f"Unknown strategy '{strategy}'. Choose from {self.DEFENSES}"

        if strategy == 'fedavg':
            new_state = self._fedavg(client_updates)
        elif strategy == 'krum':
            new_state = self._krum(client_updates, f=f)
        elif strategy == 'trimmed_mean':
            new_state = self._trimmed_mean(client_updates, trim_ratio=trim_ratio)
        elif strategy == 'median':
            new_state = self._coordinate_median(client_updates)

        self.model.load_state_dict(new_state)
        self.round += 1
        return new_state

    # ─── Aggregation Strategies ───────────────────────────────────────────────

    def _fedavg(self, updates):
        """Standard FedAvg: simple average of all updates."""
        avg = {}
        for key in updates[0]:
            stacked = torch.stack([u[key].float() for u in updates])
            avg[key] = stacked.mean(dim=0)
        return avg

    def _krum(self, updates, f=1):
        """
        Krum (Blanchard et al., 2017):
        Select the update with the smallest sum of squared distances
        to its (n - f - 2) nearest neighbors.

        f = number of assumed Byzantine workers.
        """
        n = len(updates)
        # Flatten each update to a vector
        flat = []
        for u in updates:
            vec = torch.cat([v.flatten().float() for v in u.values()])
            flat.append(vec)

        scores = []
        for i in range(n):
            dists = []
            for j in range(n):
                if i != j:
                    d = torch.norm(flat[i] - flat[j]).item() ** 2
                    dists.append(d)
            dists.sort()
            # Sum the n-f-2 smallest distances
            k = max(1, n - f - 2)
            scores.append(sum(dists[:k]))

        best_idx = int(np.argmin(scores))
        return updates[best_idx]

    def _trimmed_mean(self, updates, trim_ratio=0.2):
        """
        Trimmed Mean (Yin et al., 2018):
        For each parameter coordinate, remove the top and bottom
        `trim_ratio` fraction of values, then average the rest.
        """
        k = max(1, int(len(updates) * trim_ratio))
        trimmed = {}
        for key in updates[0]:
            stacked = torch.stack([u[key].float() for u in updates])  # [n, ...]
            # Sort along client dimension
            sorted_vals, _ = torch.sort(stacked, dim=0)
            # Trim top and bottom k
            if 2 * k < len(updates):
                trimmed_vals = sorted_vals[k: len(updates) - k]
            else:
                trimmed_vals = sorted_vals  # fallback: not enough clients to trim
            trimmed[key] = trimmed_vals.mean(dim=0)
        return trimmed

    def _coordinate_median(self, updates):
        """
        Coordinate-wise Median:
        For each parameter, take the median across all clients.
        Robust to up to 50% Byzantine workers.
        """
        median = {}
        for key in updates[0]:
            stacked = torch.stack([u[key].float() for u in updates])
            median[key] = stacked.median(dim=0).values
        return median

    def evaluate(self, test_loader):
        """
        Evaluate global model on the test set.

        Returns:
            accuracy (float): Top-1 accuracy.
            avg_loss (float): Average cross-entropy loss.
        """
        import torch.nn.functional as F
        self.model.eval()
        correct = 0
        total   = 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out  = self.model(x)
                loss = F.cross_entropy(out, y)
                total_loss += loss.item() * len(y)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += len(y)

        accuracy = correct / total
        avg_loss = total_loss / total
        return accuracy, avg_loss
