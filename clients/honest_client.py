"""
clients/honest_client.py
Standard federated learning client — trains locally and returns honest weight updates.
"""

import copy
import torch
import torch.nn.functional as F


class HonestClient:
    """
    Represents an honest participant in the federated learning system.
    Trains on local data and returns the model weight delta (update).

    Args:
        client_id (int): Unique identifier.
        dataloader: Local DataLoader.
        model: Copy of the global model.
        lr (float): Learning rate for local SGD.
        device (str): 'cpu' or 'cuda'.
    """

    def __init__(self, client_id, dataloader, model, lr=0.01, device="cpu"):
        self.client_id  = client_id
        self.dataloader = dataloader
        self.model      = copy.deepcopy(model)
        self.lr         = lr
        self.device     = device
        self.model.to(device)

    def update_global_model(self, global_model):
        """Sync local model with latest global model before training."""
        self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))

    def train(self, epochs=3):
        """
        Perform local training for `epochs` epochs.

        Returns:
            dict: Model state_dict after local training (the 'update').
            float: Average training loss this round.
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return copy.deepcopy(self.model.state_dict()), avg_loss

    def __repr__(self):
        return f"HonestClient(id={self.client_id})"
