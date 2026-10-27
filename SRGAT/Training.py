"""Shared normal-only autoencoder training, selection, and batched scoring."""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReconstructionTraining:
    """Requires forward(x, edge_index) -> (reconstruction, latent)."""

    def fit(
        self,
        train_data,
        edge_index,
        val_data=None,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        verbose=True,
    ):
        """Train the autoencoder on normal graph windows (B, T, N, F)."""
        if len(train_data) == 0 or (val_data is not None and len(val_data) == 0):
            raise ValueError("Training and validation must contain complete windows")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        device = next(self.parameters()).device
        edge_index = edge_index.to(device)
        loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        best_loss, best_state = float("inf"), None
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                opt.zero_grad()
                recon, _ = self(batch, edge_index)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                total_loss += loss.item() * batch.size(0)

            train_loss = total_loss / len(train_data)
            history["train_loss"].append(train_loss)

            val_str = ""
            if val_data is not None:
                self.eval()
                val_loss = float(self.score(val_data, edge_index, batch_size)[0].mean())
                history["val_loss"].append(val_loss)
                val_str = f" | val_loss {val_loss:.6f}"
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1:3d}/{epochs} - train_loss {train_loss:.6f}{val_str}")

        if best_state is not None:
            self.load_state_dict(best_state)
        return history

    def score(self, x, edge_index, batch_size=32):
        """Score in bounded batches; return CPU window and bus errors."""
        if len(x) == 0 or batch_size < 1:
            raise ValueError("Scoring requires nonempty windows and positive batch_size")
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        scores = []
        with torch.no_grad():
            edges = edge_index.to(device)
            for batch in x.split(batch_size):
                batch = batch.to(device)
                recon, _ = self(batch, edges)
                scores.append(((recon - batch) ** 2).mean(dim=(1, 3)).cpu())
        self.train(was_training)
        bus_score = torch.cat(scores)
        return bus_score.mean(dim=1), bus_score

    def fit_threshold(self, x_val, edge_index, q=0.99):
        """Fit empirical anomaly detection threshold on validation windows."""
        window_score, _ = self.score(x_val, edge_index)
        threshold = torch.quantile(window_score, q)
        self.threshold.copy_(threshold.to(self.threshold.device))
        return threshold

    def detect(self, x, edge_index):
        """Flag windows exceeding the anomaly threshold."""
        window_score, bus_score = self.score(x, edge_index)
        flags = window_score > self.threshold.cpu()
        return flags, window_score
