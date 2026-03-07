from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPAutoEncoder(nn.Module):
    def __init__(self, n_in: int, latent_dim: int = 12) -> None:
        super().__init__()
        h1 = max(32, n_in * 2)
        h2 = max(16, n_in)

        self.encoder = nn.Sequential(
            nn.Linear(n_in, h1),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, n_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class AutoEncoderScorer:
    state_dict: dict[str, torch.Tensor]
    n_in: int
    q_low: float
    q_high: float
    feature_names: list[str]

    @staticmethod
    def fit(
        X: pd.DataFrame,
        seed: int = 42,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-3,
        latent_dim: int = 12,
        device: str | None = None,
    ) -> "AutoEncoderScorer":
        torch.manual_seed(seed)
        np.random.seed(seed)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n_in = int(X.shape[1])
        model = _MLPAutoEncoder(n_in=n_in, latent_dim=latent_dim).to(dev)

        x = torch.tensor(X.values, dtype=torch.float32)
        ds = TensorDataset(x)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for (xb,) in dl:
                xb = xb.to(dev)
                opt.zero_grad(set_to_none=True)
                recon = model(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            recon = model(x.to(dev)).cpu().numpy()
        errs = np.mean((recon - X.values) ** 2, axis=1)
        q_low = float(np.quantile(errs, 0.05))
        q_high = float(np.quantile(errs, 0.95))

        return AutoEncoderScorer(
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            n_in=n_in,
            q_low=q_low,
            q_high=q_high,
            feature_names=list(X.columns),
        )

    def _load_model(self, device: str) -> _MLPAutoEncoder:
        model = _MLPAutoEncoder(n_in=self.n_in).to(device)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def score(self, X: pd.DataFrame, device: str | None = None) -> np.ndarray:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self._load_model(dev)
        x = torch.tensor(X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            recon = model(x).cpu().numpy()
        errs = np.mean((recon - X.values) ** 2, axis=1)
        denom = (self.q_high - self.q_low) if (self.q_high - self.q_low) != 0 else 1.0
        scaled = (errs - self.q_low) / denom
        return np.clip(scaled, 0.0, 1.0)

