# models/wpe.py
import torch.nn as nn


class WarmPriorEstimator(nn.Module):
    """
    Student version: simple global-global weak prior
    """
    def __init__(self, embed_dim=512, proj_dim=96, device="cuda"):
        super().__init__()
        self.fc = nn.Linear(embed_dim, proj_dim)
        self.device = device
        self.to(device)

    def forward(self, desc_emb, desc_tok, cap_emb, cap_tok):
        return self.fc(desc_emb) @ self.fc(cap_emb).T
