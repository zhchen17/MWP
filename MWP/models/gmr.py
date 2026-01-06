# models/gmr.py
import torch.nn as nn


class GranularMultiRetriever(nn.Module):
    """
    Student version: global-global retriever
    """
    def __init__(self, embed_dim=512, proj_dim=96, device="cuda"):
        super().__init__()
        self.fc = nn.Linear(embed_dim, proj_dim)
        self.device = device
        self.to(device)

    def forward(self, cap_emb, cap_tok, img_emb, img_tok):
        return self.fc(cap_emb) @ img_emb.T
