# models/framework.py
import torch
import torch.nn as nn

from losses.retrieval import retrieval_loss
from losses.prior import prior_alignment_loss


class MWPFramework(nn.Module):
    def __init__(self, encoder, wpe, gmr,
                 lambda_prior=0.5):
        super().__init__()

        self.encoder = encoder
        self.wpe = wpe
        self.gmr = gmr
        self.lambda_prior = lambda_prior

        # freeze WPE
        for p in self.wpe.parameters():
            p.requires_grad = False

    def forward(self,
                cap_ids, cap_mask,
                pixel_values,
                desc_ids, desc_mask,
                category):

        cap_emb, img_emb, cap_tok, img_tok, desc_emb, desc_tok = \
            self.encoder(
                cap_ids, cap_mask, pixel_values,
                desc_ids, desc_mask
            )

        with torch.no_grad():
            S_wpe = self.wpe(desc_emb, desc_tok, cap_emb, cap_tok)

        S_gmr = self.gmr(cap_emb, cap_tok, img_emb, img_tok)

        labels = torch.arange(
            S_gmr.size(0),
            device=S_gmr.device
        )

        loss_retrieval = retrieval_loss(
            S_gmr,
            labels=labels,
            text_labels=category,
            image_labels=category,
            lambda_category=1.0
        )

        loss_prior = prior_alignment_loss(
            S_gmr, S_wpe
        )

        return loss_retrieval + self.lambda_prior * loss_prior
