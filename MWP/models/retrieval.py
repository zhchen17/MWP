# losses/retrieval.py
import torch
import torch.nn.functional as F


def retrieval_loss(similarity_matrix,
                   labels=None,
                   tau=0.07,
                   lambda_sym=0.0,
                   text_labels=None,
                   image_labels=None,
                   lambda_category=0.5):
    device = similarity_matrix.device
    B = similarity_matrix.size(0)

    if labels is None:
        labels = torch.arange(B, device=device)

    sim = similarity_matrix / tau

    # CLIP-style contrastive loss
    loss_clip = 0.5 * (
        F.cross_entropy(sim, labels) +
        F.cross_entropy(sim.T, labels)
    )

    # symmetric KL (optional)
    p = F.softmax(sim, dim=1)
    q = F.softmax(sim.T, dim=1)
    loss_symkl = 0.5 * (
        F.kl_div(p.log(), q, reduction='batchmean') +
        F.kl_div(q.log(), p, reduction='batchmean')
    )

    # category-aware loss
    loss_category = torch.tensor(0.0, device=device)
    if text_labels is not None and image_labels is not None:
        mat = (text_labels.unsqueeze(1) == image_labels.unsqueeze(0)).float()
        loss_category = -(mat * F.log_softmax(sim, dim=1)).sum(1)
        loss_category = (loss_category / mat.sum(1).clamp(min=1)).mean()

    return loss_clip + lambda_sym * loss_symkl + lambda_category * loss_category
