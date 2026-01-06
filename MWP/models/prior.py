# losses/prior.py
import torch.nn.functional as F


def prior_alignment_loss(gmr_scores, wpe_scores, temperature=0.07):
    """
    KL( p_gmr || p_wpe )
    WPE is treated as frozen teacher
    """
    wpe_scores = wpe_scores.detach()

    p_teacher = F.softmax(wpe_scores / temperature, dim=1)
    p_student = F.log_softmax(gmr_scores / temperature, dim=1)

    return F.kl_div(
        p_student,
        p_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)
