import torch
from torch.nn import functional as F


def supervised_contrastive_loss(z, identities, sessions, temperature=0.1, cross_session=True):
    """Identity SupCon, optionally restricted to cross-session positives.

    In the cross-session variant, same-identity/same-session comparisons are
    excluded from the denominator as well, rather than treated as negatives.
    """
    if temperature <= 0 or z.ndim != 2 or len(z) < 2:
        raise ValueError("Invalid temperature or embedding batch")
    if identities.shape != (len(z),) or sessions.shape != identities.shape:
        raise ValueError("Identity and session labels must match the batch")
    same_identity = identities[:, None] == identities[None, :]
    same_session = sessions[:, None] == sessions[None, :]
    off_diagonal = ~torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = same_identity & off_diagonal
    comparisons = off_diagonal
    if cross_session:
        positive = positive & ~same_session
        comparisons = comparisons & ~(same_identity & same_session)
    counts = positive.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each anchor needs a positive; use identity/session-aware batches")
    logits = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).T / temperature
    denominator = torch.logsumexp(logits.masked_fill(~comparisons, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    return -(torch.where(positive, log_probability, 0).sum(dim=1) / counts).mean()


def pairwise_contrastive_loss(z, identities, sessions, margin=1.0, cross_session=True):
    """Balanced positive/negative pair loss on normalized embeddings."""
    if not 0 < margin <= 2:
        raise ValueError("Normalized Euclidean margin must be in (0,2]")
    z = F.normalize(z, dim=-1)
    same = identities[:, None] == identities[None, :]
    pairs = torch.triu(torch.ones_like(same), diagonal=1).bool()
    positive = pairs & same
    if cross_session:
        positive &= sessions[:, None] != sessions[None, :]
    negative = pairs & ~same
    if not positive.any() or not negative.any():
        raise ValueError("Pair loss requires both positive and negative pairs")
    d2 = (z[:, None] - z[None, :]).square().sum(dim=-1)
    positive_loss = d2[positive].mean()
    negative_loss = (margin - (d2[negative] + 1e-12).sqrt()).clamp_min(0).square().mean()
    return (positive_loss + negative_loss) / 2
