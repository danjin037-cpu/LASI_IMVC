from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_view_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    q1: torch.Tensor | None = None,
    q2: torch.Tensor | None = None,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Paired cross-view contrastive loss.
    It is computed only on samples where both views are observed.
    Confident same-cluster negatives are masked to reduce false negatives.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    if q1 is not None and q2 is not None:
        conf1, pred1 = q1.max(dim=-1)
        conf2, pred2 = q2.max(dim=-1)
        same_cluster = pred1.unsqueeze(1) == pred2.unsqueeze(0)
        confident = (conf1.unsqueeze(1) > 0.5) & (conf2.unsqueeze(0) > 0.5)
        eye = torch.eye(z1.size(0), dtype=torch.bool, device=z1.device)
        logits = logits.masked_fill(same_cluster & confident & (~eye), -1e4)

    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def reconstruction_loss(x: torch.Tensor, x_rec: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction loss on selected sample-view positions."""
    mask = observed_mask.float()
    per_sample = ((x - x_rec) ** 2).mean(dim=-1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def prototype_clustering_loss(q_global: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """DEC-style self-training loss for prototype assignments."""
    q = q_global.clamp_min(1e-8)
    if target is None:
        target = (q ** 2) / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(q.log(), target.detach(), reduction="batchmean")


def cluster_balance_regularization(q_global: torch.Tensor) -> torch.Tensor:
    """Encourages all clusters to be used."""
    avg_prob = q_global.mean(dim=0).clamp_min(1e-8)
    num_clusters = avg_prob.numel()
    uniform = torch.full_like(avg_prob, 1.0 / num_clusters)
    return F.kl_div(avg_prob.log(), uniform, reduction="sum")


def view_distribution_distillation_loss(
    q_teacher: torch.Tensor,
    student_logits_list: list[torch.Tensor],
    sample_mask: torch.Tensor,
    student_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    R3+ view-wise semantic distillation on observed-complete samples only.

    q_teacher:
        [B, K], produced by legal observed-complete fused features, usually with CLIP.
        It is detached inside this function.
    student_logits_list:
        list of [B, K], each from one original view encoder followed by the shared cluster head.
    sample_mask:
        [B], True only for samples whose teacher is legal. In strict CLIP setting this should
        usually mean [view0, view1, clip] are all observed.

    This loss never reconstructs or uses a missing view of single-view samples. It only trains
    original-view encoders on samples that are already complete under the current observed_mask.
    """
    device = q_teacher.device
    idx = torch.nonzero(sample_mask.bool(), as_tuple=False).squeeze(1)
    if idx.numel() <= 1 or len(student_logits_list) == 0:
        return torch.tensor(0.0, device=device), {
            "view_distill_count": float(idx.numel()),
            "view_distill_terms": 0.0,
        }

    target = q_teacher[idx].detach().clamp_min(1e-8)
    loss = torch.tensor(0.0, device=device)
    valid_terms = 0
    for logits in student_logits_list:
        log_prob = F.log_softmax(logits[idx] / student_temperature, dim=-1)
        loss = loss + F.kl_div(log_prob, target, reduction="batchmean")
        valid_terms += 1

    if valid_terms > 0:
        loss = loss / valid_terms

    return loss, {
        "view_distill_count": float(idx.numel()),
        "view_distill_terms": float(valid_terms),
    }


def prototype_teacher_kl_loss(
    h_fused: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_prototypes: torch.Tensor | None,
    sample_mask: torch.Tensor,
    teacher_temperature: float = 0.2,
    student_temperature: float = 1.0,
    confidence_threshold: float = 0.45,
    margin_threshold: float = 0.10,
) -> tuple[torch.Tensor, dict]:
    """
    R3+ single-view prototype distillation.

    teacher_prototypes:
        [K, D], built only from legal observed-complete samples.
    sample_mask:
        [B], usually true for real single-original-view samples under current observed_mask.

    The teacher target is derived from the sample's observed fused representation and the
    observed-complete prototype bank. It does not access the sample's missing view or missing
    raw-image CLIP feature.
    """
    device = h_fused.device
    zero_stats = {
        "proto_count": 0.0,
        "proto_keep_ratio": 0.0,
        "proto_conf_mean": 0.0,
        "proto_margin_mean": 0.0,
    }

    if teacher_prototypes is None:
        return torch.tensor(0.0, device=device), zero_stats

    idx = torch.nonzero(sample_mask.bool(), as_tuple=False).squeeze(1)
    if idx.numel() <= 1:
        stats = dict(zero_stats)
        stats["proto_count"] = float(idx.numel())
        return torch.tensor(0.0, device=device), stats

    h = F.normalize(h_fused[idx], dim=-1)
    p = F.normalize(teacher_prototypes.to(device), dim=-1)

    with torch.no_grad():
        teacher_logits = torch.matmul(h, p.t()) / teacher_temperature
        q_teacher = F.softmax(teacher_logits, dim=-1)
        top2 = torch.topk(q_teacher, k=min(2, q_teacher.shape[1]), dim=-1).values
        conf = top2[:, 0]
        if top2.shape[1] > 1:
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.ones_like(conf)
        keep = (conf >= confidence_threshold) & (margin >= margin_threshold)

    stats = {
        "proto_count": float(idx.numel()),
        "proto_keep_ratio": float(keep.float().mean().item()) if keep.numel() > 0 else 0.0,
        "proto_conf_mean": float(conf.mean().item()) if conf.numel() > 0 else 0.0,
        "proto_margin_mean": float(margin.mean().item()) if margin.numel() > 0 else 0.0,
    }

    if int(keep.sum().item()) <= 1:
        return torch.tensor(0.0, device=device), stats

    student_log_prob = F.log_softmax(student_logits[idx][keep] / student_temperature, dim=-1)
    loss = F.kl_div(student_log_prob, q_teacher[keep].detach(), reduction="batchmean")
    return loss, stats
