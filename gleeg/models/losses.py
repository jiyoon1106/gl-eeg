"""손실 함수 — CORAL 순서형 + 멀티태스크 가중합."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .heads import ModelOutput

IGNORE = -100          # 중증도 라벨이 없는 샘플 표시 (PyTorch 관례와 동일)


def coral_targets(y: torch.Tensor, n_levels: int) -> torch.Tensor:
    """단계 라벨 → CORAL 이진 타깃 (B, K-1).

    out[b, j] = 1 if y[b] > j else 0

    >>> coral_targets(torch.tensor([0, 1, 2]), 3)
    tensor([[0., 0.],
            [1., 0.],
            [1., 1.]])
    """
    j = torch.arange(n_levels - 1, device=y.device)
    return (y.unsqueeze(1) > j.unsqueeze(0)).to(torch.float32)


def coral_loss(logits: torch.Tensor, y: torch.Tensor, n_levels: int) -> torch.Tensor:
    """CORAL 손실. K-1 개의 이진 분류를 더한다.

    벌점이 오차 거리에 비례한다 — 정상을 중증으로 틀리면 K-1 개가 모두 틀리고,
    경도를 중등도로 틀리면 하나만 틀린다.

    y 가 IGNORE 인 샘플은 손실에서 제외한다 (중증도 라벨이 없는 데이터셋).
    """
    mask = y != IGNORE
    if not mask.any():
        return logits.sum() * 0.0          # 그래프를 유지한 채 0

    logits, y = logits[mask], y[mask]
    targets = coral_targets(y, n_levels)
    # 레벨 축은 합, 배치 축은 평균
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(1).mean()


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """CORAL logits → 단계 (B,). sigmoid 가 0.5 를 넘은 개수."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


def multitask_loss(
    out: ModelOutput,
    y_disease: torch.Tensor,
    y_severity: Optional[torch.Tensor] = None,
    n_levels: Optional[int] = None,
    severity_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_disease + w * L_severity

    Returns
    -------
    (총손실, 로그용 항목별 값)
    """
    l_dis = F.cross_entropy(out.disease_logits, y_disease)
    parts = {"disease": float(l_dis.detach())}

    total = l_dis
    if out.severity_logits is not None and y_severity is not None:
        if n_levels is None:
            raise ValueError("중증도 손실을 쓰려면 n_levels 가 필요합니다")
        l_sev = coral_loss(out.severity_logits, y_severity, n_levels)
        total = total + severity_weight * l_sev
        parts["severity"] = float(l_sev.detach())

    parts["total"] = float(total.detach())
    return total, parts


def distillation_loss(
    student_out: ModelOutput,
    teacher_out: ModelOutput,
    temperature: float = 3.0,
    embed_weight: float = 1.0,
) -> torch.Tensor:
    """지식 증류 — 19채널 teacher → 전두엽 저채널 student.

    soft label(KL)과 중간층 임베딩(MSE)을 함께 맞춘다. teacher 는 학습이 끝나면
    버리고 student 만 제품에 들어간다.
    """
    t = temperature
    kl = F.kl_div(
        F.log_softmax(student_out.disease_logits / t, dim=1),
        F.log_softmax(teacher_out.disease_logits.detach() / t, dim=1),
        reduction="batchmean",
        log_target=True,
    ) * (t * t)

    mse = F.mse_loss(student_out.embedding, teacher_out.embedding.detach())
    return kl + embed_weight * mse
