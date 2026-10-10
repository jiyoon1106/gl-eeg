"""인터페이스 ② — 모델 출력 규격.

모든 인코더(EEG · 손 궤적 · 음성)가 ModelOutput 네 개를 반환한다. 그러면 융합
모듈과 증류 코드가 어느 백본이 들어와도 동작한다.

중증도 헤드는 CORAL(순서형)이다. 5단계에는 순서가 있는데, 일반 softmax 는
"정상→중증 오분류"와 "경도→중등도 오분류"를 같은 벌점으로 취급하므로
의료적으로 맞지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """모든 인코더가 반환하는 네 가지."""

    disease_logits: torch.Tensor              # (B, n_disease)  softmax 전
    severity_logits: Optional[torch.Tensor]   # (B, n_levels-1) CORAL, sigmoid 전
    embedding: torch.Tensor                   # (B, embed_dim)  융합·증류가 쓴다
    quality: Optional[torch.Tensor] = None    # (B, 1) 신호 품질 (융합 가중치)

    def detach_cpu(self) -> "ModelOutput":
        f = lambda t: None if t is None else t.detach().cpu()  # noqa: E731
        return ModelOutput(
            disease_logits=f(self.disease_logits),
            severity_logits=f(self.severity_logits),
            embedding=f(self.embedding),
            quality=f(self.quality),
        )


class CoralHead(nn.Module):
    """CORAL 순서형 회귀 헤드.

    공유 가중치 하나 + 단계별 bias (n_levels-1 개). 가중치를 공유하기 때문에
    "레벨 >= 1", "레벨 >= 2", ... 예측이 서로 모순되지 않는다(rank consistency).

        출력 (B, K-1) 의 j 번째 = "레벨 > j 인가?" 의 logit
        예측 레벨 = sigmoid(logits) > 0.5 인 개수
    """

    def __init__(self, in_features: int, n_levels: int) -> None:
        super().__init__()
        if n_levels < 2:
            raise ValueError(f"n_levels 는 2 이상이어야 합니다: {n_levels}")
        self.n_levels = n_levels
        self.fc = nn.Linear(in_features, 1, bias=False)
        # bias 를 단계 순서대로 초기화해두면 학습이 안정적이다.
        self.bias = nn.Parameter(torch.arange(n_levels - 1, 0, -1, dtype=torch.float32) * 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z) + self.bias      # (B,1) + (K-1,) → (B, K-1)


class MultiTaskModel(nn.Module):
    """백본 1개 + 헤드 2개.

    백본이 두 과제의 공통 표현을 학습하므로, 모델을 따로 두는 것보다 작은
    데이터에서 유리하다.

    Parameters
    ----------
    backbone : (B, C, T) -> (B, embed_dim) 을 만족하는 모듈
    embed_dim : 백본 출력 차원
    n_disease : 질환 클래스 수
    n_levels : 중증도 단계 수. None 이면 중증도 헤드를 달지 않는다.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        n_disease: int,
        n_levels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.n_disease = n_disease
        self.n_levels = n_levels

        self.head_disease = nn.Linear(embed_dim, n_disease)
        self.head_severity = CoralHead(embed_dim, n_levels) if n_levels else None

    def forward(
        self, x: torch.Tensor, quality: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        z = self.backbone(x)
        if z.ndim > 2:                      # 백본이 (B, D, 1, 1) 등을 내면 펴준다
            z = z.flatten(1)
        return ModelOutput(
            disease_logits=self.head_disease(z),
            severity_logits=self.head_severity(z) if self.head_severity else None,
            embedding=z,
            quality=quality,                # 모델이 만들지 않는다. 전처리에서 계산해 넘긴다.
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """추론용 — 클래스 인덱스와 중증도 단계를 바로 반환."""
        from .losses import coral_predict

        self.eval()
        out = self(x)
        res = {"disease": out.disease_logits.argmax(dim=1)}
        if out.severity_logits is not None:
            res["severity"] = coral_predict(out.severity_logits)
        return res
