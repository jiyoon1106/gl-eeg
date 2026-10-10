"""백본 — Braindecode 공개 모델을 임베딩 추출기로 감싼다.

아키텍처를 직접 구현하지 않는다. Braindecode(BSD-3)의 검증된 구현을 쓰고,
아낀 시간을 증류·저채널 실험에 투입한다.

주력은 ShallowFBCSPNet 이다. 구조가 conv → square → pool → log 인데 이 네
단계가 곧 밴드파워 계산이고, 치매의 EEG 지표가 바로 대역 파워(서파 증가,
알파 감소)이기 때문이다. 도메인 지식이 아키텍처에 내장되어 있어 88명
규모에서도 학습이 된다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# n_outputs 를 embed_dim 으로 주고 그 출력을 임베딩으로 쓴다.
SUPPORTED = ("ShallowFBCSPNet", "EEGNetv4", "SincShallowNet", "EEGNeX", "Deep4Net", "FBCNet")


def build_backbone(
    name: str,
    n_chans: int,
    n_times: int,
    embed_dim: int = 64,
    **kwargs,
) -> nn.Module:
    """Braindecode 모델을 (B, C, T) -> (B, embed_dim) 모듈로 만든다."""
    try:
        import braindecode.models as bdm
    except ImportError as e:  # pragma: no cover
        raise ImportError("braindecode 가 필요합니다: pip install braindecode") from e

    # EEGNet 은 버전에 따라 이름이 다르다
    alias = {"EEGNet": "EEGNetv4"}
    name = alias.get(name, name)

    if not hasattr(bdm, name):
        raise ValueError(f"braindecode 에 {name!r} 이 없습니다. 후보: {SUPPORTED}")

    cls = getattr(bdm, name)
    model = cls(n_chans=n_chans, n_outputs=embed_dim, n_times=n_times, **kwargs)

    # 출력 형태를 확인해 둔다 (백본마다 (B,D) / (B,D,1) 등으로 다르다)
    with torch.no_grad():
        dummy = torch.zeros(2, n_chans, n_times)
        out = model(dummy)
    flat = out.flatten(1).shape[1]
    if flat != embed_dim:
        raise RuntimeError(
            f"{name} 출력이 {out.shape} → 펴면 {flat} 차원인데 embed_dim={embed_dim} 입니다. "
            "embed_dim 을 맞추거나 백본 인자를 조정하세요."
        )

    return _Flatten(model)


class _Flatten(nn.Module):
    """백본 출력을 항상 (B, D) 로 만들어 주는 얇은 래퍼."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).flatten(1)


class DummyBackbone(nn.Module):
    """테스트용 — braindecode 없이도 파이프라인을 돌려볼 수 있게.

    실제 학습에는 쓰지 않는다. 인터페이스 검증 전용.
    """

    def __init__(self, n_chans: int, n_times: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_chans, 16, kernel_size=25, stride=4),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(16 * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    """학습 파라미터 수. 88명 규모에서는 수천~수만 개가 상한이다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
