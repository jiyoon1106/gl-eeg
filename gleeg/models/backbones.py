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
SUPPORTED = ("ShallowFBCSPNet", "EEGNet", "SincShallowNet", "EEGNeX", "Deep4Net", "FBCNet")

# Braindecode 버전에 따라 클래스 이름이 다르다 (1.0 이전 EEGNetv4 -> 이후 EEGNet).
# 설치된 쪽을 실행 시점에 찾는다. 팀원 환경마다 버전이 다를 수 있기 때문이다.
ALIASES: dict[str, tuple[str, ...]] = {
    "EEGNet": ("EEGNet", "EEGNetv4"),
    "EEGNetv4": ("EEGNetv4", "EEGNet"),
}


def _resolve(bdm, name: str) -> str:
    for cand in ALIASES.get(name, (name,)):
        if hasattr(bdm, cand):
            return cand
    raise ValueError(f"braindecode 에 {name!r} 이 없습니다. 후보: {SUPPORTED}")


def accepted_kwargs(cls) -> set[str]:
    """그 백본이 실제로 받는 인자 이름.

    **kwargs 는 세지 않는다. EEGNet 은 **kwargs 를 받아놓고 모르는 인자가
    오면 TypeError 를 내기 때문에, 이름이 명시된 인자만 통과시켜야 한다.
    """
    import inspect

    params = inspect.signature(cls.__init__).parameters
    return {
        p.name
        for p in params.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"
    }


def filter_kwargs(cls, kwargs: dict, name: str = "", verbose: bool = True) -> dict:
    """백본이 받지 않는 인자를 떼어낸다.

    왜 필요한가
    -----------
    `backbone_kwargs` 는 백본마다 이름이 다르다. ShallowFBCSPNet 의
    `filter_time_length` 는 EEGNet 에 없고(`kernel_length` 다), EEGNet 은
    모르는 인자를 받으면 바로 TypeError 를 낸다. 백본을 바꿀 때마다 YAML 을
    고쳐야 하면 저채널 스윕을 돌릴 수 없으므로, 받는 쪽에서 걸러낸다.

    버린 인자는 반드시 출력한다 — 오타를 조용히 넘기면 설정이 무시된 채로
    실험이 돌아가고, 그 결과를 나중에 비교할 수 없게 된다.
    """
    ok = accepted_kwargs(cls)
    kept = {k: v for k, v in kwargs.items() if k in ok}
    dropped = sorted(set(kwargs) - ok)
    if dropped and verbose:
        print(f"  [주의] {name} 이 받지 않는 인자를 무시합니다: {dropped}")
        print(f"         (이 백본의 인자: {sorted(ok - {'n_chans', 'n_outputs', 'n_times'})})")
    return kept


def build_backbone(
    name: str,
    n_chans: int,
    n_times: int,
    embed_dim: int = 64,
    sfreq: float | None = None,
    strict_kwargs: bool = False,
    **kwargs,
) -> nn.Module:
    """Braindecode 모델을 (B, C, T) -> (B, embed_dim) 모듈로 만든다.

    sfreq 는 주파수를 직접 다루는 백본에만 전달된다. SincShallowNet(학습되는
    sinc 대역통과)과 FBCNet(필터뱅크)은 Hz 단위로 필터를 만들기 때문에
    샘플링 주파수를 모르면 생성 자체가 안 된다.

    strict_kwargs=True 면 백본이 받지 않는 인자가 있을 때 그냥 터뜨린다.
    (단일 모델을 튜닝할 때는 이쪽이 안전하다)
    """
    try:
        import braindecode.models as bdm
    except ImportError as e:  # pragma: no cover
        raise ImportError("braindecode 가 필요합니다: pip install braindecode") from e

    cls = getattr(bdm, _resolve(bdm, name))
    if sfreq is not None and "sfreq" not in kwargs and "sfreq" in accepted_kwargs(cls):
        kwargs["sfreq"] = float(sfreq)
    if not strict_kwargs:
        kwargs = filter_kwargs(cls, kwargs, name=name)
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
