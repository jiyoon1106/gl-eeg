"""epoch 분할 + 정규화.

epoch 예산
----------
ds004504 은 피험자당 599.8초다. 4초로 자르면 겹침 없이 약 150개, 50% 겹치면
약 299개가 나온다. 88명이면 13,200 ~ 26,300 epoch.

epoch 을 늘려도 독립적인 정보는 늘지 않는다. 실제 표본은 88명이고 epoch 증가는
증강 효과일 뿐이다. 그리고 Drive 로 팀에 공유해야 하므로 용량이 중요하다.

    88명 x 100 epoch x 19ch x 1000샘플 x 2바이트(float16)  ~=  334 MB

그래서 기본값은 겹침 없음 + 피험자당 100개 상한이다.
"""

from __future__ import annotations

import numpy as np


def make_epochs(
    x: np.ndarray,
    sfreq: float,
    epoch_seconds: float = 4.0,
    overlap: float = 0.0,
    max_epochs: int | None = 100,
    drop_last: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """(n_channels, n_times) -> (n_epochs, n_channels, n_epoch_times)

    Parameters
    ----------
    overlap : 0.0 = 겹침 없음, 0.5 = 50% 겹침
    max_epochs : 피험자당 상한. 넘으면 **전체 구간에서 고르게** 솎아낸다
                 (앞부분만 쓰면 녹화 초반 아티팩트에 편향된다).
    """
    if x.ndim != 2:
        raise ValueError(f"(n_channels, n_times) 를 기대합니다: {x.shape}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap 은 0 이상 1 미만: {overlap}")

    n_ch, n_times = x.shape
    win = int(round(epoch_seconds * sfreq))
    step = max(1, int(round(win * (1.0 - overlap))))

    if n_times < win:
        raise ValueError(
            f"녹화가 epoch 보다 짧습니다: {n_times/sfreq:.1f}s < {epoch_seconds}s"
        )

    starts = np.arange(0, n_times - win + 1, step)
    if not drop_last and starts[-1] + win < n_times:
        starts = np.append(starts, n_times - win)

    if max_epochs is not None and len(starts) > max_epochs:
        # 균등 솎아내기 — 녹화 전체에서 고르게 뽑는다
        idx = np.linspace(0, len(starts) - 1, max_epochs).round().astype(int)
        starts = starts[np.unique(idx)]

    return np.stack([x[:, s : s + win] for s in starts]).astype(np.float32)


def robust_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """채널별 median / IQR 정규화. 이상치에 강하다.

    피험자 **전체 녹화**를 기준으로 계산해 모든 epoch 에 같은 값을 적용한다.
    epoch 별로 하면 epoch 간 상대적 진폭 차이가 사라진다.

    주의: 피험자별 정규화는 전극 임피던스 차이를 없애주지만, 질환에 따른
    절대 파워 차이도 함께 지운다. 그래서 scale_only 도 선택지로 둔다.
    """
    med = np.median(x, axis=-1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=-1, keepdims=True)
    iqr = q75 - q25
    return ((x - med) / (iqr + eps)).astype(np.float32)


def normalize(x: np.ndarray, mode: str = "robust_zscore") -> np.ndarray:
    """(n_channels, n_times) 를 정규화. epoch 분할 **전에** 적용한다.

    mode
    ----
    robust_zscore : 채널별 median/IQR  (기본)
    scale_only    : 100 µV 로 나누기만 — 절대 파워 정보 보존
    none          : 그대로
    """
    if mode == "robust_zscore":
        return robust_zscore(x)
    if mode == "scale_only":
        return (x / 100.0).astype(np.float32)
    if mode in ("none", None):
        return x.astype(np.float32)
    raise ValueError(f"알 수 없는 정규화 방식: {mode!r}")


def epoch_subject(
    x_uv: np.ndarray,
    sfreq: float,
    epoch_seconds: float = 4.0,
    overlap: float = 0.0,
    max_epochs: int | None = 100,
    normalize_mode: str = "robust_zscore",
) -> np.ndarray:
    """한 피험자의 연속 신호 -> 정규화된 epoch 묶음.

    순서가 중요하다: **정규화를 먼저, epoch 분할을 나중에.**
    """
    x = normalize(x_uv, normalize_mode)
    return make_epochs(
        x, sfreq, epoch_seconds=epoch_seconds, overlap=overlap, max_epochs=max_epochs
    )
