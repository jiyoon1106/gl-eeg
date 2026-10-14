"""신호 전처리 — 리샘플 · 재참조 · 필터.

ds004504 실측 사양
------------------
    샘플링  500 Hz
    채널    19개, 10-20 표준 이름 그대로 (별칭 변환 불필요)
    참조    A1 A2 (양쪽 귀)
    필터    0.4~50 Hz 가 이미 적용됨  →  델타(0.5~4Hz)가 살아 있다
    길이    599.8 초 (약 10분)

500 → 250 Hz 리샘플은 안전하다. 신호가 50Hz 이하인데 250Hz 의 나이퀴스트가
125Hz 라 에일리어싱이 생기지 않는다.
"""

from __future__ import annotations

import numpy as np

# MNE 는 데이터를 볼트로 돌려준다. 마이크로볼트로 맞춘다.
V_TO_UV = 1e6


def preprocess_raw(
    raw,
    channels: list[str],
    sfreq_target: float = 250.0,
    reference: str | None = "average",
    bandpass: tuple[float | None, float | None] | None = None,
    verbose: bool = False,
):
    """Raw 를 표준 사양으로 맞춘다. 원본을 건드리지 않고 복사본을 반환한다.

    Parameters
    ----------
    channels : 사용할 채널. montage.resolve_channels() 가 정한 순서 그대로
               적용되므로, 공간 필터 가중치 매핑이 학습·추론 간에 일치한다.
    reference : "average" | None
                ds004504 은 A1 A2 참조다. 데이터셋마다 참조가 다르므로
                average 로 통일해야 합쳐 쓸 수 있다.
    bandpass : (저주파, 고주파). None 이면 건드리지 않는다.
               ds004504 은 이미 0.4~50Hz 라 기본은 None.
    """
    import mne

    v = "error" if not verbose else "info"
    raw = raw.copy()

    # ① 채널 선택 + 순서 고정  ← 반드시 리샘플보다 먼저 (불필요한 연산 절약)
    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise ValueError(f"{missing} 채널이 없습니다. 보유: {raw.ch_names}")
    raw.pick(channels)
    raw.reorder_channels(channels)

    raw.load_data(verbose=v)

    # ② 밴드패스 (요청한 경우만)
    if bandpass is not None:
        lo, hi = bandpass
        raw.filter(l_freq=lo, h_freq=hi, verbose=v)

    # ③ 재참조
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose=v)
    elif reference is not None:
        raise ValueError(f"지원하지 않는 참조: {reference!r}")

    # ④ 리샘플
    if sfreq_target and abs(raw.info["sfreq"] - sfreq_target) > 1e-6:
        raw.resample(sfreq_target, verbose=v)

    return raw


def get_data_uv(raw) -> np.ndarray:
    """(n_channels, n_times) float32, 단위는 마이크로볼트."""
    return (raw.get_data() * V_TO_UV).astype(np.float32)


def signal_quality(x: np.ndarray, clip_uv: float = 150.0) -> float:
    """신호 품질 점수 0~1 — 융합 가중치에 쓸 값.

    모델이 학습하는 값이 아니다. 전처리에서 계산해 ModelOutput.quality 로
    넘긴다. 지금은 "진폭이 생리적 범위를 벗어난 샘플의 비율"로 단순하게 잡고,
    필요하면 아티팩트 검출을 추가한다.

    Parameters
    ----------
    clip_uv : 이 값을 넘으면 아티팩트로 본다. 두피 EEG 는 보통 ±100 µV 안쪽.
    """
    bad = np.mean(np.abs(x) > clip_uv)
    return float(1.0 - bad)
