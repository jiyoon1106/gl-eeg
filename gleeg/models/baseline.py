"""고전 베이스라인 — 밴드파워 + 부스팅 트리.

신경망이 아니라 **비교 기준점**이다. 최종 제출 모델이 아니다.
비교 대상 없는 정확도 수치는 해석할 수 없기 때문에 넣는다.

설령 이쪽이 더 높게 나와도 제출 모델은 신경망으로 가고, 그 격차 자체를
"소규모 임상 데이터에서 딥러닝의 한계"로 분석해 보고서에 싣는다.

구현에 sklearn 만 쓴다 (xgboost 추가 설치 불필요).
"""

from __future__ import annotations

import numpy as np

# 임상에서 쓰는 EEG 대역. 치매 지표는 서파(delta/theta) 증가와 알파 감소다.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpower_features(
    X: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    relative: bool = True,
) -> np.ndarray:
    """(N, C, T) -> (N, C * n_bands * 2) 특징.

    채널별 대역 파워 + theta/alpha 비율을 뽑는다. ShallowFBCSPNet 이 학습으로
    찾아내는 것을 손으로 계산하는 셈이다.

    Parameters
    ----------
    relative : True 면 전체 파워로 나눈 상대 파워. 피험자 간 진폭 차이에 강하다.
    """
    from scipy.signal import welch

    bands = bands or BANDS
    N, C, T = X.shape
    nperseg = min(T, int(sfreq * 2))          # 2초 창 -> 0.5Hz 분해능

    freqs, psd = welch(X.astype(np.float64), fs=sfreq, nperseg=nperseg, axis=-1)

    powers = []
    for lo, hi in bands.values():
        m = (freqs >= lo) & (freqs < hi)
        powers.append(psd[:, :, m].sum(axis=-1))        # (N, C)
    P = np.stack(powers, axis=-1)                        # (N, C, n_bands)

    total = P.sum(axis=-1, keepdims=True) + 1e-12
    feats = [P / total if relative else np.log(P + 1e-12)]

    # theta/alpha 비율 — 인지 저하의 대표 지표
    names = list(bands)
    if "theta" in names and "alpha" in names:
        th, al = P[:, :, names.index("theta")], P[:, :, names.index("alpha")]
        feats.append((th / (al + 1e-12))[:, :, None])

    # 로그 전체 파워 — 상대 파워만 쓰면 사라지는 정보
    feats.append(np.log(total + 1e-12))

    return np.concatenate(feats, axis=-1).reshape(N, -1).astype(np.float32)


def make_classifier(seed: int = 42):
    """부스팅 트리. 특징 수가 100~200개라 트리 계열이 잘 맞는다."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def make_regressor(seed: int = 42):
    """중증도 원점수 회귀 -> 구간화. 순서형 출력을 고전 방식으로 흉내낸다."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
