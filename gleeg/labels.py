"""라벨 변환 — 원점수를 단계로 바꾸는 유일한 지점.

중증도 단계를 바꾸고 싶으면 config/exp/*.yaml 의 severity_bins 만 고친다.
이 파일의 함수는 그 설정을 받아 동작할 뿐이므로 수정할 필요가 없다.
"""

from __future__ import annotations

import numpy as np

# ds004504 의 Group 컬럼 → 표준 질환 코드
#   participants.json 기준:
#     A = Alzheimer Disease Group
#     F = Frontotemporal Dementia Group
#     C = Healthy Group
DS004504_GROUP_MAP = {"A": "AD", "F": "FTD", "C": "CN"}


def disease_to_index(disease: str, classes: list[str]) -> int:
    """질환 코드 → 정수 인덱스. classes 순서가 모델 출력 순서를 정한다."""
    try:
        return classes.index(disease)
    except ValueError as e:
        raise ValueError(f"{disease!r} 는 이 실험의 분류 대상이 아닙니다: {classes}") from e


def to_level(score: float | None, scale: str | None, bins: dict) -> int | None:
    """원점수 → 중증도 단계.

    Parameters
    ----------
    score : 원점수 (예: MMSE 22). None 이면 None 을 반환한다.
    scale : 척도명 (예: "MMSE"). bins["scale"] 과 같아야 한다.
    bins  : config 의 severity_bins
            {"scale": "MMSE", "edges": [20, 28],
             "labels": ["moderate_or_worse", "mild", "normal"]}

    Returns
    -------
    0 .. len(labels)-1 범위의 정수. 작을수록 중증.

    Examples
    --------
    >>> bins = {"scale": "MMSE", "edges": [20, 28],
    ...         "labels": ["moderate_or_worse", "mild", "normal"]}
    >>> to_level(16, "MMSE", bins), to_level(22, "MMSE", bins), to_level(30, "MMSE", bins)
    (0, 1, 2)
    """
    if score is None:
        return None

    expected = bins["scale"]
    if scale != expected:
        raise ValueError(
            f"척도 불일치: 데이터는 {scale!r}, 설정은 {expected!r}. "
            "척도가 다른 데이터셋을 섞을 때는 같은 척도끼리만 구간화할 것."
        )

    edges = bins["edges"]
    n_levels = len(bins["labels"])
    if len(edges) != n_levels - 1:
        raise ValueError(
            f"edges 는 labels 보다 하나 적어야 합니다: "
            f"edges={len(edges)}, labels={n_levels}"
        )

    return int(np.digitize(score, edges))


def n_levels(bins: dict) -> int:
    """설정이 정의한 단계 수. CORAL 헤드의 출력 차원 = n_levels - 1."""
    return len(bins["labels"])


def level_name(level: int, bins: dict) -> str:
    return bins["labels"][level]


def describe_bins(bins: dict) -> str:
    """구간을 사람이 읽을 수 있게 — 보고서·로그용."""
    edges, labels = bins["edges"], bins["labels"]
    parts = [f"{labels[0]}: {bins['scale']} < {edges[0]}"]
    for i in range(1, len(edges)):
        parts.append(f"{labels[i]}: {edges[i-1]} <= {bins['scale']} < {edges[i]}")
    parts.append(f"{labels[-1]}: {bins['scale']} >= {edges[-1]}")
    return " | ".join(parts)
